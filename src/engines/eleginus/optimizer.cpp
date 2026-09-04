#include "eleginus/formula.hpp"
#include "eleginus/game.hpp"
#include <algorithm>
#include <atomic>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <hdf5.h>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <torch/torch.h>
#include <vector>

namespace {
	struct Options {
		std::filesystem::path data;
		std::filesystem::path output;
		std::string device = "auto";
		std::size_t batch = 4096;
		std::size_t chunk = 16384;
		std::size_t workers = 1;
		std::uint64_t maxSteps = 0;
		std::uint64_t logEvery = 50;
		std::uint64_t saveEvery = 0;
		std::uint64_t seed = 2026;
		int epochs = 3;
		int validationPermille = 10;
		double learningRate = 1.0e-3;
		double weightDecay = 0.0;
		double gradClip = 1.0;
		double cpScale = 150.0;
		double cpClip = 2000.0;
		double huberDelta = 100.0;
	};

	struct Chunk {
		std::uint64_t first = 0;
		std::vector<eleginus::PackedBoard> boards;
		std::vector<std::int32_t> scores;
	};

	struct Stats {
		struct Snapshot {
			double loss = 0.0;
			double mae = 0.0;
		};

		torch::Tensor sums;
		std::uint64_t count = 0;

		void add(const torch::Tensor &errors, const torch::Tensor &losses) {
			if (errors.numel() == 0) return;
			const auto next = torch::stack({losses.detach().sum(), errors.detach().abs().sum()});
			if (sums.defined()) sums += next;
			else sums = next;
			count += static_cast<std::uint64_t>(errors.numel());
		}

		Snapshot snapshot() const {
			if (count == 0) return {};
			const auto host = sums.to(torch::kCPU);
			const auto values = host.accessor<float, 1>();
			return {values[0] / static_cast<double>(count), values[1] / static_cast<double>(count)};
		}
	};

	class H5Reader {
	public:
		explicit H5Reader(const std::filesystem::path &path)
			: file(check(H5Fopen(path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), "open HDF5 file")),
			  states(check(H5Dopen2(file, "states", H5P_DEFAULT), "open states")),
			  centipawns(check(H5Dopen2(file, "centipawns", H5P_DEFAULT), "open centipawns")) {
			const auto stateShape = shape(states);
			const auto scoreShape = shape(centipawns);
			if (stateShape.size() != 2 || stateShape[1] != eleginus::packedBoardSize) throw std::runtime_error("states have an incompatible packed-board width");
			if (scoreShape.size() != 1 || scoreShape[0] != stateShape[0]) throw std::runtime_error("centipawns must align with states");
			rows = stateShape[0];
		}

		~H5Reader() {
			if (states >= 0) H5Dclose(states);
			if (centipawns >= 0) H5Dclose(centipawns);
			if (file >= 0) H5Fclose(file);
		}

		std::uint64_t size() const noexcept { return rows; }

		Chunk read(std::uint64_t first, std::size_t requested) const {
			const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(requested, rows - first));
			Chunk result;
			result.first = first;
			result.boards.resize(count);
			result.scores.resize(count);
			readSlice(states, H5T_NATIVE_UINT8, result.boards.data(), {first, 0}, {count, eleginus::packedBoardSize});
			readSlice(centipawns, H5T_NATIVE_INT32, result.scores.data(), {first}, {count});
			return result;
		}

	private:
		static hid_t check(hid_t value, std::string_view action) {
			if (value < 0) throw std::runtime_error(std::string(action) + " failed");
			return value;
		}

		static std::vector<hsize_t> shape(hid_t dataset) {
			const hid_t space = check(H5Dget_space(dataset), "open dataset space");
			const int rank = H5Sget_simple_extent_ndims(space);
			if (rank < 0) {
				H5Sclose(space);
				throw std::runtime_error("read dataset rank failed");
			}
			std::vector<hsize_t> dimensions(static_cast<std::size_t>(rank));
			const int status = H5Sget_simple_extent_dims(space, dimensions.data(), nullptr);
			H5Sclose(space);
			if (status < 0) throw std::runtime_error("read dataset shape failed");
			return dimensions;
		}

		static void readSlice(hid_t dataset, hid_t type, void *data, const std::vector<hsize_t> &start, const std::vector<hsize_t> &count) {
			const hid_t fileSpace = check(H5Dget_space(dataset), "open dataset space");
			if (H5Sselect_hyperslab(fileSpace, H5S_SELECT_SET, start.data(), nullptr, count.data(), nullptr) < 0) {
				H5Sclose(fileSpace);
				throw std::runtime_error("select dataset slice failed");
			}
			const hid_t memorySpace = check(H5Screate_simple(static_cast<int>(count.size()), count.data(), nullptr), "create memory space");
			const herr_t status = H5Dread(dataset, type, memorySpace, fileSpace, H5P_DEFAULT, data);
			H5Sclose(memorySpace);
			H5Sclose(fileSpace);
			if (status < 0) throw std::runtime_error("read dataset slice failed");
		}

		hid_t file = -1;
		hid_t states = -1;
		hid_t centipawns = -1;
		std::uint64_t rows = 0;
	};

	double number(std::string_view text, std::string_view name) {
		std::string value(text);
		std::size_t used = 0;
		const double result = std::stod(value, &used);
		if (used != value.size() || !std::isfinite(result)) throw std::invalid_argument("invalid " + std::string(name));
		return result;
	}

	std::uint64_t unsignedValue(std::string_view text, std::string_view name) {
		std::uint64_t value = 0;
		const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
		if (error != std::errc{} || end != text.data() + text.size()) throw std::invalid_argument("invalid " + std::string(name));
		return value;
	}

	Options parse(int argc, char **argv) {
		Options options;
		for (int i = 1; i < argc; ++i) {
			const std::string_view key = argv[i];
			auto next = [&]() -> std::string_view {
				if (++i >= argc) throw std::invalid_argument("missing value after " + std::string(key));
				return argv[i];
			};
			if (key == "--data") options.data = next();
			else if (key == "--out") options.output = next();
			else if (key == "--device") options.device = next();
			else if (key == "--epochs") options.epochs = static_cast<int>(unsignedValue(next(), key));
			else if (key == "--batch-size") options.batch = static_cast<std::size_t>(unsignedValue(next(), key));
			else if (key == "--chunk-rows") options.chunk = static_cast<std::size_t>(unsignedValue(next(), key));
			else if (key == "--workers") options.workers = static_cast<std::size_t>(unsignedValue(next(), key));
			else if (key == "--max-steps") options.maxSteps = unsignedValue(next(), key);
			else if (key == "--log-every") options.logEvery = unsignedValue(next(), key);
			else if (key == "--save-every") options.saveEvery = unsignedValue(next(), key);
			else if (key == "--seed") options.seed = unsignedValue(next(), key);
			else if (key == "--validation-permille") options.validationPermille = static_cast<int>(unsignedValue(next(), key));
			else if (key == "--lr") options.learningRate = number(next(), key);
			else if (key == "--weight-decay") options.weightDecay = number(next(), key);
			else if (key == "--grad-clip") options.gradClip = number(next(), key);
			else if (key == "--cp-scale") options.cpScale = number(next(), key);
			else if (key == "--cp-clip") options.cpClip = number(next(), key);
			else if (key == "--huber-delta") options.huberDelta = number(next(), key);
			else if (key == "--help") {
				std::cout << "Usage: optimizer --data <positions.eleginus.h5> --out <weights.tsv> [options]\n"
					<< "  --device <auto|cpu|cuda> --epochs <n> --batch-size <n> --chunk-rows <n> --workers <n>\n"
					<< "  --max-steps <n|0=all> --lr <x> --weight-decay <x> --grad-clip <x>\n"
					<< "  --cp-scale <x> --cp-clip <x> --huber-delta <x> --validation-permille <0..999>\n"
					<< "  --save-every <n> --log-every <n> --seed <n>\n";
				std::exit(0);
			} else {
				throw std::invalid_argument("unknown option: " + std::string(key));
			}
		}
		if (options.data.empty() || options.output.empty()) throw std::invalid_argument("--data and --out are required");
		if (options.epochs <= 0 || options.batch == 0 || options.chunk == 0 || options.workers == 0) throw std::invalid_argument("epochs, batch, chunk and workers must be positive");
		if (options.validationPermille < 0 || options.validationPermille >= 1000) throw std::invalid_argument("--validation-permille must be in [0, 999]");
		if (options.learningRate <= 0.0 || options.weightDecay < 0.0 || options.gradClip < 0.0 || options.cpScale <= 0.0 || options.cpClip <= 0.0 ||
			options.huberDelta <= 0.0) {
			throw std::invalid_argument("optimizer scalar options are outside their valid range");
		}
		return options;
	}

	torch::Device resolveDevice(const std::string &requested) {
		if (requested == "auto") return torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
		if (requested == "cuda") {
			if (!torch::cuda::is_available()) throw std::runtime_error("CUDA optimizer was requested but CUDA is unavailable");
			return torch::Device(torch::kCUDA);
		}
		if (requested == "cpu") return torch::Device(torch::kCPU);
		throw std::invalid_argument("--device must be auto, cpu or cuda");
	}

	std::uint64_t mix(std::uint64_t value) noexcept {
		value += 0x9e3779b97f4a7c15ULL;
		value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
		value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
		return value ^ (value >> 31);
	}

	void save(const std::filesystem::path &path, const torch::Tensor &base, const torch::Tensor &material, const torch::Tensor &pressureCenter,
		const torch::Tensor &pressureWidth, const torch::Tensor &winnable, const torch::Tensor &scale) {
		if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
		const auto hostBase = base.detach().to(torch::kCPU).contiguous();
		const auto hostMaterial = material.detach().to(torch::kCPU).contiguous();
		const auto baseView = hostBase.accessor<float, 1>();
		const auto materialView = hostMaterial.accessor<float, 2>();
		const auto temporary = path.string() + ".tmp";
		std::ofstream output(temporary, std::ios::trunc);
		if (!output) throw std::runtime_error("cannot create optimizer output: " + temporary);
		output.precision(9);
		output << "index\tbase\tpawn\tknight\tbishop\trook\tqueen\n";
		for (std::int64_t formula = 0; formula < hostBase.size(0); ++formula) {
			output << formula << '\t' << baseView[formula];
			for (std::int64_t type = 0; type < hostMaterial.size(1); ++type) output << '\t' << materialView[formula][type];
			output << '\n';
		}
		const auto hostWin = winnable.detach().to(torch::kCPU).contiguous();
		const auto hostScale = scale.detach().to(torch::kCPU).contiguous();
		const float hostCenter = pressureCenter.detach().to(torch::kCPU).item<float>();
		const float hostWidth = pressureWidth.detach().to(torch::kCPU).item<float>();
		output << "globals\npressure_center\t" << hostCenter << "\npressure_width\t" << hostWidth << '\n';
		constexpr std::array<std::string_view, 7> winNames{{"win_pawns", "win_symmetric", "win_asymmetric", "win_pawn_ending", "win_strong_pawns",
			"win_opposite_passers", "win_bias"}};
		constexpr std::array<std::string_view, 5> scaleNames{{"scale_pawnless", "scale_pure_opposite", "scale_mixed_opposite", "scale_passer_step",
			"scale_pawn_step"}};
		for (std::size_t i = 0; i < winNames.size(); ++i) output << winNames[i] << '\t' << hostWin[static_cast<std::int64_t>(i)].item<float>() << '\n';
		for (std::size_t i = 0; i < scaleNames.size(); ++i) output << scaleNames[i] << '\t' << hostScale[static_cast<std::int64_t>(i)].item<float>() << '\n';
		output.close();
		if (!output) throw std::runtime_error("write optimizer output failed");
		std::filesystem::remove(path);
		std::filesystem::rename(temporary, path);
	}

	struct Batch {
		std::vector<std::int32_t> signals;
		std::vector<float> material;
		std::vector<float> pressure;
		std::vector<float> winnable;
		std::vector<float> scale;
		std::vector<float> targets;
		std::vector<std::int64_t> training;
		std::vector<std::int64_t> validation;
		std::size_t pressureIndex = 0;
	};

	Batch extract(const Chunk &chunk, std::span<const std::size_t> order, std::size_t first, std::size_t count, std::size_t formulas, const Options &options) {
		Batch result;
		result.signals.resize(count * formulas);
		result.material.resize(count * 5);
		result.pressure.resize(count * 2);
		result.winnable.resize(count * 2 * 7);
		result.scale.resize(count * 2 * 5);
		result.targets.resize(count);
		std::atomic<std::size_t> next{0};
		std::atomic<std::size_t> pressureIndex{std::numeric_limits<std::size_t>::max()};
		std::exception_ptr failure;
		std::atomic<bool> failed{false};
		auto work = [&] {
			try {
				for (;;) {
					const std::size_t local = next.fetch_add(1, std::memory_order_relaxed);
					if (local >= count || failed.load(std::memory_order_relaxed)) break;
					const std::size_t row = order[first + local];
					eleginus::FormulaContext context;
					std::span<std::int32_t> signals(result.signals.data() + local * formulas, formulas);
					eleginus::FormulaSet::features(eleginus::unpackBoard(chunk.boards[row]), signals, context);
					std::copy(context.material.begin(), context.material.end(), result.material.begin() + local * 5);
					std::copy(context.pressure.begin(), context.pressure.end(), result.pressure.begin() + local * 2);
					for (std::size_t branch = 0; branch < 2; ++branch) {
						std::copy(context.winnable[branch].begin(), context.winnable[branch].end(), result.winnable.begin() + local * 14 + branch * 7);
						std::copy(context.scale[branch].begin(), context.scale[branch].end(), result.scale.begin() + local * 10 + branch * 5);
					}
					auto expected = std::numeric_limits<std::size_t>::max();
					if (!pressureIndex.compare_exchange_strong(expected, context.pressureIndex) && expected != context.pressureIndex) {
						throw std::runtime_error("king-pressure formula index changed between positions");
					}
					result.targets[local] = static_cast<float>(std::clamp<double>(chunk.scores[row], -options.cpClip, options.cpClip));
				}
			} catch (...) {
				if (!failed.exchange(true)) failure = std::current_exception();
			}
		};

		const std::size_t active = std::min(options.workers, count);
		std::vector<std::thread> threads;
		threads.reserve(active > 0 ? active - 1 : 0);
		for (std::size_t i = 1; i < active; ++i) threads.emplace_back(work);
		work();
		for (auto &thread : threads) thread.join();
		if (failure) std::rethrow_exception(failure);
		result.pressureIndex = pressureIndex.load();
		if (result.pressureIndex >= formulas) throw std::runtime_error("king-pressure formula index was not recorded");

		for (std::size_t local = 0; local < count; ++local) {
			const std::uint64_t row = chunk.first + order[first + local];
			auto &destination = static_cast<int>(mix(row ^ options.seed) % 1000) < options.validationPermille ? result.validation : result.training;
			destination.push_back(static_cast<std::int64_t>(local));
		}
		return result;
	}

	torch::Tensor forward(const torch::Tensor &signals, const torch::Tensor &coordinates, const torch::Tensor &pressure, const torch::Tensor &winContext,
		const torch::Tensor &scaleContext, std::size_t pressureIndex, const torch::Tensor &base, const torch::Tensor &material,
		const torch::Tensor &pressureCenter, const torch::Tensor &pressureWidth, const torch::Tensor &winnable, const torch::Tensor &scale) {
		auto score = torch::matmul(signals, base) + (torch::matmul(signals, material) * coordinates).sum(1);
		const auto width = pressureWidth.clamp_min(0.05);
		const auto boundedPressure = pressure.clamp(0, 64);
		auto pressureSignal = 4096.0 * (torch::sigmoid((boundedPressure.select(1, 0) - pressureCenter) / width) -
			torch::sigmoid((boundedPressure.select(1, 1) - pressureCenter) / width));
		pressureSignal = pressureSignal + (pressureSignal.round() - pressureSignal).detach();
		const auto index = static_cast<std::int64_t>(pressureIndex);
		const auto pressureWeight = base[index] + (material[index] * coordinates).sum(1);
		score += (pressureSignal - signals.select(1, index)) * pressureWeight;

		const auto positive = score.ge(0).unsqueeze(1);
		const auto winFacts = torch::where(positive, winContext.select(1, 0), winContext.select(1, 1));
		const auto winValue = (winFacts * winnable).sum(1);
		const auto adjusted = torch::where(score.gt(0), (score + winValue).clamp_min(0), torch::where(score.lt(0), (score - winValue).clamp_max(0), score));

		const auto scaleFacts = torch::where(positive, scaleContext.select(1, 0), scaleContext.select(1, 1));
		auto factor = torch::where(scaleFacts.select(1, 0).ne(0), scale[0], torch::ones_like(score));
		const auto pure = scale[1] + scale[4] * scaleFacts.select(1, 3) + scale[3] * scaleFacts.select(1, 4);
		const auto mixed = scale[2] + scale[3] * scaleFacts.select(1, 4);
		const auto bishop = torch::where(scaleFacts.select(1, 1).ne(0), pure,
			torch::where(scaleFacts.select(1, 2).ne(0), mixed, torch::ones_like(score)));
		factor = torch::minimum(factor, bishop).clamp(0, 1);
		return adjusted * factor;
	}

	torch::Tensor huber(const torch::Tensor &errors, double delta) {
		const auto magnitude = errors.abs();
		return torch::where(magnitude <= delta, 0.5 * errors.square() / delta, magnitude - 0.5 * delta);
	}

	void projectGlobals(torch::Tensor &pressureCenter, torch::Tensor &pressureWidth, torch::Tensor &scale) {
		torch::NoGradGuard guard;
		pressureCenter.clamp_(0.0, 64.0);
		pressureWidth.clamp_(0.05, 32.0);
		scale.clamp_(0.0, 1.0);
	}
} // namespace

int main(int argc, char **argv) {
	try {
		const Options options = parse(argc, argv);
		const H5Reader data(options.data);
		if (data.size() == 0) throw std::runtime_error("optimizer dataset is empty");
		const auto compute = resolveDevice(options.device);
		torch::manual_seed(static_cast<std::int64_t>(options.seed));

		const auto initial = eleginus::FormulaSet::parameters();
		const auto globalInitial = eleginus::FormulaSet::globals();
		std::vector<float> baseValues(initial.size());
		std::vector<float> materialValues(initial.size() * 5);
		for (std::size_t formula = 0; formula < initial.size(); ++formula) {
			baseValues[formula] = initial[formula].base;
			std::copy(initial[formula].material.begin(), initial[formula].material.end(), materialValues.begin() + formula * 5);
		}
		auto base = torch::from_blob(baseValues.data(), {static_cast<std::int64_t>(initial.size())}, torch::kFloat32).clone().to(compute).set_requires_grad(true);
		auto material = torch::from_blob(materialValues.data(), {static_cast<std::int64_t>(initial.size()), 5}, torch::kFloat32).clone().to(compute).set_requires_grad(true);
		auto pressureCenter = torch::tensor(globalInitial.pressureCenter, torch::TensorOptions().dtype(torch::kFloat32).device(compute)).set_requires_grad(true);
		auto pressureWidth = torch::tensor(globalInitial.pressureWidth, torch::TensorOptions().dtype(torch::kFloat32).device(compute)).set_requires_grad(true);
		auto winValues = globalInitial.winnable;
		auto scaleValues = globalInitial.scale;
		auto winnable = torch::from_blob(winValues.data(), {7}, torch::kFloat32).clone().to(compute).set_requires_grad(true);
		auto scale = torch::from_blob(scaleValues.data(), {5}, torch::kFloat32).clone().to(compute).set_requires_grad(true);
		std::vector<torch::Tensor> trainable{base, material, pressureCenter, pressureWidth, winnable, scale};
		torch::optim::AdamW optimizer(trainable, torch::optim::AdamWOptions(options.learningRate).weight_decay(options.weightDecay));
		std::uint64_t step = 0;
		bool stopped = false;

		std::cout << "optimizer start: data=" << options.data.string() << " out=" << options.output.string() << " rows=" << data.size()
			<< " formulas=" << initial.size() << " parameters=" << initial.size() * 6 + 14 << " device=" << compute.str()
			<< " epochs=" << options.epochs << " batch_size=" << options.batch << " workers=" << options.workers << '\n';

		for (int epoch = 0; epoch < options.epochs && !stopped; ++epoch) {
			const std::size_t chunks = static_cast<std::size_t>((data.size() + options.chunk - 1) / options.chunk);
			std::vector<std::size_t> chunkOrder(chunks);
			std::iota(chunkOrder.begin(), chunkOrder.end(), 0);
			std::mt19937_64 random(options.seed + static_cast<std::uint64_t>(epoch));
			std::shuffle(chunkOrder.begin(), chunkOrder.end(), random);
			Stats training;
			Stats validation;

			for (std::size_t chunkIndex : chunkOrder) {
				const std::uint64_t firstRow = static_cast<std::uint64_t>(chunkIndex) * options.chunk;
				const Chunk chunk = data.read(firstRow, options.chunk);
				std::vector<std::size_t> order(chunk.boards.size());
				std::iota(order.begin(), order.end(), 0);
				std::shuffle(order.begin(), order.end(), random);

				for (std::size_t first = 0; first < order.size(); first += options.batch) {
					const std::size_t count = std::min(options.batch, order.size() - first);
					Batch batch = extract(chunk, order, first, count, initial.size(), options);
					auto signals = torch::from_blob(batch.signals.data(), {static_cast<std::int64_t>(count), static_cast<std::int64_t>(initial.size())}, torch::kInt32)
						.to(torch::TensorOptions().device(compute).dtype(torch::kFloat32));
					auto coordinates = torch::from_blob(batch.material.data(), {static_cast<std::int64_t>(count), 5}, torch::kFloat32).to(compute);
					auto pressure = torch::from_blob(batch.pressure.data(), {static_cast<std::int64_t>(count), 2}, torch::kFloat32).to(compute);
					auto winContext = torch::from_blob(batch.winnable.data(), {static_cast<std::int64_t>(count), 2, 7}, torch::kFloat32).to(compute);
					auto scaleContext = torch::from_blob(batch.scale.data(), {static_cast<std::int64_t>(count), 2, 5}, torch::kFloat32).to(compute);
					auto targets = torch::from_blob(batch.targets.data(), {static_cast<std::int64_t>(count)}, torch::kFloat32).to(compute);
					auto estimates = options.cpScale * forward(signals, coordinates, pressure, winContext, scaleContext, batch.pressureIndex, base, material,
						pressureCenter, pressureWidth, winnable, scale);

					if (!batch.validation.empty()) {
						torch::NoGradGuard guard;
						auto index = torch::from_blob(batch.validation.data(), {static_cast<std::int64_t>(batch.validation.size())}, torch::kInt64).to(compute);
						auto errors = estimates.index_select(0, index) - targets.index_select(0, index);
						validation.add(errors, huber(errors, options.huberDelta));
					}
					if (batch.training.empty()) continue;

					auto index = torch::from_blob(batch.training.data(), {static_cast<std::int64_t>(batch.training.size())}, torch::kInt64).to(compute);
					auto errors = estimates.index_select(0, index) - targets.index_select(0, index);
					auto losses = huber(errors, options.huberDelta);
					training.add(errors, losses);
					optimizer.zero_grad();
					losses.mean().backward();
					if (options.gradClip > 0.0) torch::nn::utils::clip_grad_norm_(trainable, options.gradClip);
					optimizer.step();
					projectGlobals(pressureCenter, pressureWidth, scale);
					++step;

					if (options.logEvery != 0 && step % options.logEvery == 0) {
						const auto trainStats = training.snapshot();
						const auto validationStats = validation.snapshot();
						std::cout << "optimizer step: epoch=" << epoch << " step=" << step << " train_huber=" << trainStats.loss
							<< " train_mae_cp=" << trainStats.mae << " validation_huber=" << validationStats.loss
							<< " validation_mae_cp=" << validationStats.mae << '\n';
					}
					if (options.saveEvery != 0 && step % options.saveEvery == 0) {
						save(options.output, base, material, pressureCenter, pressureWidth, winnable, scale);
					}
					if (options.maxSteps != 0 && step >= options.maxSteps) {
						stopped = true;
						break;
					}
				}
				if (stopped) break;
			}
			save(options.output, base, material, pressureCenter, pressureWidth, winnable, scale);
			const auto trainStats = training.snapshot();
			const auto validationStats = validation.snapshot();
			std::cout << "optimizer epoch: epoch=" << epoch << " steps=" << step << " train_huber=" << trainStats.loss
				<< " train_mae_cp=" << trainStats.mae << " validation_huber=" << validationStats.loss
				<< " validation_mae_cp=" << validationStats.mae << '\n';
		}

		save(options.output, base, material, pressureCenter, pressureWidth, winnable, scale);
		std::cout << "optimizer complete: steps=" << step << " output=" << options.output.string() << '\n';
		return 0;
	} catch (const c10::Error &error) {
		std::cerr << "optimizer error: " << error.what_without_backtrace() << '\n';
		return 1;
	} catch (const std::exception &error) {
		std::cerr << "optimizer error: " << error.what() << '\n';
		return 1;
	}
}
