// Trains the complete Eleginus HCN from compact supervised HDF5 positions.

#include "eleginus/model.hpp"
#include <torch/torch.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <future>
#include <hdf5.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
	#define NOMINMAX
	#define WIN32_LEAN_AND_MEAN
	#include <windows.h>
#endif

namespace {

	constexpr hsize_t planeCount = 4;
	std::atomic<bool> halt{false};
	static_assert(std::atomic<bool>::is_always_lock_free);

	bool stopped() noexcept { return halt.load(std::memory_order_relaxed); }
	void interrupt(int) noexcept { halt.store(true, std::memory_order_relaxed); }

#ifdef _WIN32
	BOOL WINAPI console(DWORD event) {
		if (event != CTRL_C_EVENT && event != CTRL_BREAK_EVENT) return FALSE;
		interrupt(0);
		return TRUE;
	}
#endif

	struct Options {
		std::filesystem::path data = "data/eleginus.h5";
		std::filesystem::path out = "models/eleginus/eleginus.pth";
		std::filesystem::path init;
		std::string device = "auto";
		int epochs = 1;
		int batch = 4096;
		int workers = static_cast<int>(std::clamp(std::thread::hardware_concurrency() / 2U, 1U, 8U));
		std::int64_t maxSteps = -1;
		float lr = 1.0e-3F;
		float decay = 1.0e-6F;
		float clip = 1.0F;
		int save = 5000;
		int log = 50;
		std::uint64_t seed = 2026;
	};

	Options parse(int argc, char **argv) {
		Options options;
		for (int i = 1; i < argc; ++i) {
			const std::string key = argv[i];
			if (key == "--help") {
				std::cout << "usage: train --data <eleginus.h5> --out <model.pth> [options]\n";
				std::cout << "  --init <model.pth> --epochs 1 --batch-size 4096 --max-steps -1\n";
				std::cout << "  --device auto --workers N --lr 0.001 --weight-decay 0.000001\n";
				std::cout << "  --grad-clip 1 --save-every 5000 --log-every 50 --seed 2026\n";
				std::exit(0);
			}
			if (++i == argc) throw std::invalid_argument("missing value after " + key);
			const std::string value = argv[i];
			if (key == "--data") options.data = value;
			else if (key == "--out") options.out = value;
			else if (key == "--init") options.init = value;
			else if (key == "--epochs") options.epochs = std::stoi(value);
			else if (key == "--batch-size") options.batch = std::stoi(value);
			else if (key == "--max-steps") options.maxSteps = std::stoll(value);
			else if (key == "--workers") options.workers = std::stoi(value);
			else if (key == "--device") options.device = value;
			else if (key == "--lr") options.lr = std::stof(value);
			else if (key == "--weight-decay") options.decay = std::stof(value);
			else if (key == "--grad-clip") options.clip = std::stof(value);
			else if (key == "--save-every") options.save = std::stoi(value);
			else if (key == "--log-every") options.log = std::stoi(value);
			else if (key == "--seed") options.seed = std::stoull(value);
			else throw std::invalid_argument("unknown option: " + key);
		}
		for (float value : {options.lr, options.decay, options.clip}) {
			if (!std::isfinite(value)) throw std::invalid_argument("nonfinite training option");
		}
		if (options.data.empty() || options.out.empty() || (options.device != "auto" && options.device != "cpu" && options.device != "cuda") ||
			options.epochs < 1 || options.batch < 1 || options.batch > 65536 || options.workers < 1 || options.workers > 256 ||
			options.maxSteps == 0 || options.maxSteps < -1 || options.lr <= 0 || options.decay < 0 || options.lr * options.decay >= 1 ||
			options.clip <= 0 || options.save < 0 || options.log < 1) {
			throw std::invalid_argument("invalid or incomplete Eleginus training options");
		}
		return options;
	}

	void require(herr_t status, const std::string &operation) {
		if (status < 0) throw std::runtime_error("HDF5 operation failed: " + operation);
	}

	hid_t requireId(hid_t id, const std::string &operation) {
		if (id < 0) throw std::runtime_error("HDF5 operation failed: " + operation);
		return id;
	}

	std::string stringAttribute(hid_t object, const char *name) {
		if (H5Aexists(object, name) <= 0) throw std::runtime_error(std::string("HDF5 missing attribute: ") + name);
		const hid_t attribute = requireId(H5Aopen(object, name, H5P_DEFAULT), "open attribute");
		const hid_t type = requireId(H5Aget_type(attribute), "get attribute type");
		std::vector<char> text(H5Tget_size(type) + 1, '\0');
		require(H5Aread(attribute, type, text.data()), "read attribute");
		H5Tclose(type);
		H5Aclose(attribute);
		return text.data();
	}

	std::int64_t integerAttribute(hid_t object, const char *name) {
		if (H5Aexists(object, name) <= 0) throw std::runtime_error(std::string("HDF5 missing attribute: ") + name);
		const hid_t attribute = requireId(H5Aopen(object, name, H5P_DEFAULT), "open attribute");
		std::int64_t value = 0;
		require(H5Aread(attribute, H5T_NATIVE_INT64, &value), "read attribute");
		H5Aclose(attribute);
		return value;
	}

	struct RawChunk {
		std::vector<std::uint64_t> pieces;
		std::vector<std::uint8_t> states;
		std::vector<float> values;
		std::size_t size() const noexcept { return states.size(); }
	};

	class Dataset {
	public:
		explicit Dataset(const std::filesystem::path &path) {
			file = requireId(H5Fopen(path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), "open training data");
			if (stringAttribute(file, "arch_type") != "eleginus" || stringAttribute(file, "state_encoding") != "piece-code-bitplanes" ||
				stringAttribute(file, "target_schema") != "white-expected-score" || integerAttribute(file, "formula_count") != eleginus::kFormulaCount) {
				throw std::runtime_error("HDF5 schema does not match Eleginus supervised training");
			}
			pieces = requireId(H5Dopen2(file, "pieces", H5P_DEFAULT), "open pieces");
			states = requireId(H5Dopen2(file, "states", H5P_DEFAULT), "open states");
			values = requireId(H5Dopen2(file, "values", H5P_DEFAULT), "open values");
			const auto pieceShape = shape(pieces, 2);
			const auto stateShape = shape(states, 1);
			const auto valueShape = shape(values, 1);
			if (pieceShape[1] != planeCount || pieceShape[0] == 0 || stateShape[0] != pieceShape[0] || valueShape[0] != pieceShape[0]) {
				throw std::runtime_error("Eleginus datasets have incompatible shapes");
			}
			rows = static_cast<std::int64_t>(pieceShape[0]);
			const hid_t properties = requireId(H5Dget_create_plist(pieces), "get pieces properties");
			if (H5Pget_layout(properties) == H5D_CHUNKED) {
				hsize_t dimensions[2]{};
				if (H5Pget_chunk(properties, 2, dimensions) != 2 || dimensions[0] == 0) {
					H5Pclose(properties);
					throw std::runtime_error("pieces dataset has an invalid chunk shape");
				}
				chunkRows = static_cast<std::int64_t>(dimensions[0]);
			} else {
				chunkRows = rows;
			}
			H5Pclose(properties);
		}

		~Dataset() {
			if (pieces >= 0) H5Dclose(pieces);
			if (states >= 0) H5Dclose(states);
			if (values >= 0) H5Dclose(values);
			if (file >= 0) H5Fclose(file);
		}

		Dataset(const Dataset &) = delete;
		Dataset &operator=(const Dataset &) = delete;

		std::int64_t count() const noexcept { return rows; }
		std::int64_t chunk() const noexcept { return chunkRows; }

		RawChunk read(std::int64_t first, std::int64_t count) const {
			if (first < 0 || count <= 0 || first > rows - count) throw std::out_of_range("HDF5 row range");
			RawChunk result;
			result.pieces.resize(static_cast<std::size_t>(count) * planeCount);
			result.states.resize(static_cast<std::size_t>(count));
			result.values.resize(static_cast<std::size_t>(count));
			readRange(pieces, H5T_NATIVE_UINT64, result.pieces.data(), first, count, planeCount);
			readRange(states, H5T_NATIVE_UINT8, result.states.data(), first, count, 0);
			readRange(values, H5T_NATIVE_FLOAT, result.values.data(), first, count, 0);
			if (std::any_of(result.values.begin(), result.values.end(), [](float value) { return !std::isfinite(value) || value < 0 || value > 1; })) {
				throw std::runtime_error("HDF5 target lies outside [0,1]");
			}
			return result;
		}

	private:
		static std::vector<hsize_t> shape(hid_t dataset, int rank) {
			const hid_t space = requireId(H5Dget_space(dataset), "get dataset shape");
			if (H5Sget_simple_extent_ndims(space) != rank) {
				H5Sclose(space);
				throw std::runtime_error("HDF5 dataset has the wrong rank");
			}
			std::vector<hsize_t> result(static_cast<std::size_t>(rank));
			H5Sget_simple_extent_dims(space, result.data(), nullptr);
			H5Sclose(space);
			return result;
		}

		static void readRange(hid_t dataset, hid_t type, void *destination, std::int64_t first, std::int64_t count, hsize_t width) {
			const hid_t source = requireId(H5Dget_space(dataset), "get dataset range");
			if (width) {
				const hsize_t start[] = {static_cast<hsize_t>(first), 0};
				const hsize_t dimensions[] = {static_cast<hsize_t>(count), width};
				require(H5Sselect_hyperslab(source, H5S_SELECT_SET, start, nullptr, dimensions, nullptr), "select dataset range");
				const hid_t memory = requireId(H5Screate_simple(2, dimensions, nullptr), "create range memory");
				require(H5Dread(dataset, type, memory, source, H5P_DEFAULT, destination), "read dataset range");
				H5Sclose(memory);
			} else {
				const hsize_t start[] = {static_cast<hsize_t>(first)};
				const hsize_t dimensions[] = {static_cast<hsize_t>(count)};
				require(H5Sselect_hyperslab(source, H5S_SELECT_SET, start, nullptr, dimensions, nullptr), "select dataset range");
				const hid_t memory = requireId(H5Screate_simple(1, dimensions, nullptr), "create range memory");
				require(H5Dread(dataset, type, memory, source, H5P_DEFAULT, destination), "read dataset range");
				H5Sclose(memory);
			}
			H5Sclose(source);
		}

		hid_t file = -1;
		hid_t pieces = -1;
		hid_t states = -1;
		hid_t values = -1;
		std::int64_t rows = 0;
		std::int64_t chunkRows = 1;
	};

	class PackedBoard final : public chess::Board {
	public:
		PackedBoard() : chess::Board(ProtectedCtor::CREATE) {}

		void load(const std::uint64_t *planes, std::uint8_t state) {
			if (state & 0xE0U) throw std::runtime_error("HDF5 position contains invalid state bits");
			prev_states_.clear();
			pieces_bb_.fill(chess::Bitboard(0));
			occ_bb_.fill(chess::Bitboard(0));
			board_.fill(chess::Piece::NONE);
			key_ = 0;
			cr_.clear();
			plies_ = 0;
			stm_ = chess::Color(state & 1U);
			ep_sq_ = chess::Square::NO_SQ;
			hfm_ = 0;
			chess960_ = false;
			castling_path = {};
			for (int square = 0; square < 64; ++square) {
				unsigned code = 0;
				for (unsigned bit = 0; bit < planeCount; ++bit) code |= static_cast<unsigned>((planes[bit] >> square) & 1ULL) << bit;
				if (code == 0) continue;
				if (code > 12) throw std::runtime_error("HDF5 position contains an invalid piece code");
				const int packed = static_cast<int>(code - 1);
				const int type = packed % 6;
				const int color = packed / 6;
				const auto piece = chess::Piece(chess::PieceType(static_cast<chess::PieceType::underlying>(type)), chess::Color(color));
				const chess::Bitboard bit(1ULL << square);
				board_[static_cast<std::size_t>(square)] = piece;
				pieces_bb_[static_cast<std::size_t>(type)] |= bit;
				occ_bb_[static_cast<std::size_t>(color)] |= bit;
			}
			if (pieces_bb_[5].count() != 2 || (pieces_bb_[5] & occ_bb_[0]).count() != 1 || (pieces_bb_[5] & occ_bb_[1]).count() != 1) {
				throw std::runtime_error("HDF5 position does not contain one king per side");
			}
			for (int color = 0; color < 2; ++color) {
				for (int wing = 0; wing < 2; ++wing) {
					if (!(state & (1U << (1 + 2 * color + wing)))) continue;
					const auto side = wing == 0 ? CastlingRights::Side::KING_SIDE : CastlingRights::Side::QUEEN_SIDE;
					cr_.setCastlingRight(chess::Color(color), side, wing == 0 ? chess::File::FILE_H : chess::File::FILE_A);
				}
			}
		}
	};

	struct DenseBatch {
		torch::Tensor scores;
		torch::Tensor conditions;
		torch::Tensor targets;
	};

	class FormulaPool {
	public:
		explicit FormulaPool(int count) {
			threads.reserve(static_cast<std::size_t>(count));
			for (int id = 0; id < count; ++id) threads.emplace_back([this, id] { worker(static_cast<std::size_t>(id)); });
		}

		~FormulaPool() {
			{
				std::lock_guard lock(mutex);
				closing = true;
			}
			ready.notify_all();
			for (auto &thread : threads) thread.join();
		}

		DenseBatch build(const RawChunk &raw, const std::vector<std::size_t> &order, std::size_t first, std::size_t count, bool pinned) {
			const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(pinned);
			DenseBatch result{
				torch::zeros({static_cast<std::int64_t>(count), static_cast<std::int64_t>(eleginus::kFormulaCount)}, options),
				torch::zeros({static_cast<std::int64_t>(count), static_cast<std::int64_t>(eleginus::kFormulaCount)}, options),
				torch::empty({static_cast<std::int64_t>(count)}, options),
			};
			{
				std::lock_guard lock(mutex);
				job = {&raw, &order, first, count, result.scores.data_ptr<float>(), result.conditions.data_ptr<float>(), result.targets.data_ptr<float>()};
				finished = 0;
				failure = nullptr;
				++generation;
			}
			ready.notify_all();
			std::unique_lock lock(mutex);
			done.wait(lock, [&] { return finished == threads.size(); });
			if (failure) std::rethrow_exception(failure);
			return result;
		}

	private:
		struct Job {
			const RawChunk *raw = nullptr;
			const std::vector<std::size_t> *order = nullptr;
			std::size_t first = 0;
			std::size_t count = 0;
			float *scores = nullptr;
			float *conditions = nullptr;
			float *targets = nullptr;
		};

		void worker(std::size_t id) {
			std::uint64_t seen = 0;
			PackedBoard board;
			std::vector<eleginus::Feature> features;
			features.reserve(eleginus::kFormulaCount);
			while (true) {
				Job current;
				{
					std::unique_lock lock(mutex);
					ready.wait(lock, [&] { return closing || generation != seen; });
					if (closing) return;
					seen = generation;
					current = job;
				}
				try {
					for (std::size_t row = id; row < current.count; row += threads.size()) {
						const std::size_t source = (*current.order)[current.first + row];
						board.load(current.raw->pieces.data() + source * planeCount, current.raw->states[source]);
						eleginus::FormulaSet::evaluate(board, features);
						const auto offset = row * eleginus::kFormulaCount;
						for (const auto &feature : features) {
							current.scores[offset + feature.index] = static_cast<float>(feature.score);
							current.conditions[offset + feature.index] = static_cast<float>(feature.condition);
						}
						current.targets[row] = current.raw->values[source];
					}
				} catch (...) {
					std::lock_guard lock(mutex);
					if (!failure) failure = std::current_exception();
				}
				{
					std::lock_guard lock(mutex);
					++finished;
				}
				done.notify_one();
			}
		}

		std::vector<std::thread> threads;
		std::mutex mutex;
		std::condition_variable ready;
		std::condition_variable done;
		Job job;
		std::uint64_t generation = 0;
		std::size_t finished = 0;
		bool closing = false;
		std::exception_ptr failure;
	};

	torch::Device resolveDevice(const std::string &requested) {
		if (requested == "auto") return torch::Device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
		if (requested == "cuda" && !torch::cuda::is_available()) throw std::runtime_error("CUDA training was requested but CUDA is unavailable");
		return torch::Device(requested);
	}

	class Solver {
	public:
		Solver(const eleginus::Model &model, const Options &options) : device(resolveDevice(options.device)), clip(options.clip) {
			constexpr float relationScale = static_cast<float>(eleginus::kFormulaCount);
			const auto cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
			const auto initialValues = eleginus::Model::initial();
			initial = torch::from_blob(const_cast<float *>(initialValues.data()), {static_cast<std::int64_t>(eleginus::kFormulaCount)}, cpu).clone().to(device);
			base = torch::from_blob(const_cast<float *>(model.base().data()), {static_cast<std::int64_t>(eleginus::kFormulaCount)}, cpu).clone().to(device);
			base = (base - initial).detach().set_requires_grad(true);
			relation = torch::from_blob(const_cast<float *>(model.relations().data()),
				{static_cast<std::int64_t>(eleginus::kFormulaCount), static_cast<std::int64_t>(eleginus::kFormulaCount)}, cpu).clone().to(device);
			relation = (relation * relationScale).detach().set_requires_grad(true);
			parameters = {base, relation};
			optimizer = std::make_unique<torch::optim::AdamW>(parameters, torch::optim::AdamWOptions(options.lr).weight_decay(options.decay));
			lossTotal = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
		}

		const torch::Device &trainingDevice() const noexcept { return device; }

		double step(const DenseBatch &batch, bool report) {
			constexpr float inverseRelationScale = 1.0F / static_cast<float>(eleginus::kFormulaCount);
			auto scores = batch.scores.to(device, torch::kFloat32, device.is_cuda(), false);
			auto conditions = batch.conditions.to(device, torch::kFloat32, device.is_cuda(), false);
			auto targets = batch.targets.to(device, torch::kFloat32, device.is_cuda(), false);
			auto dynamic = initial + base + inverseRelationScale * torch::matmul(conditions, relation);
			auto logits = (scores * dynamic).sum(1);
			auto loss = torch::nn::functional::binary_cross_entropy_with_logits(logits, targets);
			lossTotal.add_(loss.detach());
			++lossCount;
			double reported = 0.0;
			if (report) {
				reported = (lossTotal / lossCount).item<double>();
				if (!std::isfinite(reported)) throw std::runtime_error("nonfinite training loss");
				lossTotal.zero_();
				lossCount = 0;
			}
			optimizer->zero_grad();
			loss.backward();
			torch::nn::utils::clip_grad_norm_(parameters, clip);
			optimizer->step();
			return reported;
		}

		void save(eleginus::Model &model, const std::filesystem::path &path) const {
			constexpr float inverseRelationScale = 1.0F / static_cast<float>(eleginus::kFormulaCount);
			torch::NoGradGuard guard;
			const auto cpuBase = (initial + base).to(torch::kCPU).contiguous();
			const auto cpuRelation = (inverseRelationScale * relation).to(torch::kCPU).contiguous();
			std::vector<float> values(eleginus::kParameterCount);
			std::copy_n(cpuBase.data_ptr<float>(), eleginus::kFormulaCount, values.begin());
			std::copy_n(cpuRelation.data_ptr<float>(), eleginus::kRelationCount, values.begin() + static_cast<std::ptrdiff_t>(eleginus::kFormulaCount));
			model.update(values);
			model.save(path);
		}

	private:
		torch::Device device;
		float clip;
		torch::Tensor initial;
		torch::Tensor base;
		torch::Tensor relation;
		torch::Tensor lossTotal;
		std::int64_t lossCount = 0;
		std::vector<torch::Tensor> parameters;
		std::unique_ptr<torch::optim::AdamW> optimizer;
	};

	std::int64_t stepsPerEpoch(std::int64_t rows, std::int64_t chunk, int batch) {
		const auto full = rows / chunk;
		const auto remainder = rows % chunk;
		return full * ((chunk + batch - 1) / batch) + (remainder ? (remainder + batch - 1) / batch : 0);
	}

} // namespace

int main(int argc, char **argv) {
	try {
		const auto options = parse(argc, argv);
		std::signal(SIGINT, interrupt);
		std::signal(SIGTERM, interrupt);
		#ifdef _WIN32
		if (!SetConsoleCtrlHandler(nullptr, FALSE) || !SetConsoleCtrlHandler(console, TRUE)) throw std::runtime_error("cannot install console interrupt handler");
		#endif
		Dataset data(options.data);
		auto model = options.init.empty() ? eleginus::Model() : eleginus::Model::load(options.init);
		Solver solver(model, options);
		FormulaPool formulas(options.workers);
		std::mt19937_64 random(options.seed);
		const auto epochSteps = stepsPerEpoch(data.count(), data.chunk(), options.batch);
		std::cout << "training start: data=" << options.data.string() << " out=" << options.out.string();
		std::cout << " device=" << solver.trainingDevice() << " epochs=" << options.epochs << " batch_size=" << options.batch;
		std::cout << " workers=" << options.workers << " parameters=" << eleginus::kParameterCount << " lr=" << options.lr << std::endl;
		std::cout << "training input: rows=" << data.count() << " hdf5_chunk_rows=" << data.chunk();
		std::cout << " steps_per_epoch=" << epochSteps << " loader=chunk_shuffle_prefetch" << std::endl;
		std::int64_t globalStep = 0;
		std::int64_t trainedRows = 0;
		for (int epoch = 0; epoch < options.epochs && !stopped(); ++epoch) {
			std::vector<std::int64_t> starts;
			for (std::int64_t first = 0; first < data.count(); first += data.chunk()) starts.push_back(first);
			std::shuffle(starts.begin(), starts.end(), random);
			auto launch = [&](std::size_t position) {
				const auto first = starts[position];
				const auto count = std::min(data.chunk(), data.count() - first);
				return std::async(std::launch::async, [&data, first, count] { return data.read(first, count); });
			};
			auto pending = launch(0);
			for (std::size_t chunk = 0; chunk < starts.size() && !stopped();) {
				auto raw = pending.get();
				++chunk;
				if (chunk < starts.size()) pending = launch(chunk);
				std::vector<std::size_t> order(raw.size());
				std::iota(order.begin(), order.end(), 0);
				std::shuffle(order.begin(), order.end(), random);
				for (std::size_t first = 0; first < raw.size() && !stopped(); first += static_cast<std::size_t>(options.batch)) {
					if (options.maxSteps > 0 && globalStep >= options.maxSteps) break;
					const auto count = std::min(static_cast<std::size_t>(options.batch), raw.size() - first);
					auto batch = formulas.build(raw, order, first, count, solver.trainingDevice().is_cuda());
					const bool report = (globalStep + 1) % options.log == 0 || globalStep == 0;
					const double loss = solver.step(batch, report);
					++globalStep;
					trainedRows += static_cast<std::int64_t>(count);
					if (report) {
						std::cout << "train step: epoch=" << epoch << " global_step=" << globalStep << " loss=" << loss;
						std::cout << " lr=" << options.lr << " positions=" << trainedRows << std::endl;
					}
					if (options.save > 0 && globalStep % options.save == 0) {
						solver.save(model, options.out);
						std::cout << "saved model: step=" << globalStep << " path=" << options.out.string() << std::endl;
					}
				}
				if (options.maxSteps > 0 && globalStep >= options.maxSteps) break;
			}
			if (options.maxSteps > 0 && globalStep >= options.maxSteps) break;
		}
		if (globalStep == 0) throw std::runtime_error("training stopped before the first optimizer step");
		solver.save(model, options.out);
		std::cout << "training complete: steps=" << globalStep << " positions=" << trainedRows << " model=" << options.out.string() << std::endl;
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "training error: " << error.what() << std::endl;
		return 1;
	}
}
