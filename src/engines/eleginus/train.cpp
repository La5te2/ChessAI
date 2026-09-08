#include "eleginus/evaluate.hpp"
#include "eleginus/game.hpp"
#include "eleginus/search.hpp"
#include <torch/torch.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#ifdef _WIN32
	#define NOMINMAX
	#define WIN32_LEAN_AND_MEAN
	#include <windows.h>
#endif

namespace {
	// -------------------------- Training configuration -------------------------------
	// Command-line controls and interruption handling.

	std::atomic_bool halt{false};
	constexpr int openingPairs = 1000;
	constexpr std::size_t optimizerBatch = 16 * 16384;
	constexpr std::size_t validationStride = 20;
	constexpr std::size_t calibrationSamples = 1U << 20;
	constexpr float minimumPressureWidth = 0.02F;

	void interrupt(int) noexcept { halt.store(true, std::memory_order_relaxed); }
	bool stopped() noexcept { return halt.load(std::memory_order_relaxed); }

	#ifdef _WIN32
	BOOL WINAPI console(DWORD event) {
		if (event != CTRL_C_EVENT && event != CTRL_BREAK_EVENT) return FALSE;
		interrupt(0);
		return TRUE;
	}
	#endif

	struct Options {
		std::filesystem::path out = "models/eleginus/current.pth";
		std::filesystem::path init;
		std::filesystem::path book = "data/openings.gen.bin";
		std::string device = "auto";
		int games = 100000;
		int randomPlies = 8;
		int depth = 1;
		int evaluationDepth = 4;
		int maximumPlies = 320;
		int workers = static_cast<int>(std::clamp(std::thread::hardware_concurrency() / 2U, 1U, 4U));
		int hash = 16;
		int epochs = 100;
		int batch = 4096;
		int log = 1000;
		float lr = 1.0e-3F;
		std::uint64_t seed = 2026;
	};

	std::string valueAfter(int argc, char **argv, int &index) {
		if (++index == argc) throw std::invalid_argument(std::string("missing value after ") + argv[index - 1]);
		return argv[index];
	}

	Options parse(int argc, char **argv) {
		Options options;
		for (int index = 1; index < argc; ++index) {
			const std::string argument = argv[index];
			if (argument == "--help") {
				std::cout << "usage: train [--out parameters.pth] [--init parameters.pth] [options]\n";
				std::cout << "  --opening-book openings.gen.bin --games 100000 --random-plies 8 --depth 1\n";
				std::cout << "  --eval-depth 4 --max-plies 320 --workers 4 --hash 16\n";
				std::cout << "  --epochs 100 --batch-size 4096 --device auto --lr 0.001\n";
				std::cout << "  --log-every 1000 --seed 2026\n";
				std::exit(0);
			}
			const auto value = valueAfter(argc, argv, index);
			if (argument == "--out") options.out = value;
			else if (argument == "--init") options.init = value;
			else if (argument == "--opening-book") options.book = value;
			else if (argument == "--games") options.games = std::stoi(value);
			else if (argument == "--random-plies") options.randomPlies = std::stoi(value);
			else if (argument == "--depth") options.depth = std::stoi(value);
			else if (argument == "--eval-depth") options.evaluationDepth = std::stoi(value);
			else if (argument == "--max-plies") options.maximumPlies = std::stoi(value);
			else if (argument == "--workers") options.workers = std::stoi(value);
			else if (argument == "--hash") options.hash = std::stoi(value);
			else if (argument == "--epochs") options.epochs = std::stoi(value);
			else if (argument == "--batch-size") options.batch = std::stoi(value);
			else if (argument == "--device") options.device = value;
			else if (argument == "--lr") options.lr = std::stof(value);
			else if (argument == "--log-every") options.log = std::stoi(value);
			else if (argument == "--seed") options.seed = std::stoull(value);
			else throw std::invalid_argument("unknown option: " + argument);
		}
		if (!std::isfinite(options.lr)) throw std::invalid_argument("nonfinite training option");
		if (options.out.empty() || options.book.empty() || (options.device != "auto" && options.device != "cpu" && options.device != "cuda") ||
			options.games < 10 || options.randomPlies < 0 || options.randomPlies > 64 || options.depth < 1 || options.depth > 16 ||
			options.evaluationDepth < 1 || options.evaluationDepth > 64 || options.maximumPlies <= options.randomPlies ||
			options.maximumPlies > 320 || options.workers < 1 || options.workers > 256 || options.hash < 0 || options.hash > 4096 ||
			options.epochs < 1 || options.epochs > 10000 || options.batch < 1 || options.batch > 65536 || options.log < 1 || options.lr <= 0.0F) {
			throw std::invalid_argument("invalid or incomplete Eleginus training options");
		}
		return options;
	}

	// Derive a deterministic random stream for one generated game.
	std::uint64_t gameSeed(std::uint64_t seed, std::uint64_t game) noexcept {
		auto value = seed + 0x9e3779b97f4a7c15ULL * (game + 1);
		value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
		value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
		return value ^ (value >> 31);
	}

	// ----------------------------- Fixed WDL data -------------------------------------
	// Compact parameter-independent evaluator inputs retained for every training epoch.

	std::int16_t narrow(std::int32_t value) {
		if (value < std::numeric_limits<std::int16_t>::min() || value > std::numeric_limits<std::int16_t>::max()) {
			throw std::overflow_error("formula signal exceeds the compact training range");
		}
		return static_cast<std::int16_t>(value);
	}

	struct Sample {
		std::array<std::int16_t, eleginus::formulaCount> signals{};
		std::array<float, 5> material{};
		std::array<std::int16_t, 2> pressure{};
		std::array<std::int16_t, 14> endgame{};
		std::uint16_t pressureFormula = 0;
		float target = 0.5F;
		float initialScore = 0.0F;
	};

	// Pack one board's parameter-independent evaluation values for repeated fitting.
	Sample sample(const eleginus::Evaluator &evaluator, const chess::Board &board) {
		const auto values = evaluator.extract(board);
		Sample result;
		std::transform(values.signals.begin(), values.signals.end(), result.signals.begin(), narrow);
		result.material = values.material;
		result.pressure = {narrow(values.pressure.attacks[0]), narrow(values.pressure.attacks[1])};
		if (values.pressure.formula >= eleginus::formulaCount) throw std::runtime_error("invalid king-pressure formula index");
		result.pressureFormula = static_cast<std::uint16_t>(values.pressure.formula);
		const auto &value = values.endgame;
		const std::array<std::int32_t, 14> facts{{value.pawns, value.symmetricFiles, value.asymmetricFiles, value.pawnEnding,
			value.sidePawns[0], value.sidePawns[1], value.sidePassers[0], value.sidePassers[1], value.oppositeBishops,
			value.pawnless[0], value.pawnless[1], value.thinMaterial, value.pureOppositeBishops, value.mixedOppositeBishops}};
		std::transform(facts.begin(), facts.end(), result.endgame.begin(), narrow);
		result.initialScore = evaluator.score(values);
		return result;
	}

	struct Game {
		std::vector<Sample> positions;
		int result = -1;
	};

	// Play one game with a random prefix and retain every subsequent main-line position.
	Game play(const eleginus::Evaluator &evaluator, const Options &options, std::uint64_t id) {
		Game game;
		chess::Board board;
		std::mt19937_64 random(gameSeed(options.seed, id));
		for (int ply = 0; ply < options.randomPlies; ++ply) {
			const auto moves = eleginus::legalmoves(board);
			if (moves.empty() || eleginus::isGameOver(board)) return game;
			board.makeMove(moves[std::uniform_int_distribution<std::size_t>(0, moves.size() - 1)(random)]);
		}
		if (eleginus::isGameOver(board)) return game;

		eleginus::SearchOptions limits;
		limits.depth = options.depth;
		limits.hashMiB = static_cast<std::size_t>(options.hash);
		eleginus::Searcher searcher(evaluator, limits);
		game.positions.reserve(static_cast<std::size_t>(options.maximumPlies - options.randomPlies));
		for (int ply = options.randomPlies; ply < options.maximumPlies && !stopped() && !eleginus::isGameOver(board); ++ply) {
			game.positions.push_back(sample(evaluator, board));
			const auto result = searcher.search(board, {}, stopped);
			if (stopped() || result.move.move() == chess::Move::NO_MOVE) return Game{};
			board.makeMove(result.move);
		}

		const auto [reason, result] = board.isGameOver();
		if (stopped() || reason == chess::GameResultReason::NONE) return Game{};
		if (result == chess::GameResult::DRAW) game.result = 1;
		else {
			const auto winner = result == chess::GameResult::WIN ? board.sideToMove() : ~board.sideToMove();
			game.result = winner == chess::Color::WHITE ? 2 : 0;
		}
		const float target = 0.5F * static_cast<float>(game.result);
		for (auto &position : game.positions) position.target = target;
		return game;
	}

	struct Dataset {
		std::vector<std::vector<Sample>> games;
		std::uint64_t positions = 0;
		std::array<std::uint64_t, 3> results{};
		std::uint64_t discarded = 0;
	};

	// Generate the requested number of completed games in parallel.
	Dataset generate(const eleginus::Evaluator &evaluator, const Options &options) {
		Dataset data;
		data.games.reserve(static_cast<std::size_t>(options.games));
		std::uint64_t attempted = 0;
		std::uint64_t nextLog = static_cast<std::uint64_t>(options.log);
		while (data.games.size() < static_cast<std::size_t>(options.games) && !stopped()) {
			const int count = std::min<int>(options.workers, options.games - static_cast<int>(data.games.size()));
			std::vector<std::future<Game>> jobs;
			jobs.reserve(static_cast<std::size_t>(count));
			for (int worker = 0; worker < count; ++worker) {
				const auto id = attempted + static_cast<std::uint64_t>(worker);
				jobs.push_back(std::async(std::launch::async, [&evaluator, &options, id] { return play(evaluator, options, id); }));
			}
			attempted += static_cast<std::uint64_t>(count);
			for (auto &job : jobs) {
				auto game = job.get();
				if (game.result < 0) {
					++data.discarded;
					continue;
				}
				data.positions += game.positions.size();
				++data.results[static_cast<std::size_t>(game.result)];
				data.games.push_back(std::move(game.positions));
			}
			if (data.games.size() >= nextLog || data.games.size() == static_cast<std::size_t>(options.games)) {
				std::cout << "generation step: games=" << data.games.size() << '/' << options.games << " discarded=" << data.discarded;
				std::cout << " positions=" << data.positions << " white=" << data.results[2] << " draws=" << data.results[1];
				std::cout << " black=" << data.results[0] << std::endl;
				while (nextLog <= data.games.size()) nextLog += static_cast<std::uint64_t>(options.log);
			}
		}
		return data;
	}

	// Fit the sigmoid scale that minimizes the initial table's Texel loss.
	double calibration(const Dataset &data) {
		std::vector<std::array<float, 2>> samples;
		samples.reserve(std::min<std::uint64_t>(data.positions, calibrationSamples));
		const std::uint64_t stride = std::max<std::uint64_t>(1, (data.positions + calibrationSamples - 1) / calibrationSamples);
		std::uint64_t index = 0;
		for (const auto &game : data.games) {
			for (const auto &sample : game) {
				if (index++ % stride == 0) samples.push_back({sample.initialScore, sample.target});
			}
		}
		const auto loss = [&samples](double scale) {
			double total = 0.0;
			for (const auto &sample : samples) {
				const double argument = std::clamp(scale * sample[0], -40.0, 40.0);
				const double probability = 1.0 / (1.0 + std::exp(-argument));
				const double error = probability - sample[1];
				total += error * error;
			}
			return total / static_cast<double>(samples.size());
		};

		// Golden-section search in log space covers every practical score scale with a fixed number of data passes.
		constexpr double ratio = 0.6180339887498948482;
		double left = std::log(1.0e-4);
		double right = std::log(100.0);
		double first = right - ratio * (right - left);
		double second = left + ratio * (right - left);
		double firstLoss = loss(std::exp(first));
		double secondLoss = loss(std::exp(second));
		for (int iteration = 0; iteration < 32; ++iteration) {
			if (firstLoss <= secondLoss) {
				right = second;
				second = first;
				secondLoss = firstLoss;
				first = right - ratio * (right - left);
				firstLoss = loss(std::exp(first));
			} else {
				left = first;
				first = second;
				firstLoss = secondLoss;
				second = left + ratio * (right - left);
				secondLoss = loss(std::exp(second));
			}
		}
		return std::exp(0.5 * (left + right));
	}

	// ----------------------------- Tensor optimizer -----------------------------------
	// Dense device batches and the differentiable form of the complete evaluator.

	// Resolve automatic device selection and reject an unavailable requested CUDA device.
	torch::Device resolveDevice(const std::string &requested) {
		if (requested == "auto") return torch::Device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
		if (requested == "cuda" && !torch::cuda::is_available()) throw std::runtime_error("CUDA training was requested but CUDA is unavailable");
		return torch::Device(requested);
	}

	struct Batch {
		torch::Tensor signals;
		torch::Tensor material;
		torch::Tensor pressure;
		torch::Tensor endgame;
		torch::Tensor targets;
		std::size_t pressureFormula = eleginus::formulaCount;
	};

	// Gather compact samples into pinned tensors for one device microbatch.
	Batch batch(const std::vector<const Sample *> &samples, bool pinned) {
		const auto count = samples.size();
		const auto integers = torch::TensorOptions().dtype(torch::kInt16).device(torch::kCPU).pinned_memory(pinned);
		const auto reals = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(pinned);
		Batch result{
			torch::empty({static_cast<std::int64_t>(count), static_cast<std::int64_t>(eleginus::formulaCount)}, integers),
			torch::empty({static_cast<std::int64_t>(count), 5}, reals),
			torch::empty({static_cast<std::int64_t>(count), 2}, integers),
			torch::empty({static_cast<std::int64_t>(count), 14}, integers),
			torch::empty({static_cast<std::int64_t>(count)}, reals),
		};
		auto *signals = result.signals.data_ptr<std::int16_t>();
		auto *material = result.material.data_ptr<float>();
		auto *pressure = result.pressure.data_ptr<std::int16_t>();
		auto *endgame = result.endgame.data_ptr<std::int16_t>();
		auto *targets = result.targets.data_ptr<float>();
		for (std::size_t row = 0; row < count; ++row) {
			const auto &sample = *samples[row];
			if (result.pressureFormula == eleginus::formulaCount) result.pressureFormula = sample.pressureFormula;
			if (sample.pressureFormula != result.pressureFormula) throw std::runtime_error("inconsistent king-pressure formula index");
			std::copy(sample.signals.begin(), sample.signals.end(), signals + row * eleginus::formulaCount);
			std::copy(sample.material.begin(), sample.material.end(), material + row * sample.material.size());
			std::copy(sample.pressure.begin(), sample.pressure.end(), pressure + row * sample.pressure.size());
			std::copy(sample.endgame.begin(), sample.endgame.end(), endgame + row * sample.endgame.size());
			targets[row] = sample.target;
		}
		return result;
	}

	// Optimize the complete parameter vector and retain its lowest-validation-loss state.
	class Solver {
	public:
		Solver(const eleginus::FormulaParameters &initial, const Options &options, double calibrationScale)
			: device(resolveDevice(options.device)), scale(calibrationScale) {
			const auto values = eleginus::flattenParameters(initial);
			const auto cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
			origin = torch::from_blob(const_cast<float *>(values.data()), {static_cast<std::int64_t>(values.size())}, cpu).clone();
			parameters = origin.to(device).clone().set_requires_grad(true);
			optimizer = std::make_unique<torch::optim::Adam>(std::vector<torch::Tensor>{parameters}, torch::optim::AdamOptions(options.lr));
		}

		void learningRate(double value) {
			for (auto &group : optimizer->param_groups()) {
				static_cast<torch::optim::AdamOptions &>(group.options()).lr(value);
			}
		}
		torch::Tensor forward(const Batch &source) const {
			const auto formula = parameters.slice(0, 0, static_cast<std::int64_t>(6 * eleginus::formulaCount))
				.view({static_cast<std::int64_t>(eleginus::formulaCount), 6});
			auto signals = source.signals.to(device, torch::kFloat32, device.is_cuda(), true);
			const auto material = source.material.to(device, torch::kFloat32, device.is_cuda(), false);
			const auto pressure = source.pressure.to(device, torch::kFloat32, device.is_cuda(), false);
			const auto facts = source.endgame.to(device, torch::kFloat32, device.is_cuda(), false);
			const auto weights = formula.select(1, 0).unsqueeze(0) + torch::matmul(material, formula.slice(1, 1, 6).transpose(0, 1));
			const auto offset = static_cast<std::int64_t>(6 * eleginus::formulaCount);
			const auto center = parameters[offset];
			const auto width = torch::clamp_min(parameters[offset + 1], minimumPressureWidth);
			const auto response = [&](const torch::Tensor &attacks) {
				const auto smooth = 4096.0F * torch::sigmoid((torch::clamp(attacks, 0.0F, 64.0F) - center) / width);
				const auto rounded = torch::round(smooth);
				return smooth + (rounded - smooth).detach();
			};
			signals.select(1, static_cast<std::int64_t>(source.pressureFormula)).copy_(
				response(pressure.select(1, 0)) - response(pressure.select(1, 1)));
			auto total = (signals * weights).sum(1);
			const auto favored = total.lt(0.0F);
			const auto strongPawns = torch::where(favored, facts.select(1, 5), facts.select(1, 4));
			const auto strongPassers = torch::where(favored, facts.select(1, 7), facts.select(1, 6));
			const auto pawnless = torch::where(favored, facts.select(1, 10), facts.select(1, 9));
			const auto win = parameters.slice(0, offset + 2, offset + 9);
			const auto conversion = win[6] + win[0] * facts.select(1, 0) + win[1] * facts.select(1, 1) +
				win[2] * facts.select(1, 2) + win[3] * facts.select(1, 3) + win[4] * strongPawns +
				win[5] * facts.select(1, 8) * strongPassers;
			total = torch::where(total.gt(0.0F), torch::clamp_min(total + conversion, 0.0F),
				torch::where(total.lt(0.0F), torch::clamp_max(total - conversion, 0.0F), total));
			const auto scaling = parameters.slice(0, offset + 9, offset + 14);
			auto factor = torch::ones_like(total);
			factor = torch::where(pawnless.gt(0.0F) * facts.select(1, 11).gt(0.0F), torch::minimum(factor, scaling[0]), factor);
			const auto pure = scaling[1] + scaling[4] * strongPawns + scaling[3] * strongPassers;
			const auto mixed = scaling[2] + scaling[3] * strongPassers;
			factor = torch::where(facts.select(1, 12).gt(0.0F), torch::minimum(factor, pure),
				torch::where(facts.select(1, 13).gt(0.0F), torch::minimum(factor, mixed), factor));
			return total * torch::clamp(factor, 0.0F, 1.0F);
		}

		double train(const std::vector<const Sample *> &samples) {
			auto data = batch(samples, device.is_cuda());
			const auto expected = data.targets.to(device, torch::kFloat32, device.is_cuda(), false);
			const auto probability = torch::sigmoid(static_cast<float>(scale) * forward(data));
			const auto loss = torch::square(probability - expected).sum();
			const double value = loss.item<double>();
			if (!std::isfinite(value)) throw std::runtime_error("training produced a nonfinite loss");
			loss.backward();
			pending += samples.size();
			if (pending >= optimizerBatch) step();
			return value;
		}

		double loss(const std::vector<const Sample *> &samples) const {
			torch::NoGradGuard guard;
			auto data = batch(samples, device.is_cuda());
			const auto expected = data.targets.to(device, torch::kFloat32, device.is_cuda(), false);
			const auto probability = torch::sigmoid(static_cast<float>(scale) * forward(data));
			return torch::square(probability - expected).sum().item<double>();
		}

		void finishEpoch() { step(); }

		std::array<double, 2> change() const {
			torch::NoGradGuard guard;
			const auto values = parameters.detach().to(torch::kCPU).contiguous() - origin;
			const auto *data = values.data_ptr<float>();
			double square = 0.0;
			double maximum = 0.0;
			for (std::int64_t index = 0; index < values.numel(); ++index) {
				const double value = std::abs(static_cast<double>(data[index]));
				square += value * value;
				maximum = std::max(maximum, value);
			}
			return {std::sqrt(square / static_cast<double>(values.numel())), maximum};
		}

		eleginus::FormulaParameters result() const {
			torch::NoGradGuard guard;
			const auto tensor = parameters.detach().to(torch::kCPU).contiguous();
			eleginus::ParameterValues values{};
			std::copy_n(tensor.data_ptr<float>(), values.size(), values.begin());
			return eleginus::expandParameters(values);
		}

		void retain(double loss) {
			if (loss >= bestLoss) return;
			bestLoss = loss;
			best = parameters.detach().clone();
		}

		void restore() {
			torch::NoGradGuard guard;
			parameters.copy_(best);
		}

	private:
		void step() {
			if (pending == 0) return;
			parameters.grad().div_(static_cast<double>(pending));
			optimizer->step();
			optimizer->zero_grad();
			{
				torch::NoGradGuard guard;
				const auto offset = static_cast<std::int64_t>(6 * eleginus::formulaCount);
				parameters[offset].clamp_(0.0F, 64.0F);
				parameters[offset + 1].clamp_(minimumPressureWidth, 64.0F);
				parameters.slice(0, offset + 9, offset + 14).clamp_(0.0F, 1.0F);
			}
			pending = 0;
		}

		torch::Device device;
		double scale = 2.0;
		torch::Tensor origin;
		torch::Tensor parameters;
		torch::Tensor best;
		std::unique_ptr<torch::optim::Adam> optimizer;
		double bestLoss = std::numeric_limits<double>::infinity();
		std::uint64_t pending = 0;
	};

	// Measure one parameter state on a fixed collection of complete games.
	double measure(const Dataset &data, const std::vector<std::size_t> &order, Solver &solver, std::size_t batchSize) {
		std::vector<const Sample *> samples;
		samples.reserve(batchSize);
		double loss = 0.0;
		std::uint64_t count = 0;
		for (const auto game : order) {
			for (const auto &position : data.games[game]) {
				samples.push_back(&position);
				if (samples.size() == batchSize) {
					loss += solver.loss(samples);
					count += samples.size();
					samples.clear();
				}
			}
		}
		if (!samples.empty()) {
			loss += solver.loss(samples);
			count += samples.size();
		}
		return loss / static_cast<double>(count);
	}

	// Fit every training game per epoch and restore the best validation state.
	void optimize(const Dataset &data, Solver &solver, const Options &options, std::mt19937_64 &random) {
		std::vector<std::size_t> order;
		std::vector<std::size_t> validation;
		order.reserve(data.games.size());
		validation.reserve(data.games.size() / validationStride + 1);
		for (std::size_t game = 0; game < data.games.size(); ++game) {
			(game % validationStride == 0 ? validation : order).push_back(game);
		}
		std::vector<const Sample *> samples;
		const auto batchSize = static_cast<std::size_t>(options.batch);
		samples.reserve(batchSize);
		const double initialValidation = measure(data, validation, solver, batchSize);
		solver.retain(initialValidation);
		std::cout << "tuning validation: epoch=0/" << options.epochs << " loss=" << initialValidation << std::endl;
		for (int epoch = 0; epoch < options.epochs && !stopped(); ++epoch) {
			const double progress = static_cast<double>(epoch) / static_cast<double>(options.epochs);
			const double lr = 0.5 * options.lr * (1.0 + std::cos(std::numbers::pi * progress));
			solver.learningRate(lr);
			std::shuffle(order.begin(), order.end(), random);
			double loss = 0.0;
			std::uint64_t count = 0;
			for (const auto game : order) {
				for (const auto &position : data.games[game]) {
					samples.push_back(&position);
					if (samples.size() == batchSize) {
						loss += solver.train(samples);
						count += samples.size();
						samples.clear();
					}
				}
			}
			if (!samples.empty()) {
				loss += solver.train(samples);
				count += samples.size();
				samples.clear();
			}
			solver.finishEpoch();
			const double validationLoss = measure(data, validation, solver, batchSize);
			solver.retain(validationLoss);
			const auto change = solver.change();
			std::cout << "tuning epoch: epoch=" << epoch + 1 << '/' << options.epochs << " loss=" << loss / static_cast<double>(count);
			std::cout << " validation=" << validationLoss << " lr=" << lr << " positions=" << count;
			std::cout << " delta_rms=" << change[0] << " delta_max=" << change[1] << std::endl;
		}
		if (!stopped()) solver.restore();
	}

	// -------------------------- Paired-match acceptance -------------------------------
	// Polyglot opening traversal and one final candidate-versus-baseline match.

	// Decode one big-endian Polyglot field.
	std::uint64_t bigEndian(const unsigned char *bytes, int count) noexcept {
		std::uint64_t value = 0;
		for (int index = 0; index < count; ++index) value = (value << 8) | bytes[index];
		return value;
	}

	// Encode a legal move in Polyglot move format.
	std::uint16_t bookMove(chess::Move move) noexcept {
		const int promotion = move.typeOf() == chess::Move::PROMOTION ? static_cast<int>(move.promotionType()) : 0;
		return static_cast<std::uint16_t>(move.to().index() | (move.from().index() << 6) | (promotion << 12));
	}

	// Traverse the Polyglot tree and return its leaf positions in stable order.
	std::vector<chess::Board> openings(const std::filesystem::path &path) {
		std::ifstream stream(path, std::ios::binary);
		if (!stream) throw std::runtime_error("cannot open opening book: " + path.string());
		std::unordered_map<std::uint64_t, std::vector<std::uint16_t>> book;
		std::array<unsigned char, 16> entry{};
		while (stream.read(reinterpret_cast<char *>(entry.data()), entry.size())) {
			if (bigEndian(entry.data() + 10, 2) != 0) {
				book[bigEndian(entry.data(), 8)].push_back(static_cast<std::uint16_t>(bigEndian(entry.data() + 8, 2)));
			}
		}
		if (stream.bad() || stream.gcount() != 0 || !book.contains(chess::Board().hash())) throw std::runtime_error("invalid opening book");
		std::vector<chess::Board> pending{chess::Board()};
		std::vector<chess::Board> result;
		std::unordered_set<std::uint64_t> seen;
		while (!pending.empty()) {
			auto board = std::move(pending.back());
			pending.pop_back();
			if (!seen.insert(board.hash()).second) continue;
			const auto row = book.find(board.hash());
			if (row == book.end()) {
				if (eleginus::isGameOver(board)) throw std::runtime_error("opening book contains a terminal leaf");
				result.push_back(std::move(board));
				continue;
			}
			chess::Movelist legal;
			chess::movegen::legalmoves(legal, board);
			for (std::uint16_t code : row->second) {
				const auto move = std::find_if(legal.begin(), legal.end(), [code](chess::Move value) { return bookMove(value) == code; });
				if (move == legal.end()) throw std::runtime_error("opening book contains an illegal move");
				auto child = board;
				child.makeMove(*move);
				pending.push_back(std::move(child));
			}
		}
		if (result.size() != openingPairs) throw std::runtime_error("opening book must contain exactly 1000 leaf positions");
		std::sort(result.begin(), result.end(), [](const auto &left, const auto &right) { return left.hash() < right.hash(); });
		return result;
	}

	// Play one candidate-versus-baseline game and return the candidate's result.
	int match(const eleginus::Evaluator &candidate, const eleginus::Evaluator &baseline, chess::Board board, chess::Color candidateSide,
		const Options &options) {
		eleginus::SearchOptions limits;
		limits.depth = options.evaluationDepth;
		limits.hashMiB = static_cast<std::size_t>(options.hash);
		eleginus::Searcher candidateSearch(candidate, limits);
		eleginus::Searcher baselineSearch(baseline, limits);
		for (int ply = 0; ply < options.maximumPlies && !stopped(); ++ply) {
			const auto [reason, result] = board.isGameOver();
			if (reason != chess::GameResultReason::NONE) {
				if (result == chess::GameResult::DRAW) return 1;
				const auto winner = result == chess::GameResult::WIN ? board.sideToMove() : ~board.sideToMove();
				return winner == candidateSide ? 2 : 0;
			}
			auto &searcher = board.sideToMove() == candidateSide ? candidateSearch : baselineSearch;
			const auto resultMove = searcher.search(board, {}, stopped);
			if (resultMove.move.move() == chess::Move::NO_MOVE) throw std::runtime_error("acceptance search returned no legal move");
			board.makeMove(resultMove.move);
		}
		return stopped() ? -1 : 1;
	}

	// Convert a mean match score to logistic Elo.
	double elo(double score) noexcept {
		if (score <= 0.0) return -std::numeric_limits<double>::infinity();
		if (score >= 1.0) return std::numeric_limits<double>::infinity();
		return 400.0 * std::log10(score / (1.0 - score));
	}

	// Accept a candidate when its paired-match lower confidence endpoint exceeds zero Elo.
	bool accept(const eleginus::Evaluator &candidate, const eleginus::Evaluator &baseline, const std::vector<chess::Board> &book,
		const Options &options) {
		std::array<int, 5> pairs{};
		for (int first = 0; first < openingPairs && !stopped();) {
			const int count = std::min(options.workers, openingPairs - first);
			std::vector<std::future<int>> jobs;
			for (int worker = 0; worker < count; ++worker) {
				const auto index = static_cast<std::size_t>(first + worker);
				jobs.push_back(std::async(std::launch::async, [&, index] {
					const int white = match(candidate, baseline, book[index], chess::Color::WHITE, options);
					if (white < 0) return -1;
					const int black = match(candidate, baseline, book[index], chess::Color::BLACK, options);
					return black < 0 ? -1 : white + black;
				}));
			}
			for (auto &job : jobs) {
				const int result = job.get();
				if (result >= 0) ++pairs[static_cast<std::size_t>(result)];
			}
			first += count;
			if (first % 50 < count || first == openingPairs) std::cout << "acceptance step: games=" << 2 * first << "/2000" << std::endl;
		}
		if (stopped()) return false;
		double total = 0.0;
		for (std::size_t score = 0; score < pairs.size(); ++score) total += 0.25 * static_cast<double>(score) * pairs[score];
		const double mean = total / openingPairs;
		const double margin = std::sqrt(std::log(40.0) / (2.0 * openingPairs));
		const double low = elo(std::max(0.0, mean - margin));
		const double high = elo(std::min(1.0, mean + margin));
		std::cout << "acceptance result: score=" << mean << " elo=" << elo(mean) << " elo_ci95=[" << low << ',' << high << ']';
		std::cout << " accepted=" << (low > 0.0 ? "yes" : "no") << std::endl;
		return low > 0.0;
	}

} // namespace

// ------------------------------- Training run --------------------------------------
// Generate one fixed WDL corpus, optimize it for many epochs, then test the finished table once.
int main(int argc, char **argv) {
	try {
		const auto options = parse(argc, argv);
		std::signal(SIGINT, interrupt);
		std::signal(SIGTERM, interrupt);
		#ifdef _WIN32
		if (!SetConsoleCtrlHandler(nullptr, FALSE) || !SetConsoleCtrlHandler(console, TRUE)) {
			throw std::runtime_error("cannot install console interrupt handler");
		}
		#endif
		const auto initial = !options.init.empty() ? eleginus::loadParameters(options.init)
			: (std::filesystem::exists(options.out) ? eleginus::loadParameters(options.out) : eleginus::initialParameters());
		const eleginus::Evaluator baseline(initial);
		std::cout << "training start: out=" << options.out.string() << " formulas=" << eleginus::formulaCount;
		std::cout << " parameters=" << eleginus::parameterCount << " games=" << options.games << " epochs=" << options.epochs;
		std::cout << " device=" << options.device << " depth=" << options.depth << " eval_depth=" << options.evaluationDepth;
		std::cout << " update_batch=" << optimizerBatch << std::endl;

		auto data = generate(baseline, options);
		if (stopped()) {
			std::cout << "training stopped: incomplete generated data discarded" << std::endl;
			return 0;
		}
		const double scale = calibration(data);
		std::cout << "training data: games=" << data.games.size() << " positions=" << data.positions << " wdl_scale=" << scale << std::endl;
		std::mt19937_64 random(options.seed);
		Solver solver(initial, options, scale);
		optimize(data, solver, options, random);
		if (stopped()) {
			std::cout << "training stopped: incomplete optimization discarded" << std::endl;
			return 0;
		}

		const auto candidateParameters = solver.result();
		const eleginus::Evaluator candidate(candidateParameters);
		data = {};
		const auto book = openings(options.book);
		if (accept(candidate, baseline, book, options) && !stopped()) {
			eleginus::saveParameters(options.out, candidateParameters);
			std::cout << "published parameters: " << options.out.string() << std::endl;
		} else if (!stopped()) {
			std::cout << "training complete: candidate rejected; accepted checkpoint unchanged" << std::endl;
		}
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "training error: " << error.what() << std::endl;
		return 1;
	}
}
