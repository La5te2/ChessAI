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
	// Convert search scores from centipawns to the internal HCE scale.
	constexpr float cpScale = 150.0F;
	// Number of opening positions used by each paired acceptance match.
	constexpr int openingPairs = 1000;

	// Set the shared stop flag from process and console interrupts.
	void interrupt(int) noexcept { halt.store(true, std::memory_order_relaxed); }
	bool stopped() noexcept { return halt.load(std::memory_order_relaxed); }

	#ifdef _WIN32
	BOOL WINAPI console(DWORD event) {
		if (event != CTRL_C_EVENT && event != CTRL_BREAK_EVENT) return FALSE;
		interrupt(0);
		return TRUE;
	}
	#endif

	// Command-line controls for generation, optimization and acceptance.
	struct Options {
		std::filesystem::path out = "models/eleginus/current.pth";
		std::filesystem::path init;
		std::filesystem::path book = "data/openings.gen.bin";
		std::string device = "auto";
		int games = 100000;
		int randomPlies = 8;
		int depth = 1;
		int counterfactualDepth = 2;
		int counterfactualEvery = 16;
		int evaluationDepth = 4;
		int maximumPlies = 320;
		int workers = static_cast<int>(std::clamp(std::thread::hardware_concurrency() / 2U, 1U, 4U));
		int hash = 16;
		int epochs = 2;
		int batch = 4096;
		int iterations = 0;
		int log = 1000;
		float lr = 1.0e-4F;
		float decay = 1.0e-6F;
		float clip = 0.25F;
		std::uint64_t seed = 2026;
	};

	std::string valueAfter(int argc, char **argv, int &index) {
		if (++index == argc) throw std::invalid_argument(std::string("missing value after ") + argv[index - 1]);
		return argv[index];
	}

	// Parse and validate the command-line controls.
	Options parse(int argc, char **argv) {
		Options options;
		for (int index = 1; index < argc; ++index) {
			const std::string argument = argv[index];
			if (argument == "--help") {
				std::cout << "usage: train [--out current.pth] [--init parameters.pth] [options]\n";
				std::cout << "  --opening-book openings.gen.bin --games 100000 --random-plies 8 --depth 1\n";
				std::cout << "  --counterfactual-depth 2 --counterfactual-every 16 --eval-depth 4\n";
				std::cout << "  --max-plies 320 --workers 4 --hash 16 --epochs 2 --batch-size 4096\n";
				std::cout << "  --device auto --lr 0.0001 --weight-decay 0.000001 --grad-clip 0.25\n";
				std::cout << "  --iterations 0 --log-every 1000 --seed 2026\n";
				std::exit(0);
			}
			const auto value = valueAfter(argc, argv, index);
			if (argument == "--out") options.out = value;
			else if (argument == "--init") options.init = value;
			else if (argument == "--opening-book") options.book = value;
			else if (argument == "--games") options.games = std::stoi(value);
			else if (argument == "--random-plies") options.randomPlies = std::stoi(value);
			else if (argument == "--depth") options.depth = std::stoi(value);
			else if (argument == "--counterfactual-depth") options.counterfactualDepth = std::stoi(value);
			else if (argument == "--counterfactual-every") options.counterfactualEvery = std::stoi(value);
			else if (argument == "--eval-depth") options.evaluationDepth = std::stoi(value);
			else if (argument == "--max-plies") options.maximumPlies = std::stoi(value);
			else if (argument == "--workers") options.workers = std::stoi(value);
			else if (argument == "--hash") options.hash = std::stoi(value);
			else if (argument == "--epochs") options.epochs = std::stoi(value);
			else if (argument == "--batch-size") options.batch = std::stoi(value);
			else if (argument == "--iterations") options.iterations = std::stoi(value);
			else if (argument == "--device") options.device = value;
			else if (argument == "--lr") options.lr = std::stof(value);
			else if (argument == "--weight-decay") options.decay = std::stof(value);
			else if (argument == "--grad-clip") options.clip = std::stof(value);
			else if (argument == "--log-every") options.log = std::stoi(value);
			else if (argument == "--seed") options.seed = std::stoull(value);
			else throw std::invalid_argument("unknown option: " + argument);
		}
		for (float value : {options.lr, options.decay, options.clip}) {
			if (!std::isfinite(value)) throw std::invalid_argument("nonfinite training option");
		}
		if (options.out.empty() || options.book.empty() || (options.device != "auto" && options.device != "cpu" && options.device != "cuda") ||
			options.games < 10 || options.randomPlies < 0 || options.randomPlies > 64 || options.depth < 1 || options.depth > 16 ||
			options.counterfactualDepth < 1 || options.counterfactualDepth > 16 || options.counterfactualEvery < 1 ||
			options.evaluationDepth < 1 || options.evaluationDepth > 64 || options.maximumPlies <= options.randomPlies || options.maximumPlies > 320 ||
			options.workers < 1 || options.workers > 256 || options.hash < 0 || options.hash > 4096 || options.epochs < 1 || options.batch < 1 ||
			options.batch > 65536 || options.iterations < 0 || options.log < 1 || options.lr <= 0.0F || options.decay < 0.0F || options.clip <= 0.0F) {
			throw std::invalid_argument("invalid or incomplete Eleginus training options");
		}
		return options;
	}

	// Derive a deterministic random seed from the run seed and game id.
	std::uint64_t gameSeed(std::uint64_t seedValue, std::uint64_t game) noexcept {
		auto value = seedValue + 0x9e3779b97f4a7c15ULL * (game + 1);
		value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
		value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
		return value ^ (value >> 31);
	}

	// Map an HCE score to the [-1, 1] target range.
	float bounded(float score) noexcept { return std::tanh(score); }

	// Formula values and up to two independent targets for one board position.
	struct Sample {
		eleginus::FormulaValues values;
		std::array<float, 2> targets{};
		float perspective = 1.0F;
		std::uint8_t targetCount = 0;
	};

	// Main-line and alternative positions retained from one completed game.
	struct Game {
		std::vector<Sample> main;
		std::vector<Sample> alternatives;
		int result = -1;
	};

	// Convert a board and one side-to-move target into a training sample.
	Sample sample(const eleginus::Evaluator &evaluator, const chess::Board &board, float target) {
		Sample result;
		result.values = evaluator.extract(board);
		result.targets[0] = target;
		result.perspective = board.sideToMove() == chess::Color::WHITE ? 1.0F : -1.0F;
		result.targetCount = 1;
		return result;
	}

	// Play one complete game and retain every main-line position plus every legal alternative at sampled roots.
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
		std::vector<chess::Board> roots;
		for (int ply = options.randomPlies; ply < options.maximumPlies && !stopped() && !eleginus::isGameOver(board); ++ply) {
			Sample position;
			position.values = evaluator.extract(board);
			position.perspective = board.sideToMove() == chess::Color::WHITE ? 1.0F : -1.0F;
			const auto result = searcher.search(board, {}, stopped);
			if (stopped() || result.move.move() == chess::Move::NO_MOVE) return Game{};
			if (game.main.size() % static_cast<std::size_t>(options.counterfactualEvery) == 0) roots.push_back(board);
			game.main.push_back(std::move(position));
			board.makeMove(result.move);
		}

		const auto [reason, result] = board.isGameOver();
		if (stopped() || reason == chess::GameResultReason::NONE) return Game{};
		if (result == chess::GameResult::DRAW) game.result = 1;
		else {
			const auto winner = result == chess::GameResult::WIN ? board.sideToMove() : ~board.sideToMove();
			game.result = winner == chess::Color::WHITE ? 2 : 0;
		}

		const float whiteResult = static_cast<float>(game.result - 1);
		for (std::size_t index = 0; index < game.main.size(); ++index) {
			auto &position = game.main[index];
			position.targets[0] = position.perspective * whiteResult;
			if (index + 1 < game.main.size()) {
				const auto &next = game.main[index + 1];
				position.targets[1] = -bounded(next.perspective * evaluator.score(next.values));
			} else {
				position.targets[1] = position.targets[0];
			}
			position.targetCount = 2;
		}

		auto expandedLimits = limits;
		expandedLimits.depth = options.counterfactualDepth;
		expandedLimits.multipv = 256;
		eleginus::Searcher expandedSearcher(evaluator, expandedLimits);
		for (const auto &root : roots) {
			const auto expanded = expandedSearcher.search(root, {}, stopped);
			if (stopped() || expanded.root.empty()) return Game{};
			for (const auto &line : expanded.root) {
				auto child = root;
				child.makeMove(line.move);
				game.alternatives.push_back(sample(evaluator, child, -bounded(static_cast<float>(line.scoreCp) / cpScale)));
			}
		}
		return game;
	}

	// ----------------------------- Tensor optimizer -----------------------------------
	// Resolve the requested Torch device.
	torch::Device resolveDevice(const std::string &requested) {
		if (requested == "auto") return torch::Device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
		if (requested == "cuda" && !torch::cuda::is_available()) throw std::runtime_error("CUDA training was requested but CUDA is unavailable");
		return torch::Device(requested);
	}

	// Dense CPU tensors holding one optimizer batch before device transfer.
	struct Batch {
		torch::Tensor signals;
		torch::Tensor material;
		torch::Tensor pressure;
		torch::Tensor endgame;
		torch::Tensor perspective;
		torch::Tensor targets;
		torch::Tensor mask;
		std::size_t pressureFormula = eleginus::formulaCount;
	};

	// Pack a sample range and its active targets into contiguous float tensors.
	Batch batch(const std::vector<Sample> &samples, std::size_t first, std::size_t count, bool pinned) {
		const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(pinned);
		Batch result{
			torch::empty({static_cast<std::int64_t>(count), static_cast<std::int64_t>(eleginus::formulaCount)}, options),
			torch::empty({static_cast<std::int64_t>(count), 5}, options),
			torch::empty({static_cast<std::int64_t>(count), 2}, options),
			torch::empty({static_cast<std::int64_t>(count), 14}, options),
			torch::empty({static_cast<std::int64_t>(count)}, options),
			torch::zeros({static_cast<std::int64_t>(count), 2}, options),
			torch::zeros({static_cast<std::int64_t>(count), 2}, options),
		};
		auto *signals = result.signals.data_ptr<float>();
		auto *material = result.material.data_ptr<float>();
		auto *pressure = result.pressure.data_ptr<float>();
		auto *endgame = result.endgame.data_ptr<float>();
		auto *perspective = result.perspective.data_ptr<float>();
		auto *targets = result.targets.data_ptr<float>();
		auto *mask = result.mask.data_ptr<float>();
		for (std::size_t row = 0; row < count; ++row) {
			const auto &sample = samples[first + row];
			const auto &values = sample.values;
			if (result.pressureFormula == eleginus::formulaCount) result.pressureFormula = values.pressure.formula;
			if (values.pressure.formula != result.pressureFormula) throw std::runtime_error("inconsistent king-pressure formula index");
			for (std::size_t column = 0; column < eleginus::formulaCount; ++column) {
				signals[row * eleginus::formulaCount + column] = static_cast<float>(values.signals[column]);
			}
			std::copy(values.material.begin(), values.material.end(), material + row * 5);
			pressure[2 * row] = static_cast<float>(values.pressure.attacks[0]);
			pressure[2 * row + 1] = static_cast<float>(values.pressure.attacks[1]);
			const auto &value = values.endgame;
			const std::array<float, 14> facts{{static_cast<float>(value.pawns), static_cast<float>(value.symmetricFiles),
				static_cast<float>(value.asymmetricFiles), static_cast<float>(value.pawnEnding), static_cast<float>(value.sidePawns[0]),
				static_cast<float>(value.sidePawns[1]), static_cast<float>(value.sidePassers[0]), static_cast<float>(value.sidePassers[1]),
				static_cast<float>(value.oppositeBishops), static_cast<float>(value.pawnless[0]), static_cast<float>(value.pawnless[1]),
				static_cast<float>(value.thinMaterial), static_cast<float>(value.pureOppositeBishops),
				static_cast<float>(value.mixedOppositeBishops)}};
			std::copy(facts.begin(), facts.end(), endgame + row * facts.size());
			perspective[row] = sample.perspective;
			for (std::size_t target = 0; target < sample.targetCount; ++target) {
				targets[2 * row + target] = sample.targets[target];
				mask[2 * row + target] = 1.0F;
			}
		}
		return result;
	}

	struct TrainStats {
		double loss = 0.0;
		std::uint64_t constraints = 0;
		std::uint64_t steps = 0;
	};

	// Optimize a dense parameter delta from one accepted parameter vector.
	class Solver {
	public:
		Solver(const eleginus::FormulaParameters &parameters, const Options &options)
			: device(resolveDevice(options.device)), clip(options.clip) {
			const auto values = eleginus::flattenParameters(parameters);
			const auto cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
			origin = torch::from_blob(const_cast<float *>(values.data()), {static_cast<std::int64_t>(values.size())}, cpu).clone().to(device);
			delta = torch::zeros_like(origin).set_requires_grad(true);
			optimizer = std::make_unique<torch::optim::AdamW>(
				std::vector<torch::Tensor>{delta}, torch::optim::AdamWOptions(options.lr).weight_decay(options.decay));
		}

		const torch::Device &trainingDevice() const noexcept { return device; }

		// Evaluate a tensor batch with material-adjusted coefficients, king pressure, winnability and scaling.
		// The straight-through rounded sigmoid supplies gradients for its center and width.
		torch::Tensor forward(const Batch &source) const {
			const auto parameters = origin + delta;
			const auto formula = parameters.slice(0, 0, static_cast<std::int64_t>(6 * eleginus::formulaCount))
				.view({static_cast<std::int64_t>(eleginus::formulaCount), 6});
			auto signals = source.signals.to(device, torch::kFloat32, device.is_cuda(), true);
			const auto material = source.material.to(device, torch::kFloat32, device.is_cuda(), false);
			const auto pressure = source.pressure.to(device, torch::kFloat32, device.is_cuda(), false);
			const auto facts = source.endgame.to(device, torch::kFloat32, device.is_cuda(), false);
			const auto perspective = source.perspective.to(device, torch::kFloat32, device.is_cuda(), false);
			const auto weights = formula.select(1, 0).unsqueeze(0) + torch::matmul(material, formula.slice(1, 1, 6).transpose(0, 1));
			const auto offset = static_cast<std::int64_t>(6 * eleginus::formulaCount);
			const auto center = parameters[offset];
			const auto width = torch::clamp_min(parameters[offset + 1], 0.02);
			const auto pressureResponse = [&](const torch::Tensor &attacks) {
				const auto smooth = 4096.0F * torch::sigmoid((torch::clamp(attacks, 0.0F, 64.0F) - center) / width);
				const auto rounded = torch::round(smooth);
				return smooth + (rounded - smooth).detach();
			};
			const auto pressureSignal = pressureResponse(pressure.select(1, 0)) - pressureResponse(pressure.select(1, 1));
			signals.select(1, static_cast<std::int64_t>(source.pressureFormula)).copy_(pressureSignal);
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
			const auto scale = parameters.slice(0, offset + 9, offset + 14);
			auto factor = torch::ones_like(total);
			factor = torch::where(pawnless.gt(0.0F) * facts.select(1, 11).gt(0.0F), torch::minimum(factor, scale[0]), factor);
			const auto pure = scale[1] + scale[4] * strongPawns + scale[3] * strongPassers;
			const auto mixed = scale[2] + scale[3] * strongPassers;
			factor = torch::where(facts.select(1, 12).gt(0.0F), torch::minimum(factor, pure),
				torch::where(facts.select(1, 13).gt(0.0F), torch::minimum(factor, mixed), factor));
			return torch::tanh(total * torch::clamp(factor, 0.0F, 1.0F) * perspective);
		}

		// Optimize every active target in the supplied sample block once per epoch.
		TrainStats train(std::vector<Sample> &samples, const Options &options, std::mt19937_64 &random) {
			TrainStats stats;
			for (int epoch = 0; epoch < options.epochs && !stopped(); ++epoch) {
				std::shuffle(samples.begin(), samples.end(), random);
				for (std::size_t first = 0; first < samples.size() && !stopped(); first += static_cast<std::size_t>(options.batch)) {
					const auto count = std::min(static_cast<std::size_t>(options.batch), samples.size() - first);
					std::uint64_t constraints = 0;
					for (std::size_t row = 0; row < count; ++row) constraints += samples[first + row].targetCount;
					auto data = batch(samples, first, count, device.is_cuda());
					const auto expected = data.targets.to(device, torch::kFloat32, device.is_cuda(), false);
					const auto mask = data.mask.to(device, torch::kFloat32, device.is_cuda(), false);
					const auto difference = forward(data).unsqueeze(1) - expected;
					const auto absolute = torch::abs(difference);
					const auto elements = torch::where(absolute.lt(1.0F), 0.5F * difference * difference, absolute - 0.5F);
					const auto active = mask.sum();
					const auto loss = (elements * mask).sum() / active;
					optimizer->zero_grad();
					loss.backward();
					torch::nn::utils::clip_grad_norm_(std::vector<torch::Tensor>{delta}, clip);
					optimizer->step();
					const double value = loss.item<double>();
					if (!std::isfinite(value)) throw std::runtime_error("training produced a nonfinite loss");
					stats.loss += value * static_cast<double>(constraints);
					stats.constraints += constraints;
					++stats.steps;
				}
			}
			if (stats.constraints != 0) stats.loss /= static_cast<double>(stats.constraints);
			return stats;
		}

		// Return the optimized parameter vector with a valid sigmoid width.
		eleginus::FormulaParameters parameters() const {
			torch::NoGradGuard guard;
			const auto tensor = (origin + delta).to(torch::kCPU).contiguous();
			eleginus::ParameterValues values{};
			std::copy_n(tensor.data_ptr<float>(), values.size(), values.begin());
			values[6 * eleginus::formulaCount + 1] = std::max(values[6 * eleginus::formulaCount + 1], 0.02F);
			return eleginus::expandParameters(values);
		}

	private:
		torch::Device device;
		float clip;
		torch::Tensor origin;
		torch::Tensor delta;
		std::unique_ptr<torch::optim::AdamW> optimizer;
	};

	struct CycleStats {
		std::uint64_t games = 0;
		std::uint64_t discarded = 0;
		std::uint64_t positions = 0;
		std::uint64_t counterfactual = 0;
		std::uint64_t targets = 0;
		std::uint64_t steps = 0;
		double loss = 0.0;
	};

	// Train and release one bounded block while preserving one optimizer across the complete generation cycle.
	void flush(std::vector<Sample> &samples, Solver &solver, const Options &options, std::mt19937_64 &random, CycleStats &stats) {
		if (samples.empty()) return;
		const auto trained = solver.train(samples, options, random);
		stats.loss += trained.loss * static_cast<double>(trained.constraints);
		stats.steps += trained.steps;
		samples.clear();
	}

	// Generate independent games under one reference parameter set and optimize every retained target.
	CycleStats optimize(const eleginus::Evaluator &reference, Solver &solver, const Options &options, std::uint64_t iteration,
		std::mt19937_64 &random) {
		CycleStats stats;
		std::uint64_t attempted = 0;
		std::uint64_t nextLog = static_cast<std::uint64_t>(options.log);
		const std::size_t blockSize = static_cast<std::size_t>(options.batch) * 8;
		std::vector<Sample> samples;
		samples.reserve(blockSize);

		while (stats.games < static_cast<std::uint64_t>(options.games) && !stopped()) {
			const int count = std::min<int>(options.workers, options.games - static_cast<int>(stats.games));
			std::vector<std::future<Game>> jobs;
			jobs.reserve(static_cast<std::size_t>(count));
			for (int worker = 0; worker < count; ++worker) {
				const auto id = iteration * 1000000000ULL + attempted + static_cast<std::uint64_t>(worker);
				jobs.push_back(std::async(std::launch::async, [&reference, &options, id] { return play(reference, options, id); }));
			}
			attempted += static_cast<std::uint64_t>(count);
			for (auto &job : jobs) {
				auto game = job.get();
				if (game.result < 0) {
					++stats.discarded;
					continue;
				}
				++stats.games;
				stats.positions += game.main.size();
				stats.counterfactual += game.alternatives.size();
				stats.targets += 2 * game.main.size() + game.alternatives.size();
				for (auto &position : game.main) samples.push_back(std::move(position));
				for (auto &position : game.alternatives) samples.push_back(std::move(position));
				if (samples.size() >= blockSize) flush(samples, solver, options, random, stats);
			}
			if (stats.games >= nextLog || stats.games == static_cast<std::uint64_t>(options.games)) {
				std::cout << "generation step: games=" << stats.games << '/' << options.games << " discarded=" << stats.discarded;
				std::cout << " positions=" << stats.positions << " counterfactual=" << stats.counterfactual;
				std::cout << " targets=" << stats.targets << std::endl;
				while (nextLog <= stats.games) nextLog += static_cast<std::uint64_t>(options.log);
			}
		}
		flush(samples, solver, options, random, stats);
		const auto totalConstraints = static_cast<double>(options.epochs) * static_cast<double>(stats.targets);
		if (totalConstraints != 0.0) stats.loss /= totalConstraints;
		return stats;
	}

	// -------------------------- Paired-match acceptance -------------------------------
	// Decode one unsigned big-endian Polyglot field.
	std::uint64_t bigEndian(const unsigned char *bytes, int count) noexcept {
		std::uint64_t value = 0;
		for (int index = 0; index < count; ++index) value = (value << 8) | bytes[index];
		return value;
	}

	// Encode one legal move in Polyglot move format.
	std::uint16_t bookMove(chess::Move move) noexcept {
		const int promotion = move.typeOf() == chess::Move::PROMOTION ? static_cast<int>(move.promotionType()) : 0;
		return static_cast<std::uint16_t>(move.to().index() | (move.from().index() << 6) | (promotion << 12));
	}

	// Load and traverse the Polyglot book to obtain its 1000 leaf positions.
	std::vector<chess::Board> openings(const std::filesystem::path &path) {
		std::ifstream stream(path, std::ios::binary);
		if (!stream) throw std::runtime_error("cannot open opening book: " + path.string());
		std::unordered_map<std::uint64_t, std::vector<std::uint16_t>> book;
		std::array<unsigned char, 16> entry{};
		while (stream.read(reinterpret_cast<char *>(entry.data()), entry.size())) {
			if (bigEndian(entry.data() + 10, 2) != 0) book[bigEndian(entry.data(), 8)].push_back(static_cast<std::uint16_t>(bigEndian(entry.data() + 8, 2)));
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

	// Play one fixed-depth game and return loss, draw or win from the candidate's perspective.
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

	// Convert a mean game score to logistic Elo.
	double elo(double score) noexcept {
		if (score <= 0.0) return -std::numeric_limits<double>::infinity();
		if (score >= 1.0) return std::numeric_limits<double>::infinity();
		return 400.0 * std::log10(score / (1.0 - score));
	}

	// Run paired games and accept a candidate whose 95% lower score bound exceeds 50%.
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


// ------------------------------- Training loop -------------------------------------
// Generate a large sample cycle, optimize every target and publish the candidate after one paired match.
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
		auto current = !options.init.empty() ? eleginus::loadParameters(options.init)
			: (std::filesystem::exists(options.out) ? eleginus::loadParameters(options.out) : eleginus::initialParameters());
		const auto book = openings(options.book);
		std::mt19937_64 random(options.seed);
		std::cout << "training start: out=" << options.out.string() << " formulas=" << eleginus::formulaCount;
		std::cout << " parameters=" << eleginus::parameterCount << " games=" << options.games << " device=" << options.device;
		std::cout << " depth=" << options.depth << " counterfactual_depth=" << options.counterfactualDepth;
		std::cout << " eval_depth=" << options.evaluationDepth << std::endl;
		for (int iteration = 0; !stopped() && (options.iterations == 0 || iteration < options.iterations); ++iteration) {
			const eleginus::Evaluator reference(current);
			Solver solver(current, options);
			const auto stats = optimize(reference, solver, options, static_cast<std::uint64_t>(iteration), random);
			if (stopped()) break;
			std::cout << "optimization result: iteration=" << iteration << " loss=" << stats.loss << " targets=" << stats.targets;
			std::cout << " steps=" << stats.steps << " device=" << solver.trainingDevice() << std::endl;
			const auto candidateParameters = solver.parameters();
			const eleginus::Evaluator candidate(candidateParameters);
			if (accept(candidate, reference, book, options) && !stopped()) {
				eleginus::saveParameters(options.out, candidateParameters);
				current = candidateParameters;
				std::cout << "published parameters: " << options.out.string() << std::endl;
			}
		}
		if (stopped()) std::cout << "training stopped: accepted checkpoint unchanged by incomplete work" << std::endl;
		else if (std::filesystem::exists(options.out)) std::cout << "training complete: accepted checkpoint=" << options.out.string() << std::endl;
		else std::cout << "training complete: no candidate accepted; no checkpoint written" << std::endl;
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "training error: " << error.what() << std::endl;
		return 1;
	}
}
