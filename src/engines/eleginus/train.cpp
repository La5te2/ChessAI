#include "eleginus/game.hpp"
#include "eleginus/model.hpp"
#include "eleginus/search.hpp"
#include "eleginus/match.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <future>
#include <iostream>
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

	std::atomic<bool> halt{false};
	static_assert(std::atomic<bool>::is_always_lock_free);

	bool stopped() {
		return halt.load(std::memory_order_relaxed);
	}

	void interrupt(int) {
		halt.store(true, std::memory_order_relaxed);
	}

#ifdef _WIN32
	BOOL WINAPI console(DWORD event) {
		if (event != CTRL_C_EVENT && event != CTRL_BREAK_EVENT) return FALSE;
		interrupt(0);
		return TRUE;
	}
#endif

	struct Options {
		std::filesystem::path out = "models/eleginus/current.pth", init, book = "data/openings.gen.bin";
		int depth = 2, plies = 320, hash = 16, every = 1000, evalDepth = 4, discover = 64;
		int workers = static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
		int batch = 256, log = 10;
		float lr = 1.0e-3F, decay = 1.0e-6F, sparse = 1.0e-5F, clip = 1.0F, explore = 0.08F, temp = 80.0F;
		std::uint64_t seed = 2026;
	};

	Options parse(int argc, char **argv) {
		Options o;
		for (int i = 1; i < argc; ++i) {
			const std::string arg = argv[i];
			if (arg == "--help") {
				std::cout << "usage: train [--out models/eleginus/current.pth] [--init model.pth] [--depth 2]\n";
				std::cout << "  [--opening-book data/openings.gen.bin] [--eval-every 1000] [--eval-depth 4]\n";
				std::cout << "  [--max-plies 320] [--workers N] [--hash 16] [--batch-size 256] [--lr 0.001]\n";
				std::cout << "  [--weight-decay 0.000001] [--relation-l1 0.00001] [--discover-every 64] [--grad-clip 1]\n";
				std::cout << "  [--exploration 0.08] [--temperature 80]\n";
				std::cout << "  [--log-every 10] [--seed 2026]\n";
				std::cout << "Train until Ctrl+C. Evaluate 2000 paired games after every K completed training games.\n";
				std::cout << "Only candidates with a positive 95% Elo lower bound replace --out. No save on exit.\n";
				std::exit(0);
			}
			if (++i == argc) throw std::invalid_argument("missing value after " + arg);
			const std::string val = argv[i];
			if (arg == "--out") o.out = val;
			else if (arg == "--init") o.init = val;
			else if (arg == "--opening-book") o.book = val;
			else if (arg == "--eval-every") o.every = std::stoi(val);
			else if (arg == "--eval-depth") o.evalDepth = std::stoi(val);
			else if (arg == "--depth") o.depth = std::stoi(val);
			else if (arg == "--max-plies") o.plies = std::stoi(val);
			else if (arg == "--hash") o.hash = std::stoi(val);
			else if (arg == "--workers") o.workers = std::stoi(val);
			else if (arg == "--batch-size") o.batch = std::stoi(val);
			else if (arg == "--lr") o.lr = std::stof(val);
			else if (arg == "--weight-decay") o.decay = std::stof(val);
			else if (arg == "--relation-l1") o.sparse = std::stof(val);
			else if (arg == "--discover-every") o.discover = std::stoi(val);
			else if (arg == "--grad-clip") o.clip = std::stof(val);
			else if (arg == "--exploration") o.explore = std::stof(val);
			else if (arg == "--temperature") o.temp = std::stof(val);
			else if (arg == "--log-every") o.log = std::stoi(val);
			else if (arg == "--seed") o.seed = std::stoull(val);
			else throw std::invalid_argument("unknown option: " + arg);
		}
		for (float x : {o.lr, o.decay, o.sparse, o.clip, o.explore, o.temp}) {
			if (!std::isfinite(x)) throw std::invalid_argument("nonfinite training option");
		}
		if (o.out.empty() || o.book.empty() || o.every < 1 || o.discover < 1 || o.evalDepth < 1 || o.evalDepth > 64 || o.depth < 1 || o.depth > 16 ||
			o.plies < 1 || o.plies > 320 || o.hash < 1 || o.hash > 4096 || o.workers < 1 || o.workers > 256 || o.batch < 1 || o.batch > 65536 ||
			o.lr <= 0 || o.decay < 0 || o.lr * o.decay >= 1 || o.clip <= 0 || o.sparse < 0 || o.explore < 0 || o.explore > 1 || o.temp < 0 ||
			o.log < 1) {
			throw std::invalid_argument("invalid or incomplete Eleginus training options");
		}
		return o;
	}

	struct Sample {
		std::vector<eleginus::Feature> x;
		float y = 0;
	};

	struct Game {
		std::vector<Sample> samples;
		int result = -1; // -1 unfinished, 0 black win, 1 draw, 2 white win.
	};

	class Prox {
	public:
		explicit Prox(std::size_t n) : grad(n) {}
		void sync(std::size_t n) { grad.resize(n); }

		void step(eleginus::Model &model, std::size_t count, const Options &o) {
			if (count == 0) return;
			if (grad.size() != model.params().size()) grad.resize(model.params().size());
			double norm = 0;
			for (auto &x : grad) {
				x /= static_cast<float>(count);
				if (!std::isfinite(x)) throw std::runtime_error("nonfinite gradient");
				norm += static_cast<double>(x) * x;
			}
			const float scale = static_cast<float>(std::min(1.0, o.clip / std::max(std::sqrt(norm), 1.0e-12)));
			auto &p = model.params();
			const auto initial = eleginus::Model::initial();
			for (std::size_t i = 0; i < model.formulas(); ++i) {
				p[i] -= o.lr * (grad[i] * scale + o.decay * (p[i] - initial[i]));
				if (!std::isfinite(p[i])) throw std::runtime_error("nonfinite parameter update");
			}
			for (std::size_t i = model.formulas(); i < p.size(); ++i) {
				const float z = p[i] - o.lr * (grad[i] * scale + o.decay * p[i]);
				p[i] = std::copysign(std::max(0.0F, std::abs(z) - o.lr * o.sparse), z);
			}
			std::fill(grad.begin(), grad.end(), 0.0F);
			model.prune(0.0F);
			grad.resize(model.params().size());
		}

	public:
		std::vector<float> grad;
	};

	std::uint64_t seedFor(std::uint64_t seed, std::uint64_t game) {
		auto x = seed + 0x9e3779b97f4a7c15ULL * (game + 1);
		x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
		x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
		return x ^ (x >> 31);
	}

	chess::Move choose(const eleginus::SearchResult &r, const Options &o, std::mt19937_64 &rng) {
		if (r.root.empty()) return chess::Move(chess::Move::NO_MOVE);
		if (std::uniform_real_distribution<float>(0, 1)(rng) < o.explore) {
			return r.root[std::uniform_int_distribution<std::size_t>(0, r.root.size() - 1)(rng)].move;
		}
		if (o.temp == 0) return r.move;
		std::vector<double> mass;
		const auto best = r.root.front().score_cp;
		for (const auto &a : r.root) {
			mass.push_back(std::exp((a.score_cp - best) / static_cast<double>(o.temp)));
		}
		return r.root[std::discrete_distribution<std::size_t>(mass.begin(), mass.end())(rng)].move;
	}

	Game play(const eleginus::Model &model, const Options &o, std::uint64_t seed) {
		Game game;
		chess::Board board;
		eleginus::SearchOptions limits;
		limits.depth = o.depth;
		limits.hash_mb = static_cast<std::size_t>(o.hash);
		limits.collect_leaf = true;
		// Temperature sampling requires scores, not the bounds returned by a root null-window search.
		limits.multipv = o.temp > 0 ? 256 : 1;
		eleginus::Searcher search(model, limits);
		std::mt19937_64 rng(seed);
		for (int ply = 0; ply < o.plies && !stopped() && !eleginus::isGameOver(board); ++ply) {
			const auto result = search.search(board, {}, stopped);
			if (stopped()) break;
			const auto move = choose(result, o, rng);
			if (move.move() == chess::Move::NO_MOVE) break;
			Sample sample;
			sample.x = result.leaf;
			if (sample.x.empty()) model.extract(board, sample.x);
			game.samples.push_back(std::move(sample));
			board.makeMove(move);
		}
		const auto [reason, result] = board.isGameOver();
		if (stopped() || reason == chess::GameResultReason::NONE) {
			game.samples.clear();
			return game;
		}
		if (result == chess::GameResult::DRAW) game.result = 1;
		else {
			const auto winner = result == chess::GameResult::WIN ? board.sideToMove() : ~board.sideToMove();
			game.result = winner == chess::Color::WHITE ? 2 : 0;
		}
		const float white = 0.5F * game.result;
		for (auto &s : game.samples) {
			s.y = white;
		}
		return game;
	}

	double learn(eleginus::Model &model, Prox &solver, const std::deque<Game> &replay, std::size_t count, const Options &o, std::mt19937_64 &rng) {
		std::vector<const Sample *> pool;
		for (const auto &g : replay) {
			for (const auto &s : g.samples) {
				pool.push_back(&s);
			}
		}
		if (pool.empty() || count == 0) return 0;
		std::uniform_int_distribution<std::size_t> pick(0, pool.size() - 1);
		double loss = 0;
		std::size_t used = 0;
		eleginus::Model::Cache cache;
		for (std::size_t start = 0; start < count && !stopped(); start += o.batch) {
			const auto size = std::min(count - start, static_cast<std::size_t>(o.batch));
			for (std::size_t i = 0; i < size; ++i) {
				const auto &s = *pool[pick(rng)];
				const float h = model.forward(s.x, cache);
				if (!std::isfinite(h)) throw std::runtime_error("nonfinite training score");
				// Stable BCE on logits; terminal scores may also equal 1/2.
				loss += std::max(h, 0.0F) - h * s.y + std::log1p(std::exp(-std::abs(h)));
				const float e = std::exp(-std::abs(h));
				const float p = h >= 0 ? 1.0F / (1.0F + e) : e / (1.0F + e);
				model.backward(cache, p - s.y, solver.grad);
			}
			solver.step(model, size, o);
			used += size;
		}
		return used ? loss / static_cast<double>(used) : 0;
	}

	std::size_t discover(eleginus::Model &model, const std::deque<Game> &replay, const Options &o) {
		constexpr std::size_t sampleLimit = 2048, addLimit = 16;
		const auto n = model.formulas();
		std::size_t total = 0;
		for (const auto &game : replay) {
			total += game.samples.size();
		}
		if (total == 0 || model.relations().size() == eleginus::kRelationLimit) return 0;
		const std::size_t stride = std::max<std::size_t>(1, total / sampleLimit);
		std::vector<double> correlation(n * n);
		std::size_t visited = 0, used = 0;
		for (const auto &game : replay) {
			for (const auto &sample : game.samples) {
				if (visited++ % stride != 0 || used == sampleLimit) continue;
				const float h = model.score(sample.x);
				const float e = std::exp(-std::abs(h));
				const float p = h >= 0 ? 1.0F / (1.0F + e) : e / (1.0F + e);
				const double residual = p - sample.y;
				for (const auto &row : sample.x) {
					if (row.score == 0) continue;
					for (const auto &condition : sample.x) {
						if (condition.condition != 0) {
							correlation[static_cast<std::size_t>(row.index) * n + condition.index] +=
								residual * static_cast<double>(row.score) * condition.condition;
						}
					}
				}
				++used;
			}
		}
		if (used == 0) return 0;
		std::vector<std::uint8_t> active(n * n);
		for (const auto &relation : model.relations()) {
			active[static_cast<std::size_t>(relation.row) * n + relation.condition] = 1;
		}
		struct Candidate {
			double gain;
			std::uint16_t row, condition;
		};
		std::vector<Candidate> candidates;
		candidates.reserve(n * n - model.relations().size());
		for (std::size_t row = 0; row < n; ++row) {
			for (std::size_t condition = 0; condition < n; ++condition) {
				const auto index = row * n + condition;
				if (!active[index]) {
					const double gain = std::abs(correlation[index]) / static_cast<double>(used) - o.sparse;
					if (gain > 0) candidates.push_back({gain, static_cast<std::uint16_t>(row), static_cast<std::uint16_t>(condition)});
				}
			}
		}
		const auto count = std::min({addLimit, candidates.size(), eleginus::kRelationLimit - model.relations().size()});
		std::partial_sort(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(count), candidates.end(),
			[](const Candidate &a, const Candidate &b) { return a.gain > b.gain; });
		std::size_t added = 0;
		for (std::size_t i = 0; i < count; ++i) {
			added += model.activate(candidates[i].row, candidates[i].condition);
		}
		return added;
	}

	bool approve(const eleginus::Model &model, const eleginus::Model &baseline, const std::vector<chess::Board> &book, const Options &o) {
		std::array<int, 5> pairs{};
		eleginus::SearchOptions limits;
		limits.depth = o.evalDepth;
		limits.hash_mb = static_cast<std::size_t>(o.hash);
		std::cout << "evaluation start: games=2000 openings=1000 depth=" << o.evalDepth << std::endl;
		int logged = 0;
		for (int start = 0; start < eleginus::kOpeningPairs && !stopped();) {
			const int count = std::min(o.workers, eleginus::kOpeningPairs - start);
			std::vector<std::future<int>> jobs;
			for (int i = 0; i < count; ++i) {
				const auto index = static_cast<std::size_t>(start + i);
				jobs.push_back(std::async(std::launch::async, [&, index] {
					const int white = eleginus::match(model, baseline, book[index], chess::Color::WHITE, limits, stopped);
					if (white < 0) return -1;
					const int black = eleginus::match(model, baseline, book[index], chess::Color::BLACK, limits, stopped);
					return black < 0 ? -1 : white + black;
				}));
			}
			for (auto &job : jobs) {
				const int score = job.get();
				if (score >= 0) ++pairs[score];
			}
			start += count;
			if (!stopped() && (start - logged >= 50 || start == eleginus::kOpeningPairs)) {
				std::cout << "evaluation step: games=" << 2 * start << "/2000" << std::endl;
				logged = start;
			}
		}
		if (stopped()) {
			std::cout << "evaluation cancelled: current unchanged" << std::endl;
			return false;
		}
		const auto result = eleginus::confidence(pairs);
		const bool accepted = result.low > 0;
		std::cout << "evaluation result: score=" << result.score << " elo=" << result.elo;
		std::cout << " elo_ci95=[" << result.low << "," << result.high << "] ";
		std::cout << "accepted=" << (accepted ? "yes" : "no") << std::endl;
		return accepted;
	}

} // namespace

int main(int argc, char **argv) {
	try {
		const auto o = parse(argc, argv);
		std::signal(SIGINT, interrupt);
		std::signal(SIGTERM, interrupt);
		#ifdef _WIN32
		if (!SetConsoleCtrlHandler(nullptr, FALSE) || !SetConsoleCtrlHandler(console, TRUE)) {
			throw std::runtime_error("cannot install console interrupt handler");
		}
		#endif
		const auto book = eleginus::openings(o.book);
		const bool hasCurrent = std::filesystem::exists(o.out);
		auto baseline = hasCurrent ? eleginus::Model::load(o.out) : (o.init.empty() ? eleginus::Model() : eleginus::Model::load(o.init));
		auto model = o.init.empty() ? baseline : eleginus::Model::load(o.init);
		Prox solver(model.params().size());
		std::mt19937_64 rng(o.seed);
		std::deque<Game> replay;
		constexpr std::size_t history = 64; // Bounded RAM-only replay, never part of a checkpoint.
		std::uint64_t attempted = 0, completed = 0, discarded = 0, white = 0, black = 0, draws = 0, logged = 0, discovered = 0;
		int since = 0;
		std::size_t samples = 0;
		std::cout << "self-play start: out=" << o.out.string() << " formulas=" << model.formulas();
		std::cout << " parameters=" << model.params().size() << " depth=" << o.depth << " workers=" << o.workers;
		std::cout << " seed=" << o.seed << " eval_every=" << o.every << " eval_depth=" << o.evalDepth;
		std::cout << " baseline=" << (hasCurrent ? "current" : "initial") << std::endl;
		while (!stopped()) {
			const auto batch = std::min(o.workers, o.every - since);
			std::vector<std::future<Game>> jobs;
			for (int i = 0; i < batch; ++i) {
				const auto seed = seedFor(o.seed, attempted + i);
				jobs.push_back(std::async(std::launch::async, [&model, &o, seed] { return play(model, o, seed); }));
			}
			std::vector<Game> games;
			for (auto &job : jobs) {
				games.push_back(job.get());
			}
			if (stopped()) break;
			// All searches have finished before parameters change.
			std::size_t count = 0;
			for (auto &game : games) {
				if (game.result < 0) {
					++discarded;
					continue;
				}
				++completed;
				++since;
				white += game.result == 2;
				draws += game.result == 1;
				black += game.result == 0;
				count += game.samples.size();
				replay.push_back(std::move(game));
				if (replay.size() > history) replay.pop_front();
			}
			if (completed / static_cast<std::uint64_t>(o.discover) > discovered) {
				const auto added = discover(model, replay, o);
				solver.sync(model.params().size());
				discovered = completed / static_cast<std::uint64_t>(o.discover);
				if (added) std::cout << "relation discovery: added=" << added << " active=" << model.relations().size() << std::endl;
			}
			const auto loss = learn(model, solver, replay, count, o, rng);
			samples += count;
			attempted += batch;
			if (attempted - logged >= static_cast<std::uint64_t>(o.log) || since == o.every) {
				std::cout << "self-play step: games=" << attempted << " completed=" << completed << " discarded=" << discarded;
				std::cout << " positions=" << samples << " white=" << white << " draws=" << draws << " black=" << black;
				if (count) std::cout << " bce=" << loss;
				std::cout << std::endl;
				logged = attempted;
			}
			if (since == o.every && !stopped()) {
				if (approve(model, baseline, book, o) && !stopped()) {
					model.save(o.out);
					baseline = model;
					std::cout << "published model: " << o.out.string() << std::endl;
				}
				since = 0;
			}
		}
		std::cout << "training stopped: no progress saved; current unchanged by exit" << std::endl;
		return 0;
	} catch (const std::exception &e) {
		std::cerr << "training error: " << e.what() << '\n';
		return 1;
	}
}
