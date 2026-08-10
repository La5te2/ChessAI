// UCI adapter with lazy embedded-weight loading and cooperatively cancellable PVS.

#include "eleginus/runtime.hpp"
#include "eleginus/search.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

void apply_position(chess::Board &board, const std::string &command) {
	std::istringstream input(command);
	std::string token;
	input >> token; // position
	input >> token;
	if (token == "startpos") {
		board = chess::Board();
		input >> token;
	} else if (token == "fen") {
		std::vector<std::string> fields;
		for (int index = 0; index < 6 && input >> token; ++index)
			fields.push_back(token);
		if (fields.size() != 6)
			throw std::invalid_argument("position fen requires six fields");
		std::string fen;
		for (std::size_t index = 0; index < fields.size(); ++index) {
			if (index != 0)
				fen += ' ';
			fen += fields[index];
		}
		board = chess::Board(fen);
		input >> token;
	}
	if (token == "moves") {
		while (input >> token)
			board.makeMove(chess::uci::uciToMove(board, token));
	}
}

bool is_no_move(const chess::Move &move) noexcept {
	return move.move() == chess::Move::NO_MOVE;
}

std::string fallback_move(const chess::Board &board) {
	const auto moves = eleginus::legal_moves(board);
	return moves.empty() ? "0000" : eleginus::move_uci(moves.front());
}

class Engine {
	public:
	Engine(int argc, char **argv) {
		for (int index = 1; index < argc; ++index) {
			const std::string argument = argv[index];
			if (index + 1 >= argc)
				throw std::invalid_argument("missing value for " + argument);
			const std::string value = argv[++index];
			if (argument == "--depth")
				pvs_depth_ = std::stoi(value);
			else if (argument == "--threads")
				threads_ = std::stoi(value);
			else if (argument == "--hash-mb")
				hash_mb_ = static_cast<std::size_t>(std::stoull(value));
			else
				throw std::invalid_argument("unknown argument: " + argument);
		}
	}

	~Engine() { stop_and_join(); }

	void run() {
		for (std::string line; std::getline(std::cin, line);) {
			try {
				if (line == "uci") {
					emit_identity();
				} else if (line == "isready") {
					ensure_model();
					print("readyok");
				} else if (line.starts_with("position ")) {
					stop_and_join();
					apply_position(board_, line);
				} else if (line.starts_with("setoption ")) {
					stop_and_join();
					set_option(line);
				} else if (line.starts_with("go")) {
					go(line);
				} else if (line == "stop") {
					stop_requested_ = true;
				} else if (line == "ucinewgame") {
					stop_and_join();
					board_ = chess::Board();
					original_time_adjust_ = -1.0;
				} else if (line == "quit") {
					break;
				}
			} catch (const std::exception &error) {
				print("info string UCI command error: " + std::string(error.what()));
				if (line.starts_with("go"))
					print("bestmove " + fallback_move(board_));
				else if (line == "isready")
					print("readyok");
			}
		}
		stop_and_join();
	}

	private:
	static void print(const std::string &line) {
		static std::mutex output_mutex;
		std::lock_guard lock(output_mutex);
		std::cout << line << std::endl;
	}

	void emit_identity() const {
		print("id name Eleginus");
		print("id author Gadidae");
		print("option name Depth type spin default 4 min 1 max 64");
		print("option name Hash type spin default 64 min 1 max 4096");
		print("option name Threads type spin default 1 min 1 max 256");
		print("option name MultiPV type spin default 5 min 1 max 256");
		print("option name Move Overhead type spin default 10 min 0 max 5000");
		print("uciok");
	}

	void ensure_model() {
		if (policy_ && value_)
			return;
		auto weights = eleginus::load_embedded_runtime_model();
		policy_ = std::make_unique<eleginus::CpuPolicy>(std::move(weights.policy));
		value_ = std::make_unique<eleginus::CpuValue>(std::move(weights.value));
	}

	void stop_and_join() {
		stop_requested_ = true;
		if (search_thread_.joinable())
			search_thread_.join();
		stop_requested_ = false;
	}

	void set_option(const std::string &line) {
		if (line.starts_with("setoption name Depth value ")) {
			constexpr std::string_view prefix = "setoption name Depth value ";
			pvs_depth_ = std::clamp(std::stoi(line.substr(prefix.size())), 1, 64);
		} else if (line.starts_with("setoption name Hash value ")) {
			constexpr std::string_view prefix = "setoption name Hash value ";
			hash_mb_ = std::clamp<std::size_t>(
				static_cast<std::size_t>(std::stoull(line.substr(prefix.size()))), 1, 4096);
		} else if (line.starts_with("setoption name Threads value ")) {
			constexpr std::string_view prefix = "setoption name Threads value ";
			threads_ = std::clamp(std::stoi(line.substr(prefix.size())), 1, 256);
		} else if (line.starts_with("setoption name MultiPV value ")) {
			constexpr std::string_view prefix = "setoption name MultiPV value ";
			multipv_ = std::clamp(std::stoi(line.substr(prefix.size())), 1, 256);
		} else if (line.starts_with("setoption name Move Overhead value ")) {
			constexpr std::string_view prefix = "setoption name Move Overhead value ";
			move_overhead_ms_ = std::clamp(std::stoi(line.substr(prefix.size())), 0, 5000);
		}
	}

	int clock_movetime(int remaining, int increment, int requested_moves) {
		if (remaining <= 0)
			return 1;
		int moves_to_go = requested_moves > 0 ? std::min(requested_moves, 50) : 50;
		if (remaining < 1000)
			moves_to_go = std::max(1, static_cast<int>(remaining * 0.05));
		const double time_left = std::max(1.0,
			static_cast<double>(remaining) + static_cast<double>(increment) * (moves_to_go - 1) -
				static_cast<double>(move_overhead_ms_) * (2 + moves_to_go));
		const int game_ply = 2 * (static_cast<int>(board_.fullMoveNumber()) - 1) +
			(board_.sideToMove() == chess::Color::BLACK ? 1 : 0);
		double optimum_scale = 0.0;
		if (requested_moves == 0) {
			if (original_time_adjust_ < 0.0)
				original_time_adjust_ = 0.3272 * std::log10(time_left) - 0.4141;
			const double log_seconds =
				std::log10(std::max(1.0, static_cast<double>(remaining)) / 1000.0);
			const double optimum_constant =
				std::min(0.0029869 + 0.00033554 * log_seconds, 0.004905);
			optimum_scale = std::min(
				0.012112 + std::pow(game_ply + 3.22713, 0.46866) * optimum_constant,
				0.19404 * remaining / time_left) * original_time_adjust_;
		} else {
			optimum_scale = std::min(
				(0.88 + game_ply / 116.4) / moves_to_go,
				0.88 * remaining / time_left);
		}
		const int optimum = static_cast<int>(std::max(1.0, optimum_scale * time_left));
		return std::min(optimum, std::max(1, remaining - move_overhead_ms_));
	}

	void emit_info(const eleginus::SearchResult &result, const Clock::time_point start) const {
		if (result.depth <= 0 || is_no_move(result.move))
			return;
		const auto elapsed = std::max<std::int64_t>(1,
			std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count());
		const auto nps = static_cast<std::uint64_t>(1000ULL * result.nodes /
			static_cast<std::uint64_t>(elapsed));
		const auto count = std::min<std::size_t>(
			static_cast<std::size_t>(multipv_), result.root.size());
		for (std::size_t index = 0; index < count; ++index) {
			const auto &root = result.root[index];
			const std::string bound = root.exact_score ? "" : " upperbound";
			print("info depth " + std::to_string(result.depth) + " seldepth " +
				std::to_string(result.selective_depth) + " multipv " +
				std::to_string(index + 1) + " score cp " + std::to_string(root.score_cp) +
				bound + " nodes " + std::to_string(result.nodes) + " nps " +
				std::to_string(nps) + " time " + std::to_string(elapsed) + " pv " +
				eleginus::move_uci(root.move));
		}
	}

	void go(const std::string &line) {
		stop_and_join();
		ensure_model();
		if (eleginus::game_is_over(board_)) {
			print("bestmove 0000");
			return;
		}

		int requested_depth = pvs_depth_;
		std::uint64_t node_limit = 0;
		int movetime_ms = 0;
		int white_time_ms = -1;
		int black_time_ms = -1;
		int white_increment_ms = 0;
		int black_increment_ms = 0;
		int moves_to_go = 0;
		bool unbounded_depth = false;
		bool explicit_depth = false;
		bool infinite = false;
		std::istringstream command(line);
		std::string token;
		command >> token; // go
		while (command >> token) {
			if (token == "depth" && command >> token) {
				requested_depth = std::stoi(token);
				explicit_depth = true;
			} else if (token == "nodes" && command >> token) {
				node_limit = std::stoull(token);
				unbounded_depth = node_limit > 0;
			} else if (token == "movetime" && command >> token) {
				const int requested = std::max(0, std::stoi(token));
				movetime_ms = requested > 0 ? std::max(1, requested - move_overhead_ms_) : 0;
				unbounded_depth = movetime_ms > 0;
			} else if (token == "wtime" && command >> token) {
				white_time_ms = std::max(0, std::stoi(token));
			} else if (token == "btime" && command >> token) {
				black_time_ms = std::max(0, std::stoi(token));
			} else if (token == "winc" && command >> token) {
				white_increment_ms = std::max(0, std::stoi(token));
			} else if (token == "binc" && command >> token) {
				black_increment_ms = std::max(0, std::stoi(token));
			} else if (token == "movestogo" && command >> token) {
				moves_to_go = std::max(1, std::stoi(token));
			} else if (token == "infinite") {
				unbounded_depth = true;
				infinite = true;
			}
		}
		if (!infinite && movetime_ms == 0) {
			const bool white = board_.sideToMove() == chess::Color::WHITE;
			const int remaining = white ? white_time_ms : black_time_ms;
			const int increment = white ? white_increment_ms : black_increment_ms;
			if (remaining >= 0) {
				movetime_ms = clock_movetime(remaining, increment, moves_to_go);
				unbounded_depth = true;
			}
		}
		if (unbounded_depth && !explicit_depth)
			requested_depth = 64;

		eleginus::SearchOptions options;
		options.depth = requested_depth;
		options.hash_mb = hash_mb_;
		options.node_limit = node_limit;
		options.threads = threads_;
		options.multipv = multipv_;
		const auto board = board_;
		const auto start = Clock::now();
		const auto deadline = movetime_ms > 0
			? start + std::chrono::milliseconds(movetime_ms)
			: Clock::time_point::max();
		const auto *policy = policy_.get();
		const auto *value = value_.get();
		stop_requested_ = false;
		search_thread_ = std::thread([this, board, options, start, deadline, policy, value] {
			try {
				const auto result = eleginus::Searcher(*policy, *value, options).search(
					board,
					[this, start](const eleginus::SearchResult &partial) {
						emit_info(partial, start);
					},
					[this, deadline] {
						return stop_requested_.load() || Clock::now() >= deadline;
					});
				print("bestmove " + (is_no_move(result.move)
					? fallback_move(board)
					: eleginus::move_uci(result.move)));
			} catch (const std::exception &error) {
				print("info string go error: " + std::string(error.what()));
				print("bestmove " + fallback_move(board));
			}
		});
	}

	int pvs_depth_ = 4;
	std::size_t hash_mb_ = 64;
	int threads_ = 1;
	int multipv_ = 5;
	int move_overhead_ms_ = 10;
	double original_time_adjust_ = -1.0;
	chess::Board board_;
	std::unique_ptr<eleginus::CpuPolicy> policy_;
	std::unique_ptr<eleginus::CpuValue> value_;
	std::thread search_thread_;
	std::atomic_bool stop_requested_{false};
};

} // namespace

int main(int argc, char **argv) {
	try {
		Engine(argc, argv).run();
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "Eleginus UCI failed: " << error.what() << '\n';
		return 1;
	}
}
