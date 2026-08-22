// Exposes Gadus checkpoints as a standards-oriented UCI process for GUI and bot clients.

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include "gadus/args.hpp"
#include "gadus/checkpoint.hpp"
#include "gadus/search.hpp"

namespace {

struct EngineOptions {
	std::filesystem::path model_path;
	gadus::SearchOptions search;
	int move_overhead_ms = 10;
	int multipv = 5;
};

constexpr int kProgressIntervalMs = 300;
constexpr float kCentipawnScale = 90.0F;
constexpr float kCentipawnAngle = 1.5637541897F;

// Resolve the explicitly packaged Gadus checkpoint beside the executable without
// introducing a repository fallback or coupling the model name to the EXE name.
std::filesystem::path sidecar_model_path(const char *argv0) {
	std::filesystem::path executable = argv0 == nullptr ? std::filesystem::path() : argv0;
	std::error_code error;
	const auto absolute = std::filesystem::absolute(executable, error);
	if (!error) {
		executable = absolute;
	}
	return executable.parent_path() / "gadus.pth";
}

// Remove surrounding ASCII whitespace from protocol input and option values.
std::string trim(std::string value) {
	const auto first = value.find_first_not_of(" \t\r\n");
	if (first == std::string::npos) {
		return {};
	}
	const auto last = value.find_last_not_of(" \t\r\n");
	return value.substr(first, last - first + 1);
}

// Tokenize one UCI command on whitespace.
std::vector<std::string> split(const std::string &line) {
	std::istringstream stream(line);
	std::vector<std::string> tokens;
	for (std::string token; stream >> token;) {
		tokens.push_back(std::move(token));
	}
	return tokens;
}

// Canonicalize option names by keeping lowercase alphanumeric characters only.
std::string normalized(std::string value) {
	std::string output;
	for (const unsigned char character : value) {
		if (std::isalnum(character)) {
			output.push_back(static_cast<char>(std::tolower(character)));
		}
	}
	return output;
}

// Parse an integer option without letting malformed client input terminate the engine.
int parse_int(const std::string &value, int fallback) {
	try {
		return std::stoi(value);
	} catch (...) {
		return fallback;
	}
}

// Parse a nonnegative simulation count and clamp it to the search counter range.
int parse_simulations(const std::string &value, int fallback) {
	try {
		const auto parsed = std::stoll(value);
		return static_cast<int>(std::clamp<std::int64_t>(
			parsed, 0, std::numeric_limits<int>::max()));
	} catch (...) {
		return fallback;
	}
}

// Map the synthetic UCI depth reported by Gadus back to its simulation count.
int simulations_for_depth(const std::string &value) {
	const int depth = std::max(1, parse_int(value, 1));
	if (depth > std::numeric_limits<int>::digits) {
		return std::numeric_limits<int>::max();
	}
	return 1 << (depth - 1);
}

class UciEngine {
	public:
	// Store initial options; model loading remains lazy until readiness or search.
	explicit UciEngine(EngineOptions options) : options_(std::move(options)) {}
	~UciEngine() {
		stop_and_join();
	}

	// Dispatch UCI commands until quit or end-of-input, reporting command errors as info strings.
	void loop() {
		for (std::string line; std::getline(std::cin, line);) {
			line = trim(line);
			if (line.empty()) {
				continue;
			}
			const auto separator = line.find(' ');
			const auto command = line.substr(0, separator);
			try {
				if (command == "uci") {
					emit_identity();
				} else if (command == "isready") {
					ensure_model();
					print("readyok");
				} else if (command == "setoption") {
					stop_and_join();
					set_option(line);
				} else if (command == "ucinewgame") {
					stop_and_join();
					board_ = chess::Board();
					original_time_adjust_ = -1.0;
					if (searcher_) {
						searcher_->clear_evaluation_cache();
					}
				} else if (command == "position") {
					stop_and_join();
					set_position(line);
				} else if (command == "go") {
					go(line);
				} else if (command == "stop") {
					stop_requested_ = true;
				} else if (command == "quit") {
					break;
				} else if (command == "debug" || command == "ponderhit" ||
						   command == "register") {
					continue;
				} else {
					print("info string unknown command: " + line);
				}
			} catch (const std::exception &error) {
				print("info string " + command + " error: " + error.what());
				if (command == "go") {
					print("bestmove " + fallback_move());
				} else if (command == "isready") {
					print("readyok");
				}
			}
		}
		stop_and_join();
	}

	private:
	// Emit and flush one complete protocol line.
	static void print(const std::string &text) {
		static std::mutex output_mutex;
		std::lock_guard lock(output_mutex);
		std::cout << text << std::endl;
	}

	// Request cooperative cancellation and join the search before mutable UCI state changes.
	void stop_and_join() {
		stop_requested_ = true;
		if (search_thread_.joinable()) {
			search_thread_.join();
		}
		stop_requested_ = false;
	}

	// Advertise engine identity and every configurable UCI option before uciok.
	void emit_identity() const {
		print("id name Gadidae Gadus");
		print("id author La5te2");
		print("option name ModelPath type string default " + options_.model_path.string());
		print("option name Threads type spin default " +
			  std::to_string(options_.search.cpu_threads) + " min 1 max 256");
		print("option name Hash type spin default " +
			  std::to_string(options_.search.evaluation_cache_mb) + " min 0 max 65536");
		print("option name Sims type spin default " + std::to_string(options_.search.mcts_sims) +
			  " min 0 max " + std::to_string(std::numeric_limits<int>::max()));
		print("option name Move Overhead type spin default " +
			  std::to_string(options_.move_overhead_ms) + " min 0 max 5000");
		print("option name MultiPV type spin default " + std::to_string(options_.multipv) +
			  " min 1 max 256");
		print("uciok");
	}

	// Load or reload the checkpoint only when model path or device changed.
	void ensure_model() {
		if (options_.model_path.empty()) {
			throw std::runtime_error("ModelPath is empty");
		}
		if (model_ && searcher_ && loaded_model_path_ == options_.model_path) {
			return;
		}
		device_ = gadus::resolve_device("auto");
		model_ = gadus::load_checkpoint(options_.model_path, device_);
		searcher_ = std::make_shared<gadus::Searcher>(model_, device_, options_.search);
		loaded_model_path_ = options_.model_path;
	}

	// Parse setoption name/value and update the matching typed engine setting.
	void set_option(const std::string &line) {
		const auto name_at = line.find(" name ");
		if (name_at == std::string::npos) {
			return;
		}
		const auto value_at = line.find(" value ", name_at + 6);
		const auto name = trim(line.substr(name_at + 6, value_at == std::string::npos
													? std::string::npos
													: value_at - name_at - 6));
		const auto value = value_at == std::string::npos ? std::string() : trim(line.substr(value_at + 7));
		const auto key = normalized(name);
		if (key == "modelpath") {
			options_.model_path = value;
			model_ = nullptr;
			searcher_.reset();
		} else if (key == "threads") {
			options_.search.cpu_threads =
				std::clamp(parse_int(value, options_.search.cpu_threads), 1, 256);
		} else if (key == "hash") {
			options_.search.evaluation_cache_mb =
				std::clamp(parse_int(value, options_.search.evaluation_cache_mb), 0, 65536);
		} else if (key == "sims") {
			options_.search.mcts_sims = parse_simulations(value, options_.search.mcts_sims);
		} else if (key == "moveoverhead") {
			options_.move_overhead_ms =
				std::clamp(parse_int(value, options_.move_overhead_ms), 0, 5000);
		} else if (key == "multipv") {
			options_.multipv = std::clamp(parse_int(value, options_.multipv), 1, 256);
		} else {
			print("info string unknown option: " + name);
		}
		if (searcher_) {
			searcher_->set_options(options_.search);
		}
	}

	// Reconstruct startpos/FEN and apply the optional legal UCI move sequence.
	void set_position(const std::string &line) {
		const auto tokens = split(line);
		if (tokens.size() < 2) {
			return;
		}
		std::size_t move_at = tokens.size();
		for (std::size_t index = 2; index < tokens.size(); ++index) {
			if (tokens[index] == "moves") {
				move_at = index;
				break;
			}
		}
		if (tokens[1] == "startpos") {
			board_ = chess::Board();
		} else if (tokens[1] == "fen") {
			std::ostringstream fen;
			for (std::size_t index = 2; index < move_at; ++index) {
				if (index > 2) {
					fen << ' ';
				}
				fen << tokens[index];
			}
			board_ = chess::Board(fen.str());
		} else {
			throw std::invalid_argument("unsupported position command");
		}
		if (move_at < tokens.size()) {
			for (std::size_t index = move_at + 1; index < tokens.size(); ++index) {
				board_.makeMove(chess::uci::uciToMove(board_, tokens[index]));
			}
		}
	}

	// Parse go tokens into a lookup table, preserving valueless flags such as infinite.
	std::unordered_map<std::string, std::string> parse_go(const std::string &line) const {
		const auto tokens = split(line);
		std::unordered_map<std::string, std::string> values;
		for (std::size_t index = 1; index < tokens.size(); ++index) {
			if (tokens[index] == "infinite" || tokens[index] == "ponder") {
				values[tokens[index]] = "1";
			} else if (index + 1 < tokens.size()) {
				const auto key = tokens[index];
				values[key] = tokens[index + 1];
				++index;
			}
		}
		return values;
	}

	// Apply Stockfish's optimum-time allocation to the active side's UCI clock fields.
	int movetime_for(const std::unordered_map<std::string, std::string> &go) {
		if (go.contains("infinite")) {
			return 0;
		}
		if (const auto found = go.find("movetime"); found != go.end()) {
			const int requested = std::max(0, parse_int(found->second, 0));
			return requested > 0 ? std::max(1, requested - options_.move_overhead_ms) : 0;
		}
		const bool white = board_.sideToMove() == chess::Color::WHITE;
		const auto time_key = white ? "wtime" : "btime";
		const auto increment_key = white ? "winc" : "binc";
		if (const auto found = go.find(time_key); found != go.end()) {
			const int remaining = std::max(0, parse_int(found->second, 0));
			const int increment = go.contains(increment_key)
								  ? std::max(0, parse_int(go.at(increment_key), 0))
								  : 0;
			if (remaining == 0) {
				return 1;
			}
			const int requested_moves = go.contains("movestogo")
				? std::max(1, parse_int(go.at("movestogo"), 1))
				: 0;
			int moves_to_go = requested_moves > 0 ? std::min(requested_moves, 50) : 50;
			if (remaining < 1000) {
				moves_to_go = std::max(1, static_cast<int>(remaining * 0.05));
			}
			const double time_left = std::max(1.0,
				static_cast<double>(remaining) + static_cast<double>(increment) * (moves_to_go - 1) -
					static_cast<double>(options_.move_overhead_ms) * (2 + moves_to_go));
			const int game_ply = 2 * (static_cast<int>(board_.fullMoveNumber()) - 1) +
				(board_.sideToMove() == chess::Color::BLACK ? 1 : 0);
			double optimum_scale = 0.0;
			if (requested_moves == 0) {
				if (original_time_adjust_ < 0.0) {
					original_time_adjust_ = 0.3272 * std::log10(time_left) - 0.4141;
				}
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
			return std::min(optimum, std::max(1, remaining - options_.move_overhead_ms));
		}
		return 0;
	}

	// Expand bounded root values into the conventional centipawn display range.
	int score_cp(float value) const {
		return static_cast<int>(std::lround(
			kCentipawnScale * std::tan(kCentipawnAngle * std::clamp(value, -1.0F, 1.0F))));
	}

	// Emit final or progressive MultiPV lines using root-side values and search statistics.
	void emit_info(const gadus::SearchResult &result) const {
		const int elapsed = std::max(0, static_cast<int>(std::lround(result.elapsed_ms)));
		const int nodes = result.sims_completed;
		const int nps = static_cast<int>(1000LL * nodes / std::max(1, elapsed));
		const int depth = std::max(1, static_cast<int>(std::log2(std::max(1, nodes))) + 1);
		const int count = std::min<int>(options_.multipv, result.root.size());
		for (int index = 0; index < count; ++index) {
			const auto &row = result.root[index];
			const float value = row.visits > 0 ? row.q : result.value;
			print("info depth " + std::to_string(depth) + " seldepth " + std::to_string(depth) +
				  " multipv " + std::to_string(index + 1) + " score cp " +
				  std::to_string(score_cp(value)) + " nodes " + std::to_string(nodes) + " nps " +
				  std::to_string(nps) + " time " + std::to_string(elapsed) + " pv " +
				  gadus::move_uci(row.move));
		}
	}

	// Start search on a worker so the protocol loop can process stop while MCTS is running.
	void go(const std::string &line) {
		stop_and_join();
		if (gadus::game_is_over(board_)) {
			print("bestmove 0000");
			return;
		}
		ensure_model();
		auto search_options = options_.search;
		const auto go_values = parse_go(line);
		search_options.movetime_ms = movetime_for(go_values);
		search_options.unbounded_simulations = false;
		if (const auto nodes = go_values.find("nodes"); nodes != go_values.end()) {
			search_options.mcts_sims = parse_simulations(nodes->second, search_options.mcts_sims);
		} else if (const auto depth = go_values.find("depth"); depth != go_values.end()) {
			search_options.mcts_sims = simulations_for_depth(depth->second);
		} else if (go_values.contains("infinite") && search_options.mcts_sims > 0) {
			search_options.unbounded_simulations = true;
		}
		search_options.root_topn = std::max(options_.multipv, search_options.root_topn);
		const auto board = board_;
		searcher_->set_options(search_options);
		const auto searcher = searcher_;
		const int progress_interval_ms = kProgressIntervalMs;
		stop_requested_ = false;
		search_thread_ = std::thread([this, board, searcher, progress_interval_ms] {
			try {
				const auto result = searcher->search(
					board, [this](const gadus::SearchResult &partial) { emit_info(partial); },
					progress_interval_ms, [this] { return stop_requested_.load(); });
				emit_info(result);
				print("bestmove " + gadus::move_uci(result.move));
			} catch (const std::exception &error) {
				print("info string go error: " + std::string(error.what()));
				const auto moves = gadus::legal_moves(board);
				print("bestmove " +
					  (moves.empty() ? std::string("0000") : gadus::move_uci(moves.front())));
			}
		});
	}

	// Supply a deterministic legal move only when command recovery needs a protocol response.
	std::string fallback_move() const {
		const auto moves = gadus::legal_moves(board_);
		return moves.empty() ? "0000" : gadus::move_uci(moves.front());
	}

	EngineOptions options_;
	chess::Board board_;
	torch::Device device_{torch::kCPU};
	gadus::Model model_{nullptr};
	std::shared_ptr<gadus::Searcher> searcher_;
	std::filesystem::path loaded_model_path_;
	std::thread search_thread_;
	std::atomic_bool stop_requested_{false};
	double original_time_adjust_ = -1.0;
};

// Convert process arguments to initial UCI options before entering the protocol loop.
EngineOptions options_from_args(int argc, char **argv) {
	gadus::Args args(argc, argv);
	EngineOptions options;
	options.model_path = args.get("model");
	if (options.model_path.empty()) {
		options.model_path = sidecar_model_path(argc > 0 ? argv[0] : nullptr);
		if (!std::filesystem::is_regular_file(options.model_path)) {
			throw std::invalid_argument(
				"--model is required when the sidecar checkpoint does not exist: " +
				options.model_path.string());
		}
	}
	options.search.cpu_threads = 2;
	options.search.evaluation_cache_mb = 256;
	options.search.c_puct = 1.0;
	options.search.repetition_policy_penalty = 1.0;
	options.search.instant_mate_first = true;
	return options;
}

} // namespace

// Start the Gadus UCI process and convert fatal initialization errors to stderr/exit failure.
int main(int argc, char **argv) {
	try {
		UciEngine(options_from_args(argc, argv)).loop();
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "uci error: " << error.what() << std::endl;
		return 1;
	}
}
