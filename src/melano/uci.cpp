// Exposes Melano checkpoints as a standards-oriented UCI process for GUI and bot clients.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "melano/args.hpp"
#include "melano/checkpoint.hpp"
#include "melano/search.hpp"

namespace {

struct EngineOptions {
	std::filesystem::path model_path;
	std::string device = "auto";
	melano::SearchOptions search;
	int move_overhead_ms = 50;
	int min_movetime_ms = 50;
	int max_movetime_ms = 10000;
	double time_divisor = 30.0;
	double increment_fraction = 0.75;
	int progress_interval_ms = 750;
	int multipv = 5;
	int score_scale = 1000;
};

// Resolve the explicitly packaged Melano checkpoint beside the executable without
// introducing a repository fallback or coupling the model name to the EXE name.
std::filesystem::path sidecar_model_path(const char *argv0) {
	std::filesystem::path executable = argv0 == nullptr ? std::filesystem::path() : argv0;
	std::error_code error;
	const auto absolute = std::filesystem::absolute(executable, error);
	if (!error) {
		executable = absolute;
	}
	return executable.parent_path() / "melano.pth";
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

// Parse permissive UCI Boolean text while retaining a caller-provided fallback.
bool parse_bool(const std::string &value, bool fallback) {
	const auto key = normalized(value);
	if (key == "1" || key == "true" || key == "yes" || key == "on") {
		return true;
	}
	if (key == "0" || key == "false" || key == "no" || key == "off") {
		return false;
	}
	return fallback;
}

// Parse an integer option without letting malformed client input terminate the engine.
int parse_int(const std::string &value, int fallback) {
	try {
		return std::stoi(value);
	} catch (...) {
		return fallback;
	}
}

// Parse a floating option without letting malformed client input terminate the engine.
double parse_double(const std::string &value, double fallback) {
	try {
		return std::stod(value);
	} catch (...) {
		return fallback;
	}
}

class UciEngine {
	public:
	// Store initial options; model loading remains lazy until readiness or search.
	explicit UciEngine(EngineOptions options) : options_(std::move(options)) {}

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
					set_option(line);
				} else if (command == "ucinewgame") {
					board_ = chess::Board();
				} else if (command == "position") {
					set_position(line);
				} else if (command == "go") {
					go(line);
				} else if (command == "stop") {
					continue;
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
	}

	private:
	// Emit and flush one complete protocol line.
	static void print(const std::string &text) {
		std::cout << text << std::endl;
	}

	// Advertise engine identity and every configurable UCI option before uciok.
	void emit_identity() const {
		print("id name Gadidae Melano");
		print("id author La5te2");
		print("option name ModelPath type string default " + options_.model_path.string());
		print("option name Device type string default " + options_.device);
		print("option name SearchType type combo default " +
			  melano::search_type_name(options_.search.type) + " var closed var only-mcts");
		print("option name MCTSSims type spin default " + std::to_string(options_.search.mcts_sims) +
			  " min 0 max 1000000");
		print("option name MCTSMinSims type spin default " +
			  std::to_string(options_.search.mcts_min_sims) + " min 0 max 1000000");
		print("option name MCTSBatchSize type spin default " +
			  std::to_string(options_.search.mcts_batch_size) + " min 1 max 4096");
		print("option name MoveTimeMS type spin default " +
			  std::to_string(static_cast<int>(options_.search.movetime_ms)) +
			  " min 0 max 3600000");
		print("option name MoveOverheadMS type spin default " +
			  std::to_string(options_.move_overhead_ms) + " min 0 max 60000");
		print("option name MinMoveTimeMS type spin default " +
			  std::to_string(options_.min_movetime_ms) + " min 0 max 3600000");
		print("option name MaxMoveTimeMS type spin default " +
			  std::to_string(options_.max_movetime_ms) + " min 0 max 3600000");
		print("option name TimeDivisor type string default " + std::to_string(options_.time_divisor));
		print("option name IncrementFraction type string default " +
			  std::to_string(options_.increment_fraction));
		print("option name CPuct type string default " + std::to_string(options_.search.c_puct));
		print("option name CPuctBase type string default " +
			  std::to_string(options_.search.c_puct_base));
		print("option name CPuctFactor type string default " +
			  std::to_string(options_.search.c_puct_factor));
		print("option name FPUReduction type string default " +
			  std::to_string(options_.search.fpu_reduction));
		print("option name VirtualLoss type string default " +
			  std::to_string(options_.search.virtual_loss));
		print("option name RepetitionPolicyPenalty type string default " +
			  std::to_string(options_.search.repetition_policy_penalty));
		print(std::string("option name InstantMateFirst type check default ") +
			  (options_.search.instant_mate_first ? "true" : "false"));
		print("option name ProgressIntervalMS type spin default " +
			  std::to_string(options_.progress_interval_ms) + " min 0 max 60000");
		print("option name MultiPV type spin default " + std::to_string(options_.multipv) +
			  " min 1 max 256");
		print("option name ScoreScale type spin default " + std::to_string(options_.score_scale) +
			  " min 1 max 100000");
		print("uciok");
	}

	// Load or reload the checkpoint only when model path or device changed.
	void ensure_model() {
		if (options_.model_path.empty()) {
			throw std::runtime_error("ModelPath is empty");
		}
		if (model_ && loaded_model_path_ == options_.model_path && loaded_device_ == options_.device) {
			return;
		}
		device_ = melano::resolve_device(options_.device);
		model_ = melano::load_checkpoint(options_.model_path, device_);
		loaded_model_path_ = options_.model_path;
		loaded_device_ = options_.device;
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
		} else if (key == "device") {
			options_.device = value;
			model_ = nullptr;
		} else if (key == "searchtype") {
			options_.search.type = melano::parse_search_type(value);
		} else if (key == "mctssims") {
			options_.search.mcts_sims = std::max(0, parse_int(value, options_.search.mcts_sims));
		} else if (key == "mctsminsims") {
			options_.search.mcts_min_sims = std::max(0, parse_int(value, options_.search.mcts_min_sims));
		} else if (key == "mctsbatchsize") {
			options_.search.mcts_batch_size = std::max(1, parse_int(value, options_.search.mcts_batch_size));
		} else if (key == "movetimems") {
			options_.search.movetime_ms = std::max(0, parse_int(value, static_cast<int>(options_.search.movetime_ms)));
		} else if (key == "moveoverheadms") {
			options_.move_overhead_ms = std::max(0, parse_int(value, options_.move_overhead_ms));
		} else if (key == "minmovetimems") {
			options_.min_movetime_ms = std::max(0, parse_int(value, options_.min_movetime_ms));
		} else if (key == "maxmovetimems") {
			options_.max_movetime_ms = std::max(0, parse_int(value, options_.max_movetime_ms));
		} else if (key == "timedivisor") {
			options_.time_divisor = std::max(1.0, parse_double(value, options_.time_divisor));
		} else if (key == "incrementfraction") {
			options_.increment_fraction = std::max(0.0, parse_double(value, options_.increment_fraction));
		} else if (key == "cpuct") {
			options_.search.c_puct = std::max(0.0, parse_double(value, options_.search.c_puct));
		} else if (key == "cpuctbase") {
			options_.search.c_puct_base = std::max(1.0, parse_double(value, options_.search.c_puct_base));
		} else if (key == "cpuctfactor") {
			options_.search.c_puct_factor = std::max(0.0, parse_double(value, options_.search.c_puct_factor));
		} else if (key == "fpureduction") {
			options_.search.fpu_reduction = std::max(0.0, parse_double(value, options_.search.fpu_reduction));
		} else if (key == "virtualloss") {
			options_.search.virtual_loss = std::max(0.0, parse_double(value, options_.search.virtual_loss));
		} else if (key == "repetitionpolicypenalty") {
			options_.search.repetition_policy_penalty =
				std::clamp(parse_double(value, options_.search.repetition_policy_penalty), 0.0, 1.0);
		} else if (key == "instantmatefirst") {
			options_.search.instant_mate_first = parse_bool(value, options_.search.instant_mate_first);
		} else if (key == "progressintervalms") {
			options_.progress_interval_ms =
				std::clamp(parse_int(value, options_.progress_interval_ms), 0, 60000);
		} else if (key == "multipv") {
			options_.multipv = std::clamp(parse_int(value, options_.multipv), 1, 256);
		} else if (key == "scorescale") {
			options_.score_scale = std::clamp(parse_int(value, options_.score_scale), 1, 100000);
		} else {
			print("info string unknown option: " + name);
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
				values[tokens[index]] = tokens[++index];
			}
		}
		return values;
	}

	// Derive a bounded per-move budget from movetime or the active side's clock and increment.
	int movetime_for(const std::unordered_map<std::string, std::string> &go) const {
		if (const auto found = go.find("movetime"); found != go.end()) {
			return std::max(0, parse_int(found->second, static_cast<int>(options_.search.movetime_ms)));
		}
		const bool white = board_.sideToMove() == chess::Color::WHITE;
		const auto time_key = white ? "wtime" : "btime";
		const auto increment_key = white ? "winc" : "binc";
		if (const auto found = go.find(time_key); found != go.end()) {
			const int remaining = std::max(0, parse_int(found->second, 0));
			const int increment = go.contains(increment_key)
								  ? std::max(0, parse_int(go.at(increment_key), 0))
								  : 0;
			double budget = remaining / options_.time_divisor +
							increment * options_.increment_fraction - options_.move_overhead_ms;
			if (options_.max_movetime_ms > 0) {
				budget = std::min(budget, static_cast<double>(options_.max_movetime_ms));
			}
			if (options_.min_movetime_ms > 0) {
				budget = std::max(budget, static_cast<double>(options_.min_movetime_ms));
			}
			budget = std::min(budget, static_cast<double>(std::max(1, remaining - options_.move_overhead_ms)));
			return std::max(0, static_cast<int>(budget));
		}
		return std::max(0, static_cast<int>(options_.search.movetime_ms));
	}

	// Map bounded neural value to a monotonic centipawn-like UCI display score.
	int score_cp(float value) const {
		return static_cast<int>(std::lround(std::clamp(value, -0.999F, 0.999F) * options_.score_scale));
	}

	// Emit final or progressive MultiPV lines using root-side values and search statistics.
	void emit_info(const melano::SearchResult &result) const {
		const int elapsed = std::max(0, static_cast<int>(std::lround(result.elapsed_ms)));
		const int nodes = std::max(result.sims_completed, result.nn_batches > 0 ? 1 : 0);
		const int nps = static_cast<int>(1000LL * nodes / std::max(1, elapsed));
		const int depth = std::max(1, static_cast<int>(std::log2(std::max(1, nodes))) + 1);
		const int count = std::min<int>(options_.multipv, result.root.size());
		for (int index = 0; index < count; ++index) {
			const auto &row = result.root[index];
			const float value = row.visits > 0 ? row.q : result.value;
			print("info depth " + std::to_string(depth) + " seldepth " + std::to_string(depth) +
				  " multipv " + std::to_string(index + 1) + " score cp " +
				  std::to_string(score_cp(value)) + " nodes " + std::to_string(nodes) + " nps " +
				  std::to_string(nps) + " time " + std::to_string(elapsed) +
				  " hashfull 0 tbhits 0 pv " + melano::move_uci(row.move));
		}
	}

	// Run one search under go overrides, emit progress, then publish exactly one bestmove.
	void go(const std::string &line) {
		if (melano::game_is_over(board_)) {
			print("bestmove 0000");
			return;
		}
		ensure_model();
		auto search_options = options_.search;
		const auto go_values = parse_go(line);
		search_options.movetime_ms = movetime_for(go_values);
		if (const auto nodes = go_values.find("nodes"); nodes != go_values.end()) {
			search_options.mcts_sims = std::max(0, parse_int(nodes->second, search_options.mcts_sims));
		}
		search_options.root_topn = std::max(options_.multipv, search_options.root_topn);
		melano::Searcher searcher(model_, device_, search_options);
		const auto result = searcher.search(
			board_, [this](const melano::SearchResult &partial) { emit_info(partial); },
			options_.progress_interval_ms);
		emit_info(result);
		print("bestmove " + melano::move_uci(result.move));
	}

	// Supply a deterministic legal move only when command recovery needs a protocol response.
	std::string fallback_move() const {
		const auto moves = melano::legal_moves(board_);
		return moves.empty() ? "0000" : melano::move_uci(moves.front());
	}

	EngineOptions options_;
	chess::Board board_;
	torch::Device device_{torch::kCPU};
	melano::Model model_{nullptr};
	std::filesystem::path loaded_model_path_;
	std::string loaded_device_;
};

// Convert process arguments to initial UCI options before entering the protocol loop.
EngineOptions options_from_args(int argc, char **argv) {
	melano::Args args(argc, argv);
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
	options.device = args.get("device", options.device);
	options.search.type = melano::parse_search_type(args.get("search-type", "only-mcts"));
	options.search.mcts_sims = args.get_int("mcts-sims", options.search.mcts_sims);
	options.search.mcts_min_sims = args.get_int("mcts-min-sims", options.search.mcts_min_sims);
	options.search.mcts_batch_size = args.get_int("mcts-batch-size", options.search.mcts_batch_size);
	options.search.movetime_ms = args.get_double("movetime-ms", options.search.movetime_ms);
	options.search.c_puct = args.get_double("c-puct", options.search.c_puct);
	options.search.c_puct_base = args.get_double("c-puct-base", options.search.c_puct_base);
	options.search.c_puct_factor = args.get_double("c-puct-factor", options.search.c_puct_factor);
	options.search.fpu_reduction = args.get_double("fpu-reduction", options.search.fpu_reduction);
	options.search.virtual_loss = args.get_double("virtual-loss", options.search.virtual_loss);
	options.search.repetition_policy_penalty =
		args.get_double("repetition-policy-penalty", options.search.repetition_policy_penalty);
	options.search.instant_mate_first =
		args.get_bool("instant-mate-first", options.search.instant_mate_first);
	options.search.root_topn = args.get_int("root-topn", options.search.root_topn);
	options.progress_interval_ms =
		args.get_int("progress-interval-ms", options.progress_interval_ms);
	options.move_overhead_ms = args.get_int("move-overhead-ms", options.move_overhead_ms);
	options.min_movetime_ms = args.get_int("min-movetime-ms", options.min_movetime_ms);
	options.max_movetime_ms = args.get_int("max-movetime-ms", options.max_movetime_ms);
	options.time_divisor = args.get_double("time-divisor", options.time_divisor);
	options.increment_fraction = args.get_double("increment-fraction", options.increment_fraction);
	options.multipv = args.get_int("multipv", options.multipv);
	options.score_scale = args.get_int("score-scale", options.score_scale);
	return options;
}

} // namespace

// Start the Melano UCI process and convert fatal initialization errors to stderr/exit failure.
int main(int argc, char **argv) {
	try {
		UciEngine(options_from_args(argc, argv)).loop();
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "uci error: " << error.what() << std::endl;
		return 1;
	}
}
