#include "eleginus/game.hpp"
#include "eleginus/evaluate.hpp"
#include "eleginus/search.hpp"
#include <algorithm>
#include <atomic>
#include <bit>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {
	// Resolve the default parameter file beside the executable so the engine directory is relocatable.
	std::filesystem::path defaultParametersPath(const char *executable) {
		std::error_code error;
		auto path = std::filesystem::absolute(executable == nullptr ? "" : executable, error);
		if (error) path = executable == nullptr ? std::filesystem::path() : executable;
		return path.parent_path() / "eleginus.pth";
	}

	// Remove protocol whitespace while preserving spaces inside FEN and option values.
	std::string trim(std::string value) {
		const auto first = value.find_first_not_of(" \t\r\n");
		if (first == std::string::npos) return {};
		return value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
	}

	// Tokenize commands whose grammar is defined by whitespace-separated UCI fields.
	std::vector<std::string> split(const std::string &line) {
		std::istringstream stream(line);
		std::vector<std::string> tokens;
		for (std::string token; stream >> token;) {
			tokens.push_back(std::move(token));
		}
		return tokens;
	}

	// Normalize option names so spacing and capitalization do not affect matching.
	std::string normalized(std::string value) {
		std::string output;
		for (const unsigned char character : value) {
			if (std::isalnum(character)) output.push_back(static_cast<char>(std::tolower(character)));
		}
		return output;
	}

	// Keep malformed GUI option values from terminating the protocol loop.
	int parse_int(const std::string &value, int fallback) {
		try {
			return std::stoi(value);
		} catch (...) {
			return fallback;
		}
	}

	// Engine owns protocol state, the selected immutable evaluator and at most one search worker.
	class Engine {
	public:
		explicit Engine(std::filesystem::path parametersPath) : parametersPath(std::move(parametersPath)) {}
		~Engine() { stop(); }

		// Read UCI commands serially while searches execute on the managed worker thread.
		void loop() {
			for (std::string line; std::getline(std::cin, line);) {
				line = trim(line);
				if (line.empty()) continue;
				const auto command = line.substr(0, line.find(' '));
				try {
					if (command == "uci") {
						identity();
					} else if (command == "isready") {
						ensureEvaluator();
						print("readyok");
					} else if (command == "setoption") {
						stop();
						set_option(line);
					} else if (command == "position") {
						stop();
						set_position(line);
					} else if (command == "ucinewgame") {
						stop();
						board_ = chess::Board();
					} else if (command == "go") {
						go(line);
					} else if (command == "stop") {
						stop_requested_ = true;
					} else if (command == "quit") {
						break;
					}
				} catch (const std::exception &error) {
					print("info string " + command + " error: " + error.what());
					if (command == "go") print("bestmove 0000");
					else if (command == "isready") print("readyok");
				}
			}
			stop();
		}

	private:
		// Serialize worker and protocol output so individual UCI records remain intact.
		static void print(const std::string &text) {
			static std::mutex mutex;
			std::lock_guard lock(mutex);
			std::cout << text << std::endl;
		}

		// Publish engine identity and the supported UCI options.
		void identity() const {
			print("id name Gadidae Eleginus");
			print("id author La5te2");
			print("option name ParametersPath type string default " + parametersPath.string());
			print("option name Hash type spin default " + std::to_string(options_.hash_mb) + " min 0 max 4096");
			print("option name MultiPV type spin default " + std::to_string(options_.multipv) + " min 1 max 256");
			print("option name Move Overhead type spin default " + std::to_string(move_overhead_) + " min 0 max 5000");
			print("uciok");
		}

		// Request cancellation and join the active search before changing shared state.
		void stop() {
			stop_requested_ = true;
			if (worker_.joinable()) worker_.join();
			stop_requested_ = false;
		}

		// Parse and apply one supported setoption command.
		void set_option(const std::string &line) {
			const auto name_at = line.find(" name ");
			if (name_at == std::string::npos) return;
			const auto value_at = line.find(" value ", name_at + 6);
			const auto name = trim(line.substr(name_at + 6, value_at == std::string::npos ? std::string::npos : value_at - name_at - 6));
			const auto value = value_at == std::string::npos ? std::string() : trim(line.substr(value_at + 7));
			const auto key = normalized(name);
			if (key == "parameterspath") {
				parametersPath = value;
				evaluator.reset();
			} else if (key == "hash") {
				const auto requested = static_cast<std::size_t>(std::clamp(parse_int(value, static_cast<int>(options_.hash_mb)), 0, 4096));
				options_.hash_mb = std::bit_floor(requested);
			} else if (key == "multipv") {
				options_.multipv = std::clamp(parse_int(value, options_.multipv), 1, 256);
			} else if (key == "moveoverhead") {
				move_overhead_ = std::clamp(parse_int(value, move_overhead_), 0, 5000);
			}
		}

		// Load and prepare the selected parameter set once before it is used by search.
		void ensureEvaluator() {
			if (parametersPath.empty()) throw std::runtime_error("ParametersPath is empty");
			if (!evaluator) evaluator = std::make_shared<eleginus::Evaluator>(eleginus::loadParameters(parametersPath));
		}

		// Reconstruct the board from startpos or FEN and then replay the supplied moves.
		void set_position(const std::string &line) {
			const auto tokens = split(line);
			if (tokens.size() < 2) return;
			std::size_t moves_at = tokens.size();
			for (std::size_t index = 2; index < tokens.size(); ++index) {
				if (tokens[index] == "moves") {
					moves_at = index;
					break;
				}
			}
			if (tokens[1] == "startpos") {
				board_ = chess::Board();
			} else if (tokens[1] == "fen") {
				std::ostringstream fen;
				for (std::size_t index = 2; index < moves_at; ++index) {
					fen << (index == 2 ? "" : " ") << tokens[index];
				}
				board_ = chess::Board(fen.str());
			} else {
				throw std::invalid_argument("unsupported position command");
			}
			for (std::size_t index = moves_at + 1; index < tokens.size(); ++index) {
				board_.makeMove(chess::uci::uciToMove(board_, tokens[index]));
			}
		}

		// Collect go limits by name; later code applies only the limits supported by Eleginus.
		std::unordered_map<std::string, std::string> parse_go(const std::string &line) const {
			const auto tokens = split(line);
			std::unordered_map<std::string, std::string> values;
			for (std::size_t index = 1; index < tokens.size(); ++index) {
				if (tokens[index] == "infinite") {
					values[tokens[index]] = "1";
				} else if (index + 1 < tokens.size()) {
					const auto key = tokens[index];
					values[key] = tokens[++index];
				}
			}
			return values;
		}

		// Derive a bounded move budget from fixed movetime or clock controls.
		int allocated_time(const std::unordered_map<std::string, std::string> &values) const {
			if (values.contains("infinite")) return 0;
			if (const auto found = values.find("movetime"); found != values.end()) return std::max(1, parse_int(found->second, 1) - move_overhead_);
			const bool white = board_.sideToMove() == chess::Color::WHITE;
			const std::string time_key = white ? "wtime" : "btime";
			const std::string increment_key = white ? "winc" : "binc";
			if (!values.contains(time_key)) return 0;
			const int remaining = std::max(1, parse_int(values.at(time_key), 1));
			const int increment = values.contains(increment_key) ? std::max(0, parse_int(values.at(increment_key), 0)) : 0;
			return std::clamp(remaining / 30 + increment / 2 - move_overhead_, 1, std::max(1, remaining - move_overhead_));
		}

		// Format one completed search iteration, including all requested root lines.
		void emit_info(const eleginus::SearchResult &result) const {
			const int count = std::min<int>(options_.multipv, result.root.size());
			const auto elapsed = std::max<std::uint64_t>(1, result.elapsed_ms);
			const auto nps = static_cast<std::uint64_t>(1000.0 * static_cast<double>(result.nodes) / static_cast<double>(elapsed));
			for (int index = 0; index < count; ++index) {
				const auto &row = result.root[static_cast<std::size_t>(index)];
				const std::string score = std::abs(row.score_cp) >= 29000
					? "mate " + std::to_string((row.score_cp > 0 ? 1 : -1) * std::max(1, (30000 - std::abs(row.score_cp) + 1) / 2))
					: "cp " + std::to_string(row.score_cp);
				print("info depth " + std::to_string(result.depth) + " seldepth " + std::to_string(result.selective_depth) + " multipv " +
					std::to_string(index + 1) + " score " + score + " nodes " + std::to_string(result.nodes) + " nps " + std::to_string(nps) +
					" time " + std::to_string(result.elapsed_ms) + " pv " + eleginus::moveToUci(row.move));
			}
		}

		// Snapshot search options and launch one asynchronous search operation.
		void go(const std::string &line) {
			stop();
			ensureEvaluator();
			if (eleginus::isGameOver(board_)) {
				print("bestmove 0000");
				return;
			}
			auto search_options = options_;
			const auto values = parse_go(line);
			search_options.depth = values.contains("depth") ? std::clamp(parse_int(values.at("depth"), 6), 1, 64) : 64;
			search_options.node_limit = values.contains("nodes") ? std::stoull(values.at("nodes")) : 0;
			search_options.movetime_ms = allocated_time(values);
			const auto board = board_;
			const auto activeEvaluator = evaluator;
			stop_requested_ = false;
			worker_ = std::thread([this, board, search_options, activeEvaluator] {
				try {
					eleginus::Searcher searcher(*activeEvaluator, search_options);
					const auto result = searcher.search(
						board, [this](const eleginus::SearchResult &partial) { emit_info(partial); }, [this] { return stop_requested_.load(); });
					print("bestmove " + (result.move.move() == chess::Move::NO_MOVE ? fallback_move() : eleginus::moveToUci(result.move)));
				} catch (const std::exception &error) {
					print("info string search error: " + std::string(error.what()));
					print("bestmove " + fallback_move());
				}
			});
		}

		// Return a legal move if an exception interrupts search after go has been accepted.
		std::string fallback_move() const {
			const auto moves = eleginus::legalmoves(board_);
			return moves.empty() ? "0000" : eleginus::moveToUci(moves.front());
		}

		eleginus::SearchOptions options_;
		std::filesystem::path parametersPath;
		std::shared_ptr<const eleginus::Evaluator> evaluator;
		chess::Board board_;
		int move_overhead_ = 10;
		std::atomic_bool stop_requested_{false};
		std::thread worker_;
	};

} // namespace

// Run the UCI protocol loop until quit or end of input.
int main(int argc, char **argv) {
	try {
		auto parameters = defaultParametersPath(argc > 0 ? argv[0] : nullptr);
		if (argc == 3 && std::string(argv[1]) == "--parameters") parameters = argv[2];
		else if (argc != 1) throw std::invalid_argument("usage: uci [--parameters eleginus.pth]");
		Engine(std::move(parameters)).loop();
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "uci error: " << error.what() << '\n';
		return 1;
	}
}
