#include "eleginus/game.hpp"
#include "eleginus/model.hpp"
#include "eleginus/search.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
	#define NOMINMAX
	#include <windows.h>
#elif defined(__linux__)
	#include <unistd.h>
#endif

namespace {

	std::string trim(std::string value) {
		const auto first = value.find_first_not_of(" \t\r\n");
		if (first == std::string::npos) return {};
		return value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
	}

	std::vector<std::string> split(const std::string &line) {
		std::istringstream stream(line);
		std::vector<std::string> tokens;
		for (std::string token; stream >> token;) {
			tokens.push_back(std::move(token));
		}
		return tokens;
	}

	std::string normalized(std::string value) {
		std::string output;
		for (const unsigned char character : value) {
			if (std::isalnum(character)) output.push_back(static_cast<char>(std::tolower(character)));
		}
		return output;
	}

	int parse_int(const std::string &value, int fallback) {
		try {
			return std::stoi(value);
		} catch (...) {
			return fallback;
		}
	}

	std::filesystem::path executable_directory(const char *argument_zero) {
		#if defined(_WIN32)
		std::array<wchar_t, 32768> buffer{};
		const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
		if (length > 0 && length < buffer.size()) return std::filesystem::path(std::wstring_view(buffer.data(), length)).parent_path();
		#elif defined(__linux__)
		std::array<char, 4096> buffer{};
		const auto length = readlink("/proc/self/exe", buffer.data(), buffer.size());
		if (length > 0 && static_cast<std::size_t>(length) < buffer.size()) {
			return std::filesystem::path(std::string_view(buffer.data(), static_cast<std::size_t>(length))).parent_path();
		}
		#endif
		std::error_code error;
		const auto absolute = std::filesystem::absolute(argument_zero, error);
		return error ? std::filesystem::current_path() : absolute.parent_path();
	}

	class Engine {
	public:
		explicit Engine(std::filesystem::path model_path) : model_path_(std::move(model_path)) {}
		~Engine() { stop(); }

		void loop() {
			for (std::string line; std::getline(std::cin, line);) {
				line = trim(line);
				if (line.empty()) continue;
				const auto command = line.substr(0, line.find(' '));
				try {
					if (command == "uci") {
						identity();
					} else if (command == "isready") {
						load_model();
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
		static void print(const std::string &text) {
			static std::mutex mutex;
			std::lock_guard lock(mutex);
			std::cout << text << std::endl;
		}

		void identity() const {
			print("id name Gadidae Eleginus");
			print("id author La5te2");
			print("option name ModelPath type string default " + model_path_.string());
			print("option name Hash type spin default " + std::to_string(options_.hash_mb) + " min 0 max 4096");
			print("option name MultiPV type spin default " + std::to_string(options_.multipv) + " min 1 max 256");
			print("option name Move Overhead type spin default " + std::to_string(move_overhead_) + " min 0 max 5000");
			print("uciok");
		}

		void load_model() {
			if (model_ && loaded_path_ == model_path_) return;
			if (model_path_.empty()) throw std::runtime_error("ModelPath is empty");
			model_.emplace(eleginus::Model::load(model_path_));
			loaded_path_ = model_path_;
		}

		void stop() {
			stop_requested_ = true;
			if (worker_.joinable()) worker_.join();
			stop_requested_ = false;
		}

		void set_option(const std::string &line) {
			const auto name_at = line.find(" name ");
			if (name_at == std::string::npos) return;
			const auto value_at = line.find(" value ", name_at + 6);
			const auto name = trim(line.substr(name_at + 6, value_at == std::string::npos ? std::string::npos : value_at - name_at - 6));
			const auto value = value_at == std::string::npos ? std::string() : trim(line.substr(value_at + 7));
			const auto key = normalized(name);
			if (key == "modelpath") {
				model_path_ = value;
				loaded_path_.clear();
			} else if (key == "hash") {
				const auto requested = static_cast<std::size_t>(std::clamp(parse_int(value, static_cast<int>(options_.hash_mb)), 0, 4096));
				options_.hash_mb = std::bit_floor(requested);
			} else if (key == "multipv") {
				options_.multipv = std::clamp(parse_int(value, options_.multipv), 1, 256);
			} else if (key == "moveoverhead") {
				move_overhead_ = std::clamp(parse_int(value, move_overhead_), 0, 5000);
			}
		}

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

		void go(const std::string &line) {
			stop();
			load_model();
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
			stop_requested_ = false;
			worker_ = std::thread([this, board, search_options] {
				try {
					eleginus::Searcher searcher(*model_, search_options);
					const auto result = searcher.search(
						board, [this](const eleginus::SearchResult &partial) { emit_info(partial); }, [this] { return stop_requested_.load(); });
					print("bestmove " + (result.move.move() == chess::Move::NO_MOVE ? fallback_move() : eleginus::moveToUci(result.move)));
				} catch (const std::exception &error) {
					print("info string search error: " + std::string(error.what()));
					print("bestmove " + fallback_move());
				}
			});
		}

		std::string fallback_move() const {
			const auto moves = eleginus::legalmoves(board_);
			return moves.empty() ? "0000" : eleginus::moveToUci(moves.front());
		}

		std::filesystem::path model_path_;
		std::filesystem::path loaded_path_;
		std::optional<eleginus::Model> model_;
		eleginus::SearchOptions options_;
		chess::Board board_;
		int move_overhead_ = 10;
		std::atomic_bool stop_requested_{false};
		std::thread worker_;
	};

} // namespace

int main(int argc, char **argv) {
	try {
		std::filesystem::path model_path = executable_directory(argv[0]) / "eleginus.pth";
		for (int index = 1; index < argc; ++index) {
			if (std::string(argv[index]) == "--model" && index + 1 < argc) model_path = argv[++index];
			else throw std::invalid_argument("usage: uci [--model eleginus.pth]");
		}
		Engine(std::move(model_path)).loop();
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "uci error: " << error.what() << '\n';
		return 1;
	}
}
