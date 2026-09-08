#include "eleginus/game.hpp"
#include "eleginus/evaluate.hpp"
#include "eleginus/parameters.hpp"
#include "eleginus/search.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>
#ifdef _WIN32
	#define NOMINMAX
	#define WIN32_LEAN_AND_MEAN
	#include <windows.h>
#endif

namespace {
	// Return the running generator or engine path independently of the launch directory.
	std::filesystem::path executablePath(const char *argument) {
		#ifdef _WIN32
		std::array<wchar_t, 32768> buffer{};
		const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
		if (length == 0) throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), "cannot locate Eleginus");
		if (length >= static_cast<DWORD>(buffer.size())) throw std::runtime_error("Eleginus executable path is too long");
		return std::filesystem::path(std::wstring_view(buffer.data(), length));
		#elif defined(__linux__)
		std::error_code linkError;
		auto link = std::filesystem::read_symlink("/proc/self/exe", linkError);
		if (!linkError) return link;
		#endif
		if (argument == nullptr || *argument == '\0') throw std::runtime_error("cannot locate Eleginus");
		std::error_code absoluteError;
		auto path = std::filesystem::absolute(argument, absoluteError);
		if (absoluteError) throw std::system_error(absoluteError, "cannot locate Eleginus");
		return path;
	}
}

#if defined(ELEGINUS_GENERATOR)

namespace eleginus::runtime {
	std::span<const unsigned char> image();
}

namespace {
	// Write a complete engine to a sibling temporary file, then replace the requested output.
	void generateEngine(
		const std::filesystem::path &generator,
		const std::filesystem::path &output,
		const eleginus::FormulaParameters &parameters) {
		auto temporary = output;
		temporary += ".tmp";
		try {
			std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
			if (!stream) throw std::runtime_error("cannot create the Eleginus executable: " + temporary.string());

			const auto image = eleginus::runtime::image();
			const auto values = eleginus::flattenParameters(parameters);
			stream.write(reinterpret_cast<const char *>(image.data()), static_cast<std::streamsize>(image.size()));
			stream.write(reinterpret_cast<const char *>(values.data()), sizeof(values));
			stream.close();
			if (!stream) throw std::runtime_error("cannot write the Eleginus executable: " + temporary.string());

			std::filesystem::permissions(temporary, std::filesystem::status(generator).permissions());
			#ifdef _WIN32
			if (!MoveFileExW(temporary.c_str(), output.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
				throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), "cannot replace the Eleginus executable");
			}
			#else
			std::filesystem::rename(temporary, output);
			#endif
		} catch (...) {
			std::error_code ignored;
			std::filesystem::remove(temporary, ignored);
			throw;
		}
	}
}

// Generate a standalone Eleginus engine from initial or saved parameters.
int main(int argc, char **argv) {
	try {
		if (argc > 2) throw std::invalid_argument("usage: generator [parameters.pth]");
		const auto parameters = argc == 2 ? eleginus::loadParameters(argv[1]) : eleginus::initialParameters();
		const auto generator = executablePath(argc > 0 ? argv[0] : nullptr);
		const auto directory = generator.parent_path();
		const auto output = directory / (std::string("eleginus") + generator.extension().string());
		generateEngine(generator, output, parameters);
		std::cout << "generated Eleginus engine: " << output.string() << '\n';
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "generator error: " << error.what() << '\n';
		return 1;
	}
}

#else

namespace {
	// Read the fixed-size parameter payload appended to this executable.
	eleginus::FormulaParameters embeddedParameters(const std::filesystem::path &path) {
		std::ifstream stream(path, std::ios::binary | std::ios::ate);
		if (!stream) throw std::runtime_error("cannot read the Eleginus executable: " + path.string());
		const auto payloadSize = static_cast<std::streamoff>(sizeof(eleginus::ParameterValues));
		if (stream.tellg() < payloadSize) throw std::runtime_error("Eleginus executable has no embedded parameters");
		stream.seekg(-payloadSize, std::ios::end);
		eleginus::ParameterValues values{};
		stream.read(reinterpret_cast<char *>(values.data()), payloadSize);
		if (!stream) throw std::runtime_error("embedded Eleginus parameters are incomplete");
		return eleginus::expandParameters(values);
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
	int parseInt(const std::string &value, int fallback) {
		try {
			std::size_t parsed = 0;
			const int result = std::stoi(value, &parsed);
			return parsed == value.size() ? result : fallback;
		} catch (...) {
			return fallback;
		}
	}

	// Parse a complete unsigned integer and retain the fallback for malformed input.
	std::uint64_t parseUnsigned(const std::string &value, std::uint64_t fallback) {
		if (value.empty() || value.front() == '-') return fallback;
		try {
			std::size_t parsed = 0;
			const auto result = std::stoull(value, &parsed);
			return parsed == value.size() ? result : fallback;
		} catch (...) {
			return fallback;
		}
	}

	struct GoLimits {
		int depth = 64;
		std::uint64_t nodes = 0;
		int moveTime = -1;
		std::array<int, 2> time{{-1, -1}};
		std::array<int, 2> increment{};
		bool infinite = false;
	};

	// Engine owns protocol state, the evaluator and at most one search worker.
	class Engine {
	public:
		explicit Engine(eleginus::FormulaParameters parameters)
			: evaluator(std::make_shared<eleginus::Evaluator>(std::move(parameters))) {}
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
						print("readyok");
					} else if (command == "setoption") {
						stop();
						setOption(line);
					} else if (command == "position") {
						stop();
						setPosition(line);
					} else if (command == "ucinewgame") {
						stop();
						board = chess::Board();
					} else if (command == "go") {
						go(line);
					} else if (command == "stop") {
						stopRequested = true;
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
			print("option name Hash type spin default " + std::to_string(options.hashMiB) + " min 0 max 4096");
			print("option name MultiPV type spin default " + std::to_string(options.multipv) + " min 1 max 256");
			print("option name Move Overhead type spin default " + std::to_string(moveOverhead) + " min 0 max 5000");
			print("uciok");
		}

		// Request cancellation and join the active search before changing shared state.
		void stop() {
			stopRequested = true;
			if (worker.joinable()) worker.join();
			stopRequested = false;
		}

		// Parse and apply one supported setoption command.
		void setOption(const std::string &line) {
			const auto nameAt = line.find(" name ");
			if (nameAt == std::string::npos) return;
			const auto valueAt = line.find(" value ", nameAt + 6);
			const auto name = trim(line.substr(nameAt + 6, valueAt == std::string::npos ? std::string::npos : valueAt - nameAt - 6));
			const auto value = valueAt == std::string::npos ? std::string() : trim(line.substr(valueAt + 7));
			const auto key = normalized(name);
			if (key == "hash") {
				const auto requested = static_cast<std::size_t>(std::clamp(parseInt(value, static_cast<int>(options.hashMiB)), 0, 4096));
				options.hashMiB = std::bit_floor(requested);
			} else if (key == "multipv") {
				options.multipv = std::clamp(parseInt(value, options.multipv), 1, 256);
			} else if (key == "moveoverhead") {
				moveOverhead = std::clamp(parseInt(value, moveOverhead), 0, 5000);
			}
		}

		// Reconstruct the board from startpos or FEN and then replay the supplied moves.
		void setPosition(const std::string &line) {
			const auto tokens = split(line);
			if (tokens.size() < 2) return;
			std::size_t movesAt = tokens.size();
			for (std::size_t index = 2; index < tokens.size(); ++index) {
				if (tokens[index] == "moves") {
					movesAt = index;
					break;
				}
			}
			if (tokens[1] == "startpos") {
				board = chess::Board();
			} else if (tokens[1] == "fen") {
				std::ostringstream fen;
				for (std::size_t index = 2; index < movesAt; ++index) {
					fen << (index == 2 ? "" : " ") << tokens[index];
				}
				board = chess::Board(fen.str());
			} else {
				throw std::invalid_argument("unsupported position command");
			}
			for (std::size_t index = movesAt + 1; index < tokens.size(); ++index) {
				board.makeMove(chess::uci::uciToMove(board, tokens[index]));
			}
		}

		// Parse the search limits supported by Eleginus.
		GoLimits parseGo(const std::string &line) const {
			const auto tokens = split(line);
			GoLimits limits;
			for (std::size_t index = 1; index < tokens.size(); ++index) {
				const auto &name = tokens[index];
				if (name == "infinite") {
					limits.infinite = true;
					continue;
				}
				if (name == "ponder") continue;
				if (name == "searchmoves") throw std::invalid_argument("go searchmoves is unsupported");
				const bool hasValue = name == "depth" || name == "nodes" || name == "movetime" || name == "wtime" || name == "btime" ||
					name == "winc" || name == "binc" || name == "movestogo" || name == "mate";
				if (!hasValue) continue;
				if (index + 1 >= tokens.size()) break;
				const auto &value = tokens[++index];
				if (name == "depth") limits.depth = std::clamp(parseInt(value, 6), 1, 64);
				else if (name == "nodes") limits.nodes = parseUnsigned(value, 0);
				else if (name == "movetime") limits.moveTime = parseInt(value, 1);
				else if (name == "wtime") limits.time[0] = parseInt(value, 1);
				else if (name == "btime") limits.time[1] = parseInt(value, 1);
				else if (name == "winc") limits.increment[0] = parseInt(value, 0);
				else if (name == "binc") limits.increment[1] = parseInt(value, 0);
			}
			return limits;
		}

		// Derive a bounded move budget from fixed movetime or clock controls.
		int allocatedTime(const GoLimits &limits) const {
			if (limits.infinite) return 0;
			if (limits.moveTime >= 0) return std::max(1, limits.moveTime - moveOverhead);
			const std::size_t side = board.sideToMove() == chess::Color::WHITE ? 0 : 1;
			if (limits.time[side] < 0) return 0;
			const int remaining = std::max(1, limits.time[side]);
			const int increment = std::max(0, limits.increment[side]);
			return std::clamp(remaining / 30 + increment / 2 - moveOverhead, 1, std::max(1, remaining - moveOverhead));
		}

		// Format one completed search iteration, including all requested root lines.
		void emitInfo(const eleginus::SearchResult &result) const {
			const int count = std::min<int>(options.multipv, result.root.size());
			const auto elapsed = std::max<std::uint64_t>(1, result.elapsedMs);
			const auto nps = static_cast<std::uint64_t>(1000.0 * static_cast<double>(result.nodes) / static_cast<double>(elapsed));
			for (int index = 0; index < count; ++index) {
				const auto &row = result.root[static_cast<std::size_t>(index)];
				const std::string score = std::abs(row.scoreCp) >= 29000
					? "mate " + std::to_string((row.scoreCp > 0 ? 1 : -1) * std::max(1, (30000 - std::abs(row.scoreCp) + 1) / 2))
					: "cp " + std::to_string(row.scoreCp);
				print("info depth " + std::to_string(result.depth) + " seldepth " + std::to_string(result.selectiveDepth) + " multipv " +
					std::to_string(index + 1) + " score " + score + " nodes " + std::to_string(result.nodes) + " nps " + std::to_string(nps) +
					" time " + std::to_string(result.elapsedMs) + " pv " + eleginus::moveToUci(row.move));
			}
		}

		// Snapshot search options and launch one asynchronous search operation.
		void go(const std::string &line) {
			stop();
			if (eleginus::isGameOver(board)) {
				print("bestmove 0000");
				return;
			}
			auto searchOptions = options;
			const auto limits = parseGo(line);
			searchOptions.depth = limits.depth;
			searchOptions.nodeLimit = limits.nodes;
			searchOptions.moveTimeMs = allocatedTime(limits);
			const auto position = board;
			const auto activeEvaluator = evaluator;
			stopRequested = false;
			worker = std::thread([this, position, searchOptions, activeEvaluator] {
				try {
					eleginus::Searcher searcher(*activeEvaluator, searchOptions);
					const auto result = searcher.search(
						position, [this](const eleginus::SearchResult &partial) { emitInfo(partial); }, [this] { return stopRequested.load(); });
					print("bestmove " + (result.move.move() == chess::Move::NO_MOVE ? fallbackMove() : eleginus::moveToUci(result.move)));
				} catch (const std::exception &error) {
					print("info string search error: " + std::string(error.what()));
					print("bestmove " + fallbackMove());
				}
			});
		}

		// Return a legal move if an exception interrupts search after go has been accepted.
		std::string fallbackMove() const {
			const auto moves = eleginus::legalmoves(board);
			return moves.empty() ? "0000" : eleginus::moveToUci(moves.front());
		}

		eleginus::SearchOptions options;
		const std::shared_ptr<const eleginus::Evaluator> evaluator;
		chess::Board board;
		int moveOverhead = 10;
		std::atomic_bool stopRequested{false};
		std::thread worker;
	};

} // namespace

// Run the UCI protocol with the parameter set embedded by the generator.
int main(int argc, char **argv) {
	try {
		if (argc != 1) throw std::invalid_argument("usage: eleginus");
		auto parameters = embeddedParameters(executablePath(argc > 0 ? argv[0] : nullptr));
		Engine(std::move(parameters)).loop();
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "eleginus error: " << error.what() << '\n';
		return 1;
	}
}

#endif
