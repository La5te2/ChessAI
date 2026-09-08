#include "eleginus/game.hpp"
#include "eleginus/evaluate.hpp"
#include "eleginus/search.hpp"
#include <algorithm>
#include <array>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#ifdef _WIN32
	#define NOMINMAX
	#define WIN32_LEAN_AND_MEAN
	#include <windows.h>
#endif

namespace {

	std::string valueAfter(int argc, char **argv, int &index) {
		if (index + 1 >= argc) throw std::invalid_argument(std::string("missing value after ") + argv[index]);
		return argv[++index];
	}

	// Locate the default parameter file beside the analysis executable.
	std::filesystem::path defaultParametersPath(const char *executable) {
		#ifdef _WIN32
		std::array<wchar_t, 32768> buffer{};
		const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
		if (length > 0 && length < static_cast<DWORD>(buffer.size())) {
			return std::filesystem::path(std::wstring_view(buffer.data(), length)).parent_path() / "eleginus.pth";
		}
		#elif defined(__linux__)
		std::error_code linkError;
		auto link = std::filesystem::read_symlink("/proc/self/exe", linkError);
		if (!linkError) return link.parent_path() / "eleginus.pth";
		#endif
		std::error_code absoluteError;
		auto path = std::filesystem::absolute(executable == nullptr ? "" : executable, absoluteError);
		if (absoluteError) path = executable == nullptr ? std::filesystem::path() : executable;
		return path.parent_path() / "eleginus.pth";
	}

} // namespace

// Search one position and print each completed depth.
int main(int argc, char **argv) {
	try {
		std::string fen = chess::constants::STARTPOS;
		std::filesystem::path parameters = defaultParametersPath(argv[0]);
		eleginus::SearchOptions options;
		for (int index = 1; index < argc; ++index) {
			const std::string argument = argv[index];
			if (argument == "--parameters") {
				parameters = valueAfter(argc, argv, index);
			} else if (argument == "--fen") {
				fen = valueAfter(argc, argv, index);
			} else if (argument == "--depth") {
				options.depth = std::stoi(valueAfter(argc, argv, index));
			} else if (argument == "--hash") {
				options.hashMiB = std::stoull(valueAfter(argc, argv, index));
			} else if (argument == "--nodes") {
				options.nodeLimit = std::stoull(valueAfter(argc, argv, index));
			} else if (argument == "--multipv") {
				options.multipv = std::stoi(valueAfter(argc, argv, index));
			} else if (argument == "--help") {
				std::cout << "usage: search [--parameters eleginus.pth] [--fen FEN] [--depth 6] [--hash 64] [--nodes 0] [--multipv 1]\n";
				return 0;
			} else {
				throw std::invalid_argument("unknown option: " + argument);
			}
		}
		const chess::Board board(fen == "startpos" ? chess::constants::STARTPOS : fen);
		const eleginus::Evaluator evaluator(eleginus::loadParameters(parameters));
		eleginus::Searcher searcher(evaluator, options);
		const auto result = searcher.search(board, [](const eleginus::SearchResult &partial) {
			const auto elapsed = std::max<std::uint64_t>(1, partial.elapsedMs);
			const auto nps = static_cast<std::uint64_t>(1000.0 * static_cast<double>(partial.nodes) / static_cast<double>(elapsed));
			std::cout << "depth=" << partial.depth << " score_cp=" << partial.scoreCp << " nodes=" << partial.nodes << " nps=" << nps;
			std::cout << " time_ms=" << partial.elapsedMs;
			std::cout << " bestmove=" << (partial.move.move() == chess::Move::NO_MOVE ? "0000" : eleginus::moveToUci(partial.move)) << '\n';
		});
		std::cout << "bestmove " << (result.move.move() == chess::Move::NO_MOVE ? "0000" : eleginus::moveToUci(result.move)) << '\n';
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "search error: " << error.what() << '\n';
		return 1;
	}
}
