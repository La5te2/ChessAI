#include "eleginus/game.hpp"
#include "eleginus/model.hpp"
#include "eleginus/search.hpp"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

	std::string valueAfter(int argc, char **argv, int &index) {
		if (index + 1 >= argc) throw std::invalid_argument(std::string("missing value after ") + argv[index]);
		return argv[++index];
	}

} // namespace

int main(int argc, char **argv) {
	try {
		std::filesystem::path modelPath, exportPath;
		std::string fen = chess::constants::STARTPOS;
		eleginus::SearchOptions options;
		for (int index = 1; index < argc; ++index) {
			const std::string argument = argv[index];
			if (argument == "--model") {
				modelPath = valueAfter(argc, argv, index);
			} else if (argument == "--export-initial") {
				exportPath = valueAfter(argc, argv, index);
			} else if (argument == "--fen") {
				fen = valueAfter(argc, argv, index);
			} else if (argument == "--depth") {
				options.depth = std::stoi(valueAfter(argc, argv, index));
			} else if (argument == "--hash") {
				options.hash_mb = std::stoull(valueAfter(argc, argv, index));
			} else if (argument == "--nodes") {
				options.node_limit = std::stoull(valueAfter(argc, argv, index));
			} else if (argument == "--multipv") {
				options.multipv = std::stoi(valueAfter(argc, argv, index));
			} else if (argument == "--help") {
				std::cout << "usage: search [--model eleginus.pth] [--fen FEN] [--depth 6] [--hash 64] [--nodes 0] [--multipv 1]\n";
				std::cout << "       search --export-initial models/eleginus/eleginus.pth\n";
				return 0;
			} else {
				throw std::invalid_argument("unknown option: " + argument);
			}
		}
		if (!exportPath.empty()) {
			if (!modelPath.empty()) throw std::invalid_argument("--export-initial cannot be combined with --model");
			eleginus::Model().save(exportPath);
			std::cout << "exported " << exportPath.string() << '\n';
			return 0;
		}
		const auto model = modelPath.empty() ? eleginus::Model() : eleginus::Model::load(modelPath);
		const chess::Board board(fen == "startpos" ? chess::constants::STARTPOS : fen);
		eleginus::Searcher searcher(model, options);
		const auto result = searcher.search(board, [](const eleginus::SearchResult &partial) {
			const auto elapsed = std::max<std::uint64_t>(1, partial.elapsed_ms);
			const auto nps = static_cast<std::uint64_t>(1000.0 * static_cast<double>(partial.nodes) / static_cast<double>(elapsed));
			std::cout << "depth=" << partial.depth << " score_cp=" << partial.score_cp << " nodes=" << partial.nodes << " nps=" << nps;
			std::cout << " time_ms=" << partial.elapsed_ms;
			std::cout << " bestmove=" << (partial.move.move() == chess::Move::NO_MOVE ? "0000" : eleginus::moveToUci(partial.move)) << '\n';
		});
		std::cout << "bestmove " << (result.move.move() == chess::Move::NO_MOVE ? "0000" : eleginus::moveToUci(result.move)) << '\n';
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "search error: " << error.what() << '\n';
		return 1;
	}
}
