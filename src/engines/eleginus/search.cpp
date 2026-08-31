#include "eleginus/game.hpp"
#include "eleginus/model.hpp"
#include "eleginus/search.hpp"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

std::string value_after(int argc, char **argv, int &index) {
	if (index + 1 >= argc) {
		throw std::invalid_argument(std::string("missing value after ") + argv[index]);
	}
	return argv[++index];
}

} // namespace

int main(int argc, char **argv) {
	try {
		std::filesystem::path model_path;
		std::string fen = chess::constants::STARTPOS;
		eleginus::SearchOptions options;
		for (int index = 1; index < argc; ++index) {
			const std::string argument = argv[index];
			if (argument == "--model") {
				model_path = value_after(argc, argv, index);
			} else if (argument == "--fen") {
				fen = value_after(argc, argv, index);
			} else if (argument == "--depth") {
				options.depth = std::stoi(value_after(argc, argv, index));
			} else if (argument == "--hash") {
				options.hash_mb = std::stoull(value_after(argc, argv, index));
			} else if (argument == "--nodes") {
				options.node_limit = std::stoull(value_after(argc, argv, index));
			} else if (argument == "--multipv") {
				options.multipv = std::stoi(value_after(argc, argv, index));
			} else if (argument == "--help") {
				std::cout << "usage: search [--model eleginus.pth] [--fen FEN] [--depth 6] [--hash 64] [--nodes 0] [--multipv 1]\n";
				return 0;
			} else {
				throw std::invalid_argument("unknown option: " + argument);
			}
		}
		const auto model = model_path.empty() ? eleginus::Model() : eleginus::Model::load(model_path);
		const chess::Board board(fen);
		eleginus::Searcher searcher(model, options);
		const auto result = searcher.search(board, [](const eleginus::SearchResult &partial) {
			const auto elapsed = std::max<std::uint64_t>(1, partial.elapsed_ms);
			const auto nps = static_cast<std::uint64_t>(1000.0 * static_cast<double>(partial.nodes) / static_cast<double>(elapsed));
			std::cout << "depth=" << partial.depth << " score_cp=" << partial.score_cp << " nodes=" << partial.nodes
					  << " nps=" << nps << " time_ms=" << partial.elapsed_ms
					  << " bestmove=" << (partial.move.move() == chess::Move::NO_MOVE ? "0000" : eleginus::moveToUci(partial.move)) << '\n';
		});
		std::cout << "bestmove " << (result.move.move() == chess::Move::NO_MOVE ? "0000" : eleginus::moveToUci(result.move)) << '\n';
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "search error: " << error.what() << '\n';
		return 1;
	}
}
