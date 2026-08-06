// Read-only command-line analysis with the embedded Eleginus Value model.

#include "eleginus/runtime.hpp"
#include "eleginus/search.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char **argv) {
	try {
		std::string fen = "startpos";
		int expansions = 32;
		for (int index = 1; index < argc; ++index) {
			const std::string argument = argv[index];
			if (argument == "--help") {
				std::cout << "Usage: search [--fen startpos|FEN] [--expansions N]\n";
				return 0;
			}
			if (index + 1 >= argc) {
				throw std::invalid_argument("missing value for " + argument);
			}
			const std::string value = argv[++index];
			if (argument == "--fen")
				fen = value;
			else if (argument == "--expansions")
				expansions = std::stoi(value);
			else
				throw std::invalid_argument("unknown argument: " + argument);
		}
		eleginus::CpuValue value(eleginus::load_embedded_runtime_model());
		eleginus::SearchOptions options;
		options.expansions = expansions;
		const chess::Board board = fen == "startpos" ? chess::Board() : chess::Board(fen);
		const auto result = eleginus::Searcher(value, options).search(board);
		std::cout << "bestmove " << eleginus::move_uci(result.move) << " value=" << result.value
				  << " expanded=" << result.expanded_nodes
				  << " evaluated=" << result.evaluated_nodes << '\n';
		for (const auto &root : result.root) {
			std::cout << eleginus::move_uci(root.move) << " v=" << root.value
					  << " nodes=" << root.subtree_nodes << '\n';
		}
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "Eleginus search failed: " << error.what() << '\n';
		return 1;
	}
}
