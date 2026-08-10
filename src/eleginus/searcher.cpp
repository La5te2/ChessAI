// Read-only command-line analysis with embedded independent Policy/Value models.

#include "eleginus/runtime.hpp"
#include "eleginus/search.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

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
		auto weights = eleginus::load_embedded_runtime_model();
		eleginus::CpuPolicy policy(std::move(weights.policy));
		eleginus::CpuValue value(std::move(weights.value));
		eleginus::SearchOptions options;
		options.expansions = expansions;
		const chess::Board board = fen == "startpos" ? chess::Board() : chess::Board(fen);
		const auto result = eleginus::Searcher(policy, value, options).search(board);
		std::cout << "bestmove " << eleginus::move_uci(result.move) << " value=" << result.value
				  << " expanded=" << result.expanded_nodes
				  << " evaluated=" << result.evaluated_nodes << '\n';
		for (const auto &root : result.root) {
			std::cout << eleginus::move_uci(root.move) << " p=" << root.prior
					  << " v=" << root.value
					  << " nodes=" << root.subtree_nodes << '\n';
		}
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "Eleginus search failed: " << error.what() << '\n';
		return 1;
	}
}
