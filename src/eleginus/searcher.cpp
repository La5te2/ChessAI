// Read-only command-line PVS analysis with an embedded Value model.

#include "eleginus/runtime.hpp"
#include "eleginus/search.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

int main(int argc, char **argv) {
	try {
		std::string fen = "startpos";
		int depth = 4;
		int threads = 1;
		int multipv = 1;
		std::size_t hash_mb = 64;
		for (int index = 1; index < argc; ++index) {
			const std::string argument = argv[index];
			if (argument == "--help") {
				std::cout << "Usage: search [--fen startpos|FEN] [--depth N] [--threads N] "
							 "[--multipv N] [--hash-mb N]\n";
				return 0;
			}
			if (index + 1 >= argc) {
				throw std::invalid_argument("missing value for " + argument);
			}
			const std::string value = argv[++index];
			if (argument == "--fen")
				fen = value;
			else if (argument == "--depth")
				depth = std::stoi(value);
			else if (argument == "--threads")
				threads = std::stoi(value);
			else if (argument == "--multipv")
				multipv = std::stoi(value);
			else if (argument == "--hash-mb")
				hash_mb = static_cast<std::size_t>(std::stoull(value));
			else
				throw std::invalid_argument("unknown argument: " + argument);
		}
		auto weights = eleginus::load_embedded_runtime_model();
		eleginus::CpuValue value(std::move(weights.value));
		eleginus::SearchOptions options;
		options.depth = depth;
		options.threads = threads;
		options.multipv = multipv;
		options.hash_mb = hash_mb;
		const chess::Board board = fen == "startpos" ? chess::Board() : chess::Board(fen);
		const auto result = eleginus::Searcher(value, options).search(board);
		std::cout << "bestmove " << eleginus::move_uci(result.move)
				  << " score_cp=" << result.score_cp
				  << " depth=" << result.depth
				  << " seldepth=" << result.selective_depth
				  << " nodes=" << result.nodes
				  << " evaluated=" << result.evaluated_nodes << '\n';
		for (const auto &root : result.root) {
			std::cout << eleginus::move_uci(root.move) << " order=" << root.order
					  << " score_cp=" << root.score_cp
					  << " bound=" << (root.exact_score ? "exact" : "upper")
					  << " nodes=" << root.nodes << '\n';
		}
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "Eleginus search failed: " << error.what() << '\n';
		return 1;
	}
}
