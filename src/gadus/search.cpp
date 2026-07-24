// Gadus single-position analysis CLI; the reusable algorithm is implemented in searcher.cpp.

#include <iomanip>
#include <iostream>
#include "gadus/args.hpp"
#include "gadus/checkpoint.hpp"
#include "gadus/search.hpp"

// Parse a FEN/search configuration, load one checkpoint, and print ranked root diagnostics.
int main(int argc, char **argv) {
	try {
		gadus::Args args(argc, argv);
		if (args.has("help")) {
			std::cout
				<< "Usage: search --model <gadus.pth> [--fen <fen>] [options]\n"
				<< "  --device <auto|cpu|cuda> --precision <fp32|bf16>\n"
				<< "  --search-type <closed|only-mcts>\n"
				<< "  --mcts-sims <n> --mcts-min-sims <n> --mcts-batch-size <n> --movetime-ms "
				   "<ms>\n"
				<< "  --c-puct <x> --c-puct-base <x> --c-puct-factor <x> --fpu-reduction <x>\n"
				<< "  --virtual-loss <x> --repetition-policy-penalty <x>\n"
				<< "  --instant-mate-first <0|1> --root-topn <n>\n";
			return 0;
		}
		const auto model_path = args.get("model", "models/gadus/gadus.pth");
		const auto device = gadus::resolve_device(args.get("device", "auto"));
		gadus::SearchOptions options;
		options.precision =
			gadus::parse_compute_precision(args.get("precision", "fp32"));
		options.type = gadus::parse_search_type(args.get("search-type", "only-mcts"));
		options.mcts_sims = args.get_int("mcts-sims", options.mcts_sims);
		options.mcts_min_sims = args.get_int("mcts-min-sims", options.mcts_min_sims);
		options.mcts_batch_size = args.get_int("mcts-batch-size", options.mcts_batch_size);
		options.movetime_ms = args.get_double("movetime-ms", options.movetime_ms);
		options.c_puct = args.get_double("c-puct", options.c_puct);
		options.c_puct_base = args.get_double("c-puct-base", options.c_puct_base);
		options.c_puct_factor = args.get_double("c-puct-factor", options.c_puct_factor);
		options.fpu_reduction = args.get_double("fpu-reduction", options.fpu_reduction);
		options.virtual_loss = args.get_double("virtual-loss", options.virtual_loss);
		options.repetition_policy_penalty =
			args.get_double("repetition-policy-penalty", options.repetition_policy_penalty);
		options.instant_mate_first =
			args.get_bool("instant-mate-first", options.instant_mate_first);
		options.root_topn = args.get_int("root-topn", options.root_topn);

		auto model = gadus::load_checkpoint(model_path, device);
		gadus::Searcher searcher(model, device, options);
		const std::string fen = args.get("fen", std::string(chess::constants::STARTPOS));
		chess::Board board(fen);
		const auto result = searcher.search(board);

		std::cout << std::fixed << std::setprecision(6);
		std::cout << "fen: " << board.getFen() << '\n';
		std::cout << "best: " << gadus::move_san(board, result.move) << ' '
				  << gadus::move_uci(result.move) << '\n';
		std::cout << "value: " << result.value << '\n';
		std::cout << "mcts: " << result.sims_completed << " / " << result.dynamic_target << " / "
				  << options.mcts_sims << '\n';
		std::cout << "uncertainty: " << result.uncertainty << '\n';
		std::cout << "expanded_nodes: " << result.expanded_nodes << '\n';
		std::cout << "nn_batches: " << result.nn_batches << '\n';
		std::cout << "elapsed_ms: " << result.elapsed_ms << '\n';
		std::cout << "root:\n";
		for (std::size_t index = 0; index < result.root.size(); ++index) {
			const auto &row = result.root[index];
			std::cout << index + 1 << ". " << gadus::move_san(board, row.move) << ' '
					  << gadus::move_uci(row.move) << " p=" << row.probability
					  << " decision=" << row.decision_score << " prior=" << row.prior
					  << " visits=" << row.visits << " q=" << row.q;
			if (row.instant_mate) {
				std::cout << " imf";
			}
			if (row.repetition_penalized) {
				std::cout << " rpp";
			}
			std::cout << '\n';
		}
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "search error: " << error.what() << std::endl;
		return 1;
	}
}
