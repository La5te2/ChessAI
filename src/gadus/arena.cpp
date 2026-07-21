#include <iostream>

#include "gadus/arena.hpp"
#include "gadus/args.hpp"

int main(int argc, char **argv) {
	try {
		gadus::Args args(argc, argv);
		if (args.has("help")) {
			std::cout
				<< "Usage: arena --candidate <model> --baseline <model> [options]\n"
				<< "  --device <auto|cpu|cuda> --games <n> --games-in-flight <n> --max-plies <n>\n"
				<< "  --opening-book <path|empty> --book-plies <n> --max-book-positions <n>\n"
				<< "  --search-type <closed|only-mcts> --sims <n> --mcts-min-sims <n>\n"
				<< "  --mcts-batch-size <n> --movetime-ms <ms> --c-puct <x> --c-puct-base <x>\n"
				<< "  --c-puct-factor <x> --fpu-reduction <x> --virtual-loss <x>\n"
				<< "  --repetition-policy-penalty <x> --instant-mate-first <0|1>\n"
				<< "  --min-net-wins <n> --pgn-output <path> --seed <n> --log-every <games>\n";
			return 0;
		}
		gadus::ArenaOptions options;
		options.candidate = args.get("candidate");
		options.baseline = args.get("baseline");
		options.device = args.get("device", options.device);
		options.games = args.get_int("games", options.games);
		options.games_in_flight = args.get_int("games-in-flight", options.games_in_flight);
		options.max_plies = args.get_int("max-plies", options.max_plies);
		options.opening_book = args.get("opening-book", options.opening_book);
		options.book_plies = args.get_int("book-plies", options.book_plies);
		options.max_book_positions = args.get_int("max-book-positions", options.max_book_positions);
		options.seed = static_cast<std::uint64_t>(args.get_int64("seed", options.seed));
		options.min_net_wins = args.get_int("min-net-wins", options.min_net_wins);
		options.log_every = args.get_int("log-every", options.log_every);
		options.pgn_output = args.get("pgn-output", options.pgn_output.string());

		auto &search = options.search;
		search.type = gadus::parse_search_type(args.get("search-type", "only-mcts"));
		search.mcts_sims = args.get_int("sims", search.mcts_sims);
		search.mcts_min_sims = args.get_int("mcts-min-sims", search.mcts_min_sims);
		search.mcts_batch_size = args.get_int("mcts-batch-size", search.mcts_batch_size);
		search.movetime_ms = args.get_double("movetime-ms", search.movetime_ms);
		search.c_puct = args.get_double("c-puct", search.c_puct);
		search.c_puct_base = args.get_double("c-puct-base", search.c_puct_base);
		search.c_puct_factor = args.get_double("c-puct-factor", search.c_puct_factor);
		search.fpu_reduction = args.get_double("fpu-reduction", search.fpu_reduction);
		search.virtual_loss = args.get_double("virtual-loss", search.virtual_loss);
		search.repetition_policy_penalty =
			args.get_double("repetition-policy-penalty", search.repetition_policy_penalty);
		search.instant_mate_first = args.get_bool("instant-mate-first", search.instant_mate_first);

		std::cout << "arena summary:\n" << gadus::evaluate_models(options).dump(2) << std::endl;
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "arena error: " << error.what() << std::endl;
		return 1;
	}
}
