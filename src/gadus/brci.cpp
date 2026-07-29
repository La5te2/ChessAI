// Gadus BRCI command-line entry point; bellman.cpp contains the learning algorithm.

#include <iostream>
#include "gadus/args.hpp"
#include "gadus/brci.hpp"

// Parse BRCI sampling, optimizer, restricted-graph, and Arena controls.
int main(int argc, char **argv) {
	try {
		gadus::Args args(argc, argv);
		if (args.has("help")) {
			std::cout
				<< "Usage: brci --model <gadus.pth> [options]\n"
				<< "  --iterations <n> --games-per-iter <n> --games-in-flight <n> --max-plies <n>\n"
				<< "  --opening-book <path|empty> --startpos-fraction <x>\n"
				<< "  --inference-batch-size <n> --behavior-temperature <x>\n"
				<< "  --epochs <n> --train-max-steps <n> --batch-size <n> --lr <x>\n"
				<< "  --eval-games <n> --eval-games-in-flight <n> --eval-max-plies <n>\n"
				<< "  --eval-opening-book <path|empty> --eval-search-type <closed|only-mcts>\n"
				<< "  --eval-sims <n> --eval-mcts-batch-size <n> --eval-movetime-ms <ms>\n"
				<< "  --eval-min-net-wins <n>\n"
				<< "  --device <auto|cpu|cuda> --precision <fp32|bf16> --seed <n>\n";
			return 0;
		}

		gadus::BrciOptions options;
		options.model = args.get("model", options.model.string());
		options.device = args.get("device", options.device);
		options.precision =
			gadus::parse_compute_precision(args.get("precision", "fp32"));
		options.iterations = args.get_int("iterations", options.iterations);
		options.games_per_iter = args.get_int("games-per-iter", options.games_per_iter);
		options.games_in_flight = args.get_int("games-in-flight", options.games_in_flight);
		options.max_plies = args.get_int("max-plies", options.max_plies);
		options.opening_book = args.get("opening-book", options.opening_book);
		options.startpos_fraction =
			args.get_double("startpos-fraction", options.startpos_fraction);
		options.book_plies = args.get_int("book-plies", options.book_plies);
		options.max_book_positions =
			args.get_int("max-book-positions", options.max_book_positions);
		options.inference_batch_size =
			args.get_int("inference-batch-size", options.inference_batch_size);
		options.behavior_temperature =
			args.get_double("behavior-temperature", options.behavior_temperature);
		options.epochs = args.get_int("epochs", options.epochs);
		options.train_max_steps =
			args.get_int64("train-max-steps", options.train_max_steps);
		options.batch_size = args.get_int("batch-size", options.batch_size);
		options.learning_rate = args.get_double("lr", options.learning_rate);
		options.log_every = args.get_int("log-every", options.log_every);
		options.seed = static_cast<std::uint64_t>(args.get_int64("seed", options.seed));

		auto &arena = options.arena;
		arena.games = args.get_int("eval-games", arena.games);
		arena.games_in_flight =
			args.get_int("eval-games-in-flight", arena.games_in_flight);
		arena.max_plies = args.get_int("eval-max-plies", arena.max_plies);
		arena.opening_book = args.get("eval-opening-book", arena.opening_book);
		arena.book_plies = args.get_int("eval-book-plies", arena.book_plies);
		arena.max_book_positions =
			args.get_int("eval-max-book-positions", arena.max_book_positions);
		arena.min_net_wins = args.get_int("eval-min-net-wins", 4);
		arena.log_every = options.log_every;
		auto &search = arena.search;
		search.type =
			gadus::parse_search_type(args.get("eval-search-type", "only-mcts"));
		search.mcts_sims = args.get_int("eval-sims", search.mcts_sims);
		search.mcts_batch_size =
			args.get_int("eval-mcts-batch-size", search.mcts_batch_size);
		search.movetime_ms = args.get_double("eval-movetime-ms", search.movetime_ms);
		search.precision = options.precision;
		search.repetition_policy_penalty =
			args.get_double("eval-repetition-policy-penalty",
							search.repetition_policy_penalty);
		search.instant_mate_first =
			args.get_bool("eval-instant-mate-first", search.instant_mate_first);

		gadus::run_brci(options);
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "brci error: " << error.what() << std::endl;
		return 1;
	}
}
