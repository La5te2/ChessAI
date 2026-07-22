#include <iostream>

#include "melano/args.hpp"
#include "melano/fcpi.hpp"

int main(int argc, char **argv) {
	try {
		melano::Args args(argc, argv);
		if (args.has("help")) {
			std::cout
				<< "Usage: fcpi --model <melano.pth> [options]\n"
				<< "  --iterations <n> --games-per-iter <n> --games-in-flight <n> --max-plies <n>\n"
				<< "  --positions-per-game <n> --opening-book <path|empty> --startpos-fraction "
				   "<x>\n"
				<< "  --counterfactual-topk <n> --counterfactual-min-plies <n>\n"
				<< "  --counterfactual-max-plies <n> --counterfactual-target-average-plies <x>\n"
				<< "  --counterfactual-lambda <x> --td-lambda <x> --behavior-temperature <x>\n"
				<< "  --uniform-mix <x> --behavior-advantage-weight <x>\n"
				<< "  --policy-temperature <x> --prior-power <x> --successor-weight <x>\n"
				<< "  --played-return-weight <x> --policy-weight <x> --value-weight <x>\n"
				<< "  --dueling-q-weight <x>\n"
				<< "  --kl-weight <x> --entropy-weight <x> --epochs <n> --train-max-steps <n>\n"
				<< "  --batch-size <n> --lr <x> --weight-decay <x> --grad-clip <x>\n"
				<< "  --eval-games <n> --eval-games-in-flight <n> --eval-max-plies <n>\n"
				<< "  --eval-opening-book <path|empty> --eval-search-type <closed|only-mcts>\n"
				<< "  --eval-sims <n> --eval-mcts-batch-size <n> --eval-movetime-ms <ms>\n"
				<< "  --eval-min-net-wins <n> --device <auto|cpu|cuda> --seed <n> --log-every "
				   "<n>\n";
			return 0;
		}
		melano::FcpiOptions options;
		options.model = args.get("model", options.model.string());
		options.device = args.get("device", options.device);
		options.iterations = args.get_int("iterations", options.iterations);
		options.games_per_iter = args.get_int("games-per-iter", options.games_per_iter);
		options.games_in_flight = args.get_int("games-in-flight", options.games_in_flight);
		options.max_plies = args.get_int("max-plies", options.max_plies);
		options.positions_per_game = args.get_int("positions-per-game", options.positions_per_game);
		options.opening_book = args.get("opening-book", options.opening_book);
		options.startpos_fraction = args.get_double("startpos-fraction", options.startpos_fraction);
		options.book_plies = args.get_int("book-plies", options.book_plies);
		options.max_book_positions = args.get_int("max-book-positions", options.max_book_positions);
		options.inference_batch_size =
			args.get_int("inference-batch-size", options.inference_batch_size);
		options.target_records_per_batch =
			args.get_int("target-records-per-batch", options.target_records_per_batch);
		options.counterfactual_topk =
			args.get_int("counterfactual-topk", options.counterfactual_topk);
		options.counterfactual_min_plies =
			args.get_int("counterfactual-min-plies", options.counterfactual_min_plies);
		options.counterfactual_max_plies =
			args.get_int("counterfactual-max-plies", options.counterfactual_max_plies);
		options.counterfactual_target_average_plies = args.get_double(
			"counterfactual-target-average-plies", options.counterfactual_target_average_plies);
		options.counterfactual_lambda =
			args.get_double("counterfactual-lambda", options.counterfactual_lambda);
		options.td_lambda = args.get_double("td-lambda", options.td_lambda);
		options.behavior_temperature =
			args.get_double("behavior-temperature", options.behavior_temperature);
		options.uniform_mix = args.get_double("uniform-mix", options.uniform_mix);
		options.behavior_advantage_weight =
			args.get_double("behavior-advantage-weight", options.behavior_advantage_weight);
		options.policy_temperature =
			args.get_double("policy-temperature", options.policy_temperature);
		options.prior_power = args.get_double("prior-power", options.prior_power);
		options.successor_weight = args.get_double("successor-weight", options.successor_weight);
		options.played_return_weight =
			args.get_double("played-return-weight", options.played_return_weight);
		options.policy_weight = args.get_double("policy-weight", options.policy_weight);
		options.value_weight = args.get_double("value-weight", options.value_weight);
		options.dueling_q_weight =
			args.get_double("dueling-q-weight", options.dueling_q_weight);
		options.kl_weight = args.get_double("kl-weight", options.kl_weight);
		options.entropy_weight = args.get_double("entropy-weight", options.entropy_weight);
		options.epochs = args.get_int("epochs", options.epochs);
		options.train_max_steps = args.get_int64("train-max-steps", options.train_max_steps);
		options.batch_size = args.get_int("batch-size", options.batch_size);
		options.learning_rate = args.get_double("lr", options.learning_rate);
		options.weight_decay = args.get_double("weight-decay", options.weight_decay);
		options.grad_clip = args.get_double("grad-clip", options.grad_clip);
		options.log_every = args.get_int("log-every", options.log_every);
		options.seed = static_cast<std::uint64_t>(args.get_int64("seed", options.seed));

		auto &arena = options.arena;
		arena.games = args.get_int("eval-games", arena.games);
		arena.games_in_flight = args.get_int("eval-games-in-flight", arena.games_in_flight);
		arena.max_plies = args.get_int("eval-max-plies", arena.max_plies);
		arena.opening_book = args.get("eval-opening-book", arena.opening_book);
		arena.book_plies = args.get_int("eval-book-plies", arena.book_plies);
		arena.max_book_positions =
			args.get_int("eval-max-book-positions", arena.max_book_positions);
		arena.min_net_wins = args.get_int("eval-min-net-wins", 4);
		arena.log_every = options.log_every;
		auto &search = arena.search;
		search.type = melano::parse_search_type(args.get("eval-search-type", "closed"));
		search.mcts_sims = args.get_int("eval-sims", search.mcts_sims);
		search.mcts_batch_size = args.get_int("eval-mcts-batch-size", search.mcts_batch_size);
		search.movetime_ms = args.get_double("eval-movetime-ms", search.movetime_ms);
		search.c_puct = args.get_double("eval-c-puct", search.c_puct);
		search.c_puct_base = args.get_double("eval-c-puct-base", search.c_puct_base);
		search.c_puct_factor = args.get_double("eval-c-puct-factor", search.c_puct_factor);
		search.fpu_reduction = args.get_double("eval-fpu-reduction", search.fpu_reduction);
		search.repetition_policy_penalty =
			args.get_double("eval-repetition-policy-penalty", search.repetition_policy_penalty);
		search.instant_mate_first =
			args.get_bool("eval-instant-mate-first", search.instant_mate_first);

		melano::run_fcpi(options);
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "fcpi error: " << error.what() << std::endl;
		return 1;
	}
}
