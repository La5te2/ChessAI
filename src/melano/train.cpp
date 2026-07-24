// Melano one-shot supervised training command-line entry point.

#include <iostream>
#include "melano/args.hpp"
#include "melano/dataset.hpp"

// Parse training hyperparameters and train a fresh policy/value/advantage checkpoint.
int main(int argc, char **argv) {
	try {
		melano::Args args(argc, argv);
		if (args.has("help")) {
			std::cout << "Usage: train --data <games.melano.h5> --out <melano.pth> [options]\n"
					  << "  --channels <n> --blocks <n> --epochs <n> --batch-size <n>\n"
					  << "  --max-steps <n> --lr <x> --weight-decay <x> --value-weight <x>\n"
					  << "  --dueling-q-weight <x> --dynamics-weight <x> "
						 "--imagined-value-weight <x>\n"
					  << "  --save-every <steps> --log-every <steps> --seed <n> --device "
						 "<auto|cpu|cuda>\n"
					  << "  --precision <fp32|bf16>\n";
			return 0;
		}
		melano::TrainOptions options;
		options.data = args.get("data", options.data.string());
		options.output = args.get("out", options.output.string());
		options.channels = args.get_int("channels", options.channels);
		options.blocks = args.get_int("blocks", options.blocks);
		options.epochs = args.get_int("epochs", options.epochs);
		options.batch_size = args.get_int("batch-size", options.batch_size);
		options.max_steps = args.get_int64("max-steps", options.max_steps);
		options.learning_rate = args.get_double("lr", options.learning_rate);
		options.weight_decay = args.get_double("weight-decay", options.weight_decay);
		options.value_weight = args.get_double("value-weight", options.value_weight);
		options.dueling_q_weight =
			args.get_double("dueling-q-weight", options.dueling_q_weight);
		options.dynamics_weight = args.get_double("dynamics-weight", options.dynamics_weight);
		options.imagined_value_weight =
			args.get_double("imagined-value-weight", options.imagined_value_weight);
		options.save_every = args.get_int("save-every", options.save_every);
		options.log_every = args.get_int("log-every", options.log_every);
		options.seed = static_cast<std::uint64_t>(args.get_int64("seed", options.seed));
		options.device = args.get("device", options.device);
		options.precision =
			melano::parse_compute_precision(args.get("precision", "fp32"));
		melano::train_supervised(options);
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "train error: " << error.what() << std::endl;
		return 1;
	}
}
