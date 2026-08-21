// Gadus PGN or Lichess-evaluation to HDF5 preprocessing command-line entry point.

#include <iostream>
#include "gadus/args.hpp"
#include "gadus/dataset.hpp"

// Parse preprocessing controls and build one architecture-locked HDF5 dataset.
int main(int argc, char **argv) {
	try {
		gadus::Args args(argc, argv);
		if (args.has("help")) {
			std::cout
				<< "Usage: preprocess --source <pgn|lichess-eval> --input <path|-> "
					  "--output <games.gadus.h5> [options]\n"
				<< "  --max-games <n> --chunk-size <n> --has-cmt <0|1>\n"
				<< "  --max-positions <n> (lichess-eval)\n"
				<< "  --compression-level <0..9> --log-every <accepted records>\n";
			return 0;
		}
		gadus::PreprocessOptions options;
		options.source = args.get("source", options.source);
		options.input = args.get("input", options.input.string());
		options.output = args.get("output", options.output.string());
		options.max_games = args.get_int64("max-games", options.max_games);
		options.max_positions = args.get_int64("max-positions", options.max_positions);
		options.chunk_size = args.get_int("chunk-size", options.chunk_size);
		options.has_comments = args.get_int("has-cmt", options.has_comments);
		options.compression_level = args.get_int("compression-level", options.compression_level);
		options.log_every = args.get_int("log-every", options.log_every);
		if (options.source == "pgn") {
			gadus::preprocess_pgn(options);
		} else if (options.source == "lichess-eval") {
			gadus::preprocess_lichess_evaluations(options);
		} else {
			throw std::invalid_argument("--source must be pgn or lichess-eval");
		}
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "preprocess error: " << error.what() << std::endl;
		return 1;
	}
}
