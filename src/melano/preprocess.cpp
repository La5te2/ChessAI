// Melano PGN-to-HDF5 preprocessing command-line entry point.

#include <iostream>
#include "melano/args.hpp"
#include "melano/dataset.hpp"

// Parse preprocessing controls and build one architecture-locked P/V/A HDF5 dataset.
int main(int argc, char **argv) {
	try {
		melano::Args args(argc, argv);
		if (args.has("help")) {
			std::cout
				<< "Usage: preprocess --input <games.pgn> --output <games.melano.h5> [options]\n"
				<< "  --max-games <n> --chunk-size <n> --has-cmt <0|1>\n"
				<< "  --compression-level <0..9> --log-every <games>\n";
			return 0;
		}
		melano::PreprocessOptions options;
		options.input = args.get("input", options.input.string());
		options.output = args.get("output", options.output.string());
		options.max_games = args.get_int64("max-games", options.max_games);
		options.chunk_size = args.get_int("chunk-size", options.chunk_size);
		options.has_comments = args.get_int("has-cmt", options.has_comments);
		options.compression_level = args.get_int("compression-level", options.compression_level);
		options.log_every = args.get_int("log-every", options.log_every);
		melano::preprocess_pgn(options);
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "preprocess error: " << error.what() << std::endl;
		return 1;
	}
}
