// Eleginus PGN-to-HDF5 preprocessing command-line entry point.

#include <iostream>

#include "eleginus/args.hpp"
#include "eleginus/dataset.hpp"

int main(int argc, char **argv) {
	try {
		eleginus::Args args(argc, argv);
		if (args.has("help")) {
			std::cout
				<< "Usage: preprocess --input <games.pgn> --output <games.eleginus.h5> [options]\n"
				<< "  --max-games <n> --chunk-size <n> --has-cmt <0|1>\n"
				<< "  --compression-level <0..9> --log-every <games>\n";
			return 0;
		}
		eleginus::PreprocessOptions options;
		options.input = args.get("input", options.input.string());
		options.output = args.get("output", options.output.string());
		options.max_games = args.get_int64("max-games", options.max_games);
		options.chunk_size = args.get_int("chunk-size", options.chunk_size);
		options.has_comments = args.get_int("has-cmt", options.has_comments);
		options.compression_level =
			args.get_int("compression-level", options.compression_level);
		options.log_every = args.get_int("log-every", options.log_every);
		eleginus::preprocess_pgn(options);
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "Eleginus preprocess failed: " << error.what() << std::endl;
		return 1;
	}
}
