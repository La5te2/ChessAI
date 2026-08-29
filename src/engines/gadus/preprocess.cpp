// Gadus JSONL preprocessing command-line entry point.

#include "gadus/args.hpp"
#include "gadus/dataset.hpp"
#include <iostream>

// Parse preprocessing controls and build one architecture-locked HDF5 dataset.
int main(int argc, char **argv) {
	try {
		gadus::Args args(argc, argv);
		if (args.has("help")) {
			std::cout << "Usage: preprocess --input <path|-> --output <games.gadus.h5> [options]\n"
			          << "  --max-positions <n> --chunk-size <n>\n"
			          << "  --compression-level <0..9> --log-every <accepted records>\n";
			return 0;
		}
		gadus::PreprocessOptions options;
		options.input = args.get("input", options.input.string());
		options.output = args.get("output", options.output.string());
		options.max_positions = args.get_int64("max-positions", options.max_positions);
		options.chunk_size = args.get_int("chunk-size", options.chunk_size);
		options.compression_level = args.get_int("compression-level", options.compression_level);
		options.log_every = args.get_int("log-every", options.log_every);
		gadus::preprocess(options);
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "preprocess error: " << error.what() << std::endl;
		return 1;
	}
}
