// Embeds one Eleginus training checkpoint into a Torch-free executable template.

#include <iostream>

#include "eleginus/args.hpp"
#include "eleginus/checkpoint.hpp"

int main(int argc, char **argv) {
	try {
		eleginus::Args args(argc, argv);
		if (args.has("help")) {
			std::cout
				<< "Usage: embed --model <model.pth> --input <uci-template> --output <engine>\n";
			return 0;
		}
		const auto model = std::filesystem::path(args.get("model", "models/eleginus/eleginus.pth"));
		const auto input = std::filesystem::path(args.get("input", "build/eleginus/uci"));
		const auto output = std::filesystem::path(args.get("output", "models/eleginus/eleginus"));
		eleginus::embed_checkpoint_atomic(model, input, output);
		std::cout << "Eleginus executable embedded: model=" << model.string()
				  << " input=" << input.string() << " output=" << output.string() << std::endl;
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "Eleginus embed failed: " << error.what() << std::endl;
		return 1;
	}
}
