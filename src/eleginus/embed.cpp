// Embeds one Eleginus training checkpoint into a selected Torch-free executable.

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "eleginus/args.hpp"
#include "eleginus/checkpoint.hpp"

int main(int argc, char **argv) {
	try {
		eleginus::Args args(argc, argv);
		if (args.has("help")) {
			std::cout
				<< "Usage: embed --model <model.pth> [--type uci|search] [--output <engine>]\n";
			return 0;
		}
		const auto model = std::filesystem::path(args.get("model", "models/eleginus/eleginus.pth"));
		const std::string type = args.get("type", "uci");
		if (type != "uci" && type != "search")
			throw std::invalid_argument("Eleginus embed type must be uci or search");
		if (args.has("input"))
			throw std::invalid_argument("Eleginus embed selects its template through --type");
		auto input = std::filesystem::weakly_canonical(argv[0]).parent_path() / type;
		std::filesystem::path default_output = type == "uci"
			? std::filesystem::path("models/eleginus/eleginus")
			: std::filesystem::path("models/eleginus/eleginus_search");
#ifdef _WIN32
		input += ".exe";
		default_output += ".exe";
#endif
		const auto output = std::filesystem::path(args.get("output", default_output.string()));
		eleginus::embed_checkpoint_atomic(model, input, output);
		std::cout << "Eleginus executable embedded: model=" << model.string()
				  << " type=" << type << " output=" << output.string() << std::endl;
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "Eleginus embed failed: " << error.what() << std::endl;
		return 1;
	}
}
