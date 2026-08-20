// Eleginus supervised Value training command-line entry point.

#include <iostream>

#include <torch/torch.h>

#include "eleginus/args.hpp"
#include "eleginus/checkpoint.hpp"
#include "eleginus/dataset.hpp"

namespace {

torch::Device resolve_device(const std::string &requested) {
	if (requested == "auto")
		return torch::Device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
	if (requested == "cpu")
		return torch::Device(torch::kCPU);
	if (requested.starts_with("cuda")) {
		if (!torch::cuda::is_available())
			throw std::runtime_error("CUDA was requested but is unavailable");
		return torch::Device(requested);
	}
	throw std::invalid_argument("unsupported Eleginus device: " + requested);
}

} // namespace

int main(int argc, char **argv) {
	try {
		eleginus::Args args(argc, argv);
		if (args.has("help")) {
			std::cout
				<< "Usage: train --data <games.eleginus.h5> --out <model.pth> [options]\n"
				<< "  --model <existing.pth> --epochs <n> --batch-size <n> --max-steps <n>\n"
				<< "  --lr <x> --weight-decay <x> --device <auto|cpu|cuda>\n"
				<< "  --seed <n> --log-every <steps>\n";
			return 0;
		}
		const auto device = resolve_device(args.get("device", "auto"));
		const auto seed = static_cast<std::uint64_t>(args.get_int64("seed", 2026));
		eleginus::Model model;
		if (const auto existing = args.optional("model"); existing && !existing->empty())
			model = eleginus::load_checkpoint(*existing, device);
		else
			model = eleginus::make_model(device, seed);

		eleginus::TrainOptions options;
		options.data = args.get("data", "data/games.eleginus.h5");
		options.epochs = args.get_int("epochs", options.epochs);
		options.batch_size = args.get_int("batch-size", options.batch_size);
		options.max_steps = args.get_int64("max-steps", options.max_steps);
		options.learning_rate = args.get_double("lr", options.learning_rate);
		options.weight_decay = args.get_double("weight-decay", options.weight_decay);
		options.log_every = args.get_int("log-every", options.log_every);
		options.seed = seed;
		const auto output = std::filesystem::path(args.get("out", "models/eleginus/eleginus.pth"));
		options.output = output;
		const auto stats = eleginus::train_from_h5(model, options, device);
		std::cout << "Eleginus supervised training complete: rows=" << stats.samples
				  << " steps=" << stats.steps
				  << " checkpoint=" << output.string() << std::endl;
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "Eleginus train failed: " << error.what() << std::endl;
		return 1;
	}
}
