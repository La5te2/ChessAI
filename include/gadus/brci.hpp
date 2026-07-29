#pragma once

// Gadus Bellman-restricted counterfactual iteration configuration and entry point.

#include <cstdint>
#include <filesystem>
#include <string>
#include <torch/types.h>
#include "gadus/arena.hpp"
#include "gadus/precision.hpp"

namespace gadus {

/// Computes BRCI cross-entropy over variable-width legal-action rows.
torch::Tensor brci_masked_policy_loss(const torch::Tensor &selected_logits,
									  const torch::Tensor &targets,
									  const torch::Tensor &counts,
									  const torch::Tensor &weights);

/// Returns the minimum-norm convex weight on the Policy backbone gradient.
double brci_common_descent_lambda(double policy_norm_squared,
								  double value_norm_squared,
								  double policy_value_dot);

struct BrciOptions {
	std::filesystem::path model = "models/gadus/gadus.pth";
	std::string device = "auto";
	ComputePrecision precision = ComputePrecision::Fp32;
	int iterations = 1;
	int games_per_iter = 500;
	int games_in_flight = 64;
	int max_plies = 240;
	std::string opening_book = "data/openings.gen.bin";
	double startpos_fraction = 0.5;
	int book_plies = 8;
	int max_book_positions = 50000;
	int inference_batch_size = 64;
	double behavior_temperature = 1.0;
	int epochs = 15;
	std::int64_t train_max_steps = 2000;
	int batch_size = 256;
	double learning_rate = 2e-5;
	ArenaOptions arena;
	int log_every = 50;
	std::uint64_t seed = 2026;
};

/// Runs terminal-anchored BRCI and promotes candidates that pass graph-error and Arena gates.
void run_brci(const BrciOptions &options);

} // namespace gadus
