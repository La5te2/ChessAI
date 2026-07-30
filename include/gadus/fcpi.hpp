#pragma once

// Gadus FCPI self-play, counterfactual target construction, training, and promotion loop.

#include <cstdint>
#include <filesystem>
#include <string>
#include "gadus/arena.hpp"
#include "gadus/precision.hpp"

namespace gadus {

struct FcpiOptions {
	std::filesystem::path model = "models/gadus/gadus.pth";
	std::string device = "auto";
	ComputePrecision precision = ComputePrecision::Fp32;
	int iterations = 1;
	// Number of games in each of current-self, previous-accept-1, and previous-accept-2.
	int games_per_iter = 500;
	int games_in_flight = 64;
	int max_plies = 240;
	std::string opening_book = "data/openings.gen.bin";
	double startpos_fraction = 0.5;
	int book_plies = 8;
	int max_book_positions = 50000;
	int inference_batch_size = 64;
	int target_records_per_batch = 256;
	int counterfactual_budget = 24;
	double behavior_temperature = 1.0;
	int epochs = 15;
	std::int64_t train_max_steps = 2000;
	int batch_size = 256;
	double learning_rate = 2e-5;
	ArenaOptions arena;
	int log_every = 50;
	std::uint64_t seed = 2026;
};

/// Runs all configured FCPI iterations and advances current.pth only after arena acceptance.
void run_fcpi(const FcpiOptions &options);

} // namespace gadus
