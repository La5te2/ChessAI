#pragma once

// Melano FCPI self-play, PVA counterfactual training, and promotion loop.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
#include "melano/arena.hpp"
#include "melano/precision.hpp"

namespace melano {

struct FcpiOptions {
	std::filesystem::path model = "models/melano.pth";
	std::string device = "auto";
	ComputePrecision precision = ComputePrecision::Fp32;
	int iterations = 1;
	int games_per_iter = 500;
	int games_in_flight = 64;
	int max_plies = 240;
	int positions_per_game = 200;
	std::string opening_book = "data/openings.gen.bin";
	double startpos_fraction = 0.5;
	int book_plies = 8;
	int max_book_positions = 50000;
	int inference_batch_size = 64;
	int target_records_per_batch = 256;
	int counterfactual_budget = 24;
	double td_lambda = 0.85;
	double behavior_temperature = 0.85;
	double uniform_mix = 0.02;
	double policy_weight = 1.0;
	double value_weight = 1.0;
	double dueling_q_weight = 0.5;
	double dynamics_weight = 0.25;
	double imagined_value_weight = 0.25;
	int epochs = 15;
	std::int64_t train_max_steps = 2000;
	int batch_size = 256;
	double learning_rate = 2e-5;
	double weight_decay = 1e-4;
	double grad_clip = 1.0;
	ArenaOptions arena;
	int log_every = 50;
	std::uint64_t seed = 2026;
};

/// Runs all configured FCPI iterations and advances current.pth only after arena acceptance.
void run_fcpi(const FcpiOptions &options);

} // namespace melano
