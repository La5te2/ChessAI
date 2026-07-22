#pragma once

// Paired, batched Gadus-vs-Gadus evaluation used by promotion gates.

#include <cstdint>
#include <filesystem>
#include <string>
#include <nlohmann/json.hpp>
#include "gadus/search.hpp"

namespace gadus {

struct ArenaOptions {
	std::filesystem::path candidate;
	std::filesystem::path baseline;
	std::string device = "auto";
	int games = 100;
	int games_in_flight = 32;
	int max_plies = 240;
	std::string opening_book = "data/openings.gen.bin";
	int book_plies = 8;
	int max_book_positions = 50000;
	std::uint64_t seed = 2026;
	int min_net_wins = 0;
	int log_every = 1;
	std::filesystem::path pgn_output;
	SearchOptions search;
};

/// Plays a color-balanced match and returns result statistics and acceptance state as JSON.
nlohmann::json evaluate_models(const ArenaOptions &options);

} // namespace gadus
