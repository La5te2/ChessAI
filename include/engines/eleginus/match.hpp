#pragma once

#include "eleginus/search.hpp"
#include <array>
#include <filesystem>
#include <vector>

namespace eleginus {

	inline constexpr int kOpeningPairs = 1000;

	struct MatchScore {
		double score = 0.5, elo = 0, low = 0, high = 0;
	};

	std::vector<chess::Board> openings(const std::filesystem::path &path);
	int match(const Model &candidate, const Model &baseline, chess::Board board, chess::Color side, SearchOptions options, const SearchCancel &cancel);
	MatchScore confidence(const std::array<int, 5> &pairs);

} // namespace eleginus
