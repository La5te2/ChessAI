#pragma once

// Value-guided best-first minimax search over incrementally evaluated NNUE states.

#include <cstdint>
#include <vector>

#include "eleginus/nnue.hpp"

namespace eleginus {

struct SearchOptions {
	int expansions = 32;
	int max_depth = 64;
	float depth_penalty = 0.12F;
	float uncertainty_weight = 0.20F;
};

struct RootMove {
	chess::Move move;
	float value = 0.5F;
	int subtree_nodes = 0;
};

struct SearchResult {
	chess::Move move{chess::Move::NO_MOVE};
	float value = 0.5F;
	int expanded_nodes = 0;
	int evaluated_nodes = 0;
	std::vector<RootMove> root;
};

class Searcher {
	public:
	Searcher(const CpuValue &value, SearchOptions options = {});
	SearchResult search(const chess::Board &board) const;

	private:
	const CpuValue *value_;
	SearchOptions options_;
};

} // namespace eleginus
