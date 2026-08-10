#pragma once

// Policy-guided best-first minimax over independent incremental Policy/Value states.

#include <cstdint>
#include <vector>

#include "eleginus/nnue.hpp"

namespace eleginus {

struct SearchOptions {
	int expansions = 32;
	int max_depth = 64;
};

struct RootMove {
	chess::Move move;
	float prior = 0.0F;
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
	Searcher(const CpuPolicy &policy, const CpuValue &value, SearchOptions options = {});
	SearchResult search(const chess::Board &board) const;

	private:
	const CpuPolicy *policy_;
	const CpuValue *value_;
	SearchOptions options_;
};

} // namespace eleginus
