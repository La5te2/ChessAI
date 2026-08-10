#pragma once

// Value-led principal-variation search with Policy move ordering.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "eleginus/nnue.hpp"

namespace eleginus {

struct SearchOptions {
	int depth = 4;
	int quiescence_depth = 8;
	std::size_t hash_mb = 64;
	std::uint64_t node_limit = 0;
	int threads = 1;
	int multipv = 1;
};

struct RootMove {
	chess::Move move;
	float prior = 0.0F;
	int score_cp = 0;
	std::uint64_t nodes = 0;
	bool exact_score = false;
};

struct SearchResult {
	chess::Move move{chess::Move::NO_MOVE};
	int score_cp = 0;
	int depth = 0;
	int selective_depth = 0;
	std::uint64_t nodes = 0;
	std::uint64_t evaluated_nodes = 0;
	std::vector<RootMove> root;
};

using SearchProgressCallback = std::function<void(const SearchResult &)>;
using SearchCancelCallback = std::function<bool()>;

class Searcher {
	public:
	Searcher(const CpuPolicy &policy, const CpuValue &value, SearchOptions options = {});
	SearchResult search(const chess::Board &board,
						const SearchProgressCallback &progress = {},
						const SearchCancelCallback &cancel = {}) const;

	private:
	const CpuPolicy *policy_;
	const CpuValue *value_;
	SearchOptions options_;
};

} // namespace eleginus
