#pragma once

#include "eleginus/model.hpp"
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace eleginus {

struct SearchOptions {
	int depth = 6;
	int quiescence_depth = 8;
	std::size_t hash_mb = 64;
	std::uint64_t node_limit = 0;
	int movetime_ms = 0;
	int multipv = 1;
	float uncertainty_threshold = 0.35F;
	int uncertainty_extensions = 1;
	bool capture_principal_variation = false;
};

struct RootMove {
	chess::Move move{chess::Move::NO_MOVE};
	int score_cp = 0;
	std::uint64_t nodes = 0;
};

struct SearchResult {
	chess::Move move{chess::Move::NO_MOVE};
	int score_cp = 0;
	int depth = 0;
	int selective_depth = 0;
	std::uint64_t nodes = 0;
	std::uint64_t evaluated_nodes = 0;
	std::uint64_t elapsed_ms = 0;
	std::vector<RootMove> root;
	std::vector<chess::Move> principal_variation;
};

using SearchProgress = std::function<void(const SearchResult &)>;
using SearchCancel = std::function<bool()>;

class SearchState;

class Searcher {
public:
	explicit Searcher(const Model &model, SearchOptions options = {});
	~Searcher();
	Searcher(Searcher &&) noexcept;
	Searcher &operator=(Searcher &&) noexcept;
	Searcher(const Searcher &) = delete;
	Searcher &operator=(const Searcher &) = delete;

	SearchResult search(const chess::Board &board, const SearchProgress &progress = {}, const SearchCancel &cancel = {});

private:
	const Model *model_;
	SearchOptions options_;
	std::unique_ptr<SearchState> state_;
};

} // namespace eleginus
