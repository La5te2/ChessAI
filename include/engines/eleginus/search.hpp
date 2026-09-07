#pragma once

#include "chess.hpp"
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace eleginus {
	class Evaluator;

	struct SearchOptions {
		int depth = 6;
		int quiescence_depth = 8;
		std::size_t hash_mb = 64;
		std::uint64_t node_limit = 0;
		int movetime_ms = 0;
		int multipv = 1;
	};

	struct RootMove {
		chess::Move move{chess::Move::NO_MOVE};
		int score_cp = 0;
	};

	struct SearchResult {
		chess::Move move{chess::Move::NO_MOVE};
		int score_cp = 0;
		int depth = 0;
		int selective_depth = 0;
		std::uint64_t nodes = 0;
		std::uint64_t elapsed_ms = 0;
		std::vector<RootMove> root;
	};

	using SearchProgress = std::function<void(const SearchResult &)>;
	using SearchCancel = std::function<bool()>;

	class SearchState;

	class Searcher {
	public:
		// Validate the options and allocate the persistent transposition table.
		Searcher(const Evaluator &evaluator, SearchOptions options = {});
		~Searcher();
		Searcher(const Searcher &) = delete;
		Searcher &operator=(const Searcher &) = delete;

		// Search one board by iterative deepening and report each completed depth through progress.
		SearchResult search(const chess::Board &board, const SearchProgress &progress = {}, const SearchCancel &cancel = {});

	private:
		const Evaluator *evaluator;
		SearchOptions opts;
		std::unique_ptr<SearchState> state;
	};

} // namespace eleginus
