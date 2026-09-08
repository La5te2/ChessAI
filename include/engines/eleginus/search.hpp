#pragma once

#include "chess.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace eleginus {
	class Evaluator;

	struct SearchOptions {
		int depth = 6;
		int quiescenceDepth = 8;
		std::size_t hashMiB = 64;
		std::uint64_t nodeLimit = 0;
		int moveTimeMs = 0;
		int multipv = 1;
	};

	struct RootMove {
		chess::Move move{chess::Move::NO_MOVE};
		int scoreCp = 0;
	};

	struct SearchResult {
		chess::Move move{chess::Move::NO_MOVE};
		int scoreCp = 0;
		int depth = 0;
		int selectiveDepth = 0;
		std::uint64_t nodes = 0;
		std::uint64_t elapsedMs = 0;
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
