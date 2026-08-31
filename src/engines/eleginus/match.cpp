#include "eleginus/match.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace eleginus {

	namespace {

		std::uint64_t bigEndian(const unsigned char *bytes, int n) {
			std::uint64_t value = 0;
			for (int i = 0; i < n; ++i)
				value = (value << 8) | bytes[i];
			return value;
		}

		std::uint16_t bookMove(chess::Move move) {
			const int promotion = move.typeOf() == chess::Move::PROMOTION ? static_cast<int>(move.promotionType()) : 0;
			// Polyglot and chess.hpp both encode castling as king-to-rook.
			return static_cast<std::uint16_t>(move.to().index() | (move.from().index() << 6) | (promotion << 12));
		}

		double elo(double score) {
			if (score <= 0)
				return -std::numeric_limits<double>::infinity();
			if (score >= 1)
				return std::numeric_limits<double>::infinity();
			return 400 * std::log10(score / (1 - score));
		}

	} // namespace

	std::vector<chess::Board> openings(const std::filesystem::path &path) {
		std::ifstream in(path, std::ios::binary);
		if (!in)
			throw std::runtime_error("cannot open opening book: " + path.string());
		std::unordered_map<std::uint64_t, std::vector<std::uint16_t>> book;
		std::array<unsigned char, 16> entry{};
		while (in.read(reinterpret_cast<char *>(entry.data()), entry.size())) {
			if (bigEndian(entry.data() + 10, 2) != 0)
				book[bigEndian(entry.data(), 8)].push_back(static_cast<std::uint16_t>(bigEndian(entry.data() + 8, 2)));
		}
		if (in.bad() || in.gcount() != 0)
			throw std::runtime_error("invalid Polyglot opening book");
		if (!book.contains(chess::Board().hash()))
			throw std::runtime_error("opening book has no start-position moves");
		std::vector<chess::Board> pending{chess::Board()}, result;
		std::unordered_set<std::uint64_t> seen;
		while (!pending.empty()) {
			auto board = std::move(pending.back());
			pending.pop_back();
			if (!seen.insert(board.hash()).second)
				continue;
			const auto row = book.find(board.hash());
			if (row == book.end()) {
				if (board.isGameOver().first != chess::GameResultReason::NONE)
					throw std::runtime_error("opening book contains a terminal start position");
				result.push_back(std::move(board));
				continue;
			}
			chess::Movelist legal;
			chess::movegen::legalmoves(legal, board);
			for (const auto code : row->second) {
				const auto move = std::find_if(legal.begin(), legal.end(), [code](auto m) { return bookMove(m) == code; });
				if (move == legal.end())
					throw std::runtime_error("opening book contains an illegal move");
				auto child = board;
				child.makeMove(*move);
				pending.push_back(std::move(child));
			}
		}
		if (result.size() != kOpeningPairs)
			throw std::runtime_error("opening book must contain exactly 1000 distinct leaf positions; found " + std::to_string(result.size()));
		std::sort(result.begin(), result.end(), [](const auto &a, const auto &b) { return a.hash() < b.hash(); });
		return result;
	}

	int match(const Model &candidate, const Model &baseline, chess::Board board, chess::Color side, SearchOptions options, const SearchCancel &cancel) {
		options.multipv = 1;
		Searcher first(candidate, options), second(baseline, options);
		for (;;) {
			if (cancel && cancel())
				return -1;
			const auto [reason, result] = board.isGameOver();
			if (reason != chess::GameResultReason::NONE) {
				if (result == chess::GameResult::DRAW)
					return 1;
				const auto winner = result == chess::GameResult::WIN ? board.sideToMove() : ~board.sideToMove();
				return winner == side ? 2 : 0;
			}
			auto &search = board.sideToMove() == side ? first : second;
			const auto resultMove = search.search(board, {}, cancel);
			if (cancel && cancel())
				return -1;
			if (resultMove.move.move() == chess::Move::NO_MOVE)
				throw std::runtime_error("match search returned no move for a nonterminal position");
			board.makeMove(resultMove.move);
		}
	}

	MatchScore confidence(const std::array<int, 5> &pairs) {
		int count = 0;
		double total = 0;
		for (int i = 0; i < 5; ++i) {
			if (pairs[i] < 0)
				throw std::invalid_argument("negative opening-pair count");
			count += pairs[i];
			total += 0.25 * i * pairs[i];
		}
		if (count != kOpeningPairs)
			throw std::invalid_argument("confidence requires 1000 completed opening pairs");
		const double score = total / count;
		// Two-sided 95% Hoeffding interval for independent opening-pair scores in [0, 1].
		const double margin = std::sqrt(std::log(40.0) / (2 * count));
		return {score, elo(score), elo(std::max(0.0, score - margin)), elo(std::min(1.0, score + margin))};
	}

} // namespace eleginus
