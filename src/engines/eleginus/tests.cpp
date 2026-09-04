#include "eleginus/formula.hpp"
#include "eleginus/game.hpp"
#include "eleginus/search.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

	void require(bool ok, const char *message) {
		if (!ok) throw std::runtime_error(message);
	}

	float reconstruct(std::span<const std::int32_t> signals, const eleginus::FormulaContext &context,
		const std::vector<eleginus::FormulaParameter> &parameters) {
		float score = 0.0F;
		for (std::size_t i = 0; i < parameters.size(); ++i) {
			float weight = parameters[i].base;
			for (std::size_t type = 0; type < context.material.size(); ++type) {
				weight = std::fma(parameters[i].material[type], context.material[type], weight);
			}
			score = std::fma(weight, static_cast<float>(signals[i]), score);
		}

		const auto globals = eleginus::FormulaSet::globals();
		if (score != 0.0F) {
			const std::size_t branch = score < 0.0F;
			const float win = std::inner_product(context.winnable[branch].begin(), context.winnable[branch].end(), globals.winnable.begin(), 0.0F);
			score = score > 0.0F ? std::max(0.0F, score + win) : std::min(0.0F, score - win);
			const auto &facts = context.scale[branch];
			float factor = facts[0] != 0.0F ? globals.scale[0] : 1.0F;
			if (facts[1] != 0.0F) factor = std::min(factor, globals.scale[1] + globals.scale[4] * facts[3] + globals.scale[3] * facts[4]);
			else if (facts[2] != 0.0F) factor = std::min(factor, globals.scale[2] + globals.scale[3] * facts[4]);
			score *= std::clamp(factor, 0.0F, 1.0F);
		}
		return score;
	}

	void checkFormulas() {
		// Repeated pawn keys must not retain king/occupancy/turn-dependent results.
		const std::array<chess::Board, 11> positions{
			chess::Board(),
			chess::Board("rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1"),
			chess::Board("r3k2r/ppp2ppp/2n1bn2/3qp3/3P4/2N1BN2/PPP2PPP/R2Q1RK1 b kq - 3 11"),
			chess::Board("8/2p5/3p4/1P1Pp1k1/4P3/5K2/8/8 w - - 12 42"),
			chess::Board("8/2p5/3p4/1P1Pp1k1/4P3/6K1/8/8 b - - 13 42"),
			chess::Board("4k3/8/8/8/8/8/5Q2/4K3 w - - 0 1"),
			chess::Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"),
			chess::Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
			chess::Board("4k3/P7/8/8/8/8/7p/4K3 w - - 0 1"),
			chess::Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
			chess::Board("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 4 6"),
		};
		for (int repeat = 0; repeat < 3; ++repeat) {
			for (const auto &board : positions) {
				require(std::isfinite(eleginus::FormulaSet::score(board)), "formula evaluation produced a nonfinite score");
			}
		}

		const std::array<std::pair<chess::Board, chess::Board>, 2> mirrors{{
			{chess::Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
				chess::Board("r3k2r/pppbbppp/2n2q1P/1P2p3/3pn3/BN2PNP1/P1PPQPB1/R3K2R b KQkq - 0 1")},
			{chess::Board("8/2p5/3p4/1P1Pp1k1/4P3/5K2/8/8 w - - 12 42"),
				chess::Board("8/8/5k2/4p3/1p1pP1K1/3P4/2P5/8 b - - 12 42")},
		}};
		for (const auto &[board, mirror] : mirrors) {
			const float sum = eleginus::FormulaSet::score(board) + eleginus::FormulaSet::score(mirror);
			require(std::abs(sum) < 1.0e-5F, "formula evaluation lost color-mirror antisymmetry");
		}

		const auto parameters = eleginus::FormulaSet::parameters();
		std::vector<std::int32_t> signals(parameters.size());
		for (const auto &board : positions) {
			const auto packed = eleginus::packBoard(board);
			const auto restored = eleginus::unpackBoard(packed);
			require(eleginus::packBoard(restored) == packed, "packed board codec is not reversible");
			eleginus::FormulaContext context;
			eleginus::FormulaSet::features(restored, signals, context);
			require(context.pressureIndex < signals.size(), "king-pressure formula index is invalid");
			require(std::all_of(context.material.begin(), context.material.end(), [](float value) { return std::isfinite(value); }),
				"material coordinates are not finite");
			require(std::abs(reconstruct(signals, context, parameters) - eleginus::FormulaSet::score(restored)) < 1.0e-5F,
				"extracted formula context does not reconstruct the production score");
		}

		// External FEN can be structurally invalid; formula extraction must remain memory-safe.
		require(std::isfinite(eleginus::FormulaSet::score(chess::Board("4k3/P1P1P1P1/1P1P1P1P/P1P1P1P1/1P1P1P1P/P7/8/1B2K3 w - - 0 1"))),
			"malformed position escaped formula evaluation");
	}

} // namespace

int main() {
	try {
		checkFormulas();

		eleginus::SearchOptions options;
		options.depth = 4;
		options.hash_mb = 1;
		chess::Board mate("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1");
		const auto result = eleginus::Searcher(options).search(mate);
		require(result.move.move() != chess::Move::NO_MOVE, "search returned no move in a nonterminal position");
		mate.makeMove(result.move);
		chess::Movelist replies;
		chess::movegen::legalmoves(replies, mate);
		require(mate.inCheck() && replies.empty(), "search missed an immediate checkmate");
		require(result.root.front().move == result.move, "reported PV differs from bestmove");
		const auto lastPlyMate = eleginus::Searcher(options).search(chess::Board("7k/6Q1/6K1/8/8/8/8/8 b - - 100 1"));
		require(lastPlyMate.move.move() == chess::Move::NO_MOVE && lastPlyMate.score_cp < 0, "fifty-move adjudication overrode checkmate");

		options.depth = 4;
		options.quiescence_depth = 0;
		options.multipv = 256;
		const auto rows = eleginus::Searcher(options).search(chess::Board("7k/5K2/8/6Q1/8/8/8/8 w - - 0 1")).root;
		const auto draw = std::find_if(rows.begin(), rows.end(), [](const auto &row) { return chess::uci::moveToUci(row.move) == "g5g6"; });
		require(draw != rows.end() && draw->score_cp == 0, "qsearch evaluated a stalemate as a nonterminal position");

		// This low-material tree has no LMR; single-PV windows and full candidate scores must agree.
		const chess::Board ending("8/8/4k3/2p5/2P5/3K4/8/8 w - - 0 1");
		options.depth = 4;
		const auto full = eleginus::Searcher(options).search(ending);
		options.multipv = 2;
		const auto pair = eleginus::Searcher(options).search(ending);
		require(pair.root.size() == 2, "MultiPV returned the wrong number of root lines");
		require(std::equal(pair.root.begin(), pair.root.end(), full.root.begin(),
					[](const auto &left, const auto &right) { return left.move == right.move && left.score_cp == right.score_cp; }),
			"selective MultiPV differs from exhaustive root search in a fixed tree");
		options.multipv = 1;
		const auto narrow = eleginus::Searcher(options).search(ending);
		require(narrow.score_cp == full.score_cp, "aspiration changed a fixed-tree score");

		options.depth = 8;
		int published = 0, probes = 0;
		int completedScore = 0;
		chess::Move completedMove;
		const auto progress = [&](const auto &r) {
			published = r.depth;
			completedScore = r.score_cp;
			completedMove = r.move;
		};
		const auto cancel = [&] { return published >= 3 && ++probes >= 3; };
		const auto interrupted = eleginus::Searcher(options).search(chess::Board(), progress, cancel);
		const bool retained = interrupted.depth == 3 && interrupted.score_cp == completedScore && interrupted.move == completedMove;
		require(retained, "interruption published an incomplete iteration");

		std::cout << "Eleginus tests passed\n";
		return 0;
	} catch (const std::exception &e) {
		std::cerr << "Eleginus tests failed: " << e.what() << '\n';
		return 1;
	}
}
