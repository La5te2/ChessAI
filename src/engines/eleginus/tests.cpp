#include "eleginus/model.hpp"
#include "eleginus/search.hpp"
#include "eleginus/match.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

	void require(bool ok, const char *message) {
		if (!ok)
			throw std::runtime_error(message);
	}

	void checkGrad(eleginus::Model model) {
		const auto &formulas = eleginus::FormulaSet::fixed();
		require(formulas.size() == eleginus::kFormulaCount, "fixed formula set has the wrong dimensions");
		require(formulas.names().size() == eleginus::kFormulaCount && formulas.names()[0] == "tempo" && formulas.names()[1] == "material.pawn" &&
		        formulas.names()[6] == "pst.pawn.a1" && formulas.names()[390] == "bishopPair" && formulas.names()[621] == "endgame.oppositeBishopPassers",
		    "formula catalog no longer matches the fixed coordinate order");
		std::vector<eleginus::Feature> x;
		model.extract(chess::Board("r3k2r/ppp2ppp/2n1bn2/3qp3/3P4/2N1BN2/PPP2PPP/R2Q1RK1 b kq - 3 11"), x);
		require(x.size() > 1 && model.activate(x[0].index, x[1].index), "cannot activate a graybox relation");
		auto &p = model.params();
		eleginus::Model::Cache cache;
		const float score = model.forward(x, cache);
		std::vector<float> grad(p.size());
		model.backward(cache, 1.0F, grad);
		for (const std::size_t i : {static_cast<std::size_t>(x.front().index), p.size() - 1}) {
			require(std::abs(grad[i]) > 1.0e-7F, "a graybox parameter has no gradient");
			const auto saved = p[i];
			constexpr float eps = 0.001F;
			p[i] = saved + eps;
			const float plus = model.score(x);
			p[i] = saved - eps;
			const float minus = model.score(x);
			p[i] = saved;
			const float numerical = (plus - minus) / (2.0F * eps);
			require(std::abs(numerical - grad[i]) < 0.002F * (1 + std::abs(grad[i])), "native backward failed finite differences");
		}
		std::vector<float> w;
		model.weights(x, w);
		float explicitScore = 0;
		for (const auto &f : x)
			explicitScore += w[f.index] * static_cast<float>(f.score);
		require(std::abs(score - explicitScore) < 1.0e-5F, "reordered evaluation changed dynamic weighted sum");

		const auto bce = [](float h) { return std::max(h, 0.0F) - h + std::log1p(std::exp(-std::abs(h))); };
		const float delta = -1.0F / (1.0F + std::exp(score));
		for (std::size_t i = 0; i < p.size(); ++i)
			p[i] -= 1.0e-4F * delta * grad[i];
		require(bce(model.score(x)) < bce(score), "native gradient does not decrease BCE");

		// Repeated pawn keys must not retain king/occupancy/turn-dependent results.
		const std::array<chess::Board, 11> positions{chess::Board(), chess::Board("rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1"),
		    chess::Board("r3k2r/ppp2ppp/2n1bn2/3qp3/3P4/2N1BN2/PPP2PPP/R2Q1RK1 b kq - 3 11"), chess::Board("8/2p5/3p4/1P1Pp1k1/4P3/5K2/8/8 w - - 12 42"),
		    chess::Board("8/2p5/3p4/1P1Pp1k1/4P3/6K1/8/8 b - - 13 42"), chess::Board("4k3/8/8/8/8/8/5Q2/4K3 w - - 0 1"), chess::Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"),
		    chess::Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"), chess::Board("4k3/P7/8/8/8/8/7p/4K3 w - - 0 1"),
		    chess::Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
		    chess::Board("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 4 6")};
		for (int revision = 0; revision < 2; ++revision) {
			for (int repeat = 0; repeat < 3; ++repeat)
				for (const auto &board : positions) {
					model.extract(board, x);
					for (const auto &f : x) {
						require(f.index < model.formulas(), "formula execution changed output indices");
						require(f.condition >= std::abs(f.score) && ((f.condition - f.score) & 1) == 0, "formula score and condition cannot reconstruct both activations");
					}
					const float white = model.score(x);
					const float expected = board.sideToMove() == chess::Color::WHITE ? white : -white;
					require(std::abs(model.score(board) - expected) < 2.0e-5F * (1 + std::abs(expected)), "direct graybox output changed sparse evaluation");
				}
			// A new search reads the model parameters again.
			p[0] += 0.125F;
			p.back() += 0.05F;
		}

		chess::Board walk;
		for (unsigned ply = 1; ply < 80; ++ply) {
			model.extract(walk, x);
			const float white = model.score(x);
			const float expected = walk.sideToMove() == chess::Color::WHITE ? white : -white;
			require(std::abs(model.score(walk) - expected) < 2.0e-5F * (1 + std::abs(expected)), "attack-cache reuse changed formula evaluation");
			chess::Movelist moves;
			chess::movegen::legalmoves(moves, walk);
			if (moves.empty())
				break;
			walk.makeMove(moves[(17U * ply + 5U) % moves.size()]);
		}
	}

} // namespace

int main() {
	try {
		eleginus::Model model;
		const chess::Board queen("4k3/8/8/8/8/8/4Q3/4K3 w - - 0 1");
		const chess::Board enemy("4k3/8/8/8/8/8/4Q3/4K3 b - - 0 1");
		require(model.centipawns(queen) > 600 && model.centipawns(enemy) < -600, "initial material score or perspective is incorrect");

		std::vector<eleginus::Feature> x;
		model.extract(queen, x);
		require(std::isfinite(model.score(x)), "initial kernel model produced a nonfinite score");
		checkGrad(model);

		const auto path = std::filesystem::temp_directory_path() / "eleginus-test.pth";
		model.activate(x.front().index, x.back().index);
		model.params().back() += 0.02F;
		model.save(path);
		const auto loaded = eleginus::Model::load(path);
		std::filesystem::remove(path);
		require(loaded.params() == model.params() && loaded.score(x) == model.score(x), "checkpoint changed graybox parameters or output");

		eleginus::SearchOptions options;
		options.depth = 4;
		options.hash_mb = 1;
		const auto result = eleginus::Searcher(model, options).search(chess::Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"));
		require(result.move.move() != chess::Move::NO_MOVE && result.score_cp == 29999, "search missed the shortest forced mate");
		require(result.root.front().move == result.move, "reported PV differs from bestmove");
		const auto lastPlyMate = eleginus::Searcher(model, options).search(chess::Board("7k/6Q1/6K1/8/8/8/8/8 b - - 100 1"));
		require(lastPlyMate.score_cp == -30000, "fifty-move adjudication overrode checkmate");

		options.depth = 1;
		options.quiescence_depth = 0;
		options.multipv = 256;
		const auto rows = eleginus::Searcher(model, options).search(chess::Board("7k/5K2/8/6Q1/8/8/8/8 w - - 0 1")).root;
		const auto draw = std::find_if(rows.begin(), rows.end(), [](const auto &row) { return chess::uci::moveToUci(row.move) == "g5g6"; });
		require(draw != rows.end() && draw->score_cp == 0, "qsearch evaluated a stalemate as a nonterminal position");

		// This low-material tree has no LMR; single-PV windows and full candidate scores must agree.
		const chess::Board ending("8/8/4k3/2p5/2P5/3K4/8/8 w - - 0 1");
		options.depth = 4;
		const auto full = eleginus::Searcher(model, options).search(ending);
		options.multipv = 1;
		const auto narrow = eleginus::Searcher(model, options).search(ending);
		require(narrow.score_cp == full.score_cp, "aspiration changed a fixed-tree score");

		options.depth = 8;
		int published = 0, probes = 0;
		int completedScore = 0;
		chess::Move completedMove;
		const auto interrupted = eleginus::Searcher(model, options)
		                             .search(
		                                 chess::Board(),
		                                 [&](const auto &r) {
			                                 published = r.depth;
			                                 completedScore = r.score_cp;
			                                 completedMove = r.move;
		                                 },
		                                 [&] { return published >= 3 && ++probes >= 3; });
		require(interrupted.depth == 3 && interrupted.score_cp == completedScore && interrupted.move == completedMove, "interruption published an incomplete iteration");

		const auto tied = eleginus::confidence({0, 0, 1000, 0, 0});
		const auto ahead = eleginus::confidence({0, 0, 760, 240, 0});
		const auto behind = eleginus::confidence({0, 240, 760, 0, 0});
		require(tied.elo == 0 && tied.low < 0 && tied.high > 0, "tied opening pairs acquired a significant Elo advantage");
		require(ahead.low > 0 && behind.high < 0 && std::abs(ahead.low + behind.high) < 1.0e-10, "paired confidence has an incorrect perspective");
		bool incomplete = false;
		try {
			eleginus::confidence({0, 0, 999, 0, 0});
		} catch (const std::invalid_argument &) {
			incomplete = true;
		}
		require(incomplete, "incomplete evaluation was allowed to produce an acceptance interval");
		const chess::Board mated("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1");
		require(eleginus::match(model, model, mated, chess::Color::WHITE, options, {}) == 2 && eleginus::match(model, model, mated, chess::Color::BLACK, options, {}) == 0,
		    "match result was not reported from the candidate's perspective");
		require(eleginus::match(model, model, chess::Board(), chess::Color::WHITE, options, [] { return true; }) == -1, "cancelled match acquired a result");
		std::cout << "Eleginus tests passed\n";
		return 0;
	} catch (const std::exception &e) {
		std::cerr << "Eleginus tests failed: " << e.what() << '\n';
		return 1;
	}
}
