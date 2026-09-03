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
		if (!ok) throw std::runtime_error(message);
	}

	void checkGrad(eleginus::Model model) {
		require(eleginus::Model::initial().size() == eleginus::kFormulaCount, "initial formula weights have the wrong dimensions");
		std::vector<eleginus::Feature> x;
		model.extract(chess::Board("r3k2r/ppp2ppp/2n1bn2/3qp3/3P4/2N1BN2/PPP2PPP/R2Q1RK1 b kq - 3 11"), x);
		require(x.size() > 1, "network gradient test needs two active formulas");
		auto p = model.params();
		for (std::size_t i = model.formulas(); i < p.size(); ++i) {
			p[i] = (i & 1U) == 0 ? 0.0003F : -0.0002F;
		}
		model.update(p);
		const float score = model.score(x);
		std::vector<float> grad(p.size());
		for (const auto &row : x) {
			grad[row.index] += static_cast<float>(row.score);
			for (const auto &condition : x) {
				grad[model.relationIndex(row.index, condition.index)] += static_cast<float>(row.score) * condition.condition;
			}
		}
		const auto relation = model.relationIndex(x.front().index, x.back().index);
		for (const std::size_t i : {static_cast<std::size_t>(x.front().index), relation}) {
			require(std::abs(grad[i]) > 1.0e-7F, "a network parameter has no gradient");
			const auto saved = p[i];
			constexpr float eps = 0.001F;
			p[i] = saved + eps;
			model.update(p);
			const float plus = model.score(x);
			p[i] = saved - eps;
			model.update(p);
			const float minus = model.score(x);
			p[i] = saved;
			model.update(p);
			const float numerical = (plus - minus) / (2.0F * eps);
			require(std::abs(numerical - grad[i]) < 0.002F * (1 + std::abs(grad[i])), "native backward failed finite differences");
		}
		std::vector<float> w(model.base().begin(), model.base().end());
		for (const auto &condition : x) {
			const auto offset = static_cast<std::size_t>(condition.index) * eleginus::kFormulaCount;
			for (std::size_t row = 0; row < eleginus::kFormulaCount; ++row) {
				w[row] += model.relations()[offset + row] * static_cast<float>(condition.condition);
			}
		}
		float explicitScore = 0;
		for (const auto &f : x) {
			explicitScore += w[f.index] * static_cast<float>(f.score);
		}
		require(std::abs(score - explicitScore) < 1.0e-5F, "reordered evaluation changed dynamic weighted sum");

		const auto bce = [](float h) { return std::max(h, 0.0F) - h + std::log1p(std::exp(-std::abs(h))); };
		const float delta = -1.0F / (1.0F + std::exp(score));
		for (std::size_t i = 0; i < p.size(); ++i) {
			p[i] -= 1.0e-4F * delta * grad[i];
		}
		model.update(p);
		require(bce(model.score(x)) < bce(score), "native gradient does not decrease BCE");

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
		for (int revision = 0; revision < 2; ++revision) {
			for (int repeat = 0; repeat < 3; ++repeat) {
				for (const auto &board : positions) {
					model.extract(board, x);
					for (const auto &f : x) {
						require(f.index < model.formulas(), "formula execution changed output indices");
						const bool reconstructible = f.condition >= std::abs(f.score) && ((f.condition - f.score) & 1) == 0;
						require(reconstructible, "formula score and condition cannot reconstruct both activations");
					}
					const float white = model.score(x);
					const float expected = board.sideToMove() == chess::Color::WHITE ? white : -white;
					const float direct = model.score(board);
					require(std::abs(direct - expected) < 2.0e-5F * (1 + std::abs(expected)), "direct network output changed formula evaluation");
				}
			}
			// Repeat after changing both a base weight and a relation weight.
			p[0] += 0.125F;
			p.back() += 0.05F;
			model.update(p);
		}

		// External FEN can be structurally invalid; formula extraction must remain memory-safe.
		model.extract(chess::Board("4k3/P1P1P1P1/1P1P1P1P/P1P1P1P1/1P1P1P1P/P7/8/1B2K3 w - - 0 1"), x);
		for (const auto &f : x) require(f.index < model.formulas(), "malformed position escaped the formula dimensions");

		chess::Board walk;
		eleginus::Accumulator accumulator(model);
		accumulator.reset(walk);
		for (unsigned ply = 1; ply < 80; ++ply) {
			model.extract(walk, x);
			const float white = model.score(x);
			const float expected = walk.sideToMove() == chess::Color::WHITE ? white : -white;
			require(std::abs(model.score(walk) - expected) < 2.0e-5F * (1 + std::abs(expected)), "attack-cache reuse changed formula evaluation");
			require(std::abs(accumulator.score(walk) - expected) < 2.0e-4F * (1 + std::abs(expected)), "incremental accumulator changed model output");
			chess::Movelist moves;
			chess::movegen::legalmoves(moves, walk);
			if (moves.empty()) break;
			const auto move = moves[(17U * ply + 5U) % moves.size()];
			walk.makeMove(move);
			accumulator.push();
		}

		chess::Board restore;
		accumulator.reset(restore);
		chess::Movelist firstMoves;
		chess::movegen::legalmoves(firstMoves, restore);
		const auto first = firstMoves.front();
		restore.makeMove(first);
		accumulator.push();
		chess::Movelist secondMoves;
		chess::movegen::legalmoves(secondMoves, restore);
		const auto second = secondMoves.front();
		restore.makeMove(second);
		accumulator.push();
		const float child = accumulator.score(restore);
		require(std::abs(child - model.score(restore)) < 2.0e-4F * (1 + std::abs(child)), "lazy accumulator changed a grandchild score");
		accumulator.pop();
		restore.unmakeMove(second);
		const float parent = accumulator.score(restore);
		require(std::abs(parent - model.score(restore)) < 2.0e-4F * (1 + std::abs(parent)), "lazy accumulator failed to materialize its parent");
		accumulator.pop();
		restore.unmakeMove(first);
		const float root = accumulator.score(restore);
		require(std::abs(root - model.score(restore)) < 2.0e-4F * (1 + std::abs(root)), "accumulator rollback changed the root score");
	}

} // namespace

int main() {
	try {
		eleginus::Model model;
		require(eleginus::centipawns(1.0F) == 150 && eleginus::centipawns(-1.0F) == -150, "centipawn conversion differs from the supervised target scale");
		std::vector<eleginus::Feature> x;
		model.extract(chess::Board(), x);
		require(std::isfinite(model.score(x)), "initial formula model produced a nonfinite score");
		checkGrad(model);

		const auto path = std::filesystem::temp_directory_path() / "eleginus-test.pth";
		auto values = model.params();
		values[model.relationIndex(x.front().index, x.back().index)] += 0.02F;
		model.update(values);
		model.save(path);
		const auto loaded = eleginus::Model::load(path);
		std::filesystem::remove(path);
		require(loaded.params() == model.params() && loaded.score(x) == model.score(x), "checkpoint changed network parameters or output");

		eleginus::SearchOptions options;
		options.depth = 4;
		options.hash_mb = 1;
		chess::Board mate("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1");
		const auto result = eleginus::Searcher(model, options).search(mate);
		require(result.move.move() != chess::Move::NO_MOVE, "search returned no move in a nonterminal position");
		mate.makeMove(result.move);
		chess::Movelist replies;
		chess::movegen::legalmoves(replies, mate);
		require(mate.inCheck() && replies.empty(), "search missed an immediate checkmate");
		require(result.root.front().move == result.move, "reported PV differs from bestmove");
		const auto lastPlyMate = eleginus::Searcher(model, options).search(chess::Board("7k/6Q1/6K1/8/8/8/8/8 b - - 100 1"));
		require(lastPlyMate.move.move() == chess::Move::NO_MOVE && lastPlyMate.score_cp < 0, "fifty-move adjudication overrode checkmate");

		options.depth = 4;
		options.quiescence_depth = 0;
		options.multipv = 256;
		const auto rows = eleginus::Searcher(model, options).search(chess::Board("7k/5K2/8/6Q1/8/8/8/8 w - - 0 1")).root;
		const auto draw = std::find_if(rows.begin(), rows.end(), [](const auto &row) { return chess::uci::moveToUci(row.move) == "g5g6"; });
		require(draw != rows.end() && draw->score_cp == 0, "qsearch evaluated a stalemate as a nonterminal position");

		// This low-material tree has no LMR; single-PV windows and full candidate scores must agree.
		const chess::Board ending("8/8/4k3/2p5/2P5/3K4/8/8 w - - 0 1");
		options.depth = 4;
		const auto full = eleginus::Searcher(model, options).search(ending);
		options.multipv = 2;
		const auto pair = eleginus::Searcher(model, options).search(ending);
		require(pair.root.size() == 2, "MultiPV returned the wrong number of root lines");
		require(std::equal(pair.root.begin(), pair.root.end(), full.root.begin(),
					[](const auto &left, const auto &right) { return left.move == right.move && left.score_cp == right.score_cp; }),
			"selective MultiPV differs from exhaustive root search in a fixed tree");
		options.multipv = 1;
		const auto narrow = eleginus::Searcher(model, options).search(ending);
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
		const auto interrupted = eleginus::Searcher(model, options).search(chess::Board(), progress, cancel);
		const bool retained = interrupted.depth == 3 && interrupted.score_cp == completedScore && interrupted.move == completedMove;
		require(retained, "interruption published an incomplete iteration");

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
		const bool perspective = eleginus::match(model, model, mated, chess::Color::WHITE, options, {}) == 2 &&
			eleginus::match(model, model, mated, chess::Color::BLACK, options, {}) == 0;
		require(perspective, "match result was not reported from the candidate's perspective");
		const auto cancelled = eleginus::match(model, model, chess::Board(), chess::Color::WHITE, options, [] { return true; });
		require(cancelled == -1, "cancelled match acquired a result");
		std::cout << "Eleginus tests passed\n";
		return 0;
	} catch (const std::exception &e) {
		std::cerr << "Eleginus tests failed: " << e.what() << '\n';
		return 1;
	}
}
