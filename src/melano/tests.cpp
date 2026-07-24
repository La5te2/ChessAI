// Focused Melano smoke tests for codecs, P/V/A gradients, checkpoints, and search.

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_set>
#include "melano/checkpoint.hpp"
#include "melano/dataset.hpp"
#include "melano/fcpi.hpp"
#include "melano/game.hpp"
#include "melano/model.hpp"
#include "melano/search.hpp"

namespace {

// Fail one test with a concise message instead of introducing a test-framework dependency.
void require(bool condition, const char *message) {
	if (!condition) {
		throw std::runtime_error(message);
	}
}

// Verify every legal action survives move->index->legal-move decoding in this position.
void require_move_codec(const chess::Board &board) {
	std::unordered_set<int> actions;
	for (const auto &move : melano::legal_moves(board)) {
		const int action = melano::move_to_index(move);
		require(action >= 0 && action < melano::kActionSize,
				"move codec produced an out-of-range action");
		require(actions.insert(action).second, "legal moves share an action index");
		require(melano::index_to_move(action, board) == move, "move codec round trip failed");
	}
}

// Ensure one backward pass produced finite gradients for every participating parameter.
void require_finite_gradients(const melano::Model &model) {
	for (const auto &parameter : model->parameters()) {
		require(parameter.grad().defined(), "model parameter has no gradient");
		require(torch::isfinite(parameter.grad()).all().item<bool>(),
				"model gradient contains a non-finite value");
	}
}

} // namespace

// Exercise the complete minimal Melano inference/training/checkpoint/search surface.
int main() {
	try {
		require(melano::parse_compute_precision("fp32") == melano::ComputePrecision::Fp32,
				"fp32 precision parsing failed");
		require(melano::parse_compute_precision("bf16") == melano::ComputePrecision::Bf16,
				"bf16 precision parsing failed");
		require(std::string(melano::compute_precision_name(melano::ComputePrecision::Bf16)) ==
					"bf16",
				"bf16 precision name mismatch");

		chess::Board board;
		require(board.hash() == 0x463b96181691fc9cULL, "Polyglot start-position hash mismatch");
		const auto packed = melano::encode_state(board);
		require(packed[0] == 4 && packed[4] == 6 && packed[8] == 1,
				"white piece token mismatch");
		require(packed[48] == 7 && packed[60] == 12 && packed[63] == 10,
				"black piece token mismatch");
		require(packed[64] == 1, "side-to-move token mismatch");
		require(packed[65] == 15, "castling token mismatch");
		require(packed[66] == 0, "en-passant token mismatch");

		require_move_codec(board);
		chess::Board promotion("8/P7/8/8/8/8/8/k6K w - - 0 1");
		require_move_codec(promotion);
		chess::Board castling("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
		require_move_codec(castling);
		const auto king_side_castle = chess::uci::uciToMove(castling, "e1g1");
		const auto queen_side_castle = chess::uci::uciToMove(castling, "e1c1");
		require(melano::move_to_index(king_side_castle) == 4 * 64 + 6,
				"king-side castling policy index mismatch");
		require(melano::move_to_index(queen_side_castle) == 4 * 64 + 2,
				"queen-side castling policy index mismatch");
		chess::Board black_castling("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1");
		require_move_codec(black_castling);
		const auto black_king_side = chess::uci::uciToMove(black_castling, "e8g8");
		const auto black_queen_side = chess::uci::uciToMove(black_castling, "e8c8");
		require(melano::move_to_index(black_king_side) == 60 * 64 + 62,
				"black king-side castling policy index mismatch");
		require(melano::move_to_index(black_queen_side) == 60 * 64 + 58,
				"black queen-side castling policy index mismatch");
		chess::Board black_promotion("k6K/8/8/8/8/8/p7/8 b - - 0 1");
		require_move_codec(black_promotion);
		chess::Board en_passant("8/8/8/3pP3/8/8/8/K6k w - d6 0 1");
		require_move_codec(en_passant);
		chess::Board black_en_passant("8/8/8/8/3pP3/8/8/K6k b - e3 0 1");
		require_move_codec(black_en_passant);

		// Exercise both colors and varied tactical states beyond hand-picked special moves.
		chess::Board walk;
		for (int ply = 0; ply < 256; ++ply) {
			if (melano::game_is_over(walk)) {
				walk = chess::Board();
			}
			require_move_codec(walk);
			const auto moves = melano::legal_moves(walk);
			require(!moves.empty(), "non-terminal codec walk has no legal moves");
			walk.makeMove(moves[(static_cast<std::size_t>(ply) * 37 + 11) % moves.size()]);
		}

		chess::Board checkmate("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1");
		require(melano::game_is_over(checkmate), "checkmate was not detected");
		require(melano::terminal_value_side_to_move(checkmate) == -1.0F,
				"checkmate side-to-move value mismatch");
		require(melano::game_result(checkmate) == "1-0", "checkmate result mismatch");
		require(melano::game_termination(checkmate) == "checkmate",
				"checkmate termination mismatch");

		chess::Board stalemate("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
		require(melano::game_is_over(stalemate), "stalemate was not detected");
		require(melano::terminal_value_side_to_move(stalemate) == 0.0F,
				"stalemate side-to-move value mismatch");
		require(melano::game_termination(stalemate) == "stalemate",
				"stalemate termination mismatch");

		chess::Board insufficient("7k/8/8/8/8/8/8/K7 w - - 0 1");
		require(melano::game_termination(insufficient) == "insufficient material",
				"insufficient-material termination mismatch");

		chess::Board fifty_move("7k/8/8/8/8/8/6R1/K7 b - - 100 1");
		require(melano::game_termination(fifty_move) == "fifty move rule",
				"fifty-move termination mismatch");

		chess::Board repetition;
		for (const char *uci : {"g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6",
							"f3g1", "f6g8"}) {
			repetition.makeMove(chess::uci::uciToMove(repetition, uci));
		}
		require(repetition.isRepetition(2), "threefold repetition count mismatch");
		require(!repetition.isRepetition(3), "threefold repetition was counted as fourfold");
		require(melano::game_termination(repetition) == "threefold repetition",
				"threefold-repetition termination mismatch");

		// A first annotated move has no previous score but does have an exact successor value.
		const auto pgn = std::filesystem::temp_directory_path() / "melanotest.pgn";
		const auto h5 = std::filesystem::temp_directory_path() / "melanotest.h5";
		{
			std::ofstream output(pgn);
			output << "[Event \"Melano test\"]\n"
					  "[Result \"1-0\"]\n\n"
					  "1. e4 {+0.60/12} e5 {+0.20/12} 1-0\n";
		}
		melano::PreprocessOptions preprocess;
		preprocess.input = pgn;
		preprocess.output = h5;
		preprocess.max_games = 1;
		preprocess.chunk_size = 2;
		preprocess.compression_level = 0;
		preprocess.log_every = 0;
		melano::preprocess_pgn(preprocess);
		{
			melano::SupervisedH5 supervised(h5);
			require(supervised.info().length == 2, "annotated PGN row count mismatch");
			const auto supervised_batch = supervised.read({0, 1});
			const auto expected_initial = melano::encode_boards({board}).index({0});
			require(torch::equal(supervised_batch.states.index({0}), expected_initial),
					"HDF5 initial state differs from live state codec");
			auto after_e4 = board;
			const auto e4 = chess::uci::uciToMove(after_e4, "e2e4");
			after_e4.makeMove(e4);
			const auto expected_after_e4 = melano::encode_boards({after_e4}).index({0});
			require(torch::equal(supervised_batch.next_states.index({0}), expected_after_e4),
					"HDF5 successor state differs from live state codec");
			require(supervised_batch.moves.index({0}).item<std::int64_t>() ==
						melano::move_to_index(e4),
					"HDF5 policy target differs from live move codec");
			const float expected_successor = -static_cast<float>(std::tanh(0.60 / 3.0));
			require(std::abs(supervised_batch.values.index({0}).item<float>()) < 1e-6F,
					"first unanchored value must remain neutral");
			require(std::abs(supervised_batch.next_values.index({0}).item<float>() -
							 expected_successor) < 1e-6F,
					"first annotated successor value was lost");
		}
		std::filesystem::remove(pgn);
		std::filesystem::remove(h5);

		auto model = melano::Model(8, 1);
		auto states = melano::encode_boards({board, board});
		auto tokens = model->encode(states);
		auto output = model->predict(tokens);
		require(output.policy.sizes() == torch::IntArrayRef({2, melano::kActionSize}),
				"policy shape mismatch");
		require(output.value.sizes() == torch::IntArrayRef({2, 1}), "value shape mismatch");
		require(output.advantages.sizes() == torch::IntArrayRef({2, melano::kActionSize}),
				"advantage shape mismatch");
		require(torch::isfinite(output.policy).all().item<bool>(),
				"policy contains a non-finite value");
		require(torch::isfinite(output.value).all().item<bool>(),
				"value contains a non-finite value");
		require(torch::isfinite(output.advantages).all().item<bool>(),
				"advantage contains a non-finite value");
		require(output.value.abs().max().item<float>() <= 1.000001F, "value range mismatch");
		require(output.advantages.max().item<float>() <= 1e-6F &&
				output.advantages.min().item<float>() >= -2.000001F,
				"advantage range mismatch");
		auto actions = torch::tensor(
			{melano::move_to_index(chess::uci::uciToMove(board, "e2e4")),
			 melano::move_to_index(chess::uci::uciToMove(board, "g1f3"))},
			torch::kInt64);
		auto successor = model->transition(tokens, actions);
		require(successor.sizes() == torch::IntArrayRef({2, melano::kTokenCount, 8}),
				"latent successor shape mismatch");
		auto imagined = model->predict(successor);
		(output.policy.mean() + output.value.mean() + output.advantages.mean() +
		 successor.mean() + imagined.value.mean())
			.backward();
		require_finite_gradients(model);

		const auto checkpoint = std::filesystem::temp_directory_path() / "melanotest.pth";
		model->eval();
		auto reference = model->forward(melano::encode_boards({board}));
		auto reference_latent = model->encode(melano::encode_boards({board}));
		auto reference_successor = model->transition(reference_latent, actions.index({0}).reshape({1}));
		melano::save_checkpoint_atomic(checkpoint, model, {8, 1});
		melano::ArchitectureInfo info;
		auto loaded = melano::load_checkpoint(checkpoint, torch::Device(torch::kCPU), &info);
		require(info.channels == 8 && info.blocks == 1, "checkpoint architecture mismatch");
		loaded->eval();
		auto loaded_output = loaded->forward(melano::encode_boards({board}));
		require(loaded_output.policy.size(1) == melano::kActionSize,
				"loaded model output mismatch");
		require(torch::allclose(reference.policy, loaded_output.policy),
				"checkpoint changed policy output");
		require(torch::allclose(reference.value, loaded_output.value),
				"checkpoint changed value output");
		require(torch::allclose(reference.advantages, loaded_output.advantages),
				"checkpoint changed advantage output");
		auto loaded_latent = loaded->encode(melano::encode_boards({board}));
		auto loaded_successor = loaded->transition(loaded_latent, actions.index({0}).reshape({1}));
		require(torch::allclose(reference_successor, loaded_successor),
				"checkpoint changed latent transition output");

		// Closed search ranks legal actions with policy and Melano's advantage prior.
		melano::SearchOptions closed_options;
		closed_options.type = melano::SearchType::Closed;
		closed_options.mcts_sims = 0;
		closed_options.root_topn = 4;
		melano::Searcher closed_searcher(loaded, torch::Device(torch::kCPU), closed_options);
		const auto closed_result = closed_searcher.search(board);
		require(closed_result.root.size() == 4, "closed search root size mismatch");
		require(closed_result.sims_completed == 0, "closed search unexpectedly ran MCTS");
		require(melano::index_to_move(melano::move_to_index(closed_result.move), board) ==
					closed_result.move,
				"closed search selected an illegal move");

		// Four simulations cover PUCT selection, neural expansion, and value backup.
		melano::SearchOptions mcts_options = closed_options;
		mcts_options.type = melano::SearchType::OnlyMcts;
		mcts_options.mcts_sims = 4;
		mcts_options.mcts_min_sims = 4;
		mcts_options.mcts_batch_size = 2;
		melano::Searcher mcts_searcher(loaded, torch::Device(torch::kCPU), mcts_options);
		const auto mcts_result = mcts_searcher.search(board);
		require(mcts_result.sims_completed == 4, "MCTS simulation budget mismatch");
		require(mcts_result.expanded_nodes > 0, "MCTS did not expand a node");
		require(mcts_result.exact_evaluations >= 1,
				"K=2 MCTS did not retain an exact latent anchor");
		require(mcts_result.latent_evaluations > 0,
				"K=2 MCTS did not evaluate odd-depth latent successors");
		std::filesystem::remove(checkpoint);

		std::cout << "melanotests passed" << std::endl;
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "melanotests failed: " << error.what() << std::endl;
		return 1;
	}
}
