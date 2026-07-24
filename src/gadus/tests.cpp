// Focused Gadus smoke tests for codecs, gradients, checkpoint round-trips, and search.

#include <filesystem>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include "gadus/checkpoint.hpp"
#include "gadus/game.hpp"
#include "gadus/model.hpp"
#include "gadus/search.hpp"

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
	for (const auto &move : gadus::legal_moves(board)) {
		const int action = gadus::move_to_index(move);
		require(action >= 0 && action < gadus::kActionSize,
				"move codec produced an out-of-range action");
		require(actions.insert(action).second, "legal moves share an action index");
		require(gadus::index_to_move(action, board) == move, "move codec round trip failed");
	}
}

// Ensure one backward pass produced finite gradients for every participating parameter.
void require_finite_gradients(const gadus::Model &model) {
	for (const auto &parameter : model->parameters()) {
		require(parameter.grad().defined(), "model parameter has no gradient");
		require(torch::isfinite(parameter.grad()).all().item<bool>(),
				"model gradient contains a non-finite value");
	}
}

} // namespace

// Exercise the complete minimal Gadus inference/training/checkpoint/search surface.
int main() {
	try {
		chess::Board board;
		require(gadus::parse_compute_precision("fp32") == gadus::ComputePrecision::Fp32,
				"fp32 precision parsing failed");
		require(gadus::parse_compute_precision("bf16") == gadus::ComputePrecision::Bf16,
				"bf16 precision parsing failed");
		require(std::string(gadus::compute_precision_name(gadus::ComputePrecision::Bf16)) ==
					"bf16",
				"bf16 precision name mismatch");
		require(board.hash() == 0x463b96181691fc9cULL, "Polyglot start-position hash mismatch");
		const auto packed = gadus::encode_state(board);
		require(packed[5 * 8] == 0x08, "white king state plane mismatch");
		require(packed[11 * 8 + 7] == 0x08, "black king state plane mismatch");
		for (int rank = 0; rank < 8; ++rank) {
			require(packed[12 * 8 + rank] == 0xFF, "side-to-move state plane mismatch");
		}

		require_move_codec(board);
		chess::Board promotion("8/P7/8/8/8/8/8/k6K w - - 0 1");
		require_move_codec(promotion);
		chess::Board castling("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
		require_move_codec(castling);
		chess::Board en_passant("8/8/8/3pP3/8/8/8/K6k w - d6 0 1");
		require_move_codec(en_passant);

		chess::Board checkmate("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1");
		require(gadus::game_is_over(checkmate), "checkmate was not detected");
		require(gadus::terminal_value_side_to_move(checkmate) == -1.0F,
				"checkmate side-to-move value mismatch");
		require(gadus::game_result(checkmate) == "1-0", "checkmate result mismatch");
		require(gadus::game_termination(checkmate) == "checkmate",
				"checkmate termination mismatch");

		chess::Board stalemate("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
		require(gadus::game_is_over(stalemate), "stalemate was not detected");
		require(gadus::terminal_value_side_to_move(stalemate) == 0.0F,
				"stalemate side-to-move value mismatch");
		require(gadus::game_termination(stalemate) == "stalemate",
				"stalemate termination mismatch");

		chess::Board insufficient("7k/8/8/8/8/8/8/K7 w - - 0 1");
		require(gadus::game_termination(insufficient) == "insufficient material",
				"insufficient-material termination mismatch");

		chess::Board fifty_move("7k/8/8/8/8/8/6R1/K7 b - - 100 1");
		require(gadus::game_termination(fifty_move) == "fifty move rule",
				"fifty-move termination mismatch");

		chess::Board repetition;
		for (const char *uci : {"g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6",
							"f3g1", "f6g8"}) {
			repetition.makeMove(chess::uci::uciToMove(repetition, uci));
		}
		require(repetition.isRepetition(2), "threefold repetition count mismatch");
		require(!repetition.isRepetition(3), "threefold repetition was counted as fourfold");
		require(gadus::game_termination(repetition) == "threefold repetition",
				"threefold-repetition termination mismatch");

		auto model = gadus::Model(8, 1);
		auto [policy, value] = model->forward(gadus::encode_boards({board, board}));
		require(policy.sizes() == torch::IntArrayRef({2, gadus::kActionSize}),
				"policy shape mismatch");
		require(value.sizes() == torch::IntArrayRef({2, 1}), "value shape mismatch");
		require(torch::isfinite(policy).all().item<bool>(), "policy contains a non-finite value");
		require(torch::isfinite(value).all().item<bool>(), "value contains a non-finite value");
		require(value.abs().max().item<float>() <= 1.000001F, "value range mismatch");
		(policy.mean() + value.mean()).backward();
		require_finite_gradients(model);

		const auto checkpoint = std::filesystem::temp_directory_path() / "gadustest.pth";
		model->eval();
		auto reference = model->forward(gadus::encode_boards({board}));
		gadus::save_checkpoint_atomic(checkpoint, model, {8, 1});
		gadus::ArchitectureInfo info;
		auto loaded = gadus::load_checkpoint(checkpoint, torch::Device(torch::kCPU), &info);
		require(info.channels == 8 && info.blocks == 1, "checkpoint architecture mismatch");
		loaded->eval();
		auto loaded_output = loaded->forward(gadus::encode_boards({board}));
		require(loaded_output.first.size(1) == gadus::kActionSize, "loaded model output mismatch");
		require(torch::allclose(reference.first, loaded_output.first),
				"checkpoint changed policy output");
		require(torch::allclose(reference.second, loaded_output.second),
				"checkpoint changed value output");

		// Closed search ranks legal policy actions without constructing an MCTS tree.
		gadus::SearchOptions closed_options;
		closed_options.type = gadus::SearchType::Closed;
		closed_options.mcts_sims = 0;
		closed_options.root_topn = 4;
		gadus::Searcher closed_searcher(loaded, torch::Device(torch::kCPU), closed_options);
		const auto closed_result = closed_searcher.search(board);
		require(closed_result.root.size() == 4, "closed search root size mismatch");
		require(closed_result.sims_completed == 0, "closed search unexpectedly ran MCTS");
		require(gadus::index_to_move(gadus::move_to_index(closed_result.move), board) ==
					closed_result.move,
				"closed search selected an illegal move");
		auto full_probabilities =
			torch::softmax(reference.first, 1).squeeze(0).to(torch::kCPU).contiguous();
		std::vector<float> full_policy(gadus::kActionSize);
		std::copy_n(full_probabilities.data_ptr<float>(), gadus::kActionSize,
					full_policy.begin());
		const auto expected_policy = gadus::normalize_legal_policy(full_policy, board);
		for (int action = 0; action < gadus::kActionSize; ++action) {
			require(std::abs(closed_result.policy[action] - expected_policy[action]) < 1e-5F,
					"compact legal-policy transfer changed closed search probabilities");
		}

		// Four simulations exercise selection, batched expansion, and value backup.
		gadus::SearchOptions mcts_options = closed_options;
		mcts_options.type = gadus::SearchType::OnlyMcts;
		mcts_options.mcts_sims = 4;
		mcts_options.mcts_min_sims = 4;
		mcts_options.mcts_batch_size = 2;
		gadus::Searcher mcts_searcher(loaded, torch::Device(torch::kCPU), mcts_options);
		const auto mcts_result = mcts_searcher.search(board);
		require(mcts_result.sims_completed == 4, "MCTS simulation budget mismatch");
		require(mcts_result.expanded_nodes > 0, "MCTS did not expand a node");
		std::filesystem::remove(checkpoint);

		std::cout << "gadustests passed" << std::endl;
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "gadustests failed: " << error.what() << std::endl;
		return 1;
	}
}
