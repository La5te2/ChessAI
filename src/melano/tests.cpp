#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <unordered_set>

#include "melano/checkpoint.hpp"
#include "melano/game.hpp"
#include "melano/model.hpp"

namespace {

void require(bool condition, const char *message) {
	if (!condition) {
		throw std::runtime_error(message);
	}
}

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

void require_finite_gradients(const melano::Model &model) {
	for (const auto &parameter : model->parameters()) {
		require(parameter.grad().defined(), "model parameter has no gradient");
		require(torch::isfinite(parameter.grad()).all().item<bool>(),
				"model gradient contains a non-finite value");
	}
}

} // namespace

int main() {
	try {
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
		chess::Board en_passant("8/8/8/3pP3/8/8/8/K6k w - d6 0 1");
		require_move_codec(en_passant);

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

		auto model = melano::Model(8, 1);
		auto output = model->forward(melano::encode_boards({board, board}));
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
		(output.policy.mean() + output.value.mean() + output.advantages.mean()).backward();
		require_finite_gradients(model);

		const auto checkpoint = std::filesystem::temp_directory_path() / "melanotest.pth";
		model->eval();
		auto reference = model->forward(melano::encode_boards({board}));
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
		std::filesystem::remove(checkpoint);

		std::cout << "melanotests passed" << std::endl;
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "melanotests failed: " << error.what() << std::endl;
		return 1;
	}
}
