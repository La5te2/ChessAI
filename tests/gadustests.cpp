#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "gadus/checkpoint.hpp"
#include "gadus/game.hpp"
#include "gadus/model.hpp"

namespace {

void require(bool condition, const char *message) {
	if (!condition) {
		throw std::runtime_error(message);
	}
}

} // namespace

int main() {
	try {
		chess::Board board;
		require(board.hash() == 0x463b96181691fc9cULL, "Polyglot start-position hash mismatch");
		const auto packed = gadus::encode_state(board);
		require(packed[5 * 8] == 0x08, "white king state plane mismatch");
		require(packed[11 * 8 + 7] == 0x08, "black king state plane mismatch");
		for (int rank = 0; rank < 8; ++rank) {
			require(packed[12 * 8 + rank] == 0xFF, "side-to-move state plane mismatch");
		}

		for (const auto &move : gadus::legal_moves(board)) {
			const int action = gadus::move_to_index(move);
			require(gadus::index_to_move(action, board) == move, "move codec round trip failed");
		}

		auto model = gadus::Model(8, 1);
		auto [policy, value] = model->forward(gadus::encode_boards({board, board}));
		require(policy.sizes() == torch::IntArrayRef({2, gadus::kActionSize}),
				"policy shape mismatch");
		require(value.sizes() == torch::IntArrayRef({2, 1}), "value shape mismatch");
		(policy.mean() + value.mean()).backward();

		const auto checkpoint = std::filesystem::temp_directory_path() / "gadustest.pth";
		gadus::save_checkpoint_atomic(checkpoint, model, {8, 1, 2, 3, "test"});
		gadus::CheckpointInfo info;
		auto loaded = gadus::load_checkpoint(checkpoint, torch::Device(torch::kCPU), &info);
		require(info.channels == 8 && info.blocks == 1 && info.epoch == 2 && info.global_step == 3,
				"checkpoint metadata mismatch");
		auto loaded_output = loaded->forward(gadus::encode_boards({board}));
		require(loaded_output.first.size(1) == gadus::kActionSize, "loaded model output mismatch");
		std::filesystem::remove(checkpoint);

		std::cout << "gadustests passed" << std::endl;
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "gadustests failed: " << error.what() << std::endl;
		return 1;
	}
}
