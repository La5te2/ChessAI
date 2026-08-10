// CTest entrypoint for codecs, independent Policy/Value inference, storage and BFM invariants.

#include "eleginus/checkpoint.hpp"
#include "eleginus/dataset.hpp"
#include "eleginus/search.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {

void require(bool condition, const char *message) {
	if (!condition) {
		throw std::runtime_error(message);
	}
}

void require_codec(const chess::Board &board) {
	for (const auto &move : eleginus::legal_moves(board)) {
		const int action = eleginus::move_to_index(move, board.sideToMove());
		require(action >= 0 && action < eleginus::kActionSize, "action index out of range");
		require(eleginus::index_to_move(action, board) == move, "move codec round trip failed");
	}
}

float max_difference(const std::vector<float> &left, const std::vector<float> &right) {
	require(left.size() == right.size(), "vector size mismatch");
	float difference = 0.0F;
	for (std::size_t index = 0; index < left.size(); ++index) {
		difference = std::max(difference, std::abs(left[index] - right[index]));
	}
	return difference;
}

} // namespace

int main() {
	try {
		chess::Board board;
		require_codec(board);
		require_codec(chess::Board("8/P7/8/8/8/8/8/k6K w - - 0 1"));
		require_codec(chess::Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"));
		const auto encoded = eleginus::encode_features(board);
		for (const auto &perspective : encoded.perspective) {
			for (const int feature : perspective) {
				require(feature >= 0 && feature <= eleginus::kPaddingFeature,
						"feature index out of range");
			}
		}

		auto model = eleginus::make_model(torch::Device(torch::kCPU), 7);
		model->eval();
		eleginus::CpuPolicy policy = eleginus::snapshot_policy(model->policy);
		eleginus::CpuValue value = eleginus::snapshot_value(model->value);
		const auto moves = eleginus::legal_moves(board);
		const auto policy_accumulator = policy.refresh(board);
		const auto value_accumulator = value.refresh(board);
		std::vector<eleginus::EncodedFeatures> batch{encoded};
		auto [features, side] =
			eleginus::encode_feature_batch(batch, torch::Device(torch::kCPU));
		const float torch_value = model->value->forward(features, side).item<float>();
		require(std::abs(value.evaluate(value_accumulator) - torch_value) < 1.0e-5F,
				"custom Value inference differs from LibTorch");
		std::vector<std::int64_t> action_indices;
		action_indices.reserve(moves.size());
		for (const auto &move : moves)
			action_indices.push_back(eleginus::move_to_index(move, board.sideToMove()));
		auto indices = torch::tensor(action_indices, torch::TensorOptions().dtype(torch::kInt64));
		auto torch_policy = model->policy->forward(features, side)
			.index_select(1, indices).softmax(1).squeeze(0).contiguous();
		const auto native_policy = policy.evaluate(policy_accumulator, moves);
		const std::vector<float> torch_policy_values(
			torch_policy.data_ptr<float>(), torch_policy.data_ptr<float>() + torch_policy.numel());
		require(max_difference(native_policy, torch_policy_values) < 1.0e-5F,
			"custom Policy inference differs from LibTorch");

		auto after = board;
		after.makeMove(chess::uci::uciToMove(after, "e2e4"));
		const auto black_e5 = chess::uci::uciToMove(after, "e7e5");
		require(eleginus::move_to_index(chess::uci::uciToMove(board, "e2e4"),
					board.sideToMove()) ==
				eleginus::move_to_index(black_e5, after.sideToMove()),
			"side-relative move encoding differs between colors");
		const auto policy_updated = policy.update(policy_accumulator, board, after);
		const auto policy_refreshed = policy.refresh(after);
		const auto value_updated = value.update(value_accumulator, board, after);
		const auto value_refreshed = value.refresh(after);
		for (int perspective = 0; perspective < eleginus::kPerspectiveCount; ++perspective) {
			require(max_difference(
						policy_updated.perspective[static_cast<std::size_t>(perspective)],
						policy_refreshed.perspective[static_cast<std::size_t>(perspective)]) < 1.0e-5F,
					"incremental Policy accumulator differs from refresh");
			require(max_difference(
						value_updated.perspective[static_cast<std::size_t>(perspective)],
						value_refreshed.perspective[static_cast<std::size_t>(perspective)]) < 1.0e-5F,
					"incremental Value accumulator differs from refresh");
		}

		eleginus::SearchOptions search_options;
		search_options.expansions = 2;
		const auto search = eleginus::Searcher(policy, value, search_options).search(board);
		require(search.move.move() != chess::Move::NO_MOVE, "BFM returned no move");
		require(search.expanded_nodes == 2, "BFM expansion budget mismatch");
		require(search.root.size() == moves.size(), "BFM root move count mismatch");

		const auto checkpoint = std::filesystem::temp_directory_path() / "eleginus-test.pth";
		const auto executable = std::filesystem::temp_directory_path() / "eleginus-template.bin";
		const auto embedded = std::filesystem::temp_directory_path() / "eleginus-embedded.bin";
		eleginus::save_checkpoint_atomic(checkpoint, model);
		{
			std::ofstream output(executable, std::ios::binary | std::ios::trunc);
			output << "Eleginus executable template";
		}
		eleginus::embed_checkpoint_atomic(checkpoint, executable, embedded);
		auto runtime_weights = eleginus::load_embedded_runtime_model(embedded);
		eleginus::CpuPolicy runtime_policy(std::move(runtime_weights.policy));
		eleginus::CpuValue runtime_value(std::move(runtime_weights.value));
		require(std::abs(runtime_value.evaluate(runtime_value.refresh(board)) - torch_value) < 1.0e-5F,
				"native runtime checkpoint changed Value output");
		require(max_difference(runtime_policy.evaluate(runtime_policy.refresh(board), moves),
			native_policy) < 1.0e-5F, "native runtime checkpoint changed Policy output");
		auto loaded = eleginus::load_checkpoint(checkpoint, torch::Device(torch::kCPU));
		auto loaded_value = loaded->value->forward(features, side);
		require(torch::allclose(loaded_value, model->value->forward(features, side)),
				"checkpoint changed Value output");
		require(torch::allclose(loaded->policy->forward(features, side),
			model->policy->forward(features, side)), "checkpoint changed Policy output");
		std::filesystem::remove(checkpoint);
		std::filesystem::remove(executable);
		std::filesystem::remove(embedded);

		const auto dataset_path =
			std::filesystem::temp_directory_path() / "eleginus-value-test.h5";
		const auto after_encoded = eleginus::encode_features(after);
		const auto black_move = chess::uci::uciToMove(after, "e7e5");
		{
			eleginus::WriterOptions writer_options;
			writer_options.output = dataset_path;
			writer_options.compression_level = 0;
			writer_options.source = "unit_test";
			eleginus::H5Writer writer(writer_options);
			writer.append({encoded, after_encoded},
				{static_cast<std::uint16_t>(eleginus::move_to_index(
					 moves.front(), board.sideToMove())),
				 static_cast<std::uint16_t>(eleginus::move_to_index(
					 black_move, after.sideToMove()))},
				{0.75F, 0.25F});
			writer.flush();
		}
		{
			eleginus::H5Dataset dataset(dataset_path);
			require(dataset.info().length == 2, "Eleginus HDF5 row count mismatch");
			require(dataset.info().source == "unit_test", "Eleginus HDF5 source mismatch");
			auto stored = dataset.read_contiguous(0, 2);
			require(stored.moves[1].item<std::int64_t>() ==
				eleginus::move_to_index(black_move, after.sideToMove()),
				"Eleginus HDF5 move mismatch");
			require(std::abs(stored.values[0].item<float>() - 0.75F) < 1.0e-6F,
				"Eleginus HDF5 Value mismatch");
			auto [original_features, original_side] = eleginus::encode_feature_batch(
				{encoded, after_encoded}, torch::Device(torch::kCPU));
			auto [stored_features, stored_side] =
				eleginus::encode_feature_batch(stored.features, torch::Device(torch::kCPU));
			require(torch::allclose(model->value->forward(original_features, original_side),
				model->value->forward(stored_features, stored_side)),
				"Eleginus HDF5 side-to-move orientation changed Value input");
			require(torch::allclose(model->policy->forward(original_features, original_side),
				model->policy->forward(stored_features, stored_side)),
				"Eleginus HDF5 side-to-move orientation changed Policy input");
		}
		std::filesystem::remove(dataset_path);

		chess::Board mate("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1");
		require(eleginus::game_is_over(mate), "checkmate not detected");
		require(eleginus::terminal_score_side_to_move(mate) == 0.0F,
				"checkmate score mismatch");
		std::cout << "eleginustests passed\n";
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "eleginustests failed: " << error.what() << '\n';
		return 1;
	}
}
