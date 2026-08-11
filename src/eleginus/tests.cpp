// CTest entrypoint for codecs, independent Policy/Value inference, storage and PVS invariants.

#include "eleginus/checkpoint.hpp"
#include "eleginus/dataset.hpp"
#include "eleginus/search.hpp"
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>
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

float max_relation_difference(const eleginus::RelationAccumulator &left,
							  const eleginus::RelationAccumulator &right) {
	float difference = 0.0F;
	for (int perspective = 0; perspective < eleginus::kPerspectiveCount; ++perspective) {
		for (int channel = 0; channel < eleginus::kValueAttentionWidth; ++channel) {
			difference = std::max(difference, std::abs(
				left.perspective[static_cast<std::size_t>(perspective)]
					[static_cast<std::size_t>(channel)] -
				right.perspective[static_cast<std::size_t>(perspective)]
					[static_cast<std::size_t>(channel)]));
		}
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
				require(feature >= 0 && feature <= eleginus::kEncodedPaddingFeature,
						"feature index out of range");
			}
			for (const int feature : eleginus::canonicalize_features(perspective)) {
				require(feature >= 0 && feature <= eleginus::kPaddingFeature,
					"canonical feature index out of range");
			}
		}
		const auto left = eleginus::encode_features(
			chess::Board("4k3/8/8/8/8/2N5/8/2K5 w - - 0 1"));
		const auto right = eleginus::encode_features(
			chess::Board("3k4/8/8/8/8/5N2/8/5K2 w - - 0 1"));
		for (int perspective = 0; perspective < eleginus::kPerspectiveCount; ++perspective) {
			require(eleginus::canonicalize_features(left.perspective[perspective]) ==
					eleginus::canonicalize_features(right.perspective[perspective]),
				"horizontal feature canonicalization mismatch");
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
		auto [value_batch_features, value_batch_side] =
			eleginus::encode_feature_batch({encoded, encoded}, torch::Device(torch::kCPU));
		auto torch_values = model->value->forward(value_batch_features, value_batch_side);
		require(torch_values.dim() == 1 && torch_values.size(0) == 2,
			"Value batch output shape mismatch");
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
						value_updated.features.perspective[static_cast<std::size_t>(perspective)],
						value_refreshed.features.perspective[static_cast<std::size_t>(perspective)]) <
					1.0e-5F,
					"incremental Value accumulator differs from refresh");
		}
		require(max_relation_difference(value_updated.relations,
			value_refreshed.relations) < 1.0e-4F,
			"incremental Value relation state differs from refresh");
		require(std::abs(value.evaluate(value_updated) - value.evaluate(value_refreshed)) < 1.0e-5F,
			"incremental Value output differs from refresh");

		auto require_value_update = [&](const chess::Board &before, const char *uci) {
			auto next = before;
			next.makeMove(chess::uci::uciToMove(next, uci));
			const auto updated = value.update(value.refresh(before), before, next);
			const auto refreshed = value.refresh(next);
			for (int perspective = 0; perspective < eleginus::kPerspectiveCount; ++perspective) {
				require(max_difference(
					updated.features.perspective[static_cast<std::size_t>(perspective)],
					refreshed.features.perspective[static_cast<std::size_t>(perspective)]) <
					1.0e-5F,
					"incremental Value feature state differs on a special move");
			}
			require(max_relation_difference(updated.relations, refreshed.relations) < 1.0e-4F,
				"incremental Value relation state differs on a special move");
			require(std::abs(value.evaluate(updated) - value.evaluate(refreshed)) < 1.0e-5F,
				"incremental Value output differs on a special move");
		};
		require_value_update(
			chess::Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"), "e1g1");
		require_value_update(
			chess::Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"), "e5d6");
		require_value_update(
			chess::Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1"), "a7a8q");

		chess::Board sequence;
		auto sequence_state = value.refresh(sequence);
		for (const char *uci : {"e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6",
				 "b5a4", "g8f6", "e1g1", "f8e7", "f1e1", "b7b5", "a4b3"}) {
			auto next = sequence;
			next.makeMove(chess::uci::uciToMove(next, uci));
			sequence_state = value.update(sequence_state, sequence, next);
			const auto refreshed = value.refresh(next);
			require(max_relation_difference(sequence_state.relations,
				refreshed.relations) < 2.0e-4F,
				"cumulative Value relation updates drifted from refresh");
			require(std::abs(value.evaluate(sequence_state) - value.evaluate(refreshed)) < 1.0e-5F,
				"cumulative Value updates changed the evaluated result");
			sequence = std::move(next);
		}

		eleginus::SearchOptions search_options;
		search_options.depth = 1;
		search_options.quiescence_depth = 1;
		search_options.hash_mb = 1;
		const auto search = eleginus::Searcher(policy, value, search_options).search(board);
		require(search.move.move() != chess::Move::NO_MOVE, "PVS returned no move");
		require(search.depth == 1, "PVS depth mismatch");
		require(search.nodes >= moves.size(), "PVS node count mismatch");
		require(search.root.size() == moves.size(), "PVS root move count mismatch");
		search_options.depth = 4;
		int completed_depths = 0;
		const auto cancelled = eleginus::Searcher(policy, value, search_options).search(
			board,
			[&](const eleginus::SearchResult &) { ++completed_depths; },
			[&] { return completed_depths >= 1; });
		require(cancelled.depth == 1 && completed_depths == 1,
			"PVS cancellation did not preserve the last completed iteration");
		search_options.depth = 2;
		search_options.threads = 1;
		const auto serial_search = eleginus::Searcher(policy, value, search_options).search(board);
		search_options.threads = 2;
		const auto parallel_search = eleginus::Searcher(policy, value, search_options).search(board);
		require(parallel_search.move == serial_search.move &&
			parallel_search.score_cp == serial_search.score_cp,
			"parallel root PVS changed the completed search result");
		require(parallel_search.root.size() == serial_search.root.size(),
			"parallel root PVS omitted legal root moves");
		search_options.depth = 1;
		search_options.multipv = 3;
		const auto multipv_search = eleginus::Searcher(policy, value, search_options).search(board);
		require(multipv_search.root.size() == moves.size(),
			"MultiPV root search omitted legal root moves");
		require(std::all_of(multipv_search.root.begin(), multipv_search.root.end(),
			[](const eleginus::RootMove &move) { return move.exact_score; }),
			"MultiPV root search returned bounded rather than exact scores");

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
		auto training_model = eleginus::make_model(torch::Device(torch::kCPU), 11);
		const auto attention_before =
			training_model->value->attention->weight.detach().clone();
		const auto output_before = training_model->value->output->weight.detach().clone();
		eleginus::TrainOptions training_options;
		training_options.data = dataset_path;
		training_options.epochs = 1;
		training_options.batch_size = 2;
		training_options.max_steps = 1;
		training_options.log_every = 0;
		const auto training_stats = eleginus::train_from_h5(
			training_model, training_options, torch::Device(torch::kCPU));
		require(training_stats.steps == 1 && training_stats.samples == 2,
			"Eleginus training batch mismatch");
		const auto attention_change =
			(training_model->value->attention->weight.detach() - attention_before).abs();
		for (int part = 0; part < 3; ++part) {
			const auto begin = part * eleginus::kValueAttentionWidth;
			const auto end = begin + eleginus::kValueAttentionWidth;
			require(attention_change.index({torch::indexing::Slice(),
				torch::indexing::Slice(begin, end)}).max().item<float>() > 0.0F,
				"Eleginus relation parameters received no Value gradient");
		}
		require((training_model->value->output->weight.detach() - output_before)
				.abs().max().item<float>() > 0.0F,
			"Eleginus Value output received no gradient");
		std::filesystem::remove(dataset_path);

		const auto pgn_path =
			std::filesystem::temp_directory_path() / "eleginus-preprocess-test.pgn";
		const auto preprocessed_path =
			std::filesystem::temp_directory_path() / "eleginus-preprocess-test.h5";
		{
			std::ofstream pgn(pgn_path);
			pgn << "[Event \"root comment\"]\n"
				   "[Result \"1-0\"]\n\n"
				   "{opening note} 1. e4\n"
				   "{0s}\n"
				   "e5\n"
				   "{+0.20/12 0s}\n"
				   "2. Nf3\n"
				   "{+0.15/12 0s}\n"
				   "Nc6\n"
				   "{+0.10/12 0s} 1-0\n\n"
				   "[Event \"invalid move\"]\n"
				   "[Result \"1-0\"]\n\n"
				   "1. e4 e5 2. Qa9 1-0\n";
		}
		{
			eleginus::PreprocessOptions preprocess_options;
			preprocess_options.input = pgn_path;
			preprocess_options.output = preprocessed_path;
			preprocess_options.has_comments = 1;
			preprocess_options.compression_level = 0;
			preprocess_options.log_every = 0;
			eleginus::preprocess_pgn(preprocess_options);
			eleginus::H5Dataset dataset(preprocessed_path);
			require(dataset.info().length == 4,
				"Eleginus preprocessing rejected multiline comments or retained invalid moves");
			auto stored = dataset.read_contiguous(0, 4);
			require(std::abs(stored.values[0].item<float>() - 0.5F) < 1.0e-6F &&
					std::abs(stored.values[1].item<float>() - 0.5F) < 1.0e-6F,
				"Eleginus preprocessing changed neutral opening targets");
			require(stored.values[2].item<float>() > 0.5F &&
					stored.values[3].item<float>() < 0.5F,
				"Eleginus preprocessing lost multiline evaluation comments");
			chess::Board replay;
			for (std::int64_t row = 0; row < 4; ++row) {
				const auto expected = chess::uci::parseSan(
					replay, std::array<std::string_view, 4>{"e4", "e5", "Nf3", "Nc6"}
						[static_cast<std::size_t>(row)]);
				require(stored.moves[row].item<std::int64_t>() ==
						eleginus::move_to_index(expected, replay.sideToMove()),
					"Eleginus preprocessing changed a move after a root comment");
				replay.makeMove(expected);
			}
		}
		std::filesystem::remove(pgn_path);
		std::filesystem::remove(preprocessed_path);

		chess::Board mate("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1");
		require(eleginus::game_is_over(mate), "checkmate not detected");
		const auto mate_search = eleginus::Searcher(policy, value, search_options).search(mate);
		require(mate_search.score_cp < -29000, "checkmate score mismatch");
		std::cout << "eleginustests passed\n";
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "eleginustests failed: " << error.what() << '\n';
		return 1;
	}
}
