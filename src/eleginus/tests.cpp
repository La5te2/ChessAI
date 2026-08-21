// CTest entrypoint for Value inference, storage and PVS invariants.

#include "eleginus/checkpoint.hpp"
#include "eleginus/dataset.hpp"
#include "eleginus/search.hpp"
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace {

void require(bool condition, const char *message) {
	if (!condition) {
		throw std::runtime_error(message);
	}
}

template <typename Left, typename Right>
float max_difference(const Left &left, const Right &right) {
	require(left.size() == right.size(), "array size mismatch");
	float difference = 0.0F;
	for (std::size_t index = 0; index < left.size(); ++index) {
		difference = std::max(difference, std::abs(left[index] - right[index]));
	}
	return difference;
}

float max_control_difference(const eleginus::ControlAccumulator &left,
							 const eleginus::ControlAccumulator &right) {
	float difference = 0.0F;
	for (int perspective = 0; perspective < eleginus::kPerspectiveCount; ++perspective) {
		for (int ownership = 0; ownership < 2; ++ownership) {
			for (int square = 0; square < 64; ++square) {
				for (int channel = 0; channel < eleginus::kControlWidth; ++channel)
					difference = std::max(difference, std::abs(
						left.field[static_cast<std::size_t>(perspective * 2 + ownership)]
							[static_cast<std::size_t>(square)][static_cast<std::size_t>(channel)] -
						right.field[static_cast<std::size_t>(perspective * 2 + ownership)]
							[static_cast<std::size_t>(square)][static_cast<std::size_t>(channel)]));
			}
		}
		for (int square = 0; square < 64; ++square) {
			for (int channel = 0; channel < eleginus::kControlLocalWidth; ++channel)
				difference = std::max(difference, std::abs(
					left.local[static_cast<std::size_t>(perspective)]
						[static_cast<std::size_t>(square)][static_cast<std::size_t>(channel)] -
					right.local[static_cast<std::size_t>(perspective)]
						[static_cast<std::size_t>(square)][static_cast<std::size_t>(channel)]));
		}
		for (int channel = 0; channel < eleginus::kControlLocalWidth; ++channel)
			difference = std::max(difference, std::abs(
				left.mean[static_cast<std::size_t>(perspective)][static_cast<std::size_t>(channel)] -
				right.mean[static_cast<std::size_t>(perspective)][static_cast<std::size_t>(channel)]));
		for (int channel = 0; channel < eleginus::kControlAttentionWidth; ++channel)
			difference = std::max(difference, std::abs(
				left.attention[static_cast<std::size_t>(perspective)]
					[static_cast<std::size_t>(channel)] -
				right.attention[static_cast<std::size_t>(perspective)]
					[static_cast<std::size_t>(channel)]));
		for (int head = 0; head < eleginus::kControlAttentionHeads; ++head) {
			for (int channel = 0; channel < eleginus::kControlAttentionHeadWidth; ++channel)
				difference = std::max(difference, std::abs(
					left.attention_numerator[static_cast<std::size_t>(perspective)]
						[static_cast<std::size_t>(head)][static_cast<std::size_t>(channel)] -
					right.attention_numerator[static_cast<std::size_t>(perspective)]
						[static_cast<std::size_t>(head)][static_cast<std::size_t>(channel)]));
			difference = std::max(difference, std::abs(
				left.attention_denominator[static_cast<std::size_t>(perspective)]
					[static_cast<std::size_t>(head)] -
				right.attention_denominator[static_cast<std::size_t>(perspective)]
					[static_cast<std::size_t>(head)]));
		}
		for (int channel = 0; channel < eleginus::kMaterialFeatureWidth; ++channel)
			difference = std::max(difference, std::abs(
				left.material[static_cast<std::size_t>(perspective)]
					[static_cast<std::size_t>(channel)] -
				right.material[static_cast<std::size_t>(perspective)]
					[static_cast<std::size_t>(channel)]));
	}
	return difference;
}

void require_accumulator_close(const eleginus::ValueAccumulator &left,
							   const eleginus::ValueAccumulator &right, float tolerance,
							   const char *context) {
	require(left.encoded.perspective == right.encoded.perspective &&
		left.encoded.white_to_move == right.encoded.white_to_move,
		"incremental Value encoded state differs from refresh");
	require(left.features.white_to_move == right.features.white_to_move &&
		left.features.piece_count == right.features.piece_count,
		"incremental Value feature metadata differs from refresh");
	for (int perspective = 0; perspective < eleginus::kPerspectiveCount; ++perspective) {
		require(max_difference(
			left.features.perspective[static_cast<std::size_t>(perspective)],
			right.features.perspective[static_cast<std::size_t>(perspective)]) < tolerance,
			"incremental Value feature accumulator differs from refresh");
	}
	require(left.control.count == right.control.count &&
		left.control.occupancy == right.control.occupancy &&
		left.control.piece_type == right.control.piece_type &&
		left.control.attacks == right.control.attacks &&
		left.control.bucket == right.control.bucket,
		"incremental Value discrete control state differs from refresh");
	(void)context;
	require(max_control_difference(left.control, right.control) < tolerance,
		"incremental Value floating control state differs from refresh");
}

} // namespace

int main() {
	try {
		torch::manual_seed(20260820);
		chess::Board board;
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
		auto model = eleginus::make_model(torch::Device(torch::kCPU), 7);
		model->eval();
		const float edge_std = model->value->control_source->weight.std().item<float>();
		const float context_std = model->value->control_occupancy->weight.std().item<float>();
		require(edge_std > 0.08F && edge_std < 0.12F,
			"Eleginus control-edge initialization scale changed");
		require(context_std > 0.03F && context_std < 0.08F,
			"Eleginus control-context initialization scale changed");
		require(torch::allclose(model->value->control_local->bias,
			torch::full_like(model->value->control_local->bias, 0.25F)),
			"Eleginus local control bias changed");
		require(model->value->material->weight.numel() == eleginus::kMaterialFeatureWidth,
			"Eleginus material residual is not shared across Value buckets");
		require(model->value->attention_key->weight.size(0) ==
			eleginus::kControlAttentionHeads * eleginus::kControlAttentionKeyWidth,
			"Eleginus control attention does not expose every key head");
		require(model->value->attention_query->weight.size(1) ==
			eleginus::kControlAttentionHeads * eleginus::kControlAttentionKeyWidth,
			"Eleginus material bucket does not provide every attention query");
		eleginus::CpuValue value = eleginus::snapshot_value(model->value);
		const auto moves = eleginus::legal_moves(board);
		const auto value_accumulator = value.refresh(board);
		std::vector<eleginus::EncodedFeatures> batch{encoded};
		auto network_batch =
			eleginus::encode_feature_batch(batch, torch::Device(torch::kCPU));
		auto value_batch =
			eleginus::encode_feature_batch({encoded, encoded}, torch::Device(torch::kCPU));
		auto torch_values = model->value->forward(value_batch);
		require(torch_values.dim() == 1 && torch_values.size(0) == 2,
			"Value batch output shape mismatch");
		const float torch_value = model->value->forward(network_batch).item<float>();
		require(std::abs(value.evaluate(value_accumulator) - torch_value) < 1.0e-5F,
				"custom Value inference differs from LibTorch");

		auto after = board;
		after.makeMove(chess::uci::uciToMove(after, "e2e4"));
		const auto value_updated = value.update(value_accumulator, board, after);
		const auto value_refreshed = value.refresh(after);
		for (int perspective = 0; perspective < eleginus::kPerspectiveCount; ++perspective) {
			require(max_difference(
						value_updated.features.perspective[static_cast<std::size_t>(perspective)],
						value_refreshed.features.perspective[static_cast<std::size_t>(perspective)]) <
					1.0e-5F,
					"incremental Value accumulator differs from refresh");
		}
		require(max_control_difference(value_updated.control,
			value_refreshed.control) < 1.0e-4F,
			"incremental Value control state differs from refresh");
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
			require(max_control_difference(updated.control, refreshed.control) < 1.0e-4F,
				"incremental Value control state differs on a special move");
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
			require(max_control_difference(sequence_state.control,
				refreshed.control) < 2.0e-4F,
				"cumulative Value control updates drifted from refresh");
			require(std::abs(value.evaluate(sequence_state) - value.evaluate(refreshed)) < 1.0e-5F,
				"cumulative Value updates changed the evaluated result");
			sequence = std::move(next);
		}

		std::mt19937 random(20260820U);
		chess::Board random_board;
		auto random_state = value.refresh(random_board);
		std::vector<eleginus::EncodedFeatures> parity_positions;
		std::vector<float> parity_values;
		for (int transition = 0; transition < 2000; ++transition) {
			auto random_moves = eleginus::legal_moves(random_board);
			if (random_moves.empty()) {
				random_board = chess::Board{};
				random_state = value.refresh(random_board);
				random_moves = eleginus::legal_moves(random_board);
			}
			const auto move = random_moves[static_cast<std::size_t>(random() % random_moves.size())];
			const auto parent_state = random_state;
			random_board.makeMove(move);
			const auto undo = value.apply(random_state, random_board);
			const auto refreshed = value.refresh(random_board);
			require_accumulator_close(random_state, refreshed, 4.0e-3F,
				"random incremental Value state differs from refresh");
			value.undo(random_state, undo);
			random_board.unmakeMove(move);
			require_accumulator_close(random_state, parent_state, 4.0e-3F,
				"random Value undo failed to restore the parent state");
			require(std::abs(value.evaluate(random_state) - value.evaluate(parent_state)) < 2.0e-5F,
				"random Value undo changed the parent evaluation");
			random_board.makeMove(move);
			(void)value.apply(random_state, random_board);
			if (transition % 32 == 0) {
				parity_positions.push_back(random_state.encoded);
				parity_values.push_back(value.evaluate(random_state));
			}
		}
		auto parity_batch = eleginus::encode_feature_batch(
			parity_positions, torch::Device(torch::kCPU));
		auto parity_tensor = model->value->forward(parity_batch).contiguous();
		require(parity_tensor.dim() == 1 &&
			parity_tensor.size(0) == static_cast<std::int64_t>(parity_values.size()),
			"random Value parity batch has the wrong shape");
		const auto *parity_data = parity_tensor.data_ptr<float>();
		for (std::size_t index = 0; index < parity_values.size(); ++index) {
			require(std::abs(parity_data[index] - parity_values[index]) < 2.0e-5F,
				"custom Value inference differs from LibTorch on a random position");
		}

		eleginus::SearchOptions search_options;
		search_options.depth = 1;
		search_options.quiescence_depth = 1;
		search_options.hash_mb = 1;
		const auto search = eleginus::Searcher(value, search_options).search(board);
		require(search.move.move() != chess::Move::NO_MOVE, "PVS returned no move");
		require(search.depth == 1, "PVS depth mismatch");
		require(search.nodes >= moves.size(), "PVS node count mismatch");
		require(search.root.size() == moves.size(), "PVS root move count mismatch");
		search_options.depth = 4;
		int completed_depths = 0;
		const auto cancelled = eleginus::Searcher(value, search_options).search(
			board,
			[&](const eleginus::SearchResult &) { ++completed_depths; },
			[&] { return completed_depths >= 1; });
		require(cancelled.depth == 1 && completed_depths == 1,
			"PVS cancellation did not preserve the last completed iteration");
		search_options.depth = 2;
		search_options.threads = 1;
		const auto serial_search = eleginus::Searcher(value, search_options).search(board);
		search_options.threads = 2;
		const auto parallel_search = eleginus::Searcher(value, search_options).search(board);
		require(parallel_search.move == serial_search.move &&
			parallel_search.score_cp == serial_search.score_cp,
			"parallel root PVS changed the completed search result");
		require(parallel_search.root.size() == serial_search.root.size(),
			"parallel root PVS omitted legal root moves");
		search_options.depth = 1;
		search_options.multipv = 3;
		const auto multipv_search = eleginus::Searcher(value, search_options).search(board);
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
		eleginus::CpuValue runtime_value(std::move(runtime_weights.value));
		require(std::abs(runtime_value.evaluate(runtime_value.refresh(board)) - torch_value) < 1.0e-5F,
				"native runtime checkpoint changed Value output");
		auto loaded = eleginus::load_checkpoint(checkpoint, torch::Device(torch::kCPU));
		auto loaded_value = loaded->value->forward(network_batch);
		require(torch::allclose(loaded_value, model->value->forward(network_batch)),
				"checkpoint changed Value output");
		std::filesystem::remove(checkpoint);
		std::filesystem::remove(executable);
		std::filesystem::remove(embedded);

		const auto dataset_path =
			std::filesystem::temp_directory_path() / "eleginus-value-test.h5";
		const auto after_encoded = eleginus::encode_features(after);
		const auto material_encoded = eleginus::encode_features(chess::Board(
			"rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"));
		{
			eleginus::WriterOptions writer_options;
			writer_options.output = dataset_path;
			writer_options.compression_level = 0;
			writer_options.source = "unit_test";
			eleginus::H5Writer writer(writer_options);
			writer.append({encoded, after_encoded, material_encoded}, {0.75F, 0.25F, 0.90F});
			writer.flush();
		}
		{
			eleginus::H5Dataset dataset(dataset_path);
			require(dataset.info().length == 3, "Eleginus HDF5 row count mismatch");
			require(dataset.info().source == "unit_test", "Eleginus HDF5 source mismatch");
			auto stored = dataset.read_contiguous(0, 3);
			require(std::abs(stored.values[0].item<float>() - 0.75F) < 1.0e-6F,
				"Eleginus HDF5 Value mismatch");
			auto original_batch = eleginus::encode_feature_batch(
				{encoded, after_encoded, material_encoded}, torch::Device(torch::kCPU));
			auto stored_batch =
				eleginus::encode_feature_batch(stored.features, torch::Device(torch::kCPU));
			require(torch::allclose(model->value->forward(original_batch),
				model->value->forward(stored_batch)),
				"Eleginus HDF5 side-to-move orientation changed Value input");
		}
		auto training_model = eleginus::make_model(torch::Device(torch::kCPU), 11);
		const auto training_checkpoint =
			std::filesystem::temp_directory_path() / "eleginus-training-test.pth";
		const auto control_before =
			training_model->value->control_source->weight.detach().clone();
		const auto output_before = training_model->value->output->weight.detach().clone();
		eleginus::TrainOptions training_options;
		training_options.data = dataset_path;
		training_options.output = training_checkpoint;
		training_options.epochs = 1;
		training_options.batch_size = 3;
		training_options.max_steps = 1;
		training_options.log_every = 1;
		const auto training_stats = eleginus::train_from_h5(
			training_model, training_options, torch::Device(torch::kCPU));
		require(training_stats.steps == 1 && training_stats.samples == 3,
			"Eleginus training batch mismatch");
		require(training_model->value->material->weight.index({0, 4}).item<float>() > 0.0F,
			"Eleginus material calibration did not learn a positive queen coefficient");
		require(std::filesystem::exists(training_checkpoint),
			"Eleginus epoch checkpoint was not written");
		const auto epoch_model = eleginus::load_checkpoint(
			training_checkpoint, torch::Device(torch::kCPU));
		require(torch::allclose(epoch_model->value->output->weight,
			training_model->value->output->weight),
			"Eleginus epoch checkpoint does not contain the trained parameters");
		const auto control_change =
			(training_model->value->control_source->weight.detach() - control_before).abs();
		require(control_change.max().item<float>() > 0.0F,
			"Eleginus control parameters received no Value gradient");
		require((training_model->value->output->weight.detach() - output_before)
				.abs().max().item<float>() > 0.0F,
			"Eleginus Value output received no gradient");
		std::filesystem::remove(training_checkpoint);
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
				   "17:00: {0s}\n"
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
		}
		std::filesystem::remove(pgn_path);
		std::filesystem::remove(preprocessed_path);

		chess::Board mate("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1");
		require(eleginus::game_is_over(mate), "checkmate not detected");
		const auto mate_search = eleginus::Searcher(value, search_options).search(mate);
		require(mate_search.score_cp < -29000, "checkmate score mismatch");
		std::cout << "eleginustests passed\n";
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "eleginustests failed: " << error.what() << '\n';
		return 1;
	}
}
