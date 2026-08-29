// CTest entrypoint for Melano invariants. It covers precision parsing, state and move codecs,
// geometry relations, terminal rules, JSONL preprocessing, HDF5 schema validation,
// Policy and Value shapes and gradients, checkpoint round trips, direct Policy search and batched
// PUCT.

#include "melano/checkpoint.hpp"
#include "melano/attention.hpp"
#include "melano/cuda.hpp"
#include "melano/dataset.hpp"
#include "melano/game.hpp"
#include "melano/model.hpp"
#include "melano/search.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_set>
#include <torch/cuda.h>

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
		const int action = melano::move_to_index(move, board.sideToMove());
		require(action >= 0 && action < melano::kActionSize, "move codec produced an out-of-range action");
		require(actions.insert(action).second, "legal moves share an action index");
		require(melano::index_to_move(action, board) == move, "move codec round trip failed");
	}
}

// Ensure one backward pass produced finite gradients for every participating parameter.
void require_finite_gradients(const melano::Model &model) {
	for (const auto &parameter : model->parameters()) {
		require(parameter.grad().defined(), "model parameter has no gradient");
		require(torch::isfinite(parameter.grad()).all().item<bool>(), "model gradient contains a non-finite value");
	}
}

// Compare matrix-organized geometry attention and its gradients with the defining formula.
void require_attention_equivalence(const torch::Device &device) {
	const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
	auto query_storage = torch::randn({2, 64, 2, 8}, options).set_requires_grad(true);
	auto key_storage = torch::randn({2, 64, 2, 8}, options).set_requires_grad(true);
	auto value_storage = torch::randn({2, 64, 2, 8}, options).set_requires_grad(true);
	auto query = query_storage.transpose(1, 2);
	auto key = key_storage.transpose(1, 2);
	auto value = value_storage.transpose(1, 2);
	auto coefficients = torch::randn({2, 2, 64}, options).set_requires_grad(true);
	auto templates = torch::randn({64, 64, 64}, options).set_requires_grad(true);
	auto reference_query = query.detach().clone().set_requires_grad(true);
	auto reference_key = key.detach().clone().set_requires_grad(true);
	auto reference_value = value.detach().clone().set_requires_grad(true);
	auto reference_coefficients = coefficients.detach().clone().set_requires_grad(true);
	auto reference_templates = templates.detach().clone().set_requires_grad(true);

	const auto organized = melano::geometry_attention(query, key, value, coefficients, templates);
	const auto reference_scores = torch::matmul(reference_query, reference_key.transpose(-2, -1)) / std::sqrt(8.0);
	const auto reference_bias = torch::matmul(reference_coefficients, reference_templates.view({64, 64 * 64})).view_as(reference_scores);
	const auto reference = torch::matmul(torch::softmax(reference_scores + reference_bias, -1), reference_value);
	require(torch::allclose(organized, reference, 2.0e-4, 2.0e-5), "matrix-organized geometry attention differs from its defining formula");

	const auto output_gradient = torch::randn_like(organized);
	organized.backward(output_gradient);
	reference.backward(output_gradient);
	for (const auto &gradients : {std::pair{query_storage.grad().transpose(1, 2), reference_query.grad()},
	                              std::pair{key_storage.grad().transpose(1, 2), reference_key.grad()},
	                              std::pair{value_storage.grad().transpose(1, 2), reference_value.grad()},
	                              std::pair{coefficients.grad(), reference_coefficients.grad()}, std::pair{templates.grad(), reference_templates.grad()}}) {
		require(torch::allclose(gradients.first, gradients.second, 1.0e-3, 1.0e-4),
		        "matrix-organized geometry-attention gradient differs from its defining formula");
	}
}

void require_gab_centering(const torch::Device &device) {
	const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
	auto query = torch::randn({2, 2, 64, 8}, options).set_requires_grad(true);
	auto key = torch::randn({2, 2, 64, 8}, options).set_requires_grad(true);
	auto value = torch::randn({2, 2, 64, 8}, options).set_requires_grad(true);
	auto coefficients = torch::randn({2, 2, 64}, options).set_requires_grad(true);
	auto templates = torch::randn({64, 64, 64}, options).set_requires_grad(true);
	auto shifted_query = query.detach().clone().set_requires_grad(true);
	auto shifted_key = key.detach().clone().set_requires_grad(true);
	auto shifted_value = value.detach().clone().set_requires_grad(true);
	auto shifted_coefficients = coefficients.detach().clone().set_requires_grad(true);
	auto shifted_templates = (templates.detach() + torch::randn({64, 64, 1}, options)).set_requires_grad(true);

	const auto output = melano::geometry_attention(query, key, value, coefficients, templates);
	const auto shifted_output = melano::geometry_attention(shifted_query, shifted_key, shifted_value, shifted_coefficients, shifted_templates);
	require(torch::allclose(output, shifted_output, 2.0e-4, 2.0e-5), "GAB row constants changed the attention output");
	const auto output_gradient = torch::randn_like(output);
	output.backward(output_gradient);
	shifted_output.backward(output_gradient);
	for (const auto &gradients : {std::pair{query.grad(), shifted_query.grad()}, std::pair{key.grad(), shifted_key.grad()},
	                              std::pair{value.grad(), shifted_value.grad()}, std::pair{coefficients.grad(), shifted_coefficients.grad()},
	                              std::pair{templates.grad(), shifted_templates.grad()}}) {
		require(torch::allclose(gradients.first, gradients.second, 1.0e-3, 1.0e-4), "GAB row constants changed an attention gradient");
	}

	auto centered = shifted_templates.detach().clone();
	melano::center_gab_rows(centered);
	require(centered.mean(-1).abs().max().item<float>() < 1.0e-6F, "GAB row centering did not produce zero means");
	const auto centered_output = melano::geometry_attention(
	    query.detach(), key.detach(), value.detach(), coefficients.detach(), centered);
	require(torch::allclose(output.detach(), centered_output, 2.0e-4, 2.0e-5), "GAB row centering changed the attention output");
}

} // namespace

// Exercise the complete minimal Melano inference/training/checkpoint/search surface.
int main() {
	try {
		require_attention_equivalence(torch::kCPU);
		require_gab_centering(torch::kCPU);
		if (torch::cuda::is_available()) {
			require_attention_equivalence(torch::kCUDA);
			require_gab_centering(torch::kCUDA);
		}
		require(melano::parse_compute_precision("fp32") == melano::ComputePrecision::Fp32, "fp32 precision parsing failed");
		require(melano::parse_compute_precision("bf16") == melano::ComputePrecision::Bf16, "bf16 precision parsing failed");
		require(std::string(melano::compute_precision_name(melano::ComputePrecision::Bf16)) == "bf16", "bf16 precision name mismatch");

		chess::Board board;
		require(board.hash() == 0x463b96181691fc9cULL, "Polyglot start-position hash mismatch");
		const auto packed = melano::encode_state(board);
		require(packed[0] == 4 && packed[4] == 6 && packed[8] == 1, "white piece token mismatch");
		require(packed[48] == 7 && packed[60] == 12 && packed[63] == 10, "black piece token mismatch");
		require(packed[64] == 15, "castling token mismatch");
		require(packed[65] == 0, "en-passant token mismatch");

		chess::Board canonical_white("4k2K/8/8/3p4/4P3/8/8/8 w - - 0 1");
		chess::Board canonical_black("8/8/8/4p3/3P4/8/8/4K2k b - - 0 1");
		require(melano::encode_state(canonical_white) == melano::encode_state(canonical_black), "side-to-move canonicalization changed an equivalent position");
		const auto white_step = chess::uci::uciToMove(canonical_white, "e4e5");
		const auto black_step = chess::uci::uciToMove(canonical_black, "e5e4");
		require(melano::move_to_index(white_step, canonical_white.sideToMove()) == melano::move_to_index(black_step, canonical_black.sideToMove()),
		    "side-to-move canonicalization changed an equivalent action");

		int compact_actions = 0;
		for (int expanded = 0; expanded < melano::kExpandedActionSize; ++expanded) {
			const int compact = melano::compact_action_index(expanded);
			if (compact < 0) {
				continue;
			}
			require(compact < melano::kActionSize, "compact action index is out of range");
			require(melano::expanded_action_index(compact) == expanded, "compact action map is not invertible");
			++compact_actions;
		}
		require(compact_actions == melano::kActionSize, "compact action map has the wrong cardinality");

		require_move_codec(board);
		chess::Board promotion("8/P7/8/8/8/8/8/k6K w - - 0 1");
		require_move_codec(promotion);
		chess::Board castling("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
		require_move_codec(castling);
		const auto king_side_castle = chess::uci::uciToMove(castling, "e1g1");
		const auto queen_side_castle = chess::uci::uciToMove(castling, "e1c1");
		require(melano::expanded_action_index(melano::move_to_index(king_side_castle, castling.sideToMove())) == 4 * 64 + 6, "king-side castling policy index mismatch");
		require(melano::expanded_action_index(melano::move_to_index(queen_side_castle, castling.sideToMove())) == 4 * 64 + 2, "queen-side castling policy index mismatch");
		chess::Board black_castling("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1");
		require_move_codec(black_castling);
		const auto black_king_side = chess::uci::uciToMove(black_castling, "e8g8");
		const auto black_queen_side = chess::uci::uciToMove(black_castling, "e8c8");
		require(melano::move_to_index(black_king_side, black_castling.sideToMove()) == melano::move_to_index(king_side_castle, castling.sideToMove()),
		    "black king-side castling did not share the canonical policy index");
		require(melano::move_to_index(black_queen_side, black_castling.sideToMove()) == melano::move_to_index(queen_side_castle, castling.sideToMove()),
		    "black queen-side castling did not share the canonical policy index");
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
		require(melano::terminal_value_side_to_move(checkmate) == -1.0F, "checkmate side-to-move value mismatch");
		require(melano::game_result(checkmate) == "1-0", "checkmate result mismatch");
		require(melano::game_termination(checkmate) == "checkmate", "checkmate termination mismatch");

		chess::Board stalemate("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
		require(melano::game_is_over(stalemate), "stalemate was not detected");
		require(melano::terminal_value_side_to_move(stalemate) == 0.0F, "stalemate side-to-move value mismatch");
		require(melano::game_termination(stalemate) == "stalemate", "stalemate termination mismatch");

		chess::Board insufficient("7k/8/8/8/8/8/8/K7 w - - 0 1");
		require(melano::game_termination(insufficient) == "insufficient material", "insufficient-material termination mismatch");
		chess::Board bishop_only("7k/8/8/8/8/8/8/KB6 w - - 0 1");
		require(melano::game_termination(bishop_only) == "insufficient material", "king-and-bishop termination mismatch");
		chess::Board knight_only("7k/8/8/8/8/8/8/KN6 w - - 0 1");
		require(melano::game_termination(knight_only) == "insufficient material", "king-and-knight termination mismatch");
		chess::Board bishop_vs_knight("6nk/8/8/8/8/8/8/KB6 w - - 0 1");
		require(!melano::game_is_over(bishop_vs_knight), "king-and-bishop versus king-and-knight was incorrectly adjudicated");

		chess::Board fifty_move("7k/8/8/8/8/8/6R1/K7 b - - 100 1");
		require(melano::game_termination(fifty_move) == "fifty move rule", "fifty-move termination mismatch");

		chess::Board repetition;
		for (const char *uci : {"g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"}) {
			repetition.makeMove(chess::uci::uciToMove(repetition, uci));
		}
		require(repetition.isRepetition(2), "threefold repetition count mismatch");
		require(!repetition.isRepetition(3), "threefold repetition was counted as fourfold");
		require(melano::game_termination(repetition) == "threefold repetition", "threefold-repetition termination mismatch");

		const auto evaluations = std::filesystem::temp_directory_path() / "melanotest.jsonl";
		const auto h5 = std::filesystem::temp_directory_path() / "melanotest.h5";
		{
			std::ofstream output(evaluations);
			output << R"({"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -","evals":[{"pvs":[{"cp":10,"line":"d2d4 d7d5"}],"knodes":100,"depth":20},{"pvs":[{"cp":30,"line":"e2e4 e7e5"}],"knodes":200,"depth":24}]})"
			       << '\n'
			       << R"({"fen":"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -","evals":[{"pvs":[{"cp":20,"line":"e7e5 g1f3"}],"knodes":150,"depth":22}]})"
			       << '\n';
		}
		melano::PreprocessOptions preprocess;
		preprocess.input = evaluations;
		preprocess.output = h5;
		preprocess.chunk_size = 1;
		preprocess.compression_level = 0;
		preprocess.log_every = 0;
		melano::preprocess_jsonl(preprocess);
		{
			melano::SupervisedH5 supervised(h5);
			const auto batch = supervised.read_contiguous(0, 2);
			const auto e4 = chess::uci::uciToMove(board, "e2e4");
			require(supervised.info().length == 2, "JSONL preprocessing produced the wrong row count");
			require(batch.states.scalar_type() == torch::kUInt8, "HDF5 states were expanded before device transfer");
			require(batch.moves.scalar_type() == torch::kInt16, "HDF5 moves were expanded before device transfer");
			require(torch::equal(batch.states.index({0}), melano::encode_boards({board}).index({0})), "JSONL state differs from live canonical encoding");
			require(batch.moves.index({0}).item<std::int64_t>() == melano::move_to_index(e4, board.sideToMove()), "JSONL preprocessing did not select the deepest PV");
			require(batch.values.index({0}).item<float>() > 0.09F, "JSONL value has the wrong scale or perspective");
		}
		const auto training_checkpoint = std::filesystem::temp_directory_path() / "melanotraintest.pth";
		melano::TrainOptions training;
		training.data = h5;
		training.output = training_checkpoint;
		training.channels = 8;
		training.blocks = 1;
		training.epochs = 1;
		training.batch_size = 2;
		training.max_steps = 1;
		training.save_every = 0;
		training.log_every = 0;
		training.device = "cpu";
		melano::train_supervised(training);
		require(std::filesystem::exists(training_checkpoint), "compact HDF5 tensors did not complete one training step");
		auto trained = melano::load_checkpoint(training_checkpoint, torch::Device(torch::kCPU));
		require(trained->geometry_templates.mean(-1).abs().max().item<float>() < 1.0e-6F, "training did not preserve row-centered GAB templates");
		std::filesystem::remove(training_checkpoint);
		std::filesystem::remove(evaluations);
		std::filesystem::remove(h5);

		auto model = melano::Model(8, 1);
		auto states = melano::encode_boards({board, board});
		require(states.scalar_type() == torch::kUInt8, "live states were expanded before device transfer");
		auto embedded = model->state_embedding->forward(states);
		require(embedded.sizes() == torch::IntArrayRef({2, melano::kBoardSquares, 8}), "state embedding did not produce 64 square tokens");
		require(model->geometry_templates.sizes() == torch::IntArrayRef({melano::kGabTemplateCount, melano::kBoardSquares, melano::kBoardSquares}),
		    "GAB template bank has the wrong shape");
		require(torch::isfinite(model->geometry_templates).all().item<bool>(), "GAB template bank contains a non-finite value");
		require(model->geometry_templates.mean(-1).abs().max().item<float>() < 1.0e-6F, "GAB template bank did not begin row-centered");
		auto first_block = model->trunk->ptr<melano::GeometryAttentionBlockImpl>(0);
		auto initial_bias = first_block->geometry_bias(embedded, model->geometry_templates);
		require(initial_bias.sizes() == torch::IntArrayRef({2, first_block->heads, melano::kBoardSquares, melano::kBoardSquares}), "GAB output has the wrong shape");
		require(torch::isfinite(initial_bias).all().item<bool>(), "GAB output contains a non-finite value");
		require(torch::allclose(initial_bias.index({0}), initial_bias.index({1}), 1e-5, 1e-6), "identical positions produced different GAB outputs");
		auto after_e4 = board;
		after_e4.makeMove(chess::uci::uciToMove(after_e4, "e2e4"));
		auto changed_embedding = model->state_embedding->forward(melano::encode_boards({after_e4}));
		auto changed_bias = first_block->geometry_bias(changed_embedding, model->geometry_templates);
		require(!torch::allclose(initial_bias.index({0}), changed_bias.index({0})), "GAB output did not depend on the board state");
		require(torch::equal(model->value_head->query, torch::zeros_like(model->value_head->query)), "Value pooling query did not begin at zero");
		auto reversed = embedded.index_select(1, torch::arange(melano::kBoardSquares - 1, -1, -1, torch::TensorOptions().dtype(torch::kInt64)));
		require(torch::allclose(model->value_head->forward(embedded), model->value_head->forward(reversed)), "zero-query Value pooling was not initially uniform");
		auto [policy, value] = model->forward(states);
		auto [training_policy, value_logit] = model->forward_training(states);
		require(policy.sizes() == torch::IntArrayRef({2, melano::kActionSize}), "policy shape mismatch");
		require(value.sizes() == torch::IntArrayRef({2, 1}), "value shape mismatch");
		require(torch::allclose(training_policy, policy), "training path changed Policy logits");
		require(torch::allclose(torch::tanh(value_logit), value), "training Value logit does not map to inference Value");
		require(torch::isfinite(policy).all().item<bool>(), "policy contains a non-finite value");
		require(torch::isfinite(value).all().item<bool>(), "value contains a non-finite value");
		require(value.abs().max().item<float>() <= 1.000001F, "value range mismatch");

		const std::vector<chess::Board> legal_test_boards{board, promotion};
		std::vector<std::vector<int>> legal_test_actions;
		std::size_t legal_test_width = 0;
		for (const auto &test_board : legal_test_boards) {
			std::vector<int> actions;
			for (const auto &move : melano::legal_moves(test_board)) {
				actions.push_back(melano::move_to_index(move, test_board.sideToMove()));
			}
			legal_test_width = std::max(legal_test_width, actions.size());
			legal_test_actions.push_back(std::move(actions));
		}
		auto legal_indices =
		    torch::zeros({static_cast<std::int64_t>(legal_test_boards.size()), static_cast<std::int64_t>(legal_test_width)}, torch::TensorOptions().dtype(torch::kInt64));
		auto legal_rows = legal_indices.accessor<std::int64_t, 2>();
		for (std::size_t row = 0; row < legal_test_actions.size(); ++row) {
			for (std::size_t column = 0; column < legal_test_actions[row].size(); ++column) {
				legal_rows[static_cast<std::int64_t>(row)][static_cast<std::int64_t>(column)] = legal_test_actions[row][column];
			}
		}
		auto legal_test_states = melano::encode_boards(legal_test_boards);
		auto [full_logits, full_values] = model->forward(legal_test_states);
		auto [legal_logits, legal_values] = model->forward_legal(legal_test_states, legal_indices);
		for (std::size_t row = 0; row < legal_test_actions.size(); ++row) {
			for (std::size_t column = 0; column < legal_test_actions[row].size(); ++column) {
				const auto batch = static_cast<std::int64_t>(row);
				const auto slot = static_cast<std::int64_t>(column);
				const auto action = legal_test_actions[row][column];
				require(std::abs(legal_logits.index({batch, slot}).item<float>() - full_logits.index({batch, action}).item<float>()) < 1.0e-5F,
				    "legal-only Policy logit differs from the complete Policy head");
			}
		}
		require(torch::allclose(legal_values, full_values), "legal-only Policy path changed the Value output");
		{
			const auto valid_width = static_cast<std::int64_t>(legal_test_actions.front().size());
			auto padded_indices = torch::full({1, valid_width + 2}, -1, torch::TensorOptions().dtype(torch::kInt16));
			auto padded_row = padded_indices.accessor<std::int16_t, 2>();
			for (std::int64_t column = 0; column < valid_width; ++column) {
				padded_row[0][column] = static_cast<std::int16_t>(legal_test_actions.front()[static_cast<std::size_t>(column)]);
			}
			torch::InferenceMode guard;
			melano::InferenceGraphs inference_graphs;
			auto [padded_policy, padded_value] =
			    inference_graphs.run(model, melano::encode_boards({board}), padded_indices, torch::Device(torch::kCPU), melano::ComputePrecision::Fp32);
			auto [valid_logits, valid_value] = model->forward_legal(
			    melano::encode_boards({board}), padded_indices.slice(1, 0, valid_width).to(torch::kInt64));
			const auto expected_policy = torch::softmax(valid_logits.to(torch::kFloat32), 1);
			require(torch::allclose(padded_policy.slice(1, 0, valid_width), expected_policy), "padded inference changed legal Policy probabilities");
			require(torch::equal(padded_policy.slice(1, valid_width), torch::zeros({1, 2}, padded_policy.options())),
			    "padded inference assigned probability to sentinel actions");
			require(torch::allclose(padded_value, valid_value), "padded inference changed the Value output");
		}
		(policy.mean() + value.mean()).backward();
		require_finite_gradients(model);

		const auto checkpoint = std::filesystem::temp_directory_path() / "melanotest.pth";
		model->eval();
		auto [reference_policy, reference_value] = model->forward(melano::encode_boards({board}));
		melano::save_checkpoint_atomic(checkpoint, model, {8, 1});
		melano::ArchitectureInfo info;
		auto loaded = melano::load_checkpoint(checkpoint, torch::Device(torch::kCPU), &info);
		require(info.channels == 8 && info.blocks == 1, "checkpoint architecture mismatch");
		loaded->eval();
		auto [loaded_policy, loaded_value] = loaded->forward(melano::encode_boards({board}));
		require(loaded_policy.size(1) == melano::kActionSize, "loaded model output mismatch");
		require(torch::allclose(reference_policy, loaded_policy), "checkpoint changed policy output");
		require(torch::allclose(reference_value, loaded_value), "checkpoint changed value output");
		auto fusion_states = melano::encode_boards(legal_test_boards);
		auto wide_indices = torch::arange(melano::kBoardSquares, torch::kInt64).unsqueeze(0).repeat({fusion_states.size(0), 1});
		auto [unfused_policy, unfused_value] = loaded->forward(fusion_states);
		auto [unfused_legal, unfused_legal_value] = loaded->forward_legal(fusion_states, legal_indices);
		auto [unfused_wide, unfused_wide_value] = loaded->forward_legal(fusion_states, wide_indices);
		loaded->fuse_for_inference();
		auto [fused_policy, fused_value] = loaded->forward(fusion_states);
		auto [fused_legal, fused_legal_value] = loaded->forward_legal(fusion_states, legal_indices);
		auto [fused_wide, fused_wide_value] = loaded->forward_legal(fusion_states, wide_indices);
		require(torch::allclose(unfused_policy, fused_policy, 1e-5, 1e-6), "inference fusion changed complete Policy logits");
		require(torch::allclose(unfused_legal, fused_legal, 1e-5, 1e-6), "inference fusion changed legal Policy logits");
		require(torch::allclose(unfused_wide, fused_wide, 1e-5, 1e-6), "inference fusion changed wide legal Policy logits");
		require(torch::allclose(unfused_value, fused_value), "inference fusion changed complete Value output");
		require(torch::allclose(unfused_legal_value, fused_legal_value), "inference fusion changed legal Value output");
		require(torch::allclose(unfused_wide_value, fused_wide_value), "inference fusion changed wide legal Value output");

		// Zero simulations rank legal actions with Policy before optional decision components.
		melano::SearchOptions policy_options;
		policy_options.mcts_sims = 0;
		policy_options.root_topn = 4;
		melano::Searcher policy_searcher(loaded, torch::Device(torch::kCPU), policy_options);
		const auto policy_result = policy_searcher.search(board);
		require(policy_result.root.size() == 4, "direct Policy root size mismatch");
		require(policy_result.sims_completed == 0, "direct Policy inference ran MCTS");
		require(melano::index_to_move(melano::move_to_index(policy_result.move, board.sideToMove()), board) == policy_result.move,
		    "direct Policy inference selected an illegal move");

		// Duplicate roots share one exact evaluation inside a batched search call.
		melano::Searcher batched_searcher(loaded, torch::Device(torch::kCPU), policy_options);
		const auto batched_results = batched_searcher.search_many({board, board});
		require(batched_results[0].nn_evaluations + batched_results[1].nn_evaluations == 1, "duplicate roots performed more than one network evaluation");
		require(batched_results[0].evaluation_reuses + batched_results[1].evaluation_reuses == 1, "duplicate root reuse was not reported");

		// Positive cache capacity reuses network output while each call builds fresh search
		// statistics.
		auto cached_options = policy_options;
		cached_options.evaluation_cache_mb = 1;
		melano::Searcher cached_searcher(loaded, torch::Device(torch::kCPU), cached_options);
		const auto first_cached = cached_searcher.search(board);
		const auto second_cached = cached_searcher.search(board);
		require(first_cached.nn_evaluations == 1 && first_cached.evaluation_reuses == 0, "first cache search did not evaluate its root");
		require(second_cached.nn_evaluations == 0 && second_cached.evaluation_reuses == 1, "second cache search did not reuse its root");
		require(second_cached.sims_completed == 0 && second_cached.expanded_nodes == 1, "evaluation cache retained MCTS statistics");
		require(torch::allclose(torch::tensor(first_cached.policy), torch::tensor(second_cached.policy), 1e-6, 1e-7), "evaluation cache changed Policy");
		require(std::abs(first_cached.value - second_cached.value) < 1e-7F, "evaluation cache changed Value");
		cached_searcher.clear_evaluation_cache();
		const auto after_clear = cached_searcher.search(board);
		require(after_clear.nn_evaluations == 1 && after_clear.evaluation_reuses == 0, "cache clear did not force root reevaluation");

		// Four simulations cover PUCT selection, neural expansion, and value backup.
		melano::SearchOptions mcts_options = policy_options;
		mcts_options.mcts_sims = 4;
		mcts_options.mcts_min_sims = 4;
		mcts_options.mcts_batch_size = 2;
		melano::Searcher mcts_searcher(loaded, torch::Device(torch::kCPU), mcts_options);
		const auto mcts_result = mcts_searcher.search(board);
		require(mcts_result.sims_completed == 4, "MCTS simulation budget mismatch");
		require(mcts_result.expanded_nodes > 0, "MCTS did not expand a node");
		require(mcts_result.nn_batches > 1, "MCTS did not evaluate exact-state leaves");

		// A searched child reuses its exact evaluation as the root of the next search call.
		auto trajectory_options = mcts_options;
		trajectory_options.evaluation_cache_mb = 1;
		melano::Searcher trajectory_searcher(loaded, torch::Device(torch::kCPU), trajectory_options);
		const auto trajectory_root = trajectory_searcher.search(board);
		auto trajectory_child = board;
		trajectory_child.makeMove(trajectory_root.move);
		const auto trajectory_next = trajectory_searcher.search(trajectory_child);
		require(trajectory_next.evaluation_reuses > 0, "trajectory cache did not reuse the searched child root");
		require(trajectory_next.sims_completed == trajectory_options.mcts_sims, "trajectory cache changed the simulation budget");
		std::filesystem::remove(checkpoint);

		std::cout << "melanotests passed" << std::endl;
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "melanotests failed: " << error.what() << std::endl;
		return 1;
	}
}
