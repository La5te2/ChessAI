// CTest entrypoint for Gadus invariants. It covers state and move codecs, terminal rules,
// Policy and Value shapes and gradients, checkpoint round trips, direct Policy search, batched MCTS,
// legal-row projection, inference fusion and cross-search evaluation reuse with fresh tree statistics.

#include <filesystem>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <numeric>
#include <unordered_set>
#include "gadus/checkpoint.hpp"
#include "gadus/dataset.hpp"
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

		gadus::ValueWeightController aligned_weight(0.5);
		const double aligned = aligned_weight.update(4.0, 1.0, 1.0);
		require(aligned > 0.5 && aligned < 1.0,
				"aligned gradients did not smoothly increase the Value weight");

		gadus::ValueWeightController conflicting_weight(0.5);
		const double conflicting = conflicting_weight.update(4.0, 1.0, -1.5);
		require(conflicting > 0.5 && conflicting < 1.5,
				"conflicting gradients did not move smoothly toward the feasible interval");

		bool rejected_zero_weight = false;
		try {
			gadus::ValueWeightController invalid_weight(0.0);
		} catch (const std::invalid_argument &) {
			rejected_zero_weight = true;
		}
		require(rejected_zero_weight, "zero Value weight was accepted");
		gadus::ValueWeightController minimum_weight(0.2);
		gadus::ValueWeightController maximum_weight(2.0);
		require(minimum_weight.value() == 0.2 && maximum_weight.value() == 2.0,
				"Value-weight boundary was not preserved");
		bool rejected_large_weight = false;
		try {
			gadus::ValueWeightController invalid_weight(2.1);
		} catch (const std::invalid_argument &) {
			rejected_large_weight = true;
		}
		require(rejected_large_weight, "out-of-range Value weight was accepted");

		// Detached CCRL comments and clock fields are metadata; invalid SAN rejects its game.
		const auto preprocess_pgn =
			std::filesystem::temp_directory_path() / "gadus-preprocess-test.pgn";
		const auto preprocess_h5 =
			std::filesystem::temp_directory_path() / "gadus-preprocess-test.h5";
		{
			std::ofstream pgn(preprocess_pgn);
			pgn << "[Event \"detached comments\"]\n"
				   "[Result \"1-0\"]\n\n"
				   "1. d4 17:00: {0s} Nf6 {0s} 2. c4 23,\n"
				   "{Both last book move 0s}\n"
				   "g6 {+0.60/16 193s} 3. Nc3\n"
				   "{(d5) +0.28/14 77s}\n"
				   "d5 {+0.27/15 177s} 1-0\n\n"
				   "[Event \"invalid move\"]\n"
				   "[Result \"1-0\"]\n\n"
				   "1. e4 e5 2. Qa9 1-0\n";
		}
		gadus::PreprocessOptions preprocess_options;
		preprocess_options.input = preprocess_pgn;
		preprocess_options.output = preprocess_h5;
		preprocess_options.has_comments = 1;
		preprocess_options.compression_level = 0;
		preprocess_options.log_every = 0;
		gadus::preprocess_pgn(preprocess_options);
		{
			gadus::SupervisedH5 supervised(preprocess_h5);
			require(supervised.info().length == 6,
					"Gadus preprocessing rejected detached comments or retained invalid SAN");
			const auto batch = supervised.read_contiguous(0, 6);
			require(batch.states.scalar_type() == torch::kUInt8 && batch.states.dim() == 3 &&
					batch.states.size(0) == 6 &&
					batch.states.size(1) == gadus::kStatePlanes && batch.states.size(2) == 8,
					"Gadus HDF5 reader did not retain packed state rows");
			const auto decoded = gadus::decode_states_device(batch.states, torch::kCPU);
			require(decoded.sizes() == torch::IntArrayRef({6, gadus::kInputPlanes, 8, 8}),
					"Gadus packed tensor decoder produced the wrong shape");
			require(std::abs(batch.values.index({0}).item<float>()) < 1e-6F &&
						std::abs(batch.values.index({1}).item<float>()) < 1e-6F &&
						std::abs(batch.values.index({2}).item<float>()) < 1e-6F,
					"Gadus preprocessing changed neutral opening targets");
			require(batch.values.index({4}).item<float>() > 0.0F &&
						batch.values.index({5}).item<float>() < 0.0F,
					"Gadus preprocessing lost detached numerical comments");
		}
		const auto trained_checkpoint =
			std::filesystem::temp_directory_path() / "gadus-adaptive-weight-test.pth";
		gadus::TrainOptions train_options;
		train_options.data = preprocess_h5;
		train_options.output = trained_checkpoint;
		train_options.channels = 4;
		train_options.blocks = 1;
		train_options.epochs = 1;
		train_options.batch_size = 6;
		train_options.max_steps = 1;
		train_options.save_every = 0;
		train_options.log_every = 0;
		train_options.device = "cpu";
		gadus::train_supervised(train_options);
		require(std::filesystem::exists(trained_checkpoint),
				"adaptive Value-weight training did not produce a checkpoint");
		std::filesystem::remove(preprocess_pgn);
		std::filesystem::remove(preprocess_h5);
		std::filesystem::remove(trained_checkpoint);

		// Lichess evaluation JSONL selects the deepest first PV and preserves score perspective.
		const auto evaluation_jsonl =
			std::filesystem::temp_directory_path() / "gadus-lichess-eval-test.jsonl";
		const auto evaluation_h5 =
			std::filesystem::temp_directory_path() / "gadus-lichess-eval-test.h5";
		{
			std::ofstream jsonl(evaluation_jsonl);
			jsonl << R"({"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -","evals":[{"depth":12,"knodes":20,"pvs":[{"cp":10,"line":"e2e4 e7e5"}]},{"depth":20,"knodes":10,"pvs":[{"cp":30,"line":"d2d4 d7d5"}]}]})"
				  << '\n';
			jsonl << R"({"fen":"r1k5/8/8/8/8/8/8/4K3 b q -","evals":[{"depth":18,"pvs":[{"cp":0,"line":"c8c8"}]}]})"
				  << '\n';
			jsonl << R"({"fen":"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -","evals":[{"depth":18,"pvs":[{"cp":120,"line":"e7e5 g1f3"}]}]})"
				  << '\n';
			jsonl << R"({"fen":"rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq -","evals":[{"depth":30,"pvs":[{"mate":-1,"line":"d8h4"}]}]})"
				  << '\n';
			jsonl << R"({"fen":"r3k2r/8/8/8/8/8/8/R3K2R w KQkq -","evals":[{"depth":16,"pvs":[{"cp":0,"line":"e1h1"}]}]})"
				  << '\n';
			jsonl << R"({"fen":"not a fen","evals":[]})" << '\n';
		}
		gadus::PreprocessOptions evaluation_options;
		evaluation_options.source = "lichess-eval";
		evaluation_options.input = evaluation_jsonl;
		evaluation_options.output = evaluation_h5;
		evaluation_options.compression_level = 0;
		evaluation_options.log_every = 0;
		gadus::preprocess_lichess_evaluations(evaluation_options);
		{
			gadus::SupervisedH5 supervised(evaluation_h5);
			require(supervised.info().length == 4,
					"Lichess evaluation preprocessing retained a malformed record");
			const auto batch = supervised.read_contiguous(0, 4);
			const auto moves = batch.moves.to(torch::kCPU).contiguous();
			const auto values = batch.values.to(torch::kCPU).contiguous();
			require(moves.index({0}).item<std::int64_t>() ==
						gadus::move_to_index(chess::uci::uciToMove(board, "d2d4")),
					"Lichess evaluation preprocessing did not select the deepest PV");
			require(std::abs(values.index({0}).item<float>() - std::tanh(0.1F)) < 1e-6F,
					"Lichess centipawn conversion changed the Gadus Value scale");
			require(values.index({1}).item<float>() < 0.0F,
					"Lichess centipawn conversion lost side-to-move perspective");
			require(std::abs(values.index({2}).item<float>() - 1.0F) < 1e-6F,
					"Lichess mate conversion lost side-to-move perspective");
			chess::Board castling_record("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
			require(moves.index({3}).item<std::int64_t>() ==
						gadus::move_to_index(chess::uci::uciToMove(castling_record, "e1g1")),
					"Lichess UCI_Chess960 castling notation was not decoded");
		}
		std::filesystem::remove(evaluation_jsonl);
		std::filesystem::remove(evaluation_h5);
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
		chess::Board canonical_white("4k2K/8/8/3p4/4P3/8/8/8 w - - 0 1");
		chess::Board canonical_black("8/8/8/4p3/3P4/8/8/4K2k b - - 0 1");
		require(torch::equal(gadus::encode_boards({canonical_white}),
						 gadus::encode_boards({canonical_black})),
				"side-to-move canonicalization changed an equivalent position");
		const auto white_step = chess::uci::uciToMove(canonical_white, "e4e5");
		const auto black_step = chess::uci::uciToMove(canonical_black, "e5e4");
		require(gadus::canonical_action_index(gadus::move_to_index(white_step),
										chess::Color::WHITE) ==
					gadus::canonical_action_index(gadus::move_to_index(black_step),
										chess::Color::BLACK),
				"side-to-move canonicalization changed an equivalent action");

		require_move_codec(board);
		chess::Board promotion("8/P7/8/8/8/8/8/k6K w - - 0 1");
		require_move_codec(promotion);
		chess::Board castling("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
		require_move_codec(castling);
		const auto king_side_castle = chess::uci::uciToMove(castling, "e1g1");
		const auto queen_side_castle = chess::uci::uciToMove(castling, "e1c1");
		require(gadus::move_to_index(king_side_castle) == 4 * 73 + 4 * 7 + 1,
				"king-side castling policy index mismatch");
		require(gadus::move_to_index(queen_side_castle) == 4 * 73 + 3 * 7 + 1,
				"queen-side castling policy index mismatch");
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
		chess::Board bishop_only("7k/8/8/8/8/8/8/KB6 w - - 0 1");
		require(gadus::game_termination(bishop_only) == "insufficient material",
				"king-and-bishop termination mismatch");
		chess::Board knight_only("7k/8/8/8/8/8/8/KN6 w - - 0 1");
		require(gadus::game_termination(knight_only) == "insufficient material",
				"king-and-knight termination mismatch");
		chess::Board bishop_vs_knight("6nk/8/8/8/8/8/8/KB6 w - - 0 1");
		require(!gadus::game_is_over(bishop_vs_knight),
				"king-and-bishop versus king-and-knight was incorrectly adjudicated");

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

		// The Policy vector must follow source square * 73 + motion pattern.
		auto relation_block = gadus::ResidualBlock(8, 1);
		const auto initial_relations = relation_block->relation_matrices();
		require(initial_relations.sizes() == torch::IntArrayRef({8, 64, 64}),
				"relation initialization produced the wrong shape");
		require(torch::allclose(initial_relations.index({0}), torch::eye(64), 1e-6, 1e-7),
				"identity geometry did not initialize the first relation group");
		relation_block->eval();
		auto relation_input = torch::randn({2, 8, 8, 8});
		auto relation_reference = relation_block->forward(relation_input);
		auto zero_corrections = torch::zeros({2, 8, 8});
		auto relation_zero =
			relation_block->forward_with_relation(relation_input, zero_corrections);
		require(torch::allclose(relation_reference, relation_zero, 1e-6, 1e-7),
				"zero dynamic relation correction changed a residual block");

		auto layout_model = gadus::Model(8, 1);
		{
			torch::NoGradGuard guard;
			for (const auto &parameter : layout_model->named_parameters()) {
				if (parameter.key() == "policy_motion_vectors" ||
					parameter.key() == "policy_action_corrections") {
					parameter.value().zero_();
				}
				if (parameter.key() == "policy_action_bias") {
					auto expected = torch::arange(
						gadus::kActionSize, parameter.value().options());
					parameter.value().copy_(expected);
				}
			}
		}
		auto layout_logits =
			layout_model->forward(gadus::encode_boards({board})).first.squeeze(0);
		require(torch::equal(
				layout_logits,
				torch::arange(gadus::kActionSize, layout_logits.options())),
			"action planes were flattened in the wrong index order");

		auto model = gadus::Model(8, 1);
		auto [policy, value] = model->forward(gadus::encode_boards({board, board}));
		require(policy.sizes() == torch::IntArrayRef({2, gadus::kActionSize}),
				"policy shape mismatch");
		require(value.sizes() == torch::IntArrayRef({2, 1}), "value shape mismatch");
		require(torch::isfinite(policy).all().item<bool>(), "policy contains a non-finite value");
		require(torch::isfinite(value).all().item<bool>(), "value contains a non-finite value");
		require(value.abs().max().item<float>() <= 1.000001F, "value range mismatch");
		require(value.scalar_type() == torch::kFloat32, "Value output is not FP32");
		(policy.mean() + value.mean()).backward();
		require_finite_gradients(model);

		const auto checkpoint = std::filesystem::temp_directory_path() / "gadustest.pth";
		model->eval();
		auto reference = model->forward(gadus::encode_boards({board}));
		const auto start_moves = gadus::legal_moves(board);
		auto legal_indices = torch::empty(
			{1, static_cast<std::int64_t>(start_moves.size())},
			torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
		auto legal_row = legal_indices.accessor<std::int64_t, 2>();
		for (std::size_t index = 0; index < start_moves.size(); ++index) {
			legal_row[0][static_cast<std::int64_t>(index)] =
				gadus::move_to_index(start_moves[index]);
		}
		auto legal_output =
			model->forward_legal(gadus::encode_boards({board}), legal_indices);
		require(torch::allclose(legal_output.first, reference.first.gather(1, legal_indices),
								1e-5, 1e-6),
				"legal-action forward changed a requested policy logit");
		require(torch::allclose(legal_output.second, reference.second),
				"legal-action forward changed Value");
		chess::Board black_position;
		black_position.makeMove(chess::uci::uciToMove(black_position, "e2e4"));
		const auto black_moves = gadus::legal_moves(black_position);
		auto black_indices = torch::empty(
			{1, static_cast<std::int64_t>(black_moves.size())},
			torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
		auto black_row = black_indices.accessor<std::int64_t, 2>();
		for (std::size_t index = 0; index < black_moves.size(); ++index) {
			black_row[0][static_cast<std::int64_t>(index)] = gadus::canonical_action_index(
				gadus::move_to_index(black_moves[index]), chess::Color::BLACK);
		}
		auto black_reference = model->forward(gadus::encode_boards({black_position}));
		auto black_legal = model->forward_legal(
			gadus::encode_boards({black_position}), black_indices);
		require(torch::allclose(black_legal.first,
							 black_reference.first.gather(1, black_indices), 1e-5, 1e-6),
				"black legal-action projection used noncanonical indices");
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
		loaded->fuse_for_inference();
		auto fused_output = loaded->forward(gadus::encode_boards({board}));
		auto fused_legal_output =
			loaded->forward_legal(gadus::encode_boards({board}), legal_indices);
		require(torch::allclose(loaded_output.first, fused_output.first, 1e-4, 1e-5),
				"Conv-BN fusion changed policy output");
		require(torch::allclose(loaded_output.second, fused_output.second, 1e-5, 1e-6),
				"Conv-BN fusion changed value output");
		require(torch::allclose(fused_legal_output.first,
								 fused_output.first.gather(1, legal_indices), 1e-4, 1e-5),
				"fused legal-action forward changed a requested policy logit");
		require(torch::allclose(fused_legal_output.second, fused_output.second, 1e-5, 1e-6),
				"fused legal-action forward changed Value");

		// Zero-simulation search ranks legal policy actions without constructing an MCTS tree.
		gadus::SearchOptions direct_options;
		direct_options.mcts_sims = 0;
		direct_options.root_topn = 4;
		gadus::Searcher direct_searcher(loaded, torch::Device(torch::kCPU), direct_options);
		const auto direct_result = direct_searcher.search(board);
		const auto compact_result = direct_searcher.evaluate_policy_many({board});
		require(compact_result.size() == 1, "compact Gadus evaluation row count mismatch");
		require(compact_result[0].legal_indices.size() == compact_result[0].legal_policy.size(),
				"compact Gadus evaluation width mismatch");
		for (std::size_t column = 0; column < compact_result[0].legal_indices.size(); ++column) {
			require(std::abs(compact_result[0].legal_policy[column] -
							 direct_result.policy[compact_result[0].legal_indices[column]]) < 1e-6F,
					"compact Gadus evaluation changed a legal probability");
		}
		require(std::abs(compact_result[0].value - direct_result.value) < 1e-6F,
				"compact Gadus evaluation changed Value");
		require(direct_result.root.size() == 4, "direct search root size mismatch");
		require(direct_result.sims_completed == 0, "direct search unexpectedly ran MCTS");
		require(gadus::index_to_move(gadus::move_to_index(direct_result.move), board) ==
					direct_result.move,
				"direct search selected an illegal move");
		auto full_probabilities =
			torch::softmax(reference.first, 1).squeeze(0).to(torch::kCPU).contiguous();
		std::vector<float> full_policy(gadus::kActionSize);
		std::copy_n(full_probabilities.data_ptr<float>(), gadus::kActionSize,
					full_policy.begin());
		const auto expected_policy = gadus::normalize_legal_policy(full_policy, board);
		for (int action = 0; action < gadus::kActionSize; ++action) {
			require(std::abs(direct_result.policy[action] - expected_policy[action]) < 1e-5F,
					"compact legal-policy transfer changed direct search probabilities");
		}

		// A persistent evaluation cache reuses only Policy/Value; every call still builds a new tree.
		auto cached_options = direct_options;
		cached_options.evaluation_cache_mb = 1;
		gadus::Searcher cached_searcher(loaded, torch::Device(torch::kCPU), cached_options);
		const auto first_cached = cached_searcher.search(board);
		const auto second_cached = cached_searcher.search(board);
		require(first_cached.nn_evaluations == 1 && first_cached.evaluation_reuses == 0,
				"first persistent-cache search did not evaluate its root");
		require(second_cached.nn_evaluations == 0 && second_cached.evaluation_reuses == 1,
				"second persistent-cache search did not reuse its root evaluation");
		require(second_cached.sims_completed == 0 && second_cached.expanded_nodes == 1,
				"persistent evaluation cache retained search-tree statistics");
		require(torch::allclose(torch::tensor(first_cached.policy),
							torch::tensor(second_cached.policy), 1e-6, 1e-7),
				"persistent evaluation cache changed Policy");
		require(std::abs(first_cached.value - second_cached.value) < 1e-7F,
				"persistent evaluation cache changed Value");
		cached_searcher.clear_evaluation_cache();
		const auto after_clear = cached_searcher.search(board);
		require(after_clear.nn_evaluations == 1 && after_clear.evaluation_reuses == 0,
				"clearing the persistent evaluation cache did not force reevaluation");

		// Four simulations exercise root allocation, batched expansion, and value backup.
		gadus::SearchOptions mcts_options = direct_options;
		mcts_options.mcts_sims = 4;
		mcts_options.mcts_batch_size = 2;
		gadus::Searcher mcts_searcher(loaded, torch::Device(torch::kCPU), mcts_options);
		const auto mcts_result = mcts_searcher.search(board);
		require(mcts_result.sims_completed == 4, "MCTS simulation budget mismatch");
		require(mcts_result.expanded_nodes > 0, "MCTS did not expand a node");

		// A budget equal to the legal width gives every root action one completed visit.
		const int active_root_width = static_cast<int>(start_moves.size());
		auto coverage_options = mcts_options;
		coverage_options.mcts_sims = active_root_width;
		coverage_options.root_topn = static_cast<int>(start_moves.size());
		gadus::Searcher coverage_searcher(loaded, torch::Device(torch::kCPU), coverage_options);
		const auto coverage_result = coverage_searcher.search(board);
		require(coverage_result.root.size() == start_moves.size(),
				"root coverage omitted a legal action from diagnostics");
		for (int index = 0; index < active_root_width; ++index)
			require(coverage_result.root[static_cast<std::size_t>(index)].visits == 1,
					"legal root action did not receive its fair visit");

		auto competition_options = coverage_options;
		competition_options.mcts_sims += 1;
		gadus::Searcher competition_searcher(
			loaded, torch::Device(torch::kCPU), competition_options);
		const auto competition_result = competition_searcher.search(board);
		const auto represented = std::count_if(
			competition_result.policy.begin(), competition_result.policy.end(),
			[](float score) { return score > 0.0F; });
		require(competition_result.sims_completed == competition_options.mcts_sims,
				"root competition changed the simulation budget");
		const int competition_visits = std::accumulate(
			competition_result.root.begin(), competition_result.root.end(), 0,
			[](int total, const gadus::RootMove &move) { return total + move.visits; });
		require(competition_visits == competition_result.sims_completed,
				"root competition visits do not match completed simulations");
		require(represented == active_root_width,
				"root competition omitted a legal action");
		for (const auto &root_move : competition_result.root)
			require(std::abs(root_move.decision_score - root_move.probability) < 1e-6F,
					"root decision score differs from the PUCT visit distribution");

		// A larger budget raises the fair floor above the one-visit bootstrap.
		const int legal_width = static_cast<int>(start_moves.size());
		int fair_budget = legal_width;
		while (static_cast<int>(std::floor(10.0 * std::log1p(
				   static_cast<double>(fair_budget) /
				   (10.0 * static_cast<double>(legal_width))))) < 2) {
			++fair_budget;
		}
		auto fair_options = coverage_options;
		fair_options.mcts_sims = fair_budget;
		gadus::Searcher fair_searcher(loaded, torch::Device(torch::kCPU), fair_options);
		const auto fair_result = fair_searcher.search(board);
		for (int index = 0; index < active_root_width; ++index)
			require(fair_result.root[static_cast<std::size_t>(index)].visits >= 2,
					"legal root action did not satisfy its fair visit floor");

		auto narrow_options = competition_options;
		narrow_options.root_topn = 1;
		gadus::Searcher narrow_searcher(loaded, torch::Device(torch::kCPU), narrow_options);
		const auto narrow_result = narrow_searcher.search(board);
		require(narrow_result.root.size() == 1, "narrow MultiPV width was ignored");
		require(narrow_result.move == competition_result.move,
				"MultiPV width changed the selected move");
		require(torch::allclose(torch::tensor(narrow_result.policy),
							torch::tensor(competition_result.policy), 1e-6, 1e-7),
				"MultiPV width changed root allocation");

		// An actually played searched move reuses its evaluation even when it differs from bestmove.
		auto trajectory_options = mcts_options;
		trajectory_options.evaluation_cache_mb = 1;
		gadus::Searcher trajectory_searcher(loaded, torch::Device(torch::kCPU), trajectory_options);
		const auto trajectory_root = trajectory_searcher.search(board);
		require(trajectory_root.root.size() > 1,
				"trajectory-cache test requires a non-best searched move");
		auto trajectory_child = board;
		trajectory_child.makeMove(trajectory_root.root[1].move);
		const auto trajectory_next = trajectory_searcher.search(trajectory_child);
		require(trajectory_next.evaluation_reuses > 0,
				"trajectory-aware cache did not reuse the searched child root");
		require(trajectory_next.sims_completed == trajectory_options.mcts_sims,
				"trajectory-aware cache changed the simulation budget");
		std::filesystem::remove(checkpoint);

		std::cout << "gadustests passed" << std::endl;
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "gadustests failed: " << error.what() << std::endl;
		return 1;
	}
}
