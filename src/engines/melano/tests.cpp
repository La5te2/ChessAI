// CTest entrypoint for Melano invariants. It covers precision parsing, state and move codecs,
// geometry relations, terminal rules, annotated-PGN preprocessing, HDF5 schema validation,
// Policy and Value shapes and gradients, checkpoint round trips, direct Policy search and batched
// PUCT.

#include "melano/checkpoint.hpp"
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
		require(packed[0] == 4 && packed[4] == 6 && packed[8] == 1, "white piece token mismatch");
		require(packed[48] == 7 && packed[60] == 12 && packed[63] == 10,
				"black piece token mismatch");
		require(packed[64] == 1, "side-to-move token mismatch");
		require(packed[65] == 15, "castling token mismatch");
		require(packed[66] == 0, "en-passant token mismatch");

		const auto relation_ids = melano::build_geometry_relation_ids().contiguous();
		const auto relation_access = relation_ids.accessor<std::int64_t, 2>();
		std::array<bool, melano::kGeometryRelations> seen_relations{};
		for (std::int64_t source = 0; source < relation_ids.size(0); ++source) {
			for (std::int64_t target = 0; target < relation_ids.size(1); ++target) {
				const auto relation = relation_access[source][target];
				require(relation >= 0 && relation < melano::kGeometryRelations,
						"geometry relation id is out of range");
				seen_relations[static_cast<std::size_t>(relation)] = true;
			}
		}
		int active_relations = 0;
		for (const bool active : seen_relations) {
			active_relations += active ? 1 : 0;
		}
		require(active_relations == melano::kGeometryRelations,
				"geometry relation layout contains an unreachable id");

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
		chess::Board bishop_only("7k/8/8/8/8/8/8/KB6 w - - 0 1");
		require(melano::game_termination(bishop_only) == "insufficient material",
				"king-and-bishop termination mismatch");
		chess::Board knight_only("7k/8/8/8/8/8/8/KN6 w - - 0 1");
		require(melano::game_termination(knight_only) == "insufficient material",
				"king-and-knight termination mismatch");
		chess::Board bishop_vs_knight("6nk/8/8/8/8/8/8/KB6 w - - 0 1");
		require(!melano::game_is_over(bishop_vs_knight),
				"king-and-bishop versus king-and-knight was incorrectly adjudicated");

		chess::Board fifty_move("7k/8/8/8/8/8/6R1/K7 b - - 100 1");
		require(melano::game_termination(fifty_move) == "fifty move rule",
				"fifty-move termination mismatch");

		chess::Board repetition;
		for (const char *uci : {"g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"}) {
			repetition.makeMove(chess::uci::uciToMove(repetition, uci));
		}
		require(repetition.isRepetition(2), "threefold repetition count mismatch");
		require(!repetition.isRepetition(3), "threefold repetition was counted as fourfold");
		require(melano::game_termination(repetition) == "threefold repetition",
				"threefold-repetition termination mismatch");

		// Detached CCRL comments and clock fields are metadata; invalid SAN rejects its game.
		const auto pgn = std::filesystem::temp_directory_path() / "melanotest.pgn";
		const auto h5 = std::filesystem::temp_directory_path() / "melanotest.h5";
		{
			std::ofstream output(pgn);
			output << "[Event \"Melano test\"]\n"
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
		melano::PreprocessOptions preprocess;
		preprocess.input = pgn;
		preprocess.output = h5;
		preprocess.chunk_size = 2;
		preprocess.compression_level = 0;
		preprocess.log_every = 0;
		melano::preprocess_pgn(preprocess);
		{
			melano::SupervisedH5 supervised(h5);
			require(supervised.info().length == 6,
					"Melano preprocessing rejected detached comments or retained invalid SAN");
			const auto supervised_batch = supervised.read_contiguous(0, 6);
			const auto expected_initial = melano::encode_boards({board}).index({0});
			require(torch::equal(supervised_batch.states.index({0}), expected_initial),
					"HDF5 initial state differs from live state codec");
			auto after_d4 = board;
			const auto d4 = chess::uci::uciToMove(after_d4, "d2d4");
			after_d4.makeMove(d4);
			const auto expected_after_d4 = melano::encode_boards({after_d4}).index({0});
			require(torch::equal(supervised_batch.states.index({1}), expected_after_d4),
					"HDF5 second state differs from live state codec");
			require(supervised_batch.moves.index({0}).item<std::int64_t>() ==
						melano::move_to_index(d4),
					"HDF5 policy target differs from live move codec");
			require(std::abs(supervised_batch.values.index({0}).item<float>()) < 1e-6F &&
						std::abs(supervised_batch.values.index({1}).item<float>()) < 1e-6F &&
						std::abs(supervised_batch.values.index({2}).item<float>()) < 1e-6F,
					"Melano preprocessing changed neutral opening targets");
			require(supervised_batch.values.index({4}).item<float>() > 0.0F &&
						supervised_batch.values.index({5}).item<float>() < 0.0F,
					"Melano preprocessing lost detached numerical comments");
		}
		std::filesystem::remove(pgn);
		std::filesystem::remove(h5);

		auto model = melano::Model(8, 1);
		auto states = melano::encode_boards({board, board});
		auto [policy, value] = model->forward(states);
		require(policy.sizes() == torch::IntArrayRef({2, melano::kActionSize}),
				"policy shape mismatch");
		require(value.sizes() == torch::IntArrayRef({2, 1}), "value shape mismatch");
		require(torch::isfinite(policy).all().item<bool>(), "policy contains a non-finite value");
		require(torch::isfinite(value).all().item<bool>(), "value contains a non-finite value");
		require(value.abs().max().item<float>() <= 1.000001F, "value range mismatch");

		const std::vector<chess::Board> legal_test_boards{board, promotion};
		std::vector<std::vector<int>> legal_test_actions;
		std::size_t legal_test_width = 0;
		for (const auto &test_board : legal_test_boards) {
			std::vector<int> actions;
			for (const auto &move : melano::legal_moves(test_board)) {
				actions.push_back(melano::move_to_index(move));
			}
			legal_test_width = std::max(legal_test_width, actions.size());
			legal_test_actions.push_back(std::move(actions));
		}
		auto legal_indices = torch::zeros({static_cast<std::int64_t>(legal_test_boards.size()),
										   static_cast<std::int64_t>(legal_test_width)},
										  torch::TensorOptions().dtype(torch::kInt64));
		auto legal_rows = legal_indices.accessor<std::int64_t, 2>();
		for (std::size_t row = 0; row < legal_test_actions.size(); ++row) {
			for (std::size_t column = 0; column < legal_test_actions[row].size(); ++column) {
				legal_rows[static_cast<std::int64_t>(row)][static_cast<std::int64_t>(column)] =
					legal_test_actions[row][column];
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
				require(std::abs(legal_logits.index({batch, slot}).item<float>() -
								 full_logits.index({batch, action}).item<float>()) < 1.0e-5F,
						"legal-only Policy logit differs from the complete Policy head");
			}
		}
		require(torch::allclose(legal_values, full_values),
				"legal-only Policy path changed the Value output");
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
		require(torch::allclose(reference_policy, loaded_policy),
				"checkpoint changed policy output");
		require(torch::allclose(reference_value, loaded_value), "checkpoint changed value output");

		// Zero simulations rank legal actions with Policy before optional decision components.
		melano::SearchOptions policy_options;
		policy_options.mcts_sims = 0;
		policy_options.root_topn = 4;
		melano::Searcher policy_searcher(loaded, torch::Device(torch::kCPU), policy_options);
		const auto policy_result = policy_searcher.search(board);
		require(policy_result.root.size() == 4, "direct Policy root size mismatch");
		require(policy_result.sims_completed == 0, "direct Policy inference ran MCTS");
		require(melano::index_to_move(melano::move_to_index(policy_result.move), board) ==
					policy_result.move,
				"direct Policy inference selected an illegal move");

		// Duplicate roots share one exact evaluation inside a batched search call.
		melano::Searcher batched_searcher(loaded, torch::Device(torch::kCPU), policy_options);
		const auto batched_results = batched_searcher.search_many({board, board});
		require(batched_results[0].nn_evaluations + batched_results[1].nn_evaluations == 1,
				"duplicate roots performed more than one network evaluation");
		require(batched_results[0].evaluation_reuses + batched_results[1].evaluation_reuses == 1,
				"duplicate root reuse was not reported");

		// Positive cache capacity reuses network output while each call builds fresh search
		// statistics.
		auto cached_options = policy_options;
		cached_options.evaluation_cache_mb = 1;
		melano::Searcher cached_searcher(loaded, torch::Device(torch::kCPU), cached_options);
		const auto first_cached = cached_searcher.search(board);
		const auto second_cached = cached_searcher.search(board);
		require(first_cached.nn_evaluations == 1 && first_cached.evaluation_reuses == 0,
				"first cache search did not evaluate its root");
		require(second_cached.nn_evaluations == 0 && second_cached.evaluation_reuses == 1,
				"second cache search did not reuse its root");
		require(second_cached.sims_completed == 0 && second_cached.expanded_nodes == 1,
				"evaluation cache retained MCTS statistics");
		require(torch::allclose(torch::tensor(first_cached.policy),
								torch::tensor(second_cached.policy), 1e-6, 1e-7),
				"evaluation cache changed Policy");
		require(std::abs(first_cached.value - second_cached.value) < 1e-7F,
				"evaluation cache changed Value");
		cached_searcher.clear_evaluation_cache();
		const auto after_clear = cached_searcher.search(board);
		require(after_clear.nn_evaluations == 1 && after_clear.evaluation_reuses == 0,
				"cache clear did not force root reevaluation");

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
		melano::Searcher trajectory_searcher(loaded, torch::Device(torch::kCPU),
											 trajectory_options);
		const auto trajectory_root = trajectory_searcher.search(board);
		auto trajectory_child = board;
		trajectory_child.makeMove(trajectory_root.move);
		const auto trajectory_next = trajectory_searcher.search(trajectory_child);
		require(trajectory_next.evaluation_reuses > 0,
				"trajectory cache did not reuse the searched child root");
		require(trajectory_next.sims_completed == trajectory_options.mcts_sims,
				"trajectory cache changed the simulation budget");
		std::filesystem::remove(checkpoint);

		std::cout << "melanotests passed" << std::endl;
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "melanotests failed: " << error.what() << std::endl;
		return 1;
	}
}
