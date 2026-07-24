// Implements Melano FCPI with advantage-aware behavior, targets, training, and arena gating.

#include "melano/fcpi.hpp"
#include <hdf5.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>
#include <torch/optim.h>
#include "melano/args.hpp"
#include "melano/checkpoint.hpp"

namespace melano {

namespace {

inline constexpr const char *kFcpiFormula = "melano_tree_consistent_latent_fcpi";

// Format iteration numbers for stable, lexically sortable artifact names.
std::string zero_pad(int value, int width) {
	std::ostringstream output;
	output << std::setfill('0') << std::setw(width) << value;
	return output.str();
}

struct Position {
	int game_id = 0;
	PackedState state{};
	std::string fen;
	float root_value = 0.0F;
	std::vector<int> legal_indices;
	std::vector<float> legal_prior;
	std::vector<float> legal_advantage;
	int played_index = 0;
	std::vector<int> candidate_indices;
	float value_target = 0.0F;
	std::vector<float> policy_target;
	std::vector<float> candidate_q;
	std::vector<float> advantage_target;
	std::vector<PackedState> candidate_next_states;
	float policy_weight = 1.0F;
	float value_weight = 1.0F;
	int aggregate_count = 1;
};

struct Trajectory {
	int game_id = 0;
	chess::Board board;
	std::vector<Position> positions;
};

struct TreeNode {
	chess::Board board;
	PackedState state{};
	float value = 0.0F;
	float backed_value = 0.0F;
	std::vector<int> legal_indices;
	std::vector<float> legal_prior;
	std::vector<float> advantages;
	std::vector<int> candidate_indices;
	std::vector<int> children;
	std::vector<float> candidate_q;
	std::vector<float> policy_target;
	int parent = -1;
	int parent_action = -1;
	int depth = 0;
	double reach_probability = 1.0;
	double priority = 1.0;
	bool terminal = false;
	bool expanded = false;
};

struct CounterfactualTree {
	std::size_t root_record = 0;
	std::vector<TreeNode> nodes;
	int remaining_budget = 0;
	int evaluated_edges = 0;
};

struct TargetSummary {
	std::int64_t trees = 0;
	std::int64_t decision_nodes = 0;
	std::int64_t evaluated_edges = 0;
	std::int64_t terminal_edges = 0;
	int max_depth = 0;
	double residual_sum = 0.0;
	std::int64_t residual_count = 0;
};

struct SamplingSpec {
	std::string fen;
};

// Convert failed HDF5 status codes to descriptive C++ exceptions.
void require_h5(herr_t status, const std::string &operation) {
	if (status < 0) {
		throw std::runtime_error("HDF5 operation failed: " + operation);
	}
}

// Validate HDF5 object handles before subsequent operations use them.
hid_t require_id(hid_t id, const std::string &operation) {
	if (id < 0) {
		throw std::runtime_error("HDF5 operation failed: " + operation);
	}
	return id;
}

// Store an FCPI schema/formula marker as an HDF5 string attribute.
void write_string_attribute(hid_t object, const char *name, const std::string &value) {
	const hid_t space = require_id(H5Screate(H5S_SCALAR), name);
	const hid_t type = require_id(H5Tcopy(H5T_C_S1), name);
	require_h5(H5Tset_size(type, value.size() + 1), name);
	const hid_t attribute =
		require_id(H5Acreate2(object, name, type, space, H5P_DEFAULT, H5P_DEFAULT), name);
	require_h5(H5Awrite(attribute, type, value.c_str()), name);
	H5Aclose(attribute);
	H5Tclose(type);
	H5Sclose(space);
}

// Write one compressed, fixed-shape FCPI tensor dataset and close all temporary handles.
void write_dataset(hid_t file, const char *name, hid_t file_type, hid_t memory_type,
				   const std::vector<hsize_t> &shape, const void *data) {
	const hid_t space =
		require_id(H5Screate_simple(static_cast<int>(shape.size()), shape.data(), nullptr), name);
	const hid_t properties = require_id(H5Pcreate(H5P_DATASET_CREATE), name);
	if (shape[0] > 0) {
		auto chunk = shape;
		chunk[0] = std::min<hsize_t>(shape[0], 4096);
		require_h5(H5Pset_chunk(properties, static_cast<int>(chunk.size()), chunk.data()), name);
		require_h5(H5Pset_deflate(properties, 1), name);
	}
	const hid_t dataset = require_id(
		H5Dcreate2(file, name, file_type, space, H5P_DEFAULT, properties, H5P_DEFAULT), name);
	require_h5(H5Dwrite(dataset, memory_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data), name);
	H5Dclose(dataset);
	H5Pclose(properties);
	H5Sclose(space);
}

// Clamp invalid weights and normalize; use uniform mass when no positive mass remains.
std::vector<float> normalize(std::vector<float> values) {
	double total = 0.0;
	for (float &value : values) {
		value = std::max(0.0F, value);
		total += value;
	}
	if (!std::isfinite(total) || total <= 0.0) {
		const float uniform = 1.0F / std::max<std::size_t>(1, values.size());
		std::fill(values.begin(), values.end(), uniform);
		return values;
	}
	for (float &value : values) {
		value = static_cast<float>(value / total);
	}
	return values;
}

// Compute softmax after max subtraction and exponent clamping for numerical stability.
std::vector<float> stable_softmax(const std::vector<double> &logits) {
	if (logits.empty()) {
		return {};
	}
	const double maximum = *std::max_element(logits.begin(), logits.end());
	std::vector<float> values(logits.size());
	for (std::size_t index = 0; index < logits.size(); ++index) {
		values[index] =
			static_cast<float>(std::exp(std::clamp(logits[index] - maximum, -80.0, 0.0)));
	}
	return normalize(std::move(values));
}

// Draw one categorical sample from an already normalized behavior distribution.
std::size_t sample_index(const std::vector<float> &probabilities, std::mt19937_64 &rng) {
	std::discrete_distribution<std::size_t> distribution(probabilities.begin(),
														 probabilities.end());
	return distribution(rng);
}

// Use model-visible packed state bytes as the aggregation and per-game deduplication key.
std::string packed_key(const PackedState &state) {
	return std::string(reinterpret_cast<const char *>(state.data()), state.size());
}

// Build a shuffled mixture of startpos and opening-book starts for one iteration.
std::vector<SamplingSpec> make_sampling_specs(const FcpiOptions &options, int iteration,
											  nlohmann::json &summary) {
	std::mt19937_64 rng(options.seed + iteration);
	const int games = std::max(1, options.games_per_iter);
	const double fraction = std::clamp(options.startpos_fraction, 0.0, 1.0);
	const int startpos_games =
		options.opening_book.empty()
			? games
			: std::clamp(static_cast<int>(std::llround(games * fraction)), 0, games);
	const int book_games = games - startpos_games;
	std::vector<SamplingSpec> specs;
	specs.reserve(games);
	for (int index = 0; index < startpos_games; ++index) {
		specs.push_back({std::string(chess::constants::STARTPOS)});
	}
	int cycles = 0;
	std::vector<std::string> positions;
	if (book_games > 0) {
		positions = load_opening_positions(options.opening_book, options.book_plies,
										   options.max_book_positions, options.seed + iteration);
		if (positions.empty()) {
			throw std::runtime_error("opening book contains no FCPI positions");
		}
		while (static_cast<int>(specs.size()) < games) {
			auto cycle = positions;
			std::shuffle(cycle.begin(), cycle.end(), rng);
			for (const auto &fen : cycle) {
				if (static_cast<int>(specs.size()) >= games) {
					break;
				}
				specs.push_back({fen});
			}
			++cycles;
		}
	}
	std::shuffle(specs.begin(), specs.end(), rng);
	summary = {
		{"games", games},
		{"startpos_games", startpos_games},
		{"book_games", book_games},
		{"book_positions", positions.size()},
		{"book_cycles", cycles},
		{"reused_book_starts", std::max(0, book_games - static_cast<int>(positions.size()))},
	};
	return specs;
}

// Evaluate arbitrarily many positions through bounded neural batches.
std::vector<SearchResult> evaluate_chunks(Searcher &searcher,
										  const std::vector<chess::Board> &boards, int batch_size) {
	std::vector<SearchResult> output;
	output.reserve(boards.size());
	for (std::size_t begin = 0; begin < boards.size(); begin += std::max(1, batch_size)) {
		const auto end = std::min(boards.size(), begin + std::max(1, batch_size));
		std::vector<chess::Board> chunk(boards.begin() + begin, boards.begin() + end);
		auto results = searcher.search_many(chunk);
		output.insert(output.end(), std::make_move_iterator(results.begin()),
					  std::make_move_iterator(results.end()));
	}
	return output;
}

// Keep the best action and an optional required action, then use Gumbel top-k
// without replacement for the remaining local tree width.
std::vector<int> choose_candidates(const std::vector<int> &legal, const std::vector<float> &scores,
								   int required, int count_requested, std::mt19937_64 &rng) {
	if (legal.empty() || legal.size() != scores.size()) {
		throw std::invalid_argument("counterfactual candidates require aligned legal scores");
	}
	const std::size_t count =
		std::min<std::size_t>(static_cast<std::size_t>(std::max(1, count_requested)), legal.size());
	std::vector<int> selected;
	selected.reserve(count);
	const auto top = std::max_element(scores.begin(), scores.end()) - scores.begin();
	selected.push_back(legal[static_cast<std::size_t>(top)]);
	if (required >= 0 && required != selected.front() && selected.size() < count) {
		selected.push_back(required);
	}
	std::uniform_real_distribution<double> uniform(
		std::nextafter(0.0, 1.0), std::nextafter(1.0, 0.0));
	std::vector<std::pair<double, int>> proposals;
	for (std::size_t index = 0; index < legal.size(); ++index) {
		if (std::find(selected.begin(), selected.end(), legal[index]) != selected.end()) {
			continue;
		}
		const double gumbel = -std::log(-std::log(uniform(rng)));
		proposals.emplace_back(
			std::log(std::clamp(static_cast<double>(scores[index]), 1e-12, 1.0)) + gumbel,
			legal[index]);
	}
	std::stable_sort(proposals.begin(), proposals.end(),
					 [](const auto &left, const auto &right) {
						 return left.first > right.first;
					 });
	for (const auto &[score, action] : proposals) {
		(void)score;
		if (selected.size() >= count) {
			break;
		}
		selected.push_back(action);
	}
	return selected;
}

// Generate closed-policy games using log(P)+w*A behavior logits and lambda-return V targets.
std::vector<Position> collect_selfplay(Model model, const torch::Device &device,
									   const FcpiOptions &options, int iteration,
									   nlohmann::json &sampling_summary) {
	SearchOptions closed;
	closed.type = SearchType::Closed;
	closed.precision = options.precision;
	closed.mcts_sims = 0;
	closed.mcts_batch_size = options.inference_batch_size;
	Searcher evaluator(model, device, closed);
	nlohmann::json starts;
	const auto specs = make_sampling_specs(options, iteration, starts);
	std::mt19937_64 rng(options.seed + iteration);
	std::vector<Trajectory> trajectories;
	trajectories.reserve(specs.size());
	int completed = 0;
	std::cout << "fcpi self-play start: iteration=" << iteration << " arch_type=" << kArchType
			  << " games=" << specs.size() << " max_plies=" << options.max_plies
			  << " device=" << device.str() << std::endl;
	std::cout << "fcpi starts: " << starts.dump() << std::endl;

	for (std::size_t group_start = 0; group_start < specs.size();
		 group_start += std::max(1, options.games_in_flight)) {
		const auto group_end =
			std::min(specs.size(), group_start + std::max(1, options.games_in_flight));
		std::vector<Trajectory> group;
		for (std::size_t index = group_start; index < group_end; ++index) {
			Trajectory trajectory;
			trajectory.game_id = static_cast<int>(index) + 1;
			trajectory.board = chess::Board(specs[index].fen);
			group.push_back(std::move(trajectory));
		}
		std::vector<bool> done(group.size(), false);
		while (std::find(done.begin(), done.end(), false) != done.end()) {
			std::vector<std::size_t> active_indices;
			std::vector<chess::Board> boards;
			for (std::size_t index = 0; index < group.size(); ++index) {
				if (!done[index]) {
					active_indices.push_back(index);
					boards.push_back(group[index].board);
				}
			}
			const auto results = evaluate_chunks(evaluator, boards, options.inference_batch_size);
			for (std::size_t row = 0; row < active_indices.size(); ++row) {
				auto &trajectory = group[active_indices[row]];
				auto &board = trajectory.board;
				const auto moves = legal_moves(board);
				std::vector<int> legal;
				std::vector<float> prior;
				std::vector<float> advantages;
				for (const auto &move : moves) {
					const int action = move_to_index(move);
					legal.push_back(action);
					prior.push_back(results[row].policy[action]);
					advantages.push_back(results[row].advantages[action]);
				}
				prior = normalize(std::move(prior));
				const double temperature = std::max(1e-4, options.behavior_temperature);
				std::vector<double> behavior_logits(prior.size());
				double advantage_scale = 0.0;
				for (const float advantage : advantages) {
					advantage_scale =
						std::max(advantage_scale, std::abs(static_cast<double>(advantage)));
				}
				advantage_scale = std::max(1e-4, advantage_scale);
				for (std::size_t index = 0; index < prior.size(); ++index) {
					behavior_logits[index] =
						(std::log(std::clamp(static_cast<double>(prior[index]), 1e-12, 1.0)) +
						 advantages[index] / advantage_scale) /
						temperature;
				}
				auto behavior = stable_softmax(behavior_logits);
				const double mix = std::clamp(options.uniform_mix, 0.0, 1.0);
				for (float &probability : behavior) {
					probability =
						static_cast<float>((1.0 - mix) * probability + mix / behavior.size());
				}
				const std::size_t choice = sample_index(behavior, rng);
				const int played = legal[choice];
				Position position;
				position.game_id = trajectory.game_id;
				position.state = encode_state(board);
				position.fen = board.getFen();
				position.root_value = results[row].value;
				position.legal_indices = legal;
				position.legal_prior = prior;
				position.legal_advantage = advantages;
				position.played_index = played;
				trajectory.positions.push_back(std::move(position));
				board.makeMove(moves[choice]);
				const bool terminal = game_is_over(board);
				const bool truncated =
					static_cast<int>(trajectory.positions.size()) >= options.max_plies;
				if (terminal || truncated) {
					done[active_indices[row]] = true;
					++completed;
					std::cout << "fcpi game: completed=" << completed << '/' << specs.size()
							  << " game_id=" << trajectory.game_id
							  << " plies=" << trajectory.positions.size()
							  << " result=" << (terminal ? game_result(board) : "bootstrap")
							  << " truncated=" << (truncated && !terminal ? "true" : "false")
							  << std::endl;
				}
			}
		}
		for (auto &trajectory : group) {
			trajectories.push_back(std::move(trajectory));
		}
	}

	std::vector<chess::Board> truncated_boards;
	std::vector<std::size_t> truncated_indices;
	for (std::size_t index = 0; index < trajectories.size(); ++index) {
		if (!game_is_over(trajectories[index].board)) {
			truncated_indices.push_back(index);
			truncated_boards.push_back(trajectories[index].board);
		}
	}
	const auto truncated_results =
		evaluate_chunks(evaluator, truncated_boards, options.inference_batch_size);
	std::unordered_map<std::size_t, float> bootstrap;
	for (std::size_t index = 0; index < truncated_indices.size(); ++index) {
		bootstrap[truncated_indices[index]] = truncated_results[index].value;
	}
	const double lambda = std::clamp(options.td_lambda, 0.0, 1.0);
	for (std::size_t trajectory_index = 0; trajectory_index < trajectories.size();
		 ++trajectory_index) {
		auto &trajectory = trajectories[trajectory_index];
		float next_return = game_is_over(trajectory.board)
								? terminal_value_side_to_move(trajectory.board)
								: bootstrap.at(trajectory_index);
		const float final_value = next_return;
		for (int index = static_cast<int>(trajectory.positions.size()) - 1; index >= 0; --index) {
			const float next_value = index + 1 == static_cast<int>(trajectory.positions.size())
										 ? final_value
										 : trajectory.positions[index + 1].root_value;
			const float current_return =
				-static_cast<float>((1.0 - lambda) * next_value + lambda * next_return);
			trajectory.positions[index].value_target = std::clamp(current_return, -1.0F, 1.0F);
			next_return = current_return;
		}
	}

	std::mt19937_64 sample_rng(options.seed + iteration + 1'000'003);
	std::vector<Position> records;
	std::int64_t source_positions = 0;
	std::int64_t unique_positions = 0;
	int capped_games = 0;
	for (auto &trajectory : trajectories) {
		source_positions += trajectory.positions.size();
		std::unordered_set<std::string> seen;
		std::vector<std::size_t> indices;
		for (std::size_t index = 0; index < trajectory.positions.size(); ++index) {
			if (seen.insert(packed_key(trajectory.positions[index].state)).second) {
				indices.push_back(index);
			}
		}
		unique_positions += indices.size();
		if (static_cast<int>(indices.size()) > options.positions_per_game) {
			std::shuffle(indices.begin(), indices.end(), sample_rng);
			indices.resize(options.positions_per_game);
			std::sort(indices.begin(), indices.end());
			++capped_games;
		}
		for (const auto index : indices) {
			records.push_back(std::move(trajectory.positions[index]));
		}
	}
	sampling_summary = {
		{"games", trajectories.size()},
		{"source_positions", source_positions},
		{"unique_positions", unique_positions},
		{"selected_positions", records.size()},
		{"positions_per_game", options.positions_per_game},
		{"capped_games", capped_games},
		{"starts", starts},
	};
	std::cout << "fcpi position sampling: " << sampling_summary.dump() << std::endl;
	return records;
}

// Materialize one exact Melano decision node with frozen P/V/A predictions.
TreeNode make_tree_node(const chess::Board &board, const SearchResult &evaluation, int parent,
						int parent_action, int depth, double reach_probability) {
	TreeNode node;
	node.board = board;
	node.state = encode_state(board);
	node.value = evaluation.value;
	node.backed_value = evaluation.value;
	node.parent = parent;
	node.parent_action = parent_action;
	node.depth = depth;
	node.reach_probability = reach_probability;
	for (const auto &move : legal_moves(board)) {
		const int action = move_to_index(move);
		node.legal_indices.push_back(action);
		node.legal_prior.push_back(evaluation.policy[action]);
		node.advantages.push_back(evaluation.advantages[action]);
	}
	node.legal_prior = normalize(std::move(node.legal_prior));
	return node;
}

// Convert Melano P/A into a scale-free proposal distribution for behavior and tree widening.
std::vector<float> pva_distribution(const std::vector<float> &prior,
									const std::vector<float> &advantages) {
	if (prior.size() != advantages.size() || prior.empty()) {
		throw std::invalid_argument("Melano P/A distribution requires aligned non-empty inputs");
	}
	double scale = 0.0;
	for (const float advantage : advantages) {
		scale = std::max(scale, std::abs(static_cast<double>(advantage)));
	}
	if (scale < 1e-4) {
		return prior;
	}
	std::vector<double> logits(prior.size());
	for (std::size_t index = 0; index < prior.size(); ++index) {
		logits[index] = std::log(std::clamp(static_cast<double>(prior[index]), 1e-12, 1.0)) +
						advantages[index] / scale;
	}
	return stable_softmax(logits);
}

// Select the unexpanded node with the largest reach-weighted Bellman residual.
int select_tree_frontier(const CounterfactualTree &tree) {
	int selected = -1;
	double best = -1.0;
	for (std::size_t index = 0; index < tree.nodes.size(); ++index) {
		const auto &node = tree.nodes[index];
		if (node.terminal || node.expanded || node.legal_indices.empty()) {
			continue;
		}
		if (node.priority > best) {
			best = node.priority;
			selected = static_cast<int>(index);
		}
	}
	return selected;
}

// Couple local width and reachable depth to one per-root edge budget.
int expansion_width(const CounterfactualTree &tree) {
	if (tree.remaining_budget <= 0) {
		return 0;
	}
	const int progressive =
		std::max(2, static_cast<int>(std::ceil(std::sqrt(tree.remaining_budget))));
	return std::min(tree.remaining_budget, progressive);
}

// Build one tree-consistent improved policy using exact child values where available
// and Melano V+A estimates for actions outside the expanded set.
std::vector<float> improve_policy(const std::vector<float> &prior,
								  const std::vector<float> &action_values) {
	double mean = 0.0;
	for (std::size_t index = 0; index < prior.size(); ++index) {
		mean += prior[index] * action_values[index];
	}
	double scale = 0.0;
	for (std::size_t index = 0; index < prior.size(); ++index) {
		scale = std::max(scale, std::abs(static_cast<double>(action_values[index]) - mean));
	}
	if (scale < 1e-4) {
		return prior;
	}
	std::vector<double> logits(prior.size());
	for (std::size_t index = 0; index < prior.size(); ++index) {
		logits[index] = std::log(std::clamp(static_cast<double>(prior[index]), 1e-12, 1.0)) +
						(action_values[index] - mean) / scale;
	}
	return stable_softmax(logits);
}

// Expand exact-board counterfactual trees. Every expanded node trains P/A and
// latent dynamics, while only real self-play roots carry a Value target.
void construct_targets(std::vector<Position> &records, Model model, const torch::Device &device,
					   const FcpiOptions &options, TargetSummary &summary) {
	if (options.counterfactual_budget < 2) {
		throw std::invalid_argument("counterfactual-budget must be at least 2");
	}
	SearchOptions closed;
	closed.type = SearchType::Closed;
	closed.precision = options.precision;
	closed.mcts_sims = 0;
	closed.mcts_batch_size = options.inference_batch_size;
	Searcher evaluator(model, device, closed);
	std::mt19937_64 rng(options.seed + 3'000'017);
	std::vector<Position> tree_records;
	tree_records.reserve(records.size() * 2);
	std::cout << "fcpi counterfactual tree start: positions=" << records.size()
			  << " budget_per_root=" << options.counterfactual_budget << std::endl;

	for (std::size_t subset_begin = 0; subset_begin < records.size();
		 subset_begin += std::max(1, options.target_records_per_batch)) {
		const auto subset_end =
			std::min(records.size(), subset_begin + std::max(1, options.target_records_per_batch));
		std::vector<CounterfactualTree> trees;
		trees.reserve(subset_end - subset_begin);
		for (std::size_t record_index = subset_begin; record_index < subset_end; ++record_index) {
			const auto &record = records[record_index];
			CounterfactualTree tree;
			tree.root_record = record_index;
			tree.remaining_budget = options.counterfactual_budget;
			TreeNode root;
			root.board = chess::Board(record.fen);
			root.state = record.state;
			root.value = record.root_value;
			root.backed_value = record.root_value;
			root.legal_indices = record.legal_indices;
			root.legal_prior = record.legal_prior;
			root.advantages = record.legal_advantage;
			tree.nodes.push_back(std::move(root));
			trees.push_back(std::move(tree));
		}

		while (true) {
			struct PendingChild {
				std::size_t tree = 0;
				int parent = 0;
				std::size_t slot = 0;
				int action = 0;
				chess::Board board;
				bool terminal = false;
			};
			std::vector<PendingChild> pending;
			std::vector<chess::Board> evaluation_boards;
			std::vector<std::size_t> evaluation_pending;
			bool expanded_any = false;
			for (std::size_t tree_index = 0; tree_index < trees.size(); ++tree_index) {
				auto &tree = trees[tree_index];
				if (tree.remaining_budget <= 0) {
					continue;
				}
				const int node_index = select_tree_frontier(tree);
				if (node_index < 0) {
					continue;
				}
				auto &node = tree.nodes[static_cast<std::size_t>(node_index)];
				const int width = std::min<int>(
					expansion_width(tree), static_cast<int>(node.legal_indices.size()));
				const int required =
					node_index == 0 ? records[tree.root_record].played_index : -1;
				const auto proposals = pva_distribution(node.legal_prior, node.advantages);
				node.candidate_indices =
					choose_candidates(node.legal_indices, proposals, required, width, rng);
				node.children.assign(node.candidate_indices.size(), -1);
				node.candidate_q.assign(node.candidate_indices.size(), node.value);
				node.expanded = true;
				tree.remaining_budget -= static_cast<int>(node.candidate_indices.size());
				tree.evaluated_edges += static_cast<int>(node.candidate_indices.size());
				expanded_any = true;
				for (std::size_t slot = 0; slot < node.candidate_indices.size(); ++slot) {
					const int action = node.candidate_indices[slot];
					const auto move = index_to_move(action, node.board);
					if (move.move() == chess::Move::NO_MOVE) {
						throw std::runtime_error("FCPI tree candidate action is illegal");
					}
					PendingChild child;
					child.tree = tree_index;
					child.parent = node_index;
					child.slot = slot;
					child.action = action;
					child.board = node.board;
					child.board.makeMove(move);
					child.terminal = game_is_over(child.board);
					const std::size_t pending_index = pending.size();
					pending.push_back(std::move(child));
					if (!pending.back().terminal) {
						evaluation_pending.push_back(pending_index);
						evaluation_boards.push_back(pending.back().board);
					}
				}
			}
			if (!expanded_any) {
				break;
			}
			const auto evaluations =
				evaluate_chunks(evaluator, evaluation_boards, options.inference_batch_size);
			std::vector<int> evaluation_row(pending.size(), -1);
			for (std::size_t row = 0; row < evaluation_pending.size(); ++row) {
				evaluation_row[evaluation_pending[row]] = static_cast<int>(row);
			}
			for (std::size_t pending_index = 0; pending_index < pending.size(); ++pending_index) {
				auto &request = pending[pending_index];
				auto &tree = trees[request.tree];
				const auto &parent = tree.nodes[static_cast<std::size_t>(request.parent)];
				const auto legal_position =
					std::find(parent.legal_indices.begin(), parent.legal_indices.end(), request.action);
				const float edge_prior =
					parent.legal_prior[static_cast<std::size_t>(legal_position -
																parent.legal_indices.begin())];
				TreeNode child;
				if (request.terminal) {
					child.board = request.board;
					child.state = encode_state(request.board);
					child.value = terminal_value_side_to_move(request.board);
					child.backed_value = child.value;
					child.terminal = true;
					++summary.terminal_edges;
				} else {
					child = make_tree_node(
						request.board,
						evaluations[static_cast<std::size_t>(evaluation_row[pending_index])],
						request.parent, request.action, parent.depth + 1,
						parent.reach_probability * edge_prior);
				}
				child.parent = request.parent;
				child.parent_action = request.action;
				child.depth = parent.depth + 1;
				child.reach_probability = parent.reach_probability * edge_prior;
				const float edge_q = -child.value;
				const double residual = std::abs(edge_q - parent.value);
				child.priority = child.reach_probability *
								 (residual + 1.0 / std::sqrt(2.0 + child.depth));
				summary.residual_sum += residual;
				++summary.residual_count;
				summary.max_depth = std::max(summary.max_depth, child.depth);
				const int child_index = static_cast<int>(tree.nodes.size());
				tree.nodes.push_back(std::move(child));
				tree.nodes[static_cast<std::size_t>(request.parent)].children[request.slot] =
					child_index;
			}
		}

		for (auto &tree : trees) {
			for (std::size_t reverse = tree.nodes.size(); reverse-- > 0;) {
				auto &node = tree.nodes[reverse];
				if (node.terminal || !node.expanded) {
					node.backed_value = node.value;
					continue;
				}
				std::vector<float> action_values(node.legal_indices.size());
				for (std::size_t index = 0; index < action_values.size(); ++index) {
					action_values[index] =
						std::clamp(node.value + node.advantages[index], -1.0F, 1.0F);
				}
				for (std::size_t slot = 0; slot < node.candidate_indices.size(); ++slot) {
					const int action = node.candidate_indices[slot];
					const auto legal = std::find(node.legal_indices.begin(),
												node.legal_indices.end(), action);
					const float q = -tree.nodes[static_cast<std::size_t>(node.children[slot])]
										 .backed_value;
					action_values[static_cast<std::size_t>(legal - node.legal_indices.begin())] = q;
					node.candidate_q[slot] = q;
				}
				node.policy_target = improve_policy(node.legal_prior, action_values);
				float policy_value = 0.0F;
				for (std::size_t index = 0; index < action_values.size(); ++index) {
					policy_value += node.policy_target[index] * action_values[index];
				}
				node.backed_value =
					std::min(node.value, std::clamp(policy_value, -1.0F, 1.0F));
			}

			const float edge_total =
				static_cast<float>(std::max(1, tree.evaluated_edges));
			for (std::size_t node_index = 0; node_index < tree.nodes.size(); ++node_index) {
				const auto &node = tree.nodes[node_index];
				if (!node.expanded) {
					continue;
				}
				const auto &root = records[tree.root_record];
				Position record;
				record.game_id = root.game_id;
				record.state = node.state;
				record.fen = node.board.getFen();
				record.root_value = node.value;
				record.legal_indices = node.legal_indices;
				record.legal_prior = node.legal_prior;
				record.legal_advantage = node.advantages;
				record.played_index = node_index == 0 ? root.played_index : -1;
				record.candidate_indices = node.candidate_indices;
				record.value_target = node_index == 0 ? root.value_target : node.value;
				record.policy_target = node.policy_target;
				record.candidate_q = node.candidate_q;
				record.policy_weight =
					static_cast<float>(node.candidate_indices.size()) / edge_total;
				record.value_weight = node_index == 0 ? 1.0F : 0.0F;
				for (std::size_t slot = 0; slot < node.children.size(); ++slot) {
					record.candidate_next_states.push_back(
						tree.nodes[static_cast<std::size_t>(node.children[slot])].state);
					record.advantage_target.push_back(std::clamp(
						node.candidate_q[slot] - record.value_target, -2.0F, 0.0F));
				}
				tree_records.push_back(std::move(record));
				++summary.decision_nodes;
			}
			++summary.trees;
			summary.evaluated_edges += tree.evaluated_edges;
		}
		if (subset_end == records.size() || subset_end % std::max(1, options.log_every) == 0) {
			std::cout << "fcpi counterfactual tree: positions=" << subset_end << '/'
					  << records.size() << " decision_nodes=" << summary.decision_nodes
					  << " evaluated_edges=" << summary.evaluated_edges << std::endl;
		}
	}
	records = std::move(tree_records);
	std::cout << "fcpi counterfactual summary: trees=" << summary.trees
			  << " decision_nodes=" << summary.decision_nodes
			  << " evaluated_edges=" << summary.evaluated_edges
			  << " terminal_edges=" << summary.terminal_edges
			  << " max_depth=" << summary.max_depth << " mean_residual="
			  << (summary.residual_count > 0 ? summary.residual_sum / summary.residual_count : 0.0)
			  << std::endl;
}

// Merge model-indistinguishable states and average their stochastic P/V/A targets.
std::vector<Position> aggregate_records(std::vector<Position> records, nlohmann::json &summary) {
	const std::size_t source_count = records.size();
	std::unordered_map<std::string, std::size_t> groups;
	std::vector<Position> output;
	for (auto &record : records) {
		const auto key = packed_key(record.state);
		const auto found = groups.find(key);
		if (found == groups.end()) {
			groups.emplace(key, output.size());
			output.push_back(std::move(record));
			continue;
		}
		auto &merged = output[found->second];
		if (merged.legal_indices != record.legal_indices) {
			throw std::runtime_error("identical encoded states produced different legal actions");
		}
		const float old_count = static_cast<float>(merged.aggregate_count);
		const float new_count = old_count + 1.0F;
		const float old_policy_weight = merged.policy_weight;
		const float new_policy_weight = old_policy_weight + record.policy_weight;
		const float old_value_weight = merged.value_weight;
		const float new_value_weight = old_value_weight + record.value_weight;
		for (std::size_t index = 0; index < merged.legal_prior.size(); ++index) {
			merged.legal_prior[index] =
				(merged.legal_prior[index] * old_count + record.legal_prior[index]) / new_count;
			if (new_policy_weight > 0.0F) {
				merged.policy_target[index] =
					(merged.policy_target[index] * old_policy_weight +
					 record.policy_target[index] * record.policy_weight) /
					new_policy_weight;
			}
		}
		for (std::size_t index = 0; index < merged.legal_advantage.size(); ++index) {
			merged.legal_advantage[index] =
				(merged.legal_advantage[index] * old_count + record.legal_advantage[index]) /
				new_count;
		}
		if (new_value_weight > 0.0F) {
			merged.value_target =
				(merged.value_target * old_value_weight +
				 record.value_target * record.value_weight) /
				new_value_weight;
		}
		for (std::size_t candidate = 0; candidate < record.candidate_indices.size(); ++candidate) {
			const int action = record.candidate_indices[candidate];
			const auto existing =
				std::find(merged.candidate_indices.begin(), merged.candidate_indices.end(), action);
			if (existing == merged.candidate_indices.end()) {
				merged.candidate_indices.push_back(action);
				merged.candidate_q.push_back(record.candidate_q[candidate]);
				merged.candidate_next_states.push_back(record.candidate_next_states[candidate]);
			} else {
				const auto index = existing - merged.candidate_indices.begin();
				if (merged.candidate_next_states[index] != record.candidate_next_states[candidate]) {
					throw std::runtime_error("identical state/action produced different successor states");
				}
				merged.candidate_q[index] =
					(merged.candidate_q[index] * old_policy_weight +
					 record.candidate_q[candidate] * record.policy_weight) /
					std::max(1e-8F, new_policy_weight);
			}
		}
		merged.policy_weight = new_policy_weight;
		merged.value_weight = new_value_weight;
		merged.aggregate_count += 1;
	}
	for (auto &record : output) {
		record.legal_prior = normalize(std::move(record.legal_prior));
		record.policy_target = normalize(std::move(record.policy_target));
		record.advantage_target.resize(record.candidate_q.size());
		for (std::size_t index = 0; index < record.candidate_q.size(); ++index) {
			record.advantage_target[index] =
				std::clamp(record.candidate_q[index] - record.value_target, -2.0F, 0.0F);
		}
	}
	summary = {
		{"source_positions", source_count},
		{"aggregated_positions", output.size()},
		{"merged_positions", source_count - output.size()},
	};
	return output;
}

// Persist generated P/V/A FCPI targets for diagnostics and iteration reproducibility.
nlohmann::json write_fcpi_h5(const std::filesystem::path &path, std::vector<Position> &records) {
	nlohmann::json aggregation;
	records = aggregate_records(std::move(records), aggregation);
	std::cout << "fcpi position aggregation: " << aggregation.dump() << std::endl;
	if (records.empty()) {
		throw std::runtime_error("FCPI generated no training positions");
	}
	const std::size_t legal_width =
		std::max_element(records.begin(), records.end(), [](const auto &left, const auto &right) {
			return left.legal_indices.size() < right.legal_indices.size();
		})->legal_indices.size();
	const std::size_t candidate_width =
		std::max_element(records.begin(), records.end(), [](const auto &left, const auto &right) {
			return left.candidate_indices.size() < right.candidate_indices.size();
		})->candidate_indices.size();
	const std::size_t count = records.size();
	std::vector<std::uint8_t> states(count * kStateFeatures);
	std::vector<std::int32_t> legal(count * legal_width, 0);
	std::vector<float> priors(count * legal_width, 0.0F);
	std::vector<float> policy(count * legal_width, 0.0F);
	std::vector<std::uint8_t> legal_counts(count);
	std::vector<float> values(count);
	std::vector<float> policy_weights(count);
	std::vector<float> value_weights(count);
	std::vector<std::int32_t> candidates(count * candidate_width, 0);
	std::vector<float> candidate_q(count * candidate_width, 0.0F);
	std::vector<float> advantage_targets(count * candidate_width, 0.0F);
	std::vector<std::uint8_t> candidate_next_states(
		count * candidate_width * static_cast<std::size_t>(kStateFeatures));
	std::vector<std::uint8_t> candidate_counts(count);
	for (std::size_t row = 0; row < count; ++row) {
		std::copy(records[row].state.begin(), records[row].state.end(),
				  states.begin() + row * kStateFeatures);
		legal_counts[row] = static_cast<std::uint8_t>(records[row].legal_indices.size());
		candidate_counts[row] = static_cast<std::uint8_t>(records[row].candidate_indices.size());
		values[row] = records[row].value_target;
		policy_weights[row] = records[row].policy_weight;
		value_weights[row] = records[row].value_weight;
		for (std::size_t column = 0; column < records[row].legal_indices.size(); ++column) {
			legal[row * legal_width + column] = records[row].legal_indices[column];
			priors[row * legal_width + column] = records[row].legal_prior[column];
			policy[row * legal_width + column] = records[row].policy_target[column];
		}
		for (std::size_t column = 0; column < records[row].candidate_indices.size(); ++column) {
			candidates[row * candidate_width + column] = records[row].candidate_indices[column];
			candidate_q[row * candidate_width + column] = records[row].candidate_q[column];
			advantage_targets[row * candidate_width + column] =
				records[row].advantage_target[column];
			std::copy(records[row].candidate_next_states[column].begin(),
					  records[row].candidate_next_states[column].end(),
					  candidate_next_states.begin() +
						  (row * candidate_width + column) * kStateFeatures);
		}
	}
	if (!path.parent_path().empty()) {
		std::filesystem::create_directories(path.parent_path());
	}
	const hid_t file = require_id(
		H5Fcreate(path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), path.string());
	write_string_attribute(file, "arch_type", kArchType);
	write_string_attribute(file, "fcpi_formula", kFcpiFormula);
	write_dataset(file, "states", H5T_STD_U8LE, H5T_NATIVE_UINT8, {count, kStateFeatures},
				  states.data());
	write_dataset(file, "legal_indices", H5T_STD_I32LE, H5T_NATIVE_INT32, {count, legal_width},
				  legal.data());
	write_dataset(file, "legal_priors", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, {count, legal_width},
				  priors.data());
	write_dataset(file, "policy_targets", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, {count, legal_width},
				  policy.data());
	write_dataset(file, "legal_counts", H5T_STD_U8LE, H5T_NATIVE_UINT8, {count},
				  legal_counts.data());
	write_dataset(file, "value_targets", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, {count}, values.data());
	write_dataset(file, "policy_weights", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, {count},
				  policy_weights.data());
	write_dataset(file, "value_weights", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, {count},
				  value_weights.data());
	write_dataset(file, "candidate_indices", H5T_STD_I32LE, H5T_NATIVE_INT32,
				  {count, candidate_width}, candidates.data());
	write_dataset(file, "candidate_q", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, {count, candidate_width},
				  candidate_q.data());
	write_dataset(file, "advantage_targets", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT,
				  {count, candidate_width}, advantage_targets.data());
	write_dataset(file, "candidate_next_states", H5T_STD_U8LE, H5T_NATIVE_UINT8,
				  {count, candidate_width, kStateFeatures}, candidate_next_states.data());
	write_dataset(file, "candidate_counts", H5T_STD_U8LE, H5T_NATIVE_UINT8, {count},
				  candidate_counts.data());
	H5Fclose(file);
	return {
		{"path", path.string()},	  {"positions", count},
		{"legal_width", legal_width}, {"counterfactual_width", candidate_width},
		{"formula", kFcpiFormula},	  {"aggregation", aggregation},
	};
}

// Train P/A/dynamics on expanded tree nodes and V only on real self-play roots.
nlohmann::json train_candidate(const std::filesystem::path &source,
							   const std::filesystem::path &candidate, Model model,
							   const torch::Device &device, std::vector<Position> &records,
							   const FcpiOptions &options) {
	model->to(device);
	model->train();
	torch::optim::AdamW optimizer(
		model->parameters(),
		torch::optim::AdamWOptions(options.learning_rate).weight_decay(options.weight_decay));
	std::vector<std::size_t> order(records.size());
	std::iota(order.begin(), order.end(), 0);
	std::mt19937_64 rng(options.seed);
	std::int64_t steps = 0;
	auto metric_totals =
		torch::zeros({6}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
	for (int epoch = 0; epoch < std::max(0, options.epochs); ++epoch) {
		std::shuffle(order.begin(), order.end(), rng);
		for (std::size_t begin = 0; begin < order.size();
			 begin += std::max(1, options.batch_size)) {
			const auto end = std::min(order.size(), begin + std::max(1, options.batch_size));
			const std::int64_t batch = static_cast<std::int64_t>(end - begin);
			std::size_t width = 1;
			std::vector<std::uint8_t> packed(batch * kStateFeatures);
			std::size_t candidate_width = 1;
			for (std::size_t index = begin; index < end; ++index) {
				width = std::max(width, records[order[index]].legal_indices.size());
				candidate_width =
					std::max(candidate_width, records[order[index]].candidate_indices.size());
				std::copy(records[order[index]].state.begin(), records[order[index]].state.end(),
						  packed.begin() + (index - begin) * kStateFeatures);
			}
			std::vector<std::uint8_t> packed_next(batch * candidate_width * kStateFeatures);
			const bool pin_memory = device.is_cuda();
			auto int_options =
				torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
			auto float_options =
				torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
			if (pin_memory) {
				int_options = int_options.pinned_memory(true);
				float_options = float_options.pinned_memory(true);
			}
			auto states = decode_states(packed.data(), batch, pin_memory).to(device, true);
			auto legal =
				torch::zeros({batch, static_cast<std::int64_t>(width)}, int_options);
			auto targets =
				torch::zeros({batch, static_cast<std::int64_t>(width)}, float_options);
			auto counts = torch::zeros({batch}, int_options);
			auto values = torch::zeros({batch}, float_options);
			auto policy_weights = torch::zeros({batch}, float_options);
			auto value_weights = torch::zeros({batch}, float_options);
			auto candidate_indices =
				torch::zeros({batch, static_cast<std::int64_t>(candidate_width)}, int_options);
			auto advantage_targets =
				torch::zeros({batch, static_cast<std::int64_t>(candidate_width)}, float_options);
			auto candidate_counts = torch::zeros({batch}, int_options);
			auto legal_access = legal.accessor<std::int64_t, 2>();
			auto target_access = targets.accessor<float, 2>();
			auto count_access = counts.accessor<std::int64_t, 1>();
			auto value_access = values.accessor<float, 1>();
			auto policy_weight_access = policy_weights.accessor<float, 1>();
			auto value_weight_access = value_weights.accessor<float, 1>();
			auto candidate_access = candidate_indices.accessor<std::int64_t, 2>();
			auto advantage_access = advantage_targets.accessor<float, 2>();
			auto candidate_count_access = candidate_counts.accessor<std::int64_t, 1>();
			for (std::size_t index = begin; index < end; ++index) {
				const auto &record = records[order[index]];
				const auto row = static_cast<std::int64_t>(index - begin);
				count_access[row] = record.legal_indices.size();
				value_access[row] = record.value_target;
				policy_weight_access[row] = record.policy_weight;
				value_weight_access[row] = record.value_weight;
				candidate_count_access[row] = record.candidate_indices.size();
				for (std::size_t column = 0; column < record.legal_indices.size(); ++column) {
					legal_access[row][column] = record.legal_indices[column];
					target_access[row][column] = record.policy_target[column];
				}
				for (std::size_t column = 0; column < record.candidate_indices.size(); ++column) {
					candidate_access[row][column] = record.candidate_indices[column];
					advantage_access[row][column] = record.advantage_target[column];
					std::copy(record.candidate_next_states[column].begin(),
							  record.candidate_next_states[column].end(),
							  packed_next.begin() +
								  (row * candidate_width + column) * kStateFeatures);
				}
			}
			auto next_states =
				decode_states(packed_next.data(),
							  batch * static_cast<std::int64_t>(candidate_width), pin_memory)
					.to(device, true);
			legal = legal.to(device, true);
			targets = targets.to(device, true);
			counts = counts.to(device, true);
			values = values.to(device, true);
			policy_weights = policy_weights.to(device, true);
			value_weights = value_weights.to(device, true);
			candidate_indices = candidate_indices.to(device, true);
			advantage_targets = advantage_targets.to(device, true);
			candidate_counts = candidate_counts.to(device, true);

			optimizer.zero_grad();
			torch::Tensor tokens;
			torch::Tensor predicted_next;
			torch::Tensor target_next;
			ModelOutput output;
			ModelOutput imagined;
			{
				AutocastGuard autocast(options.precision, device);
				tokens = model->encode(states);
				output = model->predict(tokens);
				auto repeated_tokens =
					tokens.unsqueeze(1)
						.expand({batch, static_cast<std::int64_t>(candidate_width), kTokenCount,
								 model->channels()})
						.reshape({batch * static_cast<std::int64_t>(candidate_width), kTokenCount,
								  model->channels()});
				predicted_next =
					model->transition(repeated_tokens, candidate_indices.reshape({-1}));
				{
					torch::NoGradGuard no_grad;
					target_next = model->encode(next_states).detach();
				}
				imagined = model->predict(predicted_next);
			}
			auto selected = output.policy.to(torch::kFloat32).gather(1, legal);
			auto columns = torch::arange(static_cast<std::int64_t>(width), counts.options());
			auto mask = columns.unsqueeze(0) < counts.unsqueeze(1);
			selected = selected.masked_fill(~mask, -1e9);
			auto log_probability = torch::log_softmax(selected, 1);
			auto masked_targets = targets * mask;
			masked_targets = masked_targets / masked_targets.sum(1, true).clamp_min(1e-8);
			auto policy_errors = -(masked_targets * log_probability).sum(1);
			auto policy_loss =
				(policy_errors * policy_weights).sum() / policy_weights.sum().clamp_min(1e-8);
			auto predicted_values = output.value.squeeze(1).to(torch::kFloat32);
			auto value_errors = torch::nn::functional::smooth_l1_loss(
				predicted_values, values,
				torch::nn::functional::SmoothL1LossFuncOptions().reduction(torch::kNone));
			auto value_loss =
				(value_errors * value_weights).sum() / value_weights.sum().clamp_min(1.0);
			auto selected_advantages =
				output.advantages.to(torch::kFloat32).gather(1, candidate_indices);
			auto predicted_q = torch::clamp(predicted_values.unsqueeze(1) + selected_advantages,
										 -1.0, 1.0);
			auto target_q = torch::clamp(values.unsqueeze(1) + advantage_targets, -1.0, 1.0);
			auto candidate_columns = torch::arange(
				static_cast<std::int64_t>(candidate_width), candidate_counts.options());
			auto candidate_mask = candidate_columns.unsqueeze(0) < candidate_counts.unsqueeze(1);
			auto weighted_candidate_mask =
				candidate_mask * policy_weights.unsqueeze(1);
			auto q_errors = torch::nn::functional::smooth_l1_loss(
				predicted_q, target_q,
				torch::nn::functional::SmoothL1LossFuncOptions().reduction(torch::kNone));
			auto dueling_q_loss =
				(q_errors * weighted_candidate_mask).sum() /
				weighted_candidate_mask.sum().clamp_min(1e-8);
			// Exact successor states teach the action-conditioned latent world step.
			auto predicted_next_fp32 = predicted_next.to(torch::kFloat32);
			auto target_next_fp32 = target_next.to(torch::kFloat32);
			auto predicted_unit = predicted_next_fp32 /
				predicted_next_fp32.square().sum(-1, true).sqrt().clamp_min(1e-8);
			auto target_unit = target_next_fp32 /
				target_next_fp32.square().sum(-1, true).sqrt().clamp_min(1e-8);
			auto dynamics_errors =
				(1.0 - (predicted_unit * target_unit).sum(-1).mean(-1))
					.reshape({batch, static_cast<std::int64_t>(candidate_width)});
			auto dynamics_loss =
				(dynamics_errors * weighted_candidate_mask).sum() /
				weighted_candidate_mask.sum().clamp_min(1e-8);
			auto imagined_q =
				-imagined.value.to(torch::kFloat32).squeeze(1).reshape(
					{batch, static_cast<std::int64_t>(candidate_width)});
			auto imagined_errors = torch::nn::functional::smooth_l1_loss(
				imagined_q, target_q,
				torch::nn::functional::SmoothL1LossFuncOptions().reduction(torch::kNone));
			auto imagined_value_loss =
				(imagined_errors * weighted_candidate_mask).sum() /
				weighted_candidate_mask.sum().clamp_min(1e-8);
			auto loss = options.policy_weight * policy_loss + options.value_weight * value_loss +
						options.dueling_q_weight * dueling_q_loss +
						options.dynamics_weight * dynamics_loss +
						options.imagined_value_weight * imagined_value_loss;
			loss.backward();
			if (options.grad_clip > 0.0) {
				torch::nn::utils::clip_grad_norm_(model->parameters(), options.grad_clip);
			}
			optimizer.step();
			++steps;
			metric_totals.add_(torch::stack(
				{loss.detach(), policy_loss.detach(), value_loss.detach(), dueling_q_loss.detach(),
				 dynamics_loss.detach(), imagined_value_loss.detach()}));
			if (options.log_every > 0 && (steps == 1 || steps % options.log_every == 0)) {
				auto metrics =
					torch::stack({policy_loss.detach(), value_loss.detach(),
								  dueling_q_loss.detach(), dynamics_loss.detach(),
								  imagined_value_loss.detach(), loss.detach()})
						.to(torch::kCPU)
						.contiguous();
				auto metric_values = metrics.accessor<float, 1>();
				std::cout << "fcpi train: step=" << steps
						  << " policy=" << metric_values[0]
						  << " value=" << metric_values[1]
						  << " dueling_q=" << metric_values[2]
						  << " dynamics=" << metric_values[3]
						  << " imagined_value=" << metric_values[4]
						  << " loss=" << metric_values[5] << std::endl;
			}
			if (options.train_max_steps > 0 && steps >= options.train_max_steps) {
				break;
			}
		}
		if (options.train_max_steps > 0 && steps >= options.train_max_steps) {
			break;
		}
	}
	ArchitectureInfo source_arch;
	load_checkpoint(source, torch::Device(torch::kCPU), &source_arch);
	save_checkpoint_atomic(candidate, model, {source_arch.channels, source_arch.blocks});
	const double divisor = static_cast<double>(std::max<std::int64_t>(1, steps));
	auto final_metrics = metric_totals.to(torch::kCPU).contiguous();
	auto metric_values = final_metrics.accessor<float, 1>();
	return {
		{"steps", steps},
		{"epochs_requested", options.epochs},
		{"candidate", candidate.string()},
		{"precision", compute_precision_name(options.precision)},
		{"metrics",
		 {
			 {"loss", metric_values[0] / divisor},
			 {"policy", metric_values[1] / divisor},
			 {"value", metric_values[2] / divisor},
			 {"dueling_q", metric_values[3] / divisor},
			 {"dynamics", metric_values[4] / divisor},
			 {"imagined_value", metric_values[5] / divisor},
		 }},
	};
}

} // namespace

// Create an isolated run, iterate generation/training/arena, and atomically promote accepted models.
void run_fcpi(const FcpiOptions &options) {
	if (options.iterations <= 0 || options.games_per_iter <= 0 || options.games_in_flight <= 0) {
		throw std::invalid_argument("FCPI iteration and game counts must be positive");
	}
	if (!std::filesystem::is_regular_file(options.model)) {
		throw std::runtime_error("FCPI model not found: " + options.model.string());
	}
	const auto run_id = create_run_id("fcpi");
	const auto data_dir = std::filesystem::path("data/runs") / run_id;
	const auto model_dir = std::filesystem::path("models/runs") / run_id;
	if (std::filesystem::exists(data_dir) || std::filesystem::exists(model_dir)) {
		throw std::runtime_error("FCPI run directory already exists: " + run_id);
	}
	std::filesystem::create_directories(data_dir);
	std::filesystem::create_directories(model_dir);
	const auto initial = model_dir / "initial.pth";
	const auto current = model_dir / "current.pth";
	atomic_copy(options.model, initial);
	atomic_copy(initial, current);
	const auto device = resolve_device(options.device);
	validate_compute_precision(options.precision, device);
	std::cout << "fcpi run id: " << run_id << std::endl;
	std::cout << "fcpi architecture: " << kArchType << std::endl;
	std::cout << "fcpi formula: " << kFcpiFormula << std::endl;
	std::cout << "fcpi precision: " << compute_precision_name(options.precision) << std::endl;
	std::cout << "fcpi current model: " << current.string() << std::endl;
	nlohmann::json summaries = nlohmann::json::array();

	for (int iteration = 1; iteration <= options.iterations; ++iteration) {
		std::cout << "fcpi iteration " << iteration << std::endl;
		auto model = load_checkpoint(current, device);
		nlohmann::json sampling;
		auto records = collect_selfplay(model, device, options, iteration, sampling);
		TargetSummary target_summary;
		construct_targets(records, model, device, options, target_summary);
		const auto data_path = data_dir / ("fcpi_iter_" + zero_pad(iteration, 3) + ".h5");
		auto data_summary = write_fcpi_h5(data_path, records);
		data_summary["sampling"] = sampling;
		data_summary["counterfactual"] = {
			{"budget_per_root", options.counterfactual_budget},
			{"trees", target_summary.trees},
			{"decision_nodes", target_summary.decision_nodes},
			{"evaluated_edges", target_summary.evaluated_edges},
			{"terminal_edges", target_summary.terminal_edges},
			{"max_depth", target_summary.max_depth},
		};
		const auto candidate = model_dir / ("candidate_iter_" + zero_pad(iteration, 3) + ".pth");
		auto train_summary = train_candidate(current, candidate, model, device, records, options);
		auto arena_options = options.arena;
		arena_options.candidate = candidate;
		arena_options.baseline = current;
		arena_options.device = options.device;
		arena_options.seed = options.seed + iteration;
		arena_options.search.precision = options.precision;
		auto arena_summary = evaluate_models(arena_options);
		const bool accepted = arena_summary["accepted"].get<bool>();
		if (accepted) {
			atomic_copy(candidate, current);
			std::cout << "fcpi promoted: " << current.string() << std::endl;
		} else {
			std::cout << "fcpi candidate rejected: " << candidate.string() << std::endl;
		}
		summaries.push_back({
			{"iteration", iteration},
			{"architecture", kArchType},
			{"formula", kFcpiFormula},
			{"precision", compute_precision_name(options.precision)},
			{"data", data_summary},
			{"train", train_summary},
			{"arena", arena_summary},
			{"accepted", accepted},
		});
		std::ofstream summary_file(data_dir / "summary.json");
		summary_file << nlohmann::json({
										   {"run_id", run_id},
										   {"initial_model", initial.string()},
										   {"current_model", current.string()},
										   {"summaries", summaries},
									   })
							.dump(2);
	}
}

} // namespace melano
