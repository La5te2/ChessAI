// Implements Gadus FCPI: batched self-play, adaptive counterfactual traces, and arena gating.

#include "gadus/fcpi.hpp"
#include <hdf5.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
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
#include "gadus/args.hpp"
#include "gadus/checkpoint.hpp"

namespace gadus {

namespace {

inline constexpr const char *kFcpiFormula = "counterfactual-tree";
inline constexpr double kWeightDecay = 1e-4;
inline constexpr double kGradientClip = 1.0;

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
	int behavior_action = -1;
	std::vector<int> legal_indices;
	std::vector<float> legal_prior;
	float mc_value_target = 0.0F;
	float tree_value_target = 0.0F;
	std::vector<float> policy_target;
	std::vector<float> mc_policy_advantage_sums;
	std::vector<float> mc_policy_weights;
	float policy_weight = 1.0F;
	float mc_value_weight = 0.0F;
	float tree_value_weight = 0.0F;
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
	std::vector<int> candidate_indices;
	std::vector<std::size_t> candidate_slots;
	std::vector<int> children;
	std::vector<float> policy_target;
	int parent = -1;
	int parent_action = -1;
	int depth = 0;
	double reach_probability = 1.0;
	double priority = 1.0;
	bool terminal = false;
	bool expanded = false;
};

struct FrontierEntry {
	double priority = 0.0;
	int node = 0;
};

struct FrontierEarlier {
	bool operator()(const FrontierEntry &left, const FrontierEntry &right) const {
		if (left.priority != right.priority) {
			return left.priority < right.priority;
		}
		return left.node > right.node;
	}
};

struct CounterfactualTree {
	std::size_t root_record = 0;
	std::vector<TreeNode> nodes;
	std::priority_queue<FrontierEntry, std::vector<FrontierEntry>, FrontierEarlier> frontier;
	int remaining_budget = 0;
	int evaluated_edges = 0;
};

struct BehaviorCoverage {
	std::int64_t visits = 0;
	std::unordered_map<int, std::int64_t> action_visits;
};

struct TargetSummary {
	std::int64_t trees = 0;
	std::int64_t decision_nodes = 0;
	std::int64_t evaluated_edges = 0;
	std::int64_t terminal_edges = 0;
	int max_depth = 0;
	double residual_sum = 0.0;
	std::int64_t residual_count = 0;
	double tree_value_weight_sum = 0.0;
	double tree_value_correction_sum = 0.0;
	double evaluated_policy_mass_sum = 0.0;
	double policy_total_variation_sum = 0.0;
	std::int64_t policy_top1_changes = 0;
	std::int64_t policy_diagnostic_nodes = 0;
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

// Hash model-visible bytes directly so large FCPI maps avoid allocating one string per lookup.
struct PackedStateHash {
	std::size_t operator()(const PackedState &state) const noexcept {
		std::size_t hash = sizeof(std::size_t) == 8 ? 1469598103934665603ULL : 2166136261U;
		const std::size_t prime = sizeof(std::size_t) == 8 ? 1099511628211ULL : 16777619U;
		for (const auto byte : state) {
			hash = (hash ^ static_cast<std::size_t>(byte)) * prime;
		}
		return hash;
	}
};

// Allocate repeated state visits by probability deficit. For a fixed Policy,
// every action with positive probability receives a finite quota instead of
// depending on an unbounded sequence of independent categorical samples.
std::size_t choose_behavior_action(const PackedState &state_key,
								   const std::vector<int> &legal,
								   const std::vector<float> &behavior,
								   std::unordered_map<PackedState, BehaviorCoverage, PackedStateHash> &coverage) {
	if (legal.empty() || legal.size() != behavior.size()) {
		throw std::invalid_argument("behavior selection requires aligned legal probabilities");
	}
	auto &state = coverage[state_key];
	const double next_visit = static_cast<double>(state.visits + 1);
	std::size_t selected = 0;
	double largest_deficit = -std::numeric_limits<double>::infinity();
	for (std::size_t index = 0; index < legal.size(); ++index) {
		const auto found = state.action_visits.find(legal[index]);
		const double realized =
			found == state.action_visits.end() ? 0.0 : static_cast<double>(found->second);
		const double deficit = next_visit * static_cast<double>(behavior[index]) - realized;
		if (deficit > largest_deficit) {
			largest_deficit = deficit;
			selected = index;
		}
	}
	++state.visits;
	++state.action_visits[legal[selected]];
	return selected;
}

// Build a shuffled mixture of startpos and opening-book starts for one iteration.
std::vector<SamplingSpec> make_sampling_specs(const FcpiOptions &options, int iteration,
											  nlohmann::json &summary) {
	std::mt19937_64 rng(options.seed + iteration);
	const int games = std::max(1, options.games_per_iter);
	std::vector<SamplingSpec> specs;
	specs.reserve(games);
	std::vector<std::string> positions;
	if (options.opening_book.empty()) {
		for (int index = 0; index < games; ++index) {
			specs.push_back({std::string(chess::constants::STARTPOS)});
		}
	} else {
		positions = load_reachable_opening_positions(
			options.opening_book, options.max_book_positions, options.seed + iteration);
		if (static_cast<int>(positions.size()) < games) {
			throw std::runtime_error(
				"FCPI sampling requires enough unique reachable opening states: required=" +
				std::to_string(games) + " available=" + std::to_string(positions.size()));
		}
		std::shuffle(positions.begin(), positions.end(), rng);
		for (int index = 0; index < games; ++index) {
			specs.push_back({positions[static_cast<std::size_t>(index)]});
		}
	}
	std::shuffle(specs.begin(), specs.end(), rng);
	summary = {
		{"games", games},
		{"opening_book", options.opening_book},
		{"available_unique_positions", positions.empty() ? 1 : positions.size()},
		{"selected_unique_positions", options.opening_book.empty() ? 1 : specs.size()},
	};
	return specs;
}

// Evaluate arbitrarily many positions through bounded neural batches.
std::vector<ClosedEvaluation> evaluate_chunks(Searcher &searcher,
											 const std::vector<chess::Board> &boards, int batch_size) {
	std::unordered_map<PackedState, std::size_t, PackedStateHash> unique_rows;
	std::vector<chess::Board> unique_boards;
	std::vector<std::size_t> source_rows;
	unique_boards.reserve(boards.size());
	source_rows.reserve(boards.size());
	for (const auto &board : boards) {
		const auto state = encode_state(board);
		const auto [found, inserted] = unique_rows.emplace(state, unique_boards.size());
		if (inserted) {
			unique_boards.push_back(board);
		}
		source_rows.push_back(found->second);
	}

	std::vector<ClosedEvaluation> unique_output;
	unique_output.reserve(unique_boards.size());
	for (std::size_t begin = 0; begin < unique_boards.size(); begin += std::max(1, batch_size)) {
		const auto end = std::min(unique_boards.size(), begin + std::max(1, batch_size));
		std::vector<chess::Board> chunk(unique_boards.begin() + begin,
										   unique_boards.begin() + end);
		auto results = searcher.evaluate_closed_many(chunk);
		unique_output.insert(unique_output.end(), std::make_move_iterator(results.begin()),
							 std::make_move_iterator(results.end()));
	}
	std::vector<ClosedEvaluation> output;
	output.reserve(boards.size());
	for (const auto row : source_rows) {
		output.push_back(unique_output[row]);
	}
	return output;
}

// Keep Policy top-1 and an optional required action, then fill the local expansion
// with Gumbel top-k proposals so low-prior alternatives retain a budget-dependent chance.
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
	proposals.reserve(legal.size());
	for (std::size_t index = 0; index < legal.size(); ++index) {
		if (std::find(selected.begin(), selected.end(), legal[index]) != selected.end()) {
			continue;
		}
		const double gumbel = -std::log(-std::log(uniform(rng)));
		const double key =
			std::log(std::clamp(static_cast<double>(scores[index]), 1e-12, 1.0)) + gumbel;
		proposals.emplace_back(key, legal[index]);
	}
	std::stable_sort(proposals.begin(), proposals.end(),
					 [](const auto &left, const auto &right) {
						 return left.first > right.first;
					 });
	for (const auto &[key, action] : proposals) {
		(void)key;
		if (selected.size() >= count) {
			break;
		}
		selected.push_back(action);
	}
	return selected;
}

// Generate closed-policy current-model self-play from unique states in the sampling book.
std::vector<Position> collect_selfplay(Model model, const torch::Device &device,
									   const FcpiOptions &options, int iteration,
									   nlohmann::json &sampling_summary) {
	SearchOptions closed;
	closed.type = SearchType::Closed;
	closed.precision = options.precision;
	closed.cpu_threads = options.cpu_threads;
	closed.mcts_sims = 0;
	closed.mcts_batch_size = options.inference_batch_size;
	Searcher evaluator(model, device, closed);
	nlohmann::json starts;
	const auto specs = make_sampling_specs(options, iteration, starts);
	std::unordered_map<PackedState, BehaviorCoverage, PackedStateHash> behavior_coverage;
	std::vector<Trajectory> trajectories;
	trajectories.reserve(specs.size());
	int completed = 0;
	std::cout << "fcpi self-play start: iteration=" << iteration
			  << " arch_type=" << kArchType
			  << " games=" << specs.size() << " max_plies=" << options.max_plies
			  << " device=" << device.str() << std::endl;
	std::cout << "fcpi starts: " << starts.dump() << std::endl;

	const std::size_t slot_count =
		std::min(specs.size(), static_cast<std::size_t>(std::max(1, options.games_in_flight)));
	std::vector<Trajectory> slots(slot_count);
	std::vector<bool> active(slot_count, false);
	std::size_t next_spec = 0;
	const auto fill_slot = [&](std::size_t slot) {
		Trajectory trajectory;
		trajectory.game_id = static_cast<int>(next_spec) + 1;
		trajectory.board = chess::Board(specs[next_spec].fen);
		slots[slot] = std::move(trajectory);
		active[slot] = true;
		++next_spec;
	};
	for (std::size_t slot = 0; slot < slot_count; ++slot) {
		fill_slot(slot);
	}
	while (completed < static_cast<int>(specs.size())) {
			std::vector<std::size_t> active_indices;
			std::vector<chess::Board> active_boards;
			for (std::size_t index = 0; index < slots.size(); ++index) {
				if (!active[index]) {
					continue;
				}
				active_indices.push_back(index);
				active_boards.push_back(slots[index].board);
			}

			const auto results =
				evaluate_chunks(evaluator, active_boards, options.inference_batch_size);
			for (std::size_t row = 0; row < active_indices.size(); ++row) {
				const std::size_t slot = active_indices[row];
				auto &trajectory = slots[slot];
				auto &board = trajectory.board;
				std::vector<int> legal = results[row].legal_indices;
				std::vector<float> prior = results[row].legal_policy;
				prior = normalize(std::move(prior));
				const double temperature = std::max(1e-4, options.behavior_temperature);
				std::vector<float> behavior(prior.size());
				for (std::size_t index = 0; index < prior.size(); ++index) {
					behavior[index] = static_cast<float>(std::pow(
						std::clamp(static_cast<double>(prior[index]), 1e-12, 1.0),
						1.0 / temperature));
				}
				behavior = normalize(std::move(behavior));
				const auto state = encode_state(board);
				const std::size_t choice =
					choose_behavior_action(state, legal, behavior, behavior_coverage);
				Position position;
				position.game_id = trajectory.game_id;
				position.state = state;
				position.fen = board.getFen();
				position.root_value = results[row].value;
				position.behavior_action = legal[choice];
				position.legal_indices = legal;
				position.legal_prior = prior;
				position.mc_policy_advantage_sums.assign(legal.size(), 0.0F);
				position.mc_policy_weights.assign(legal.size(), 0.0F);
				trajectory.positions.push_back(std::move(position));
				const auto move = index_to_move(legal[choice], board);
				if (move.move() == chess::Move::NO_MOVE) {
					throw std::runtime_error("FCPI selected an illegal behavior action");
				}
				board.makeMove(move);
				const bool terminal = game_is_over(board);
				const bool truncated =
					static_cast<int>(trajectory.positions.size()) >= options.max_plies;
				if (terminal || truncated) {
					++completed;
					if (completed == 1 || completed == static_cast<int>(specs.size()) ||
						(options.log_every > 0 && completed % options.log_every == 0)) {
						std::cout << "fcpi game: completed=" << completed << '/' << specs.size()
								  << " game_id=" << trajectory.game_id
								  << " plies=" << trajectory.positions.size()
								  << " result="
								  << (terminal ? game_result(board) : "truncated")
								  << " truncated="
								  << (truncated && !terminal ? "true" : "false")
								  << std::endl;
					}
					trajectories.push_back(std::move(trajectory));
					if (next_spec < specs.size()) {
						fill_slot(slot);
					} else {
						active[slot] = false;
					}
				}
			}
		}

	int terminal_games = 0;
	int truncated_games = 0;
	std::int64_t completed_positions = 0;
	std::int64_t truncated_positions = 0;
	for (auto &trajectory : trajectories) {
		const bool completed_game = game_is_over(trajectory.board);
		if (completed_game) {
			++terminal_games;
			completed_positions += trajectory.positions.size();
			float next_return = terminal_value_side_to_move(trajectory.board);
			for (int index = static_cast<int>(trajectory.positions.size()) - 1; index >= 0;
				 --index) {
				const float current_return = -next_return;
				trajectory.positions[index].mc_value_target =
					std::clamp(current_return, -1.0F, 1.0F);
				trajectory.positions[index].mc_value_weight = 1.0F;
				auto &position = trajectory.positions[index];
				const auto behavior = std::find(position.legal_indices.begin(),
												position.legal_indices.end(), position.behavior_action);
				if (behavior == position.legal_indices.end()) {
					throw std::runtime_error("FCPI behavior action is absent from legal actions");
				}
				const auto column = static_cast<std::size_t>(behavior - position.legal_indices.begin());
				// G-V_old is an action-specific factual residual, not a replacement Q value.
				// Division by two maps the natural [-2, 2] range to [-1, 1].
				position.mc_policy_advantage_sums[column] =
					std::clamp(0.5F * (current_return - position.root_value), -1.0F, 1.0F);
				position.mc_policy_weights[column] = 1.0F;
				next_return = current_return;
			}
		} else {
			++truncated_games;
			truncated_positions += trajectory.positions.size();
		}
	}

	std::vector<Position> records;
	std::int64_t source_positions = 0;
	std::int64_t unique_positions = 0;
	for (auto &trajectory : trajectories) {
		source_positions += trajectory.positions.size();
		std::unordered_set<PackedState, PackedStateHash> seen;
		std::vector<std::size_t> indices;
		for (std::size_t index = 0; index < trajectory.positions.size(); ++index) {
			if (seen.insert(trajectory.positions[index].state).second) {
				indices.push_back(index);
			}
		}
		unique_positions += indices.size();
		for (const auto index : indices) {
			records.push_back(std::move(trajectory.positions[index]));
		}
	}
	sampling_summary = {
		{"games", trajectories.size()},
		{"source_positions", source_positions},
		{"unique_positions", unique_positions},
		{"selected_positions", records.size()},
		{"completed_games", terminal_games},
		{"truncated_games", truncated_games},
		{"completed_positions", completed_positions},
		{"truncated_positions", truncated_positions},
		{"starts", starts},
	};
	std::cout << "fcpi position sampling: " << sampling_summary.dump() << std::endl;
	return records;
}

// Materialize one non-terminal decision node from an exact board and a frozen-model evaluation.
TreeNode make_tree_node(const chess::Board &board, const ClosedEvaluation &evaluation, int parent,
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
	node.legal_indices = evaluation.legal_indices;
	node.legal_prior = evaluation.legal_policy;
	node.legal_prior = normalize(std::move(node.legal_prior));
	return node;
}

// Select the unexpanded node with the largest reach-weighted Bellman residual.
int select_tree_frontier(CounterfactualTree &tree) {
	while (!tree.frontier.empty()) {
		const int selected = tree.frontier.top().node;
		tree.frontier.pop();
		const auto &node = tree.nodes[static_cast<std::size_t>(selected)];
		if (!node.terminal && !node.expanded && !node.legal_indices.empty()) {
			return selected;
		}
	}
	return -1;
}

// Derive local tree width from the remaining global budget. This couples width and
// depth without separate top-k or ply controls.
int expansion_width(const CounterfactualTree &tree) {
	if (tree.remaining_budget <= 0) {
		return 0;
	}
	const int progressive =
		std::max(2, static_cast<int>(std::ceil(std::sqrt(tree.remaining_budget))));
	return std::min(tree.remaining_budget, progressive);
}

// Solve the unit-temperature KL-regularized local improvement:
// pi+(a) proportional to prior(a) * exp(Q(a) - E_prior[Q]).
std::vector<float> improve_policy(const std::vector<float> &prior,
								  const std::vector<float> &action_values) {
	if (prior.size() != action_values.size() || prior.empty()) {
		throw std::invalid_argument("tree policy improvement requires aligned non-empty inputs");
	}
	double mean = 0.0;
	for (std::size_t index = 0; index < prior.size(); ++index) {
		mean += prior[index] * action_values[index];
	}
	std::vector<double> logits(prior.size());
	for (std::size_t index = 0; index < prior.size(); ++index) {
		logits[index] = std::log(std::clamp(static_cast<double>(prior[index]), 1e-12, 1.0)) +
						(action_values[index] - mean);
	}
	return stable_softmax(logits);
}

// Summarize how often counterfactual planning changes the frozen Policy.
void record_policy_diagnostics(TargetSummary &summary, const std::vector<float> &prior,
							   const std::vector<float> &target) {
	double l1 = 0.0;
	for (std::size_t index = 0; index < prior.size(); ++index) {
		const double target_probability =
			std::clamp(static_cast<double>(target[index]), 1e-12, 1.0);
		const double prior_probability =
			std::clamp(static_cast<double>(prior[index]), 1e-12, 1.0);
		l1 += std::abs(target_probability - prior_probability);
	}
	const auto prior_top1 = std::max_element(prior.begin(), prior.end()) - prior.begin();
	const auto target_top1 = std::max_element(target.begin(), target.end()) - target.begin();
	summary.policy_total_variation_sum += 0.5 * l1;
	summary.policy_top1_changes += prior_top1 != target_top1 ? 1 : 0;
	++summary.policy_diagnostic_nodes;
}

// Measure how much of the improved local policy is backed by explicitly evaluated
// tree edges. Unexpanded actions retain the frozen node value, so their probability
// mass must not increase confidence in the counterfactual Value target.
float evaluated_policy_mass(const TreeNode &node) {
	float mass = 0.0F;
	for (const auto slot : node.candidate_slots) {
		mass += node.policy_target[slot];
	}
	return std::clamp(mass, 0.0F, 1.0F);
}

// Build counterfactual trees with exhaustive root actions and a bounded deeper
// frontier. Every non-terminal action uses the same frozen-model estimator;
// completed trajectories remain separate Monte Carlo targets for Value evaluation.
void construct_targets(std::vector<Position> &records, Model model, const torch::Device &device,
					   const FcpiOptions &options, TargetSummary &summary) {
	if (options.counterfactual_budget < 0) {
		throw std::invalid_argument("counterfactual-budget cannot be negative");
	}
	SearchOptions closed;
	closed.type = SearchType::Closed;
	closed.precision = options.precision;
	closed.cpu_threads = options.cpu_threads;
	closed.mcts_sims = 0;
	closed.mcts_batch_size = options.inference_batch_size;
	Searcher evaluator(model, device, closed);
	std::mt19937_64 rng(options.seed + 3'000'017);
	std::vector<Position> tree_records;
	tree_records.reserve(records.size() * 2);
	std::cout << "fcpi counterfactual tree start: positions=" << records.size()
			  << " deep_budget_per_root=" << options.counterfactual_budget << std::endl;

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
			tree.nodes.push_back(std::move(root));
			tree.frontier.push({tree.nodes.front().priority, 0});
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
				const bool root_pending = !tree.nodes.front().expanded;
				if (!root_pending && tree.remaining_budget <= 0) {
					continue;
				}
				const int node_index = select_tree_frontier(tree);
				if (node_index < 0) {
					continue;
				}
				auto &node = tree.nodes[static_cast<std::size_t>(node_index)];
				if (node_index == 0) {
					// Root coverage is exhaustive. The depth budget starts after these
					// one-ply action evaluations, so no legal root move can vanish behind top-k.
					node.candidate_indices = node.legal_indices;
					node.candidate_slots.resize(node.legal_indices.size());
					std::iota(node.candidate_slots.begin(), node.candidate_slots.end(), 0);
				} else {
					const int width = std::min<int>(
						expansion_width(tree), static_cast<int>(node.legal_indices.size()));
					node.candidate_indices =
						choose_candidates(node.legal_indices, node.legal_prior, -1, width, rng);
					node.candidate_slots.reserve(node.candidate_indices.size());
					for (const int action : node.candidate_indices) {
						const auto legal =
							std::find(node.legal_indices.begin(), node.legal_indices.end(), action);
						if (legal == node.legal_indices.end()) {
							throw std::runtime_error("tree candidate is absent from legal actions");
						}
						node.candidate_slots.push_back(
							static_cast<std::size_t>(legal - node.legal_indices.begin()));
					}
					tree.remaining_budget -= static_cast<int>(node.candidate_indices.size());
				}
				node.children.assign(node.candidate_indices.size(), -1);
				node.expanded = true;
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
				const float edge_prior = parent.legal_prior[parent.candidate_slots[request.slot]];
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
				if (!tree.nodes.back().terminal && !tree.nodes.back().legal_indices.empty()) {
					tree.frontier.push({tree.nodes.back().priority, child_index});
				}
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
				std::vector<float> action_values(node.legal_indices.size(), node.value);
				for (std::size_t slot = 0; slot < node.candidate_indices.size(); ++slot) {
					const auto &child =
						tree.nodes[static_cast<std::size_t>(node.children[slot])];
					const auto legal_index = node.candidate_slots[slot];
					action_values[legal_index] = -child.backed_value;
				}

				// Evaluated actions use the backed child conclusion. Unexpanded actions
				// keep V_old(s), so limited tree coverage automatically shrinks both the
				// Policy movement and the counterfactual Value residual.
				node.policy_target = improve_policy(node.legal_prior, action_values);
				node.backed_value = 0.0F;
				for (std::size_t index = 0; index < action_values.size(); ++index) {
					node.backed_value += node.policy_target[index] * action_values[index];
				}
				node.backed_value = std::clamp(node.backed_value, -1.0F, 1.0F);
				record_policy_diagnostics(summary, node.legal_prior, node.policy_target);
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
				record.policy_target = node.policy_target;
				record.mc_policy_advantage_sums.assign(node.legal_indices.size(), 0.0F);
				record.mc_policy_weights.assign(node.legal_indices.size(), 0.0F);
				const float budget_weight =
					static_cast<float>(node.candidate_indices.size()) / edge_total;
				record.policy_weight = budget_weight;
				record.mc_value_target =
					node_index == 0 ? root.mc_value_target : 0.0F;
				record.mc_value_weight =
					node_index == 0 ? root.mc_value_weight : 0.0F;
				if (node_index == 0) {
					if (root.mc_policy_advantage_sums.size() != node.legal_indices.size() ||
						root.mc_policy_weights.size() != node.legal_indices.size()) {
						throw std::runtime_error("FCPI Monte Carlo Policy target width changed");
					}
					record.mc_policy_advantage_sums = root.mc_policy_advantage_sums;
					record.mc_policy_weights = root.mc_policy_weights;
				}

				// The target itself is coverage-aware because every unexpanded action
				// contributes V_old(s). The budget weight only distributes one unit of
				// tree supervision across the decision nodes produced by this root.
				record.tree_value_target = node.backed_value;
				const float coverage = evaluated_policy_mass(node);
				record.tree_value_weight = budget_weight;
				summary.tree_value_weight_sum += record.tree_value_weight;
				summary.tree_value_correction_sum +=
					record.tree_value_weight *
					std::abs(static_cast<double>(node.backed_value - node.value));
				summary.evaluated_policy_mass_sum += coverage;
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
	const double maximum_tree_value_weight =
		static_cast<double>(summary.trees) +
		1e-5 * static_cast<double>(std::max<std::int64_t>(1, summary.trees));
	if (summary.tree_value_weight_sum > maximum_tree_value_weight) {
		throw std::runtime_error(
			"counterfactual Value weight exceeded one unit per tree");
	}
	std::cout << "fcpi counterfactual summary: trees=" << summary.trees
			  << " decision_nodes=" << summary.decision_nodes
			  << " evaluated_edges=" << summary.evaluated_edges
			  << " terminal_edges=" << summary.terminal_edges
			  << " max_depth=" << summary.max_depth << " mean_residual="
			  << (summary.residual_count > 0 ? summary.residual_sum / summary.residual_count : 0.0)
			  << " mean_coverage="
			  << (summary.decision_nodes > 0
					  ? summary.evaluated_policy_mass_sum / summary.decision_nodes
					  : 0.0)
			  << " mean_value_correction="
			  << (summary.tree_value_weight_sum > 0.0
					  ? summary.tree_value_correction_sum / summary.tree_value_weight_sum
					  : 0.0)
			  << " mean_policy_shift="
			  << (summary.policy_diagnostic_nodes > 0
					  ? summary.policy_total_variation_sum /
							summary.policy_diagnostic_nodes
					  : 0.0)
			  << " top1_change_rate="
			  << (summary.policy_diagnostic_nodes > 0
					  ? static_cast<double>(summary.policy_top1_changes) /
							summary.policy_diagnostic_nodes
					  : 0.0)
			  << std::endl;
}

// Merge model-indistinguishable states and average their stochastic targets and priors.
std::vector<Position> aggregate_records(std::vector<Position> records, nlohmann::json &summary) {
	const std::size_t source_count = records.size();
	std::unordered_map<PackedState, std::size_t, PackedStateHash> groups;
	std::vector<Position> output;
	for (auto &record : records) {
		const auto found = groups.find(record.state);
		if (found == groups.end()) {
			groups.emplace(record.state, output.size());
			output.push_back(std::move(record));
			continue;
		}
		auto &merged = output[found->second];
		if (merged.legal_indices != record.legal_indices) {
			throw std::runtime_error("identical encoded states produced different legal actions");
		}
		if (merged.mc_policy_advantage_sums.size() != record.mc_policy_advantage_sums.size() ||
			merged.mc_policy_weights.size() != record.mc_policy_weights.size() ||
			merged.mc_policy_advantage_sums.size() != merged.legal_indices.size()) {
			throw std::runtime_error("identical encoded states produced incompatible MC Policy data");
		}
		const float old_policy_weight = merged.policy_weight;
		const float new_policy_weight = old_policy_weight + record.policy_weight;
		const float old_mc_value_weight = merged.mc_value_weight;
		const float new_mc_value_weight = old_mc_value_weight + record.mc_value_weight;
		const float old_tree_value_weight = merged.tree_value_weight;
		const float new_tree_value_weight =
			old_tree_value_weight + record.tree_value_weight;
		for (std::size_t index = 0; index < merged.legal_prior.size(); ++index) {
			if (new_policy_weight > 0.0F) {
				merged.policy_target[index] =
					(merged.policy_target[index] * old_policy_weight +
					 record.policy_target[index] * record.policy_weight) /
					new_policy_weight;
			}
			merged.mc_policy_advantage_sums[index] += record.mc_policy_advantage_sums[index];
			merged.mc_policy_weights[index] += record.mc_policy_weights[index];
		}
		if (new_mc_value_weight > 0.0F) {
			merged.mc_value_target =
				(merged.mc_value_target * old_mc_value_weight +
				 record.mc_value_target * record.mc_value_weight) /
				new_mc_value_weight;
		}
		if (new_tree_value_weight > 0.0F) {
			merged.tree_value_target =
				(merged.tree_value_target * old_tree_value_weight +
				 record.tree_value_target * record.tree_value_weight) /
				new_tree_value_weight;
		}
		merged.policy_weight = new_policy_weight;
		merged.mc_value_weight = new_mc_value_weight;
		merged.tree_value_weight = new_tree_value_weight;
	}
	for (auto &record : output) {
		record.policy_target = normalize(std::move(record.policy_target));
	}
	summary = {
		{"source_positions", source_count},
		{"aggregated_positions", output.size()},
		{"merged_positions", source_count - output.size()},
	};
	return output;
}

// Persist generated FCPI targets for diagnostics and reproducibility of each iteration.
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
	const std::size_t count = records.size();
	std::vector<std::uint8_t> states(count * kStatePlanes * 8);
	std::vector<std::int32_t> legal(count * legal_width, 0);
	std::vector<float> policy(count * legal_width, 0.0F);
	std::vector<float> mc_policy_advantage_sums(count * legal_width, 0.0F);
	std::vector<float> mc_policy_weights(count * legal_width, 0.0F);
	std::vector<std::uint8_t> legal_counts(count);
	std::vector<float> mc_values(count);
	std::vector<float> tree_values(count);
	std::vector<float> policy_weights(count);
	std::vector<float> mc_value_weights(count);
	std::vector<float> tree_value_weights(count);
	double mc_policy_samples = 0.0;
	for (std::size_t row = 0; row < count; ++row) {
		std::copy(records[row].state.begin(), records[row].state.end(),
				  states.begin() + row * kStatePlanes * 8);
		legal_counts[row] = static_cast<std::uint8_t>(records[row].legal_indices.size());
		mc_values[row] = records[row].mc_value_target;
		tree_values[row] = records[row].tree_value_target;
		policy_weights[row] = records[row].policy_weight;
		mc_value_weights[row] = records[row].mc_value_weight;
		tree_value_weights[row] = records[row].tree_value_weight;
		for (std::size_t column = 0; column < records[row].legal_indices.size(); ++column) {
			legal[row * legal_width + column] = records[row].legal_indices[column];
			policy[row * legal_width + column] = records[row].policy_target[column];
			mc_policy_advantage_sums[row * legal_width + column] =
				records[row].mc_policy_advantage_sums[column];
			mc_policy_weights[row * legal_width + column] =
				records[row].mc_policy_weights[column];
			mc_policy_samples += records[row].mc_policy_weights[column];
		}
	}
	if (!path.parent_path().empty()) {
		std::filesystem::create_directories(path.parent_path());
	}
	const hid_t file = require_id(
		H5Fcreate(path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), path.string());
	write_string_attribute(file, "arch_type", kArchType);
	write_string_attribute(file, "fcpi_formula", kFcpiFormula);
	write_dataset(file, "states", H5T_STD_U8LE, H5T_NATIVE_UINT8, {count, kStatePlanes, 8},
				  states.data());
	write_dataset(file, "legal_indices", H5T_STD_I32LE, H5T_NATIVE_INT32, {count, legal_width},
				  legal.data());
	write_dataset(file, "policy_targets", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, {count, legal_width},
				  policy.data());
	write_dataset(file, "mc_policy_advantage_sums", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT,
				  {count, legal_width}, mc_policy_advantage_sums.data());
	write_dataset(file, "mc_policy_weights", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT,
				  {count, legal_width}, mc_policy_weights.data());
	write_dataset(file, "legal_counts", H5T_STD_U8LE, H5T_NATIVE_UINT8, {count},
				  legal_counts.data());
	write_dataset(file, "mc_value_targets", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, {count},
				  mc_values.data());
	write_dataset(file, "tree_value_targets", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, {count},
				  tree_values.data());
	write_dataset(file, "policy_weights", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, {count},
				  policy_weights.data());
	write_dataset(file, "mc_value_weights", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, {count},
				  mc_value_weights.data());
	write_dataset(file, "tree_value_weights", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, {count},
				  tree_value_weights.data());
	H5Fclose(file);
	return {
		{"path", path.string()},
		{"positions", count},
		{"legal_width", legal_width},
		{"mc_policy_samples", mc_policy_samples},
		{"formula", kFcpiFormula},
		{"aggregation", aggregation},
	};
}

// Fine-tune Policy on planned targets and Value against Monte Carlo and detached tree targets.
nlohmann::json train_candidate(const std::filesystem::path &source,
								   const std::filesystem::path &candidate,
								   const torch::Device &device, std::vector<Position> &records,
								   const FcpiOptions &options) {
	ArchitectureInfo source_arch;
	auto model = load_checkpoint(source, device, &source_arch);
	model->to(device);
	// FCPI targets are generated from the frozen model in inference mode. Keep
	// BatchNorm on those same running statistics while autograd updates parameters;
	// eval() changes normalization behavior but does not disable gradients.
	model->eval();
	torch::optim::AdamW optimizer(
		model->parameters(),
		torch::optim::AdamWOptions(options.learning_rate).weight_decay(kWeightDecay));
	std::vector<std::size_t> order(records.size());
	std::iota(order.begin(), order.end(), 0);
	std::mt19937_64 rng(options.seed);
	std::int64_t steps = 0;
	auto metric_totals =
		torch::zeros({7}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
	for (int epoch = 0; epoch < std::max(0, options.epochs); ++epoch) {
		std::shuffle(order.begin(), order.end(), rng);
		for (std::size_t begin = 0; begin < order.size(); begin += std::max(1, options.batch_size)) {
			const auto end = std::min(order.size(), begin + std::max(1, options.batch_size));
			const std::int64_t batch = static_cast<std::int64_t>(end - begin);
			std::size_t width = 1;
			std::vector<std::uint8_t> packed(batch * kStatePlanes * 8);
			for (std::size_t index = begin; index < end; ++index) {
				width = std::max(width, records[order[index]].legal_indices.size());
				std::copy(records[order[index]].state.begin(), records[order[index]].state.end(),
						  packed.begin() + (index - begin) * kStatePlanes * 8);
			}
			const bool pin_memory = device.is_cuda();
			auto int_options =
				torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
			auto float_options =
				torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
			if (pin_memory) {
				int_options = int_options.pinned_memory(true);
				float_options = float_options.pinned_memory(true);
			}
			auto states = decode_states_device(packed.data(), batch, device);
			auto legal =
				torch::zeros({batch, static_cast<std::int64_t>(width)}, int_options);
			auto targets =
				torch::zeros({batch, static_cast<std::int64_t>(width)}, float_options);
			auto mc_policy_advantage_sums =
				torch::zeros({batch, static_cast<std::int64_t>(width)}, float_options);
			auto mc_policy_weights =
				torch::zeros({batch, static_cast<std::int64_t>(width)}, float_options);
			auto counts = torch::zeros({batch}, int_options);
			auto mc_values = torch::zeros({batch}, float_options);
			auto tree_values = torch::zeros({batch}, float_options);
			auto policy_weights = torch::zeros({batch}, float_options);
			auto mc_value_weights = torch::zeros({batch}, float_options);
			auto tree_value_weights = torch::zeros({batch}, float_options);
			auto legal_access = legal.accessor<std::int64_t, 2>();
			auto target_access = targets.accessor<float, 2>();
			auto mc_policy_advantage_access = mc_policy_advantage_sums.accessor<float, 2>();
			auto mc_policy_weight_access = mc_policy_weights.accessor<float, 2>();
			auto count_access = counts.accessor<std::int64_t, 1>();
			auto mc_value_access = mc_values.accessor<float, 1>();
			auto tree_value_access = tree_values.accessor<float, 1>();
			auto policy_weight_access = policy_weights.accessor<float, 1>();
			auto mc_value_weight_access = mc_value_weights.accessor<float, 1>();
			auto tree_value_weight_access = tree_value_weights.accessor<float, 1>();
			for (std::size_t index = begin; index < end; ++index) {
				const auto &record = records[order[index]];
				const auto row = static_cast<std::int64_t>(index - begin);
				count_access[row] = record.legal_indices.size();
				mc_value_access[row] = record.mc_value_target;
				tree_value_access[row] = record.tree_value_target;
				policy_weight_access[row] = record.policy_weight;
				mc_value_weight_access[row] = record.mc_value_weight;
				tree_value_weight_access[row] = record.tree_value_weight;
				for (std::size_t column = 0; column < record.legal_indices.size(); ++column) {
					legal_access[row][column] = record.legal_indices[column];
					target_access[row][column] = record.policy_target[column];
					mc_policy_advantage_access[row][column] =
						record.mc_policy_advantage_sums[column];
					mc_policy_weight_access[row][column] = record.mc_policy_weights[column];
				}
			}
			legal = legal.to(device, true);
			targets = targets.to(device, true);
			mc_policy_advantage_sums = mc_policy_advantage_sums.to(device, true);
			mc_policy_weights = mc_policy_weights.to(device, true);
			counts = counts.to(device, true);
			mc_values = mc_values.to(device, true);
			tree_values = tree_values.to(device, true);
			policy_weights = policy_weights.to(device, true);
			mc_value_weights = mc_value_weights.to(device, true);
			tree_value_weights = tree_value_weights.to(device, true);

			optimizer.zero_grad();
			torch::Tensor logits;
			torch::Tensor predicted;
			{
				AutocastGuard autocast(options.precision, device);
				std::tie(logits, predicted) = model->forward(states);
			}
			auto selected = logits.to(torch::kFloat32).gather(1, legal);
			predicted = predicted.to(torch::kFloat32);
			auto columns = torch::arange(static_cast<std::int64_t>(width), counts.options());
			auto mask = columns.unsqueeze(0) < counts.unsqueeze(1);
			selected = selected.masked_fill(~mask, -1e9);
			auto log_probability = torch::log_softmax(selected, 1);
			const double behavior_temperature = std::max(1e-4, options.behavior_temperature);
			auto behavior_log_probability =
				torch::log_softmax(selected / behavior_temperature, 1);
			auto masked_targets = targets * mask;
			masked_targets = masked_targets / masked_targets.sum(1, true).clamp_min(1e-8);
			auto policy_errors = -(masked_targets * log_probability).sum(1);
			auto counterfactual_policy_loss =
				(policy_errors * policy_weights).sum() / policy_weights.sum().clamp_min(1e-8);
			// This is a behavior-policy score-function update around the frozen Value baseline.
			// Signed advantages raise or lower only the action actually observed in a
			// completed trajectory; they never replace the counterfactual Q table.
			auto mc_policy_loss =
				-(mc_policy_advantage_sums * behavior_log_probability * mask).sum() /
				 mc_policy_weights.sum().clamp_min(1.0);
			auto policy_loss = counterfactual_policy_loss + mc_policy_loss;
			auto mc_value_errors = torch::nn::functional::smooth_l1_loss(
				predicted.squeeze(1), mc_values,
				torch::nn::functional::SmoothL1LossFuncOptions().reduction(torch::kNone));
			auto tree_value_errors = torch::nn::functional::smooth_l1_loss(
				predicted.squeeze(1), tree_values,
				torch::nn::functional::SmoothL1LossFuncOptions().reduction(torch::kNone));
			const auto mc_weight_sum = mc_value_weights.sum();
			const auto tree_weight_sum = tree_value_weights.sum();
			auto mc_value_loss =
				(mc_value_errors * mc_value_weights).sum() / mc_weight_sum.clamp_min(1.0);
			auto tree_value_loss =
				(tree_value_errors * tree_value_weights).sum() /
				tree_weight_sum.clamp_min(1e-8);

			// Monte Carlo results and counterfactual residual targets are independent
			// observations. Their accumulated weights determine their relative influence.
			auto value_loss =
				((mc_value_errors * mc_value_weights).sum() +
				 (tree_value_errors * tree_value_weights).sum()) /
				(mc_weight_sum + tree_weight_sum).clamp_min(1.0);
			auto loss = policy_loss + value_loss;
			loss.backward();
			torch::nn::utils::clip_grad_norm_(model->parameters(), kGradientClip);
			optimizer.step();
			++steps;
			metric_totals.add_(
				torch::stack({loss.detach(), policy_loss.detach(),
							  counterfactual_policy_loss.detach(), mc_policy_loss.detach(),
							  value_loss.detach(), mc_value_loss.detach(), tree_value_loss.detach()}));
			if (options.log_every > 0 && (steps == 1 || steps % options.log_every == 0)) {
				auto metrics =
					torch::stack({policy_loss.detach(), mc_value_loss.detach(),
								  tree_value_loss.detach(), value_loss.detach(),
								  loss.detach()})
						.to(torch::kCPU)
						.contiguous();
				auto metric_values = metrics.accessor<float, 1>();
				std::cout << "fcpi train: step=" << steps
						  << " policy=" << metric_values[0]
						  << " value_mc=" << metric_values[1]
						  << " value_tree=" << metric_values[2]
						  << " value=" << metric_values[3]
						  << " loss=" << metric_values[4] << std::endl;
			}
			if (options.train_max_steps > 0 && steps >= options.train_max_steps) {
				break;
			}
		}
		if (options.train_max_steps > 0 && steps >= options.train_max_steps) {
			break;
		}
	}
	save_checkpoint_atomic(candidate, model, {source_arch.channels, source_arch.blocks});
	const double divisor = static_cast<double>(std::max<std::int64_t>(1, steps));
	auto final_metrics = metric_totals.to(torch::kCPU).contiguous();
	auto metric_values = final_metrics.accessor<float, 1>();
	return {
		{"steps", steps},
		{"epochs_requested", options.epochs},
		{"candidate", candidate.string()},
		{"batch_norm_running_stats", "frozen"},
		{"metrics",
		 {
			 {"loss", metric_values[0] / divisor},
			 {"policy", metric_values[1] / divisor},
			 {"policy_counterfactual", metric_values[2] / divisor},
			 {"policy_mc", metric_values[3] / divisor},
			 {"value", metric_values[4] / divisor},
			 {"value_mc", metric_values[5] / divisor},
			 {"value_tree", metric_values[6] / divisor},
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
		std::cout << "fcpi sampling summary: " << sampling.dump() << std::endl;
		TargetSummary target_summary;
		construct_targets(records, model, device, options, target_summary);
		const auto data_path = data_dir / ("fcpi_iter_" + zero_pad(iteration, 3) + ".h5");
		auto data_summary = write_fcpi_h5(data_path, records);
		data_summary["sampling"] = sampling;
		data_summary["counterfactual"] = {
			{"deep_budget_per_root", options.counterfactual_budget},
			{"trees", target_summary.trees},
			{"decision_nodes", target_summary.decision_nodes},
			{"evaluated_edges", target_summary.evaluated_edges},
			{"terminal_edges", target_summary.terminal_edges},
			{"max_depth", target_summary.max_depth},
			{"mean_residual",
			 target_summary.residual_count > 0
				 ? target_summary.residual_sum /
					   static_cast<double>(target_summary.residual_count)
				 : 0.0},
			{"mean_coverage",
			 target_summary.decision_nodes > 0
				 ? target_summary.evaluated_policy_mass_sum /
					   static_cast<double>(target_summary.decision_nodes)
				 : 0.0},
			{"mean_value_correction",
			 target_summary.tree_value_weight_sum > 0.0
				 ? target_summary.tree_value_correction_sum /
					   target_summary.tree_value_weight_sum
				 : 0.0},
			{"mean_policy_shift",
			 target_summary.policy_diagnostic_nodes > 0
				 ? target_summary.policy_total_variation_sum /
					   target_summary.policy_diagnostic_nodes
				 : 0.0},
			{"policy_top1_change_rate",
			 target_summary.policy_diagnostic_nodes > 0
				 ? static_cast<double>(target_summary.policy_top1_changes) /
					   target_summary.policy_diagnostic_nodes
				 : 0.0},
		};
		const auto candidate = model_dir / ("candidate_iter_" + zero_pad(iteration, 3) + ".pth");
		auto train_summary = train_candidate(current, candidate, device, records, options);
		auto arena_options = options.arena;
		arena_options.candidate = candidate;
		arena_options.baseline = current;
		arena_options.device = options.device;
		arena_options.search.precision = options.precision;
		arena_options.seed = options.seed + iteration;
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
			{"data", data_summary},
			{"train", train_summary},
			{"arena", arena_summary},
			{"accepted", accepted},
		});
		std::ofstream summary_file(data_dir / "summary.json");
		summary_file << nlohmann::json({
										   {"run_id", run_id},
										   {"precision", compute_precision_name(options.precision)},
										   {"initial_model", initial.string()},
										   {"current_model", current.string()},
										   {"summaries", summaries},
									   })
							.dump(2);
	}
}

} // namespace gadus
