// Implements terminal-anchored Bellman Restricted Counterfactual Iteration for Gadus.

#include "gadus/brci.hpp"
#include <hdf5.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
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
#include "gadus/args.hpp"
#include "gadus/checkpoint.hpp"

namespace gadus {

// Compute cross-entropy only on each row's real graph edges. Padding log
// probabilities are zeroed explicitly because IEEE arithmetic defines
// 0 * -infinity as NaN.
torch::Tensor brci_masked_policy_loss(const torch::Tensor &selected_logits,
									  const torch::Tensor &targets,
									  const torch::Tensor &counts,
									  const torch::Tensor &weights) {
	if (selected_logits.dim() != 2 || targets.sizes() != selected_logits.sizes() ||
		counts.dim() != 1 || weights.dim() != 1 ||
		counts.size(0) != selected_logits.size(0) ||
		weights.size(0) != selected_logits.size(0)) {
		throw std::invalid_argument("BRCI masked policy tensors are not aligned");
	}
	auto logits = selected_logits.to(torch::kFloat32);
	auto normalized_targets = targets.to(torch::kFloat32);
	auto normalized_weights = weights.to(torch::kFloat32);
	auto columns = torch::arange(logits.size(1), counts.options());
	auto mask = columns.unsqueeze(0) < counts.unsqueeze(1);
	logits = logits.masked_fill(~mask, -std::numeric_limits<float>::infinity());
	auto log_probability = torch::log_softmax(logits, 1).masked_fill(~mask, 0.0);
	normalized_targets = normalized_targets.masked_fill(~mask, 0.0);
	normalized_targets =
		normalized_targets / normalized_targets.sum(1, true).clamp_min(1e-8);
	auto policy_errors = -(normalized_targets * log_probability).sum(1);
	return (policy_errors * normalized_weights).sum() /
		   normalized_weights.sum().clamp_min(1.0);
}

namespace {

inline constexpr const char *kBrciFormula = "bellman_restricted_counterfactual";
inline constexpr double kWeightDecay = 1e-4;
inline constexpr double kGradientClip = 1.0;
// Risk deltas below one part per million are indistinguishable from FP32/BF16
// inference and parameter-interpolation noise, so they do not constitute a strict step.
inline constexpr double kStrictTolerance = 1e-6;
inline constexpr double kProbabilityFailure = 0.05;
inline constexpr int kBacktrackingSteps = 12;

// Format iteration numbers for stable artifact names.
std::string zero_pad(int value, int width) {
	std::ostringstream output;
	output << std::setfill('0') << std::setw(width) << value;
	return output.str();
}

struct EpisodeStep {
	PackedState state{};
	int action = 0;
};

struct Episode {
	int game_id = 0;
	std::string start_fen;
	std::vector<EpisodeStep> steps;
	chess::Board final_board;
};

struct WorkingGame {
	int game_id = 0;
	std::string start_fen;
	chess::Board board;
	std::vector<EpisodeStep> steps;
};

struct SelfplayBatch {
	std::vector<Episode> completed;
	nlohmann::json summary;
};

struct EpisodeSplit {
	std::vector<Episode> training;
	std::vector<Episode> test;
};

struct GraphEdge {
	int action = 0;
	std::size_t child = 0;
};

struct GraphNode {
	PackedState state{};
	std::vector<GraphEdge> edges;
	float terminal_value = 0.0F;
	float optimal_value = 0.0F;
	int component = 0;
	int depth = 0;
	bool terminal = false;
};

// Store exact action-prefix histories. Different histories are never merged, so
// repetition rights remain part of the environment state even though the network
// observes only PackedState.
class RestrictedGraph {
	public:
	void append(const Episode &episode) {
		if (episode.steps.empty() || !game_is_over(episode.final_board)) {
			throw std::invalid_argument("restricted graph accepts only non-empty terminal episodes");
		}
		std::size_t node_index = 0;
		const auto root = roots_.find(episode.start_fen);
		if (root == roots_.end()) {
			node_index = nodes_.size();
			roots_.emplace(episode.start_fen, node_index);
			GraphNode node;
			node.state = episode.steps.front().state;
			node.component = next_component_++;
			nodes_.push_back(std::move(node));
		} else {
			node_index = root->second;
			if (nodes_[node_index].state != episode.steps.front().state) {
				throw std::runtime_error("restricted graph root state mismatch");
			}
		}

		for (std::size_t step = 0; step < episode.steps.size(); ++step) {
			auto &parent = nodes_[node_index];
			if (parent.terminal || parent.state != episode.steps[step].state) {
				throw std::runtime_error("restricted graph history is inconsistent");
			}
			const bool terminal_child = step + 1 == episode.steps.size();
			const PackedState child_state =
				terminal_child ? encode_state(episode.final_board) : episode.steps[step + 1].state;
			auto edge = std::find_if(parent.edges.begin(), parent.edges.end(),
									 [&](const GraphEdge &candidate) {
										 return candidate.action == episode.steps[step].action;
									 });
			if (edge == parent.edges.end()) {
				GraphNode child;
				child.state = child_state;
				child.component = parent.component;
				child.depth = parent.depth + 1;
				child.terminal = terminal_child;
				if (terminal_child) {
					child.terminal_value =
						std::clamp(terminal_value_side_to_move(episode.final_board), -1.0F, 1.0F);
				}
				const auto child_index = nodes_.size();
				nodes_.push_back(std::move(child));
				nodes_[node_index].edges.push_back({episode.steps[step].action, child_index});
				node_index = child_index;
			} else {
				node_index = edge->child;
				const auto &child = nodes_[node_index];
				if (child.state != child_state || child.terminal != terminal_child) {
					throw std::runtime_error("restricted graph deterministic edge mismatch");
				}
				if (terminal_child) {
					const float terminal_value =
						std::clamp(terminal_value_side_to_move(episode.final_board), -1.0F, 1.0F);
					if (std::abs(child.terminal_value - terminal_value) > 1e-6F) {
						throw std::runtime_error("restricted graph terminal value mismatch");
					}
				}
			}
		}
		++episodes_;
	}

	// Solve the finite graph exactly. Every non-terminal leaf is forbidden because
	// it would reintroduce a model bootstrap into the Bellman target.
	void solve() {
		for (std::size_t reverse = nodes_.size(); reverse-- > 0;) {
			auto &node = nodes_[reverse];
			if (node.terminal) {
				node.optimal_value = node.terminal_value;
				continue;
			}
			if (node.edges.empty()) {
				throw std::runtime_error("restricted graph contains an unanchored leaf");
			}
			float best = -std::numeric_limits<float>::infinity();
			for (const auto &edge : node.edges) {
				if (edge.child <= reverse || edge.child >= nodes_.size()) {
					throw std::runtime_error("restricted graph is not an acyclic prefix graph");
				}
				best = std::max(best, -nodes_[edge.child].optimal_value);
			}
			node.optimal_value = std::clamp(best, -1.0F, 1.0F);
		}
	}

	std::vector<std::size_t> decision_nodes() const {
		std::vector<std::size_t> indices;
		indices.reserve(nodes_.size());
		for (std::size_t index = 0; index < nodes_.size(); ++index) {
			if (!nodes_[index].terminal && !nodes_[index].edges.empty()) {
				indices.push_back(index);
			}
		}
		return indices;
	}

	const GraphNode &node(std::size_t index) const {
		return nodes_.at(index);
	}

	nlohmann::json summary() const {
		std::size_t edges = 0;
		std::size_t terminals = 0;
		std::size_t decisions = 0;
		std::size_t branching = 0;
		int max_depth = 0;
		for (const auto &node : nodes_) {
			edges += node.edges.size();
			terminals += node.terminal ? 1 : 0;
			decisions += !node.terminal && !node.edges.empty() ? 1 : 0;
			branching += node.edges.size() > 1 ? 1 : 0;
			max_depth = std::max(max_depth, node.depth);
		}
		return {
			{"episodes", episodes_},
			{"roots", roots_.size()},
			{"nodes", nodes_.size()},
			{"edges", edges},
			{"decision_nodes", decisions},
			{"branching_nodes", branching},
			{"terminal_nodes", terminals},
			{"max_depth", max_depth},
		};
	}

	private:
	std::vector<GraphNode> nodes_;
	std::unordered_map<std::string, std::size_t> roots_;
	int next_component_ = 0;
	std::int64_t episodes_ = 0;
};

struct TrainingRecord {
	int component = 0;
	PackedState state{};
	float source_value = 0.0F;
	float value_target = 0.0F;
	std::vector<int> legal_indices;
	std::vector<float> source_policy;
	std::vector<float> policy_target;
	std::vector<float> action_values;
	float policy_weight = 0.0F;
	float value_weight = 1.0F;
};

struct NetworkEvaluation {
	std::vector<float> policy;
	float value = 0.0F;
};

struct ComponentError {
	double policy_sum = 0.0;
	double policy_weight = 0.0;
	double value_sum = 0.0;
	double value_weight = 0.0;
};

struct GraphErrorMetrics {
	double policy_regret = 0.0;
	double value_mse = 0.0;
	double policy_weight = 0.0;
	double value_weight = 0.0;
	std::size_t positions = 0;
	std::unordered_map<int, ComponentError> components;
};

struct GraphImprovement {
	std::size_t eligible_components = 0;
	std::size_t policy_improved_components = 0;
	std::size_t value_improved_components = 0;
	std::size_t jointly_improved_components = 0;
	double empirical_probability = 0.0;
	double probability_lower_bound = 0.0;
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

hid_t require_id(hid_t id, const std::string &operation) {
	if (id < 0) {
		throw std::runtime_error("HDF5 operation failed: " + operation);
	}
	return id;
}

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

std::vector<float> normalize(std::vector<float> values) {
	double total = 0.0;
	for (const float value : values) {
		total += std::max(0.0F, value);
	}
	if (total <= 0.0) {
		const float uniform = values.empty() ? 0.0F : 1.0F / static_cast<float>(values.size());
		std::fill(values.begin(), values.end(), uniform);
		return values;
	}
	for (auto &value : values) {
		value = static_cast<float>(std::max(0.0F, value) / total);
	}
	return values;
}

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

std::string packed_key(const PackedState &state) {
	return std::string(reinterpret_cast<const char *>(state.data()), state.size());
}

std::size_t choose_behavior_action(const std::string &state_key,
								   const std::vector<int> &legal,
								   const std::vector<float> &behavior,
								   std::unordered_map<std::string,
													  std::unordered_map<int, std::int64_t>> &counts,
								   std::unordered_map<std::string, std::int64_t> &visits) {
	if (legal.empty() || legal.size() != behavior.size()) {
		throw std::invalid_argument("behavior selection requires aligned legal probabilities");
	}
	const double next_visit = static_cast<double>(visits[state_key] + 1);
	std::size_t selected = 0;
	double largest_deficit = -std::numeric_limits<double>::infinity();
	for (std::size_t index = 0; index < legal.size(); ++index) {
		const double realized = static_cast<double>(counts[state_key][legal[index]]);
		const double deficit = next_visit * static_cast<double>(behavior[index]) - realized;
		if (deficit > largest_deficit) {
			largest_deficit = deficit;
			selected = index;
		}
	}
	++visits[state_key];
	++counts[state_key][legal[selected]];
	return selected;
}

std::vector<SamplingSpec> make_sampling_specs(const BrciOptions &options, int iteration,
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
			throw std::runtime_error("opening book contains no BRCI positions");
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

// Generate complete games. Truncated games remain diagnostics and never become
// graph leaves because their outcomes are not exact.
SelfplayBatch collect_selfplay(Model model, const torch::Device &device,
							   const BrciOptions &options, int iteration) {
	SearchOptions closed;
	closed.type = SearchType::Closed;
	closed.precision = options.precision;
	closed.mcts_sims = 0;
	closed.mcts_batch_size = options.inference_batch_size;
	Searcher evaluator(model, device, closed);
	nlohmann::json starts;
	const auto specs = make_sampling_specs(options, iteration, starts);
	std::unordered_map<std::string, std::unordered_map<int, std::int64_t>> action_counts;
	std::unordered_map<std::string, std::int64_t> state_visits;
	std::vector<Episode> completed_episodes;
	std::int64_t source_positions = 0;
	std::int64_t completed_positions = 0;
	std::int64_t truncated_positions = 0;
	int finished = 0;
	int truncated_games = 0;

	std::cout << "brci self-play start: iteration=" << iteration << " arch_type=" << kArchType
			  << " games=" << specs.size() << " max_plies=" << options.max_plies
			  << " device=" << device.str() << std::endl;
	std::cout << "brci starts: " << starts.dump() << std::endl;

	for (std::size_t group_start = 0; group_start < specs.size();
		 group_start += std::max(1, options.games_in_flight)) {
		const auto group_end =
			std::min(specs.size(), group_start + std::max(1, options.games_in_flight));
		std::vector<WorkingGame> games;
		games.reserve(group_end - group_start);
		for (std::size_t index = group_start; index < group_end; ++index) {
			WorkingGame game;
			game.game_id =
				static_cast<int>((iteration - 1) * specs.size() + index + 1);
			game.start_fen = specs[index].fen;
			game.board = chess::Board(specs[index].fen);
			games.push_back(std::move(game));
		}
		std::vector<bool> done(games.size(), false);
		while (std::find(done.begin(), done.end(), false) != done.end()) {
			std::vector<std::size_t> active_indices;
			std::vector<chess::Board> boards;
			for (std::size_t index = 0; index < games.size(); ++index) {
				if (!done[index]) {
					active_indices.push_back(index);
					boards.push_back(games[index].board);
				}
			}
			const auto results = evaluate_chunks(evaluator, boards, options.inference_batch_size);
			for (std::size_t row = 0; row < active_indices.size(); ++row) {
				auto &game = games[active_indices[row]];
				const auto moves = legal_moves(game.board);
				if (moves.empty()) {
					throw std::runtime_error("non-terminal BRCI board has no legal move");
				}
				std::vector<int> legal;
				std::vector<float> prior;
				legal.reserve(moves.size());
				prior.reserve(moves.size());
				for (const auto &move : moves) {
					const int action = move_to_index(move);
					legal.push_back(action);
					prior.push_back(results[row].policy[action]);
				}
				prior = normalize(std::move(prior));
				const double temperature = std::max(1e-4, options.behavior_temperature);
				std::vector<float> behavior(prior.size());
				for (std::size_t index = 0; index < prior.size(); ++index) {
					behavior[index] = static_cast<float>(
						std::pow(std::clamp(static_cast<double>(prior[index]), 1e-12, 1.0),
								 1.0 / temperature));
				}
				behavior = normalize(std::move(behavior));
				const auto state = encode_state(game.board);
				const auto choice = choose_behavior_action(
					packed_key(state), legal, behavior, action_counts, state_visits);
				game.steps.push_back({state, legal[choice]});
				game.board.makeMove(moves[choice]);
				const bool terminal = game_is_over(game.board);
				const bool truncated =
					static_cast<int>(game.steps.size()) >= options.max_plies;
				if (terminal || truncated) {
					done[active_indices[row]] = true;
					++finished;
					source_positions += game.steps.size();
					if (terminal) {
						completed_positions += game.steps.size();
						Episode episode;
						episode.game_id = game.game_id;
						episode.start_fen = game.start_fen;
						episode.steps = std::move(game.steps);
						episode.final_board = game.board;
						completed_episodes.push_back(std::move(episode));
					} else {
						++truncated_games;
						truncated_positions += game.steps.size();
					}
					if (options.log_every > 0 &&
						(finished == 1 || finished % options.log_every == 0 ||
						 finished == static_cast<int>(specs.size()))) {
						std::cout << "brci game: completed=" << finished << '/' << specs.size()
								  << " terminal_games=" << completed_episodes.size()
								  << " truncated_games=" << truncated_games << std::endl;
					}
				}
			}
		}
	}

	SelfplayBatch batch;
	batch.completed = std::move(completed_episodes);
	batch.summary = {
		{"games", specs.size()},
		{"source_positions", source_positions},
		{"completed_games", batch.completed.size()},
		{"truncated_games", truncated_games},
		{"completed_positions", completed_positions},
		{"truncated_positions", truncated_positions},
		{"starts", starts},
	};
	std::cout << "brci sampling summary: " << batch.summary.dump() << std::endl;
	return batch;
}

std::string episode_key(const Episode &episode) {
	std::string key = episode.start_fen;
	key.push_back('\n');
	for (const auto &step : episode.steps) {
		key.append(std::to_string(step.action));
		key.push_back(',');
	}
	return key;
}

std::uint64_t stable_hash(const std::string &value, std::uint64_t seed) {
	std::uint64_t hash = 1469598103934665603ULL ^ seed;
	for (const unsigned char byte : value) {
		hash ^= byte;
		hash *= 1099511628211ULL;
	}
	return hash;
}

// Assign each exact terminal trajectory signature to one partition for the
// entire run. Repeated trajectories can never migrate from training to test.
EpisodeSplit split_episodes(
	std::vector<Episode> episodes, int games_per_iter, std::uint64_t seed,
	std::unordered_map<std::string, bool> &test_assignments) {
	if (episodes.size() < 2) {
		throw std::runtime_error("BRCI requires at least two terminal games per iteration");
	}
	const bool first_partition = test_assignments.empty();
	const auto modulus = static_cast<std::uint64_t>(
		std::max(2, static_cast<int>(std::llround(std::sqrt(games_per_iter)))));
	std::vector<std::string> keys;
	keys.reserve(episodes.size());
	for (const auto &episode : episodes) {
		auto key = episode_key(episode);
		keys.push_back(key);
		if (!test_assignments.contains(key)) {
			test_assignments.emplace(
				key, stable_hash(key, seed) % modulus == 0);
		}
	}
	if (first_partition) {
		std::vector<std::string> unique_keys;
		std::unordered_set<std::string> seen_keys;
		for (const auto &key : keys) {
			if (seen_keys.insert(key).second) {
				unique_keys.push_back(key);
			}
		}
		if (unique_keys.size() < 2) {
			throw std::runtime_error(
				"BRCI requires at least two distinct terminal trajectories");
		}
		bool has_training = false;
		bool has_test = false;
		for (const auto &key : unique_keys) {
			has_test = has_test || test_assignments.at(key);
			has_training = has_training || !test_assignments.at(key);
		}
		if (!has_test) {
			test_assignments[unique_keys.front()] = true;
		}
		has_training = false;
		for (const auto &key : unique_keys) {
			has_training = has_training || !test_assignments.at(key);
		}
		if (!has_training) {
			test_assignments[unique_keys.back()] = false;
		}
	}
	EpisodeSplit split;
	for (std::size_t index = 0; index < episodes.size(); ++index) {
		if (test_assignments.at(keys[index])) {
			split.test.push_back(std::move(episodes[index]));
		} else {
			split.training.push_back(std::move(episodes[index]));
		}
	}
	return split;
}

// Evaluate only the action sets represented by the restricted graph.
std::vector<NetworkEvaluation> evaluate_restricted(
	Model model, const torch::Device &device, ComputePrecision precision, int batch_size,
	const std::vector<const PackedState *> &states,
	const std::vector<const std::vector<int> *> &actions) {
	if (states.size() != actions.size()) {
		throw std::invalid_argument("restricted evaluation inputs are not aligned");
	}
	model->eval();
	std::vector<NetworkEvaluation> output(states.size());
	for (std::size_t begin = 0; begin < states.size(); begin += std::max(1, batch_size)) {
		const auto end = std::min(states.size(), begin + std::max(1, batch_size));
		const auto count = static_cast<std::int64_t>(end - begin);
		std::size_t width = 1;
		std::vector<std::uint8_t> packed(static_cast<std::size_t>(count) * kStatePlanes * 8);
		for (std::size_t row = begin; row < end; ++row) {
			if (actions[row]->empty()) {
				throw std::runtime_error("restricted evaluation received an empty action set");
			}
			width = std::max(width, actions[row]->size());
			std::copy(states[row]->begin(), states[row]->end(),
					  packed.begin() + (row - begin) * kStatePlanes * 8);
		}
		const bool pin_memory = device.is_cuda();
		auto index_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
		auto mask_options = torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU);
		if (pin_memory) {
			index_options = index_options.pinned_memory(true);
			mask_options = mask_options.pinned_memory(true);
		}
		auto legal =
			torch::zeros({count, static_cast<std::int64_t>(width)}, index_options);
		auto mask =
			torch::zeros({count, static_cast<std::int64_t>(width)}, mask_options);
		auto legal_rows = legal.accessor<std::int64_t, 2>();
		auto mask_rows = mask.accessor<bool, 2>();
		for (std::size_t row = begin; row < end; ++row) {
			for (std::size_t column = 0; column < actions[row]->size(); ++column) {
				legal_rows[static_cast<std::int64_t>(row - begin)]
						  [static_cast<std::int64_t>(column)] = actions[row]->at(column);
				mask_rows[static_cast<std::int64_t>(row - begin)]
						 [static_cast<std::int64_t>(column)] = true;
			}
		}
		torch::InferenceMode inference;
		auto inputs = decode_states(packed.data(), count, pin_memory).to(device, true);
		auto device_legal = legal.to(device, true);
		auto device_mask = mask.to(device, true);
		torch::Tensor logits;
		torch::Tensor values;
		{
			AutocastGuard autocast(precision, device);
			std::tie(logits, values) = model->forward(inputs);
		}
		auto selected = logits.to(torch::kFloat32).gather(1, device_legal);
		selected = selected.masked_fill(
			~device_mask, -std::numeric_limits<float>::infinity());
		auto probabilities =
			torch::softmax(selected, 1).to(torch::kCPU).contiguous();
		values = values.reshape({-1}).to(torch::kFloat32).to(torch::kCPU).contiguous();
		auto probability_rows = probabilities.accessor<float, 2>();
		auto value_rows = values.accessor<float, 1>();
		for (std::size_t row = begin; row < end; ++row) {
			auto &evaluation = output[row];
			evaluation.value = value_rows[static_cast<std::int64_t>(row - begin)];
			evaluation.policy.resize(actions[row]->size());
			for (std::size_t column = 0; column < actions[row]->size(); ++column) {
				evaluation.policy[column] =
					probability_rows[static_cast<std::int64_t>(row - begin)]
									[static_cast<std::int64_t>(column)];
			}
		}
	}
	return output;
}

// Apply the unit-temperature KL-regularized improvement on exact graph Q values.
std::vector<float> improve_policy(const std::vector<float> &prior,
								  const std::vector<float> &action_values) {
	if (prior.empty() || prior.size() != action_values.size()) {
		throw std::invalid_argument("BRCI policy improvement inputs are not aligned");
	}
	double mean = 0.0;
	for (std::size_t index = 0; index < prior.size(); ++index) {
		mean += static_cast<double>(prior[index]) * action_values[index];
	}
	std::vector<double> logits(prior.size());
	for (std::size_t index = 0; index < prior.size(); ++index) {
		logits[index] =
			std::log(std::clamp(static_cast<double>(prior[index]), 1e-12, 1.0)) +
			static_cast<double>(action_values[index]) - mean;
	}
	return stable_softmax(logits);
}

// Convert every solved non-terminal graph node into exact Policy and Value targets.
std::vector<TrainingRecord> materialize_graph(RestrictedGraph &graph, Model model,
											 const torch::Device &device,
											 const BrciOptions &options,
											 nlohmann::json &summary) {
	graph.solve();
	const auto indices = graph.decision_nodes();
	std::vector<std::vector<int>> action_sets(indices.size());
	std::vector<const PackedState *> states;
	std::vector<const std::vector<int> *> actions;
	states.reserve(indices.size());
	actions.reserve(indices.size());
	for (std::size_t row = 0; row < indices.size(); ++row) {
		const auto &node = graph.node(indices[row]);
		action_sets[row].reserve(node.edges.size());
		for (const auto &edge : node.edges) {
			action_sets[row].push_back(edge.action);
		}
		states.push_back(&node.state);
		actions.push_back(&action_sets[row]);
	}
	const auto evaluations = evaluate_restricted(
		model, device, options.precision, options.inference_batch_size, states, actions);

	std::vector<TrainingRecord> records;
	records.reserve(indices.size());
	std::size_t branching = 0;
	std::size_t policy_changes = 0;
	double policy_variation = 0.0;
	for (std::size_t row = 0; row < indices.size(); ++row) {
		const auto &node = graph.node(indices[row]);
		TrainingRecord record;
		record.component = node.component;
		record.state = node.state;
		record.source_value = evaluations[row].value;
		record.value_target = node.optimal_value;
		record.legal_indices = action_sets[row];
		record.source_policy = evaluations[row].policy;
		record.action_values.reserve(node.edges.size());
		for (const auto &edge : node.edges) {
			record.action_values.push_back(-graph.node(edge.child).optimal_value);
		}
		record.policy_target =
			improve_policy(record.source_policy, record.action_values);
		const auto [minimum, maximum] =
			std::minmax_element(record.action_values.begin(), record.action_values.end());
		record.policy_weight =
			node.edges.size() > 1 && *maximum - *minimum > 1e-6F ? 1.0F : 0.0F;
		if (record.policy_weight > 0.0F) {
			++branching;
			double l1 = 0.0;
			for (std::size_t index = 0; index < record.source_policy.size(); ++index) {
				l1 += std::abs(record.source_policy[index] - record.policy_target[index]);
			}
			policy_variation += 0.5 * l1;
			policy_changes +=
				std::max_element(record.source_policy.begin(), record.source_policy.end()) -
						record.source_policy.begin() !=
					std::max_element(record.policy_target.begin(), record.policy_target.end()) -
						record.policy_target.begin()
					? 1
					: 0;
		}
		records.push_back(std::move(record));
	}
	summary = graph.summary();
	summary["training_records"] = records.size();
	summary["policy_records"] = branching;
	summary["mean_policy_total_variation"] =
		branching > 0 ? policy_variation / static_cast<double>(branching) : 0.0;
	summary["policy_top1_changes"] = policy_changes;
	return records;
}

nlohmann::json write_brci_h5(const std::filesystem::path &path,
							 const std::vector<TrainingRecord> &records) {
	if (records.empty()) {
		throw std::runtime_error("BRCI graph contains no training records");
	}
	const std::size_t legal_width =
		std::max_element(records.begin(), records.end(), [](const auto &left, const auto &right) {
			return left.legal_indices.size() < right.legal_indices.size();
		})->legal_indices.size();
	const std::size_t count = records.size();
	std::vector<std::uint8_t> states(count * kStatePlanes * 8);
	std::vector<std::int32_t> legal(count * legal_width, 0);
	std::vector<float> policy(count * legal_width, 0.0F);
	std::vector<std::uint8_t> legal_counts(count);
	std::vector<float> values(count);
	std::vector<float> policy_weights(count);
	std::vector<float> value_weights(count);
	std::vector<std::int32_t> components(count);
	for (std::size_t row = 0; row < count; ++row) {
		std::copy(records[row].state.begin(), records[row].state.end(),
				  states.begin() + row * kStatePlanes * 8);
		legal_counts[row] = static_cast<std::uint8_t>(records[row].legal_indices.size());
		values[row] = records[row].value_target;
		policy_weights[row] = records[row].policy_weight;
		value_weights[row] = records[row].value_weight;
		components[row] = records[row].component;
		for (std::size_t column = 0; column < records[row].legal_indices.size(); ++column) {
			legal[row * legal_width + column] = records[row].legal_indices[column];
			policy[row * legal_width + column] = records[row].policy_target[column];
		}
	}
	if (!path.parent_path().empty()) {
		std::filesystem::create_directories(path.parent_path());
	}
	const hid_t file = require_id(
		H5Fcreate(path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), path.string());
	write_string_attribute(file, "arch_type", kArchType);
	write_string_attribute(file, "brci_formula", kBrciFormula);
	write_dataset(file, "states", H5T_STD_U8LE, H5T_NATIVE_UINT8, {count, kStatePlanes, 8},
				  states.data());
	write_dataset(file, "legal_indices", H5T_STD_I32LE, H5T_NATIVE_INT32, {count, legal_width},
				  legal.data());
	write_dataset(file, "policy_targets", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT,
				  {count, legal_width}, policy.data());
	write_dataset(file, "legal_counts", H5T_STD_U8LE, H5T_NATIVE_UINT8, {count},
				  legal_counts.data());
	write_dataset(file, "value_targets", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, {count},
				  values.data());
	write_dataset(file, "policy_weights", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, {count},
				  policy_weights.data());
	write_dataset(file, "value_weights", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, {count},
				  value_weights.data());
	write_dataset(file, "components", H5T_STD_I32LE, H5T_NATIVE_INT32, {count},
				  components.data());
	H5Fclose(file);
	return {
		{"path", path.string()},
		{"positions", count},
		{"legal_width", legal_width},
		{"formula", kBrciFormula},
	};
}

// Fit a raw proposal to the exact finite-graph targets. Parameter backtracking
// later decides how much of this optimizer displacement is admissible.
nlohmann::json train_proposal(const std::filesystem::path &source,
							  const std::filesystem::path &proposal, Model model,
							  const torch::Device &device,
							  const std::vector<TrainingRecord> &records,
							  const BrciOptions &options, int iteration) {
	model->to(device);
	model->eval();
	torch::optim::AdamW optimizer(
		model->parameters(),
		torch::optim::AdamWOptions(options.learning_rate).weight_decay(kWeightDecay));
	std::vector<std::size_t> order(records.size());
	std::iota(order.begin(), order.end(), 0);
	std::mt19937_64 rng(options.seed + iteration * 313);
	std::int64_t steps = 0;
	double total_policy = 0.0;
	double total_value = 0.0;
	double total_loss = 0.0;
	for (int epoch = 0; epoch < std::max(0, options.epochs); ++epoch) {
		std::shuffle(order.begin(), order.end(), rng);
		for (std::size_t begin = 0; begin < order.size(); begin += std::max(1, options.batch_size)) {
			const auto end = std::min(order.size(), begin + std::max(1, options.batch_size));
			const auto batch = static_cast<std::int64_t>(end - begin);
			std::size_t width = 1;
			std::vector<std::uint8_t> packed(static_cast<std::size_t>(batch) * kStatePlanes * 8);
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
			auto legal =
				torch::zeros({batch, static_cast<std::int64_t>(width)}, int_options);
			auto targets =
				torch::zeros({batch, static_cast<std::int64_t>(width)}, float_options);
			auto counts = torch::zeros({batch}, int_options);
			auto values = torch::zeros({batch}, float_options);
			auto policy_weights = torch::zeros({batch}, float_options);
			auto value_weights = torch::zeros({batch}, float_options);
			auto legal_access = legal.accessor<std::int64_t, 2>();
			auto target_access = targets.accessor<float, 2>();
			auto count_access = counts.accessor<std::int64_t, 1>();
			auto value_access = values.accessor<float, 1>();
			auto policy_weight_access = policy_weights.accessor<float, 1>();
			auto value_weight_access = value_weights.accessor<float, 1>();
			for (std::size_t index = begin; index < end; ++index) {
				const auto &record = records[order[index]];
				const auto row = static_cast<std::int64_t>(index - begin);
				count_access[row] = record.legal_indices.size();
				value_access[row] = record.value_target;
				policy_weight_access[row] = record.policy_weight;
				value_weight_access[row] = record.value_weight;
				for (std::size_t column = 0; column < record.legal_indices.size(); ++column) {
					legal_access[row][column] = record.legal_indices[column];
					target_access[row][column] = record.policy_target[column];
				}
			}
			auto states = decode_states(packed.data(), batch, pin_memory).to(device, true);
			legal = legal.to(device, true);
			targets = targets.to(device, true);
			counts = counts.to(device, true);
			values = values.to(device, true);
			policy_weights = policy_weights.to(device, true);
			value_weights = value_weights.to(device, true);

			optimizer.zero_grad();
			torch::Tensor logits;
			torch::Tensor predicted;
			{
				AutocastGuard autocast(options.precision, device);
				std::tie(logits, predicted) = model->forward(states);
			}
			auto selected = logits.to(torch::kFloat32).gather(1, legal);
			predicted = predicted.reshape({-1}).to(torch::kFloat32);
			auto policy_loss =
				brci_masked_policy_loss(selected, targets, counts, policy_weights);
			auto value_errors = torch::square(predicted - values);
			auto value_loss =
				(value_errors * value_weights).sum() / value_weights.sum().clamp_min(1.0);
			auto loss = policy_loss + value_loss;
			const auto next_step = steps + 1;
			const bool inspect =
				next_step == 1 ||
				(options.log_every > 0 && next_step % options.log_every == 0);
			if (inspect &&
				(!torch::isfinite(policy_loss).item<bool>() ||
				 !torch::isfinite(value_loss).item<bool>() ||
				 !torch::isfinite(loss).item<bool>())) {
				throw std::runtime_error("BRCI training produced a non-finite loss");
			}
			loss.backward();
			torch::nn::utils::clip_grad_norm_(model->parameters(), kGradientClip);
			optimizer.step();
			++steps;
			total_policy += policy_loss.detach().item<double>();
			total_value += value_loss.detach().item<double>();
			total_loss += loss.detach().item<double>();
			if (options.log_every > 0 && (steps == 1 || steps % options.log_every == 0)) {
				std::cout << "brci train: step=" << steps
						  << " policy=" << policy_loss.detach().item<double>()
						  << " value=" << value_loss.detach().item<double>()
						  << " loss=" << loss.detach().item<double>() << std::endl;
			}
			if (options.train_max_steps > 0 && steps >= options.train_max_steps) {
				break;
			}
		}
		if (options.train_max_steps > 0 && steps >= options.train_max_steps) {
			break;
		}
	}
	ArchitectureInfo source_architecture;
	load_checkpoint(source, torch::Device(torch::kCPU), &source_architecture);
	save_checkpoint_atomic(proposal, model, source_architecture);
	const double divisor = static_cast<double>(std::max<std::int64_t>(1, steps));
	return {
		{"steps", steps},
		{"epochs_requested", options.epochs},
		{"proposal", proposal.string()},
		{"batch_norm_running_stats", "frozen"},
		{"metrics",
		 {
			 {"loss", total_loss / divisor},
			 {"policy", total_policy / divisor},
			 {"value", total_value / divisor},
		 }},
	};
}

std::vector<NetworkEvaluation> evaluate_records(
	Model model, const torch::Device &device, const BrciOptions &options,
	const std::vector<TrainingRecord> &records) {
	std::vector<const PackedState *> states;
	std::vector<const std::vector<int> *> actions;
	states.reserve(records.size());
	actions.reserve(records.size());
	for (const auto &record : records) {
		states.push_back(&record.state);
		actions.push_back(&record.legal_indices);
	}
	return evaluate_restricted(
		model, device, options.precision, options.inference_batch_size, states, actions);
}

GraphErrorMetrics evaluate_graph_error(Model model, const torch::Device &device,
									   const BrciOptions &options,
									   const std::vector<TrainingRecord> &records) {
	const auto evaluations = evaluate_records(model, device, options, records);
	GraphErrorMetrics metrics;
	metrics.positions = records.size();
	for (std::size_t row = 0; row < records.size(); ++row) {
		const auto &record = records[row];
		const auto &evaluation = evaluations[row];
		double expected_q = 0.0;
		for (std::size_t index = 0; index < record.action_values.size(); ++index) {
			expected_q += static_cast<double>(evaluation.policy[index]) *
						  static_cast<double>(record.action_values[index]);
		}
		const double policy_regret =
			std::max(0.0, static_cast<double>(record.value_target) - expected_q);
		const double value_error =
			static_cast<double>(evaluation.value) - static_cast<double>(record.value_target);
		const double policy_weight = std::max(0.0F, record.policy_weight);
		const double value_weight = std::max(0.0F, record.value_weight);
		metrics.policy_regret += policy_weight * policy_regret;
		metrics.value_mse += value_weight * value_error * value_error;
		metrics.policy_weight += policy_weight;
		metrics.value_weight += value_weight;
		auto &component = metrics.components[record.component];
		component.policy_sum += policy_weight * policy_regret;
		component.policy_weight += policy_weight;
		component.value_sum += value_weight * value_error * value_error;
		component.value_weight += value_weight;
	}
	if (metrics.policy_weight > 0.0) {
		metrics.policy_regret /= metrics.policy_weight;
	}
	if (metrics.value_weight > 0.0) {
		metrics.value_mse /= metrics.value_weight;
	}
	return metrics;
}

GraphImprovement compare_components(const GraphErrorMetrics &source,
									const GraphErrorMetrics &candidate) {
	GraphImprovement result;
	for (const auto &[component_id, source_component] : source.components) {
		const auto found = candidate.components.find(component_id);
		if (found == candidate.components.end() || source_component.policy_weight <= 0.0 ||
			source_component.value_weight <= 0.0) {
			continue;
		}
		const auto &candidate_component = found->second;
		const double source_policy =
			source_component.policy_sum / source_component.policy_weight;
		const double source_value =
			source_component.value_sum / source_component.value_weight;
		if (source_policy <= kStrictTolerance || source_value <= kStrictTolerance) {
			continue;
		}
		const double candidate_policy =
			candidate_component.policy_sum / candidate_component.policy_weight;
		const double candidate_value =
			candidate_component.value_sum / candidate_component.value_weight;
		const bool policy_improved =
			candidate_policy + kStrictTolerance < source_policy;
		const bool value_improved =
			candidate_value + kStrictTolerance < source_value;
		++result.eligible_components;
		result.policy_improved_components += policy_improved ? 1 : 0;
		result.value_improved_components += value_improved ? 1 : 0;
		result.jointly_improved_components += policy_improved && value_improved ? 1 : 0;
	}
	if (result.eligible_components > 0) {
		result.empirical_probability =
			static_cast<double>(result.jointly_improved_components) /
			static_cast<double>(result.eligible_components);
		const double radius = std::sqrt(
			std::log(1.0 / kProbabilityFailure) /
			(2.0 * static_cast<double>(result.eligible_components)));
		result.probability_lower_bound =
			std::max(0.0, result.empirical_probability - radius);
	}
	return result;
}

Model interpolate_models(const Model &source, const Model &raw,
						 const ArchitectureInfo &architecture,
						 const torch::Device &device, double alpha) {
	auto blended = Model(architecture.channels, architecture.blocks);
	blended->to(device);
	const auto source_parameters = source->parameters();
	const auto raw_parameters = raw->parameters();
	const auto blended_parameters = blended->parameters();
	const auto source_buffers = source->buffers();
	const auto blended_buffers = blended->buffers();
	if (source_parameters.size() != raw_parameters.size() ||
		source_parameters.size() != blended_parameters.size() ||
		source_buffers.size() != blended_buffers.size()) {
		throw std::runtime_error("BRCI parameter interpolation found incompatible models");
	}
	torch::NoGradGuard no_grad;
	for (std::size_t index = 0; index < blended_parameters.size(); ++index) {
		blended_parameters[index].copy_(
			source_parameters[index] +
			alpha * (raw_parameters[index] - source_parameters[index]));
	}
	for (std::size_t index = 0; index < blended_buffers.size(); ++index) {
		blended_buffers[index].copy_(source_buffers[index]);
	}
	blended->eval();
	return blended;
}

nlohmann::json graph_error_json(const GraphErrorMetrics &metrics) {
	return {
		{"positions", metrics.positions},
		{"components", metrics.components.size()},
		{"policy_regret", metrics.policy_regret},
		{"value_mse", metrics.value_mse},
		{"policy_weight", metrics.policy_weight},
		{"value_weight", metrics.value_weight},
	};
}

nlohmann::json graph_improvement_json(const GraphImprovement &metrics) {
	return {
		{"eligible_components", metrics.eligible_components},
		{"policy_improved_components", metrics.policy_improved_components},
		{"value_improved_components", metrics.value_improved_components},
		{"jointly_improved_components", metrics.jointly_improved_components},
		{"empirical_probability", metrics.empirical_probability},
		{"probability_lower_bound", metrics.probability_lower_bound},
		{"failure_probability", kProbabilityFailure},
	};
}

// Search a finite sequence of parameter step lengths and choose the admissible
// candidate with the strongest held-out component improvement bound.
nlohmann::json select_restricted_candidate(
	const std::filesystem::path &source_path,
	const std::filesystem::path &raw_path,
	const std::filesystem::path &candidate_path,
	const torch::Device &device,
	const BrciOptions &options,
	const std::vector<TrainingRecord> &test_records) {
	if (test_records.empty()) {
		throw std::runtime_error("BRCI test graph contains no decision records");
	}
	ArchitectureInfo source_architecture;
	auto source = load_checkpoint(source_path, device, &source_architecture);
	ArchitectureInfo raw_architecture;
	auto raw = load_checkpoint(raw_path, device, &raw_architecture);
	if (source_architecture.channels != raw_architecture.channels ||
		source_architecture.blocks != raw_architecture.blocks) {
		throw std::runtime_error("BRCI source and raw candidate architectures differ");
	}
	source->eval();
	raw->eval();
	const auto source_metrics = evaluate_graph_error(source, device, options, test_records);
	if (source_metrics.policy_weight <= 0.0 || source_metrics.value_weight <= 0.0) {
		std::cout << "brci restriction unavailable: test graph has no jointly measurable "
					 "Policy regret and Value error"
				  << std::endl;
		return {
			{"accepted", false},
			{"reason", "test_graph_has_no_joint_policy_value_error"},
			{"source", graph_error_json(source_metrics)},
		};
	}

	nlohmann::json attempts = nlohmann::json::array();
	bool found = false;
	double best_alpha = 0.0;
	double best_probability = -1.0;
	double best_reduction = -1.0;
	GraphErrorMetrics best_metrics;
	GraphImprovement best_graph;
	for (int step = 0; step < kBacktrackingSteps; ++step) {
		const double alpha = std::ldexp(1.0, -step);
		auto blended =
			interpolate_models(source, raw, source_architecture, device, alpha);
		const auto candidate_metrics =
			evaluate_graph_error(blended, device, options, test_records);
		const auto graph_improvement =
			compare_components(source_metrics, candidate_metrics);
		const double policy_reduction =
			source_metrics.policy_regret - candidate_metrics.policy_regret;
		const double value_reduction =
			source_metrics.value_mse - candidate_metrics.value_mse;
		const bool policy_ok = policy_reduction > kStrictTolerance;
		const bool value_ok = value_reduction > kStrictTolerance;
		const double relative_reduction =
			policy_reduction / std::max(source_metrics.policy_regret, kStrictTolerance) +
			value_reduction / std::max(source_metrics.value_mse, kStrictTolerance);
		attempts.push_back({
			{"alpha", alpha},
			{"policy_ok", policy_ok},
			{"value_ok", value_ok},
			{"policy_regret_reduction", policy_reduction},
			{"value_mse_reduction", value_reduction},
			{"metrics", graph_error_json(candidate_metrics)},
			{"graph_improvement", graph_improvement_json(graph_improvement)},
		});
		std::cout << "brci restriction: alpha=" << alpha
				  << " policy_regret_reduction=" << policy_reduction
				  << " value_mse_reduction=" << value_reduction
				  << " graph_probability_lower_bound="
				  << graph_improvement.probability_lower_bound
				  << " admissible=" << (policy_ok && value_ok ? "true" : "false")
				  << std::endl;
		const bool better_probability =
			graph_improvement.probability_lower_bound > best_probability + kStrictTolerance;
		const bool equal_probability =
			std::abs(graph_improvement.probability_lower_bound - best_probability) <=
			kStrictTolerance;
		if (policy_ok && value_ok &&
			(!found || better_probability ||
			 (equal_probability && relative_reduction > best_reduction + kStrictTolerance))) {
			found = true;
			best_alpha = alpha;
			best_probability = graph_improvement.probability_lower_bound;
			best_reduction = relative_reduction;
			best_metrics = candidate_metrics;
			best_graph = graph_improvement;
			save_checkpoint_atomic(candidate_path, blended, source_architecture);
		}
	}
	if (!found) {
		return {
			{"accepted", false},
			{"alpha", 0.0},
			{"source", graph_error_json(source_metrics)},
			{"attempts", attempts},
		};
	}
	return {
		{"accepted", true},
		{"alpha", best_alpha},
		{"source", graph_error_json(source_metrics)},
		{"candidate", graph_error_json(best_metrics)},
		{"graph_improvement", graph_improvement_json(best_graph)},
		{"attempts", attempts},
	};
}

} // namespace

void run_brci(const BrciOptions &options) {
	if (options.iterations <= 0 || options.games_per_iter <= 0 ||
		options.games_in_flight <= 0 || options.max_plies <= 0) {
		throw std::invalid_argument("BRCI iteration, game, and ply counts must be positive");
	}
	if (!std::filesystem::is_regular_file(options.model)) {
		throw std::runtime_error("BRCI model not found: " + options.model.string());
	}
	const auto run_id = create_run_id("brci");
	const auto data_dir = std::filesystem::path("data/runs") / run_id;
	const auto model_dir = std::filesystem::path("models/runs") / run_id;
	if (std::filesystem::exists(data_dir) || std::filesystem::exists(model_dir)) {
		throw std::runtime_error("BRCI run directory already exists: " + run_id);
	}
	std::filesystem::create_directories(data_dir);
	std::filesystem::create_directories(model_dir);
	const auto initial = model_dir / "initial.pth";
	const auto current = model_dir / "current.pth";
	atomic_copy(options.model, initial);
	atomic_copy(initial, current);
	const auto device = resolve_device(options.device);
	validate_compute_precision(options.precision, device);
	std::cout << "brci run id: " << run_id << std::endl;
	std::cout << "brci architecture: " << kArchType << std::endl;
	std::cout << "brci formula: " << kBrciFormula << std::endl;
	std::cout << "brci precision: " << compute_precision_name(options.precision) << std::endl;
	std::cout << "brci current model: " << current.string() << std::endl;

	RestrictedGraph training_graph;
	RestrictedGraph test_graph;
	std::unordered_map<std::string, bool> test_assignments;
	nlohmann::json summaries = nlohmann::json::array();
	for (int iteration = 1; iteration <= options.iterations; ++iteration) {
		std::cout << "brci iteration " << iteration << std::endl;
		auto model = load_checkpoint(current, device);
		auto sampled = collect_selfplay(model, device, options, iteration);
		auto split = split_episodes(
			std::move(sampled.completed), options.games_per_iter, options.seed,
			test_assignments);
		for (const auto &episode : split.training) {
			training_graph.append(episode);
		}
		for (const auto &episode : split.test) {
			test_graph.append(episode);
		}

		nlohmann::json training_graph_summary;
		auto training_records = materialize_graph(
			training_graph, model, device, options, training_graph_summary);
		nlohmann::json test_graph_summary;
		auto test_records = materialize_graph(
			test_graph, model, device, options, test_graph_summary);
		const auto data_path =
			data_dir / ("brci_iter_" + zero_pad(iteration, 3) + ".h5");
		auto data_summary = write_brci_h5(data_path, training_records);
		data_summary["sampling"] = sampled.summary;
		data_summary["new_training_games"] = split.training.size();
		data_summary["new_test_games"] = split.test.size();
		data_summary["training_graph"] = training_graph_summary;
		data_summary["test_graph"] = test_graph_summary;

		const auto raw =
			model_dir / ("raw_iter_" + zero_pad(iteration, 3) + ".pth");
		const auto candidate =
			model_dir / ("candidate_iter_" + zero_pad(iteration, 3) + ".pth");
		auto train_summary = train_proposal(
			current, raw, model, device, training_records, options, iteration);
		auto restriction_summary = select_restricted_candidate(
			current, raw, candidate, device, options, test_records);
		const bool restriction_ok = restriction_summary["accepted"].get<bool>();
		nlohmann::json arena_summary;
		bool accepted = false;
		if (restriction_ok) {
			auto arena_options = options.arena;
			arena_options.candidate = candidate;
			arena_options.baseline = current;
			arena_options.device = options.device;
			arena_options.search.precision = options.precision;
			arena_options.seed = options.seed + iteration;
			arena_summary = evaluate_models(arena_options);
			accepted = arena_summary["accepted"].get<bool>();
		} else {
			arena_summary = {
				{"accepted", false},
				{"skipped", true},
				{"reason", "restricted_graph_error_not_improved"},
			};
		}
		if (accepted) {
			atomic_copy(candidate, current);
			std::cout << "brci promoted: " << current.string() << std::endl;
		} else {
			std::cout << "brci candidate rejected: "
					  << (restriction_ok ? candidate.string() : raw.string()) << std::endl;
		}
		summaries.push_back({
			{"iteration", iteration},
			{"architecture", kArchType},
			{"formula", kBrciFormula},
			{"data", data_summary},
			{"train", train_summary},
			{"restriction", restriction_summary},
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
