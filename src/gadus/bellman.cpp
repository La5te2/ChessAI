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
#include <optional>
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

// Find the minimum-norm point on the segment between the two shared-backbone
// gradients. Its negative is a simultaneous descent direction when one exists.
double brci_common_descent_lambda(double policy_norm_squared,
								  double value_norm_squared,
								  double policy_value_dot) {
	const double denominator =
		policy_norm_squared + value_norm_squared - 2.0 * policy_value_dot;
	if (!std::isfinite(denominator) || denominator <= 1e-20) {
		return 0.5;
	}
	return std::clamp(
		(value_norm_squared - policy_value_dot) / denominator, 0.0, 1.0);
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
	std::string rule_key;
};

struct Episode {
	int game_id = 0;
	std::string start_fen;
	std::string family_key;
	std::string final_rule_key;
	std::vector<EpisodeStep> steps;
	chess::Board final_board;
};

// Tracks the rule history that can change future terminal outcomes. FEN omits
// previous repetitions, so the restricted graph carries their counts explicitly.
struct RuleHistory {
	std::unordered_map<std::string, int> repetitions;

	void initialize(const chess::Board &board) {
		repetitions.clear();
		repetitions.emplace(board.getFen(false), 1);
	}

	void observe_after_move(const chess::Board &board) {
		if (board.halfMoveClock() == 0) {
			repetitions.clear();
		}
		++repetitions[board.getFen(false)];
	}

	std::string key(const chess::Board &board) const {
		std::vector<std::pair<std::string, int>> ordered(
			repetitions.begin(), repetitions.end());
		std::sort(ordered.begin(), ordered.end());
		std::ostringstream output;
		output << board.getFen(false) << "|hm=" << board.halfMoveClock() << "|rep=";
		for (const auto &[fen, count] : ordered) {
			output << count << ':' << fen.size() << ':' << fen << ';';
		}
		return output.str();
	}
};

struct WorkingGame {
	int game_id = 0;
	std::string start_fen;
	std::string family_key;
	chess::Board board;
	RuleHistory history;
	std::vector<EpisodeStep> steps;
	std::optional<int> excluded_first_action;
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

// Merge nodes only when their complete future-relevant rule state agrees inside
// one opening component. The resulting DAG shares move-order transpositions
// without erasing repetition or fifty-move information.
class RestrictedGraph {
	public:
	void append(const Episode &episode) {
		if (episode.steps.empty() || episode.steps.front().rule_key.empty() ||
			episode.final_rule_key.empty() || !game_is_over(episode.final_board)) {
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
			node_lookup_.emplace(
				graph_key(nodes_[node_index].component, episode.steps.front().rule_key),
				node_index);
		} else {
			node_index = root->second;
			const auto lookup = node_lookup_.find(
				graph_key(nodes_[node_index].component, episode.steps.front().rule_key));
			if (lookup == node_lookup_.end() || lookup->second != node_index ||
				nodes_[node_index].state != episode.steps.front().state) {
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
			const auto &child_rule_key =
				terminal_child ? episode.final_rule_key : episode.steps[step + 1].rule_key;
			auto edge = std::find_if(parent.edges.begin(), parent.edges.end(),
									 [&](const GraphEdge &candidate) {
										 return candidate.action == episode.steps[step].action;
									 });
			if (edge == parent.edges.end()) {
				const auto key = graph_key(parent.component, child_rule_key);
				auto existing = node_lookup_.find(key);
				std::size_t child_index = 0;
				if (existing == node_lookup_.end()) {
					GraphNode child;
					child.state = child_state;
					child.component = parent.component;
					child.depth = parent.depth + 1;
					child.terminal = terminal_child;
					if (terminal_child) {
						child.terminal_value = std::clamp(
							terminal_value_side_to_move(episode.final_board), -1.0F, 1.0F);
					}
					child_index = nodes_.size();
					nodes_.push_back(std::move(child));
					node_lookup_.emplace(key, child_index);
				} else {
					child_index = existing->second;
					auto &child = nodes_[child_index];
					if (child.component != parent.component || child.state != child_state ||
						child.terminal != terminal_child) {
						throw std::runtime_error(
							"restricted graph future-equivalent node mismatch");
					}
					if (terminal_child) {
						const float terminal_value = std::clamp(
							terminal_value_side_to_move(episode.final_board), -1.0F, 1.0F);
						if (std::abs(child.terminal_value - terminal_value) > 1e-6F) {
							throw std::runtime_error(
								"restricted graph merged terminal value mismatch");
						}
					}
					child.depth = std::min(child.depth, parent.depth + 1);
					++transposition_reuses_;
				}
				nodes_[node_index].edges.push_back({episode.steps[step].action, child_index});
				node_index = child_index;
			} else {
				node_index = edge->child;
				const auto &child = nodes_[node_index];
				const auto lookup = node_lookup_.find(
					graph_key(child.component, child_rule_key));
				if (lookup == node_lookup_.end() || lookup->second != node_index ||
					child.state != child_state || child.terminal != terminal_child) {
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
			max_observed_depth_ =
				std::max(max_observed_depth_, static_cast<int>(step + 1));
		}
		++episodes_;
	}

	// Solve the finite terminal-anchored DAG by memoized Bellman recursion.
	void solve() {
		std::vector<std::uint8_t> status(nodes_.size(), 0);
		for (const auto &[fen, root] : roots_) {
			static_cast<void>(fen);
			solve_node(root, status);
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
		for (const auto &node : nodes_) {
			edges += node.edges.size();
			terminals += node.terminal ? 1 : 0;
			decisions += !node.terminal && !node.edges.empty() ? 1 : 0;
			branching += node.edges.size() > 1 ? 1 : 0;
		}
		return {
			{"episodes", episodes_},
			{"roots", roots_.size()},
			{"nodes", nodes_.size()},
			{"edges", edges},
			{"decision_nodes", decisions},
			{"branching_nodes", branching},
			{"terminal_nodes", terminals},
			{"transposition_reuses", transposition_reuses_},
			{"max_depth", max_observed_depth_},
		};
	}

	private:
	static std::string graph_key(int component, const std::string &rule_key) {
		return std::to_string(component) + '\n' + rule_key;
	}

	float solve_node(std::size_t index, std::vector<std::uint8_t> &status) {
		if (index >= nodes_.size()) {
			throw std::runtime_error("restricted graph edge is out of range");
		}
		if (status[index] == 2) {
			return nodes_[index].optimal_value;
		}
		if (status[index] == 1) {
			throw std::runtime_error(
				"restricted graph contains a future-rule-state cycle");
		}
		status[index] = 1;
		auto &node = nodes_[index];
		if (node.terminal) {
			node.optimal_value = node.terminal_value;
		} else {
			if (node.edges.empty()) {
				throw std::runtime_error("restricted graph contains an unanchored leaf");
			}
			float best = -std::numeric_limits<float>::infinity();
			for (const auto &edge : node.edges) {
				best = std::max(best, -solve_node(edge.child, status));
			}
			node.optimal_value = std::clamp(best, -1.0F, 1.0F);
		}
		status[index] = 2;
		return node.optimal_value;
	}

	std::vector<GraphNode> nodes_;
	std::unordered_map<std::string, std::size_t> roots_;
	std::unordered_map<std::string, std::size_t> node_lookup_;
	int next_component_ = 0;
	int max_observed_depth_ = 0;
	std::int64_t episodes_ = 0;
	std::int64_t transposition_reuses_ = 0;
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

struct ValueTrainingRecord {
	PackedState state{};
	float target = 0.0F;
	float weight = 0.0F;
};

struct PolicyTrainingRecord {
	PackedState state{};
	std::vector<int> legal_indices;
	std::vector<float> target;
	float weight = 0.0F;
};

struct AggregatedTrainingData {
	std::vector<ValueTrainingRecord> values;
	std::vector<PolicyTrainingRecord> policies;
	nlohmann::json summary;
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

// Serialize one complete trajectory for deterministic family assignment.
std::string episode_key(const Episode &episode) {
	std::string key = episode.start_fen;
	key.push_back('\n');
	for (const auto &step : episode.steps) {
		key.append(std::to_string(step.action));
		key.push_back(',');
	}
	return key;
}

// Compute a platform-stable FNV-1a hash for deterministic sampling and splits.
std::uint64_t stable_hash(const std::string &value, std::uint64_t seed) {
	std::uint64_t hash = 1469598103934665603ULL ^ seed;
	for (const unsigned char byte : value) {
		hash ^= byte;
		hash *= 1099511628211ULL;
	}
	return hash;
}

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
			game.history.initialize(game.board);
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
				game.steps.push_back({state, legal[choice], game.history.key(game.board)});
				game.board.makeMove(moves[choice]);
				game.history.observe_after_move(game.board);
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
						episode.final_rule_key = game.history.key(game.board);
						episode.family_key = episode_key(episode);
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

	// Create one active sibling for every terminal primary trajectory. The branch
	// point is deterministic for a run, excludes the primary action, and the
	// resulting trajectory is admitted only when it reaches an exact terminal.
	const std::size_t primary_completed_games = completed_episodes.size();
	std::vector<WorkingGame> sibling_starts;
	sibling_starts.reserve(primary_completed_games);
	for (std::size_t episode_index = 0; episode_index < primary_completed_games;
		 ++episode_index) {
		const auto &episode = completed_episodes[episode_index];
		std::vector<std::size_t> preferred;
		std::vector<std::size_t> fallback;
		chess::Board replay(episode.start_fen);
		const std::size_t preferred_limit = std::max<std::size_t>(
			1, std::min<std::size_t>(episode.steps.size(), options.max_plies / 2));
		for (std::size_t step = 0; step < episode.steps.size(); ++step) {
			const auto moves = legal_moves(replay);
			if (moves.size() > 1) {
				fallback.push_back(step);
				if (step < preferred_limit) {
					preferred.push_back(step);
				}
			}
			const auto original =
				std::find_if(moves.begin(), moves.end(), [&](const chess::Move &move) {
					return move_to_index(move) == episode.steps[step].action;
				});
			if (original == moves.end()) {
				throw std::runtime_error("BRCI primary trajectory cannot be replayed");
			}
			replay.makeMove(*original);
		}
		const auto &candidates = preferred.empty() ? fallback : preferred;
		if (candidates.empty()) {
			continue;
		}
		const auto branch = candidates[stable_hash(
			episode.family_key, options.seed + static_cast<std::uint64_t>(iteration) * 911ULL) %
									  candidates.size()];
		WorkingGame sibling;
		sibling.game_id = episode.game_id;
		sibling.start_fen = episode.start_fen;
		sibling.family_key = episode.family_key;
		sibling.board = chess::Board(episode.start_fen);
		sibling.history.initialize(sibling.board);
		for (std::size_t step = 0; step < branch; ++step) {
			const auto moves = legal_moves(sibling.board);
			const auto original =
				std::find_if(moves.begin(), moves.end(), [&](const chess::Move &move) {
					return move_to_index(move) == episode.steps[step].action;
				});
			if (original == moves.end()) {
				throw std::runtime_error("BRCI sibling prefix cannot be replayed");
			}
			sibling.steps.push_back(
				{encode_state(sibling.board), episode.steps[step].action,
				 sibling.history.key(sibling.board)});
			sibling.board.makeMove(*original);
			sibling.history.observe_after_move(sibling.board);
		}
		sibling.excluded_first_action = episode.steps[branch].action;
		sibling_starts.push_back(std::move(sibling));
	}

	int sibling_finished = 0;
	int sibling_completed = 0;
	int sibling_truncated = 0;
	std::int64_t sibling_positions = 0;
	for (std::size_t group_start = 0; group_start < sibling_starts.size();
		 group_start += std::max(1, options.games_in_flight)) {
		const auto group_end = std::min(
			sibling_starts.size(), group_start + std::max(1, options.games_in_flight));
		std::vector<WorkingGame> games;
		games.reserve(group_end - group_start);
		for (std::size_t index = group_start; index < group_end; ++index) {
			games.push_back(std::move(sibling_starts[index]));
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
			const auto results = evaluate_chunks(
				evaluator, boards, options.inference_batch_size);
			for (std::size_t row = 0; row < active_indices.size(); ++row) {
				auto &game = games[active_indices[row]];
				const auto moves = legal_moves(game.board);
				if (moves.empty()) {
					throw std::runtime_error("non-terminal BRCI sibling has no legal move");
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
				const auto state_key = packed_key(state);
				std::size_t choice = 0;
				if (game.excluded_first_action.has_value()) {
					bool found = false;
					float best = -1.0F;
					for (std::size_t index = 0; index < legal.size(); ++index) {
						if (legal[index] != *game.excluded_first_action &&
							(!found || behavior[index] > best)) {
							found = true;
							best = behavior[index];
							choice = index;
						}
					}
					if (!found) {
						throw std::runtime_error(
							"BRCI sibling branch has no alternative legal action");
					}
					++state_visits[state_key];
					++action_counts[state_key][legal[choice]];
					game.excluded_first_action.reset();
				} else {
					choice = choose_behavior_action(
						state_key, legal, behavior, action_counts, state_visits);
				}
				game.steps.push_back(
					{state, legal[choice], game.history.key(game.board)});
				game.board.makeMove(moves[choice]);
				game.history.observe_after_move(game.board);
				const bool terminal = game_is_over(game.board);
				const bool truncated =
					static_cast<int>(game.steps.size()) >= options.max_plies;
				if (terminal || truncated) {
					done[active_indices[row]] = true;
					++sibling_finished;
					sibling_positions += game.steps.size();
					if (terminal) {
						Episode episode;
						episode.game_id = game.game_id;
						episode.start_fen = game.start_fen;
						episode.family_key = game.family_key;
						episode.steps = std::move(game.steps);
						episode.final_board = game.board;
						episode.final_rule_key = game.history.key(game.board);
						completed_episodes.push_back(std::move(episode));
						++sibling_completed;
					} else {
						++sibling_truncated;
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
		{"primary_completed_games", primary_completed_games},
		{"primary_truncated_games", truncated_games},
		{"completed_positions", completed_positions},
		{"truncated_positions", truncated_positions},
		{"sibling_attempts", sibling_starts.size()},
		{"sibling_completed_games", sibling_completed},
		{"sibling_truncated_games", sibling_truncated},
		{"sibling_positions", sibling_positions},
		{"terminal_episodes", batch.completed.size()},
		{"starts", starts},
	};
	std::cout << "brci sampling summary: " << batch.summary.dump() << std::endl;
	return batch;
}

// Assign each primary trajectory family to one partition for the entire run.
// Its actively generated sibling therefore cannot leak across the graph split.
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
		const auto key =
			episode.family_key.empty() ? episode_key(episode) : episode.family_key;
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

// Collapse targets that the network cannot distinguish. Value depends only on
// PackedState. Policy additionally retains the exact observed graph-action mask.
AggregatedTrainingData aggregate_training_records(
	const std::vector<TrainingRecord> &records) {
	struct ValueAccumulator {
		PackedState state{};
		double weighted_sum = 0.0;
		double weighted_square_sum = 0.0;
		double weight = 0.0;
		std::size_t count = 0;
	};
	struct PolicyAccumulator {
		PackedState state{};
		std::vector<int> legal_indices;
		std::vector<double> weighted_targets;
		double weight = 0.0;
		std::size_t count = 0;
	};

	std::vector<ValueAccumulator> value_accumulators;
	std::vector<PolicyAccumulator> policy_accumulators;
	std::unordered_map<std::string, std::size_t> value_groups;
	std::unordered_map<std::string, std::size_t> policy_groups;
	std::size_t source_policy_records = 0;
	for (const auto &record : records) {
		const double value_weight = std::max(0.0F, record.value_weight);
		if (value_weight > 0.0) {
			const auto key = packed_key(record.state);
			auto [found, inserted] =
				value_groups.emplace(key, value_accumulators.size());
			if (inserted) {
				ValueAccumulator accumulator;
				accumulator.state = record.state;
				value_accumulators.push_back(std::move(accumulator));
			}
			auto &accumulator = value_accumulators[found->second];
			accumulator.weighted_sum += value_weight * record.value_target;
			accumulator.weighted_square_sum +=
				value_weight * record.value_target * record.value_target;
			accumulator.weight += value_weight;
			++accumulator.count;
		}

		const double policy_weight = std::max(0.0F, record.policy_weight);
		if (policy_weight <= 0.0) {
			continue;
		}
		++source_policy_records;
		std::vector<std::size_t> permutation(record.legal_indices.size());
		std::iota(permutation.begin(), permutation.end(), 0);
		std::sort(permutation.begin(), permutation.end(), [&](std::size_t left,
															 std::size_t right) {
			return record.legal_indices[left] < record.legal_indices[right];
		});
		std::vector<int> legal;
		std::vector<float> targets;
		legal.reserve(permutation.size());
		targets.reserve(permutation.size());
		for (const auto index : permutation) {
			legal.push_back(record.legal_indices[index]);
			targets.push_back(record.policy_target[index]);
		}
		std::string key = packed_key(record.state);
		key.push_back('|');
		for (const int action : legal) {
			key.append(reinterpret_cast<const char *>(&action), sizeof(action));
		}
		auto [found, inserted] =
			policy_groups.emplace(key, policy_accumulators.size());
		if (inserted) {
			PolicyAccumulator accumulator;
			accumulator.state = record.state;
			accumulator.legal_indices = legal;
			accumulator.weighted_targets.assign(legal.size(), 0.0);
			policy_accumulators.push_back(std::move(accumulator));
		}
		auto &accumulator = policy_accumulators[found->second];
		if (accumulator.legal_indices != legal) {
			throw std::runtime_error("BRCI policy aggregation mask collision");
		}
		for (std::size_t index = 0; index < targets.size(); ++index) {
			accumulator.weighted_targets[index] += policy_weight * targets[index];
		}
		accumulator.weight += policy_weight;
		++accumulator.count;
	}

	AggregatedTrainingData aggregated;
	aggregated.values.reserve(value_accumulators.size());
	double alias_variance_sum = 0.0;
	double alias_variance_weight = 0.0;
	std::size_t value_aliases = 0;
	for (const auto &accumulator : value_accumulators) {
		ValueTrainingRecord record;
		record.state = accumulator.state;
		record.target =
			static_cast<float>(accumulator.weighted_sum / accumulator.weight);
		record.weight = static_cast<float>(accumulator.weight);
		aggregated.values.push_back(std::move(record));
		if (accumulator.count > 1) {
			const double mean = accumulator.weighted_sum / accumulator.weight;
			const double variance = std::max(
				0.0, accumulator.weighted_square_sum / accumulator.weight - mean * mean);
			alias_variance_sum += accumulator.weight * variance;
			alias_variance_weight += accumulator.weight;
			value_aliases += accumulator.count - 1;
		}
	}
	aggregated.policies.reserve(policy_accumulators.size());
	std::size_t policy_aliases = 0;
	for (const auto &accumulator : policy_accumulators) {
		PolicyTrainingRecord record;
		record.state = accumulator.state;
		record.legal_indices = accumulator.legal_indices;
		record.target.resize(accumulator.weighted_targets.size());
		for (std::size_t index = 0; index < record.target.size(); ++index) {
			record.target[index] =
				static_cast<float>(accumulator.weighted_targets[index] / accumulator.weight);
		}
		record.target = normalize(std::move(record.target));
		record.weight = static_cast<float>(accumulator.weight);
		aggregated.policies.push_back(std::move(record));
		if (accumulator.count > 1) {
			policy_aliases += accumulator.count - 1;
		}
	}
	aggregated.summary = {
		{"source_value_records", records.size()},
		{"aggregated_value_records", aggregated.values.size()},
		{"merged_value_records", value_aliases},
		{"source_policy_records", source_policy_records},
		{"aggregated_policy_records", aggregated.policies.size()},
		{"merged_policy_records", policy_aliases},
		{"mean_value_alias_variance",
		 alias_variance_weight > 0.0 ? alias_variance_sum / alias_variance_weight : 0.0},
	};
	return aggregated;
}

nlohmann::json write_brci_h5(const std::filesystem::path &path,
							 const AggregatedTrainingData &records) {
	if (records.values.empty()) {
		throw std::runtime_error("BRCI graph contains no training records");
	}
	const std::size_t value_count = records.values.size();
	const std::size_t policy_count = records.policies.size();
	std::size_t legal_width = 1;
	for (const auto &record : records.policies) {
		legal_width = std::max(legal_width, record.legal_indices.size());
	}
	std::vector<std::uint8_t> value_states(value_count * kStatePlanes * 8);
	std::vector<float> values(value_count);
	std::vector<float> value_weights(value_count);
	for (std::size_t row = 0; row < value_count; ++row) {
		std::copy(records.values[row].state.begin(), records.values[row].state.end(),
				  value_states.begin() + row * kStatePlanes * 8);
		values[row] = records.values[row].target;
		value_weights[row] = records.values[row].weight;
	}
	std::vector<std::uint8_t> policy_states(policy_count * kStatePlanes * 8);
	std::vector<std::int32_t> legal(policy_count * legal_width, 0);
	std::vector<float> policy(policy_count * legal_width, 0.0F);
	std::vector<std::uint8_t> legal_counts(policy_count);
	std::vector<float> policy_weights(policy_count);
	for (std::size_t row = 0; row < policy_count; ++row) {
		const auto &record = records.policies[row];
		std::copy(record.state.begin(), record.state.end(),
				  policy_states.begin() + row * kStatePlanes * 8);
		legal_counts[row] = static_cast<std::uint8_t>(record.legal_indices.size());
		policy_weights[row] = record.weight;
		for (std::size_t column = 0; column < record.legal_indices.size(); ++column) {
			legal[row * legal_width + column] = record.legal_indices[column];
			policy[row * legal_width + column] = record.target[column];
		}
	}
	if (!path.parent_path().empty()) {
		std::filesystem::create_directories(path.parent_path());
	}
	const hid_t file = require_id(
		H5Fcreate(path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), path.string());
	write_string_attribute(file, "arch_type", kArchType);
	write_string_attribute(file, "brci_formula", kBrciFormula);
	write_dataset(file, "value_states", H5T_STD_U8LE, H5T_NATIVE_UINT8,
				  {value_count, kStatePlanes, 8}, value_states.data());
	write_dataset(file, "value_targets", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT,
				  {value_count}, values.data());
	write_dataset(file, "value_weights", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT,
				  {value_count}, value_weights.data());
	write_dataset(file, "policy_states", H5T_STD_U8LE, H5T_NATIVE_UINT8,
				  {policy_count, kStatePlanes, 8}, policy_states.data());
	write_dataset(file, "policy_legal_indices", H5T_STD_I32LE, H5T_NATIVE_INT32,
				  {policy_count, legal_width},
				  legal.data());
	write_dataset(file, "policy_targets", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT,
				  {policy_count, legal_width}, policy.data());
	write_dataset(file, "policy_legal_counts", H5T_STD_U8LE, H5T_NATIVE_UINT8,
				  {policy_count},
				  legal_counts.data());
	write_dataset(file, "policy_weights", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT,
				  {policy_count},
				  policy_weights.data());
	H5Fclose(file);
	return {
		{"path", path.string()},
		{"value_positions", value_count},
		{"policy_positions", policy_count},
		{"legal_width", legal_width},
		{"formula", kBrciFormula},
		{"aggregation", records.summary},
	};
}

// Clone defined parameter gradients and represent absent objective paths by zero.
std::vector<torch::Tensor> clone_gradients(const std::vector<torch::Tensor> &parameters) {
	std::vector<torch::Tensor> gradients;
	gradients.reserve(parameters.size());
	for (const auto &parameter : parameters) {
		gradients.push_back(
			parameter.grad().defined() ? parameter.grad().detach().clone()
									   : torch::zeros_like(parameter));
	}
	return gradients;
}

// Restore one objective's native gradients to an architecture-specific head.
void assign_gradients(const std::vector<torch::Tensor> &parameters,
					  const std::vector<torch::Tensor> &gradients) {
	if (parameters.size() != gradients.size()) {
		throw std::invalid_argument("BRCI gradient vectors are not aligned");
	}
	for (std::size_t index = 0; index < parameters.size(); ++index) {
		parameters[index].mutable_grad() = gradients[index];
	}
}

struct CommonGradientMetrics {
	double lambda = 0.5;
	double cosine = 0.0;
};

// Backpropagate both objectives separately. Heads keep their own gradients,
// while the shared backbone receives the minimum-norm convex combination.
CommonGradientMetrics assign_common_descent_gradients(
	Model model, torch::optim::Optimizer &optimizer,
	const torch::Tensor &policy_loss, const torch::Tensor &value_loss) {
	const auto backbone_parameters = model->backbone->parameters();
	const auto policy_parameters = model->policy_head->parameters();
	const auto value_parameters = model->value_head->parameters();

	optimizer.zero_grad();
	policy_loss.backward({}, true);
	const auto policy_backbone = clone_gradients(backbone_parameters);
	const auto policy_head = clone_gradients(policy_parameters);

	optimizer.zero_grad();
	value_loss.backward();
	const auto value_backbone = clone_gradients(backbone_parameters);
	const auto value_head = clone_gradients(value_parameters);

	auto scalar_options = torch::TensorOptions()
							  .dtype(torch::kFloat64)
							  .device(backbone_parameters.front().device());
	auto policy_norm = torch::zeros({}, scalar_options);
	auto value_norm = torch::zeros({}, scalar_options);
	auto dot = torch::zeros({}, scalar_options);
	for (std::size_t index = 0; index < backbone_parameters.size(); ++index) {
		const auto policy = policy_backbone[index].to(torch::kFloat64);
		const auto value = value_backbone[index].to(torch::kFloat64);
		policy_norm += torch::sum(policy * policy);
		value_norm += torch::sum(value * value);
		dot += torch::sum(policy * value);
	}
	const double policy_norm_value = policy_norm.item<double>();
	const double value_norm_value = value_norm.item<double>();
	const double dot_value = dot.item<double>();
	const double lambda = brci_common_descent_lambda(
		policy_norm_value, value_norm_value, dot_value);
	const double denominator =
		std::sqrt(std::max(0.0, policy_norm_value * value_norm_value));
	const double cosine = denominator > 0.0 ? dot_value / denominator : 0.0;

	optimizer.zero_grad();
	for (std::size_t index = 0; index < backbone_parameters.size(); ++index) {
		backbone_parameters[index].mutable_grad() =
			lambda * policy_backbone[index] + (1.0 - lambda) * value_backbone[index];
	}
	assign_gradients(policy_parameters, policy_head);
	assign_gradients(value_parameters, value_head);
	return {lambda, cosine};
}

// Fit a raw proposal from independent Policy and Value populations. Parameter
// backtracking later decides how much optimizer displacement is admissible.
nlohmann::json train_proposal(const std::filesystem::path &source,
							  const std::filesystem::path &proposal, Model model,
							  const torch::Device &device,
							  const AggregatedTrainingData &records,
							  const BrciOptions &options, int iteration) {
	if (records.values.empty()) {
		throw std::runtime_error("BRCI training has no Value records");
	}
	model->to(device);
	model->eval();
	torch::optim::AdamW optimizer(
		model->parameters(),
		torch::optim::AdamWOptions(options.learning_rate).weight_decay(kWeightDecay));
	std::vector<std::size_t> value_order(records.values.size());
	std::vector<std::size_t> policy_order(records.policies.size());
	std::iota(value_order.begin(), value_order.end(), 0);
	std::iota(policy_order.begin(), policy_order.end(), 0);
	std::mt19937_64 rng(options.seed + iteration * 313);
	const std::size_t requested_batch =
		static_cast<std::size_t>(std::max(1, options.batch_size));
	const std::size_t policy_batch_size =
		records.policies.empty() ? 0 : std::max<std::size_t>(1, requested_batch / 2);
	const std::size_t value_batch_size =
		std::max<std::size_t>(1, requested_batch - policy_batch_size);
	std::int64_t steps = 0;
	double total_policy = 0.0;
	double total_value = 0.0;
	double total_loss = 0.0;
	double total_lambda = 0.0;
	double total_cosine = 0.0;
	for (int epoch = 0; epoch < std::max(0, options.epochs); ++epoch) {
		std::shuffle(value_order.begin(), value_order.end(), rng);
		if (!policy_order.empty()) {
			std::shuffle(policy_order.begin(), policy_order.end(), rng);
		}
		std::size_t policy_cursor = 0;
		for (std::size_t value_begin = 0; value_begin < value_order.size();
			 value_begin += value_batch_size) {
			const auto value_end =
				std::min(value_order.size(), value_begin + value_batch_size);
			const std::size_t value_count = value_end - value_begin;
			const std::size_t policy_count = policy_order.empty()
				? 0
				: std::min(policy_batch_size, policy_order.size());
			const std::size_t total_count = policy_count + value_count;
			std::vector<std::size_t> policy_rows;
			policy_rows.reserve(policy_count);
			for (std::size_t row = 0; row < policy_count; ++row) {
				policy_rows.push_back(
					policy_order[(policy_cursor + row) % policy_order.size()]);
			}
			if (!policy_order.empty()) {
				policy_cursor = (policy_cursor + policy_count) % policy_order.size();
			}

			std::size_t width = 1;
			for (const auto index : policy_rows) {
				width = std::max(width, records.policies[index].legal_indices.size());
			}
			std::vector<std::uint8_t> packed(total_count * kStatePlanes * 8);
			for (std::size_t row = 0; row < policy_count; ++row) {
				const auto &record = records.policies[policy_rows[row]];
				std::copy(record.state.begin(), record.state.end(),
						  packed.begin() + row * kStatePlanes * 8);
			}
			for (std::size_t row = 0; row < value_count; ++row) {
				const auto &record = records.values[value_order[value_begin + row]];
				std::copy(record.state.begin(), record.state.end(),
						  packed.begin() + (policy_count + row) * kStatePlanes * 8);
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
			auto legal = torch::zeros(
				{static_cast<std::int64_t>(policy_count),
				 static_cast<std::int64_t>(width)},
				int_options);
			auto targets = torch::zeros(
				{static_cast<std::int64_t>(policy_count),
				 static_cast<std::int64_t>(width)},
				float_options);
			auto counts =
				torch::zeros({static_cast<std::int64_t>(policy_count)}, int_options);
			auto policy_weights =
				torch::zeros({static_cast<std::int64_t>(policy_count)}, float_options);
			auto values =
				torch::zeros({static_cast<std::int64_t>(value_count)}, float_options);
			auto value_weights =
				torch::zeros({static_cast<std::int64_t>(value_count)}, float_options);
			auto legal_access = legal.accessor<std::int64_t, 2>();
			auto target_access = targets.accessor<float, 2>();
			auto count_access = counts.accessor<std::int64_t, 1>();
			auto policy_weight_access = policy_weights.accessor<float, 1>();
			auto value_access = values.accessor<float, 1>();
			auto value_weight_access = value_weights.accessor<float, 1>();
			for (std::size_t row = 0; row < policy_count; ++row) {
				const auto &record = records.policies[policy_rows[row]];
				count_access[row] = record.legal_indices.size();
				policy_weight_access[row] = record.weight;
				for (std::size_t column = 0; column < record.legal_indices.size(); ++column) {
					legal_access[row][column] = record.legal_indices[column];
					target_access[row][column] = record.target[column];
				}
			}
			for (std::size_t row = 0; row < value_count; ++row) {
				const auto &record = records.values[value_order[value_begin + row]];
				value_access[row] = record.target;
				value_weight_access[row] = record.weight;
			}

			auto states = decode_states(
				packed.data(), static_cast<std::int64_t>(total_count), pin_memory)
							  .to(device, true);
			legal = legal.to(device, true);
			targets = targets.to(device, true);
			counts = counts.to(device, true);
			policy_weights = policy_weights.to(device, true);
			values = values.to(device, true);
			value_weights = value_weights.to(device, true);

			torch::Tensor logits;
			torch::Tensor predicted;
			{
				AutocastGuard autocast(options.precision, device);
				std::tie(logits, predicted) = model->forward(states);
			}
			predicted = predicted.reshape({-1}).to(torch::kFloat32);
			torch::Tensor policy_loss;
			if (policy_count > 0) {
				auto selected =
					logits.narrow(0, 0, static_cast<std::int64_t>(policy_count))
						.to(torch::kFloat32)
						.gather(1, legal);
				policy_loss = brci_masked_policy_loss(
					selected, targets, counts, policy_weights);
			} else {
				policy_loss = logits.sum() * 0.0;
			}
			auto predicted_values = predicted.narrow(
				0, static_cast<std::int64_t>(policy_count),
				static_cast<std::int64_t>(value_count));
			auto value_errors = torch::square(predicted_values - values);
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
			CommonGradientMetrics gradient_metrics;
			if (policy_count > 0) {
				gradient_metrics = assign_common_descent_gradients(
					model, optimizer, policy_loss, value_loss);
			} else {
				optimizer.zero_grad();
				value_loss.backward();
				gradient_metrics.lambda = 0.0;
			}
			torch::nn::utils::clip_grad_norm_(model->parameters(), kGradientClip);
			optimizer.step();
			++steps;
			total_policy += policy_loss.detach().item<double>();
			total_value += value_loss.detach().item<double>();
			total_loss += loss.detach().item<double>();
			total_lambda += gradient_metrics.lambda;
			total_cosine += gradient_metrics.cosine;
			if (options.log_every > 0 && (steps == 1 || steps % options.log_every == 0)) {
				std::cout << "brci train: step=" << steps
						  << " policy=" << policy_loss.detach().item<double>()
						  << " value=" << value_loss.detach().item<double>()
						  << " loss=" << loss.detach().item<double>()
						  << " shared_lambda=" << gradient_metrics.lambda
						  << " shared_cosine=" << gradient_metrics.cosine << std::endl;
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
		{"batching",
		 {
			 {"policy_batch_size", policy_batch_size},
			 {"value_batch_size", value_batch_size},
			 {"policy_records", records.policies.size()},
			 {"value_records", records.values.size()},
		 }},
		{"metrics",
		 {
			 {"loss", total_loss / divisor},
			 {"policy", total_policy / divisor},
			 {"value", total_value / divisor},
			 {"shared_lambda", total_lambda / divisor},
			 {"shared_gradient_cosine", total_cosine / divisor},
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
		auto aggregated_training = aggregate_training_records(training_records);
		const auto data_path =
			data_dir / ("brci_iter_" + zero_pad(iteration, 3) + ".h5");
		auto data_summary = write_brci_h5(data_path, aggregated_training);
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
			current, raw, model, device, aggregated_training, options, iteration);
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
