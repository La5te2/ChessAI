// Implements Gadus batched PUCT; search.cpp only supplies the command-line front end.

#include "gadus/search.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace gadus {

namespace {

using Clock = std::chrono::steady_clock;

struct Evaluation {
	std::vector<std::vector<float>> policies;
	std::vector<float> values;
};

struct Node {
	// Create an edge/node pair with its policy prior and incoming legal move.
	explicit Node(float initial_prior = 0.0F, chess::Move incoming = chess::Move::NO_MOVE)
		: prior(initial_prior), move(incoming) {}

	// Return the empirical value from this node's side-to-move perspective.
	float q() const { return visits > 0 ? value_sum / static_cast<float>(visits) : 0.0F; }

	float prior = 0.0F;
	chess::Move move;
	int visits = 0;
	float value_sum = 0.0F;
	int virtual_visits = 0;
	std::vector<std::unique_ptr<Node>> children;
};

struct SelectedLeaf {
	std::size_t state_index = 0;
	Node *leaf = nullptr;
	chess::Board board;
	std::vector<Node *> path;
};

struct TreeState {
	chess::Board board;
	std::unique_ptr<Node> root = std::make_unique<Node>();
	std::vector<float> network_policy;
	float network_value = 0.0F;
	int sims_completed = 0;
	int dynamic_target = 0;
	int expanded_nodes = 0;
	int nn_batches = 0;
	double total_leaf_depth = 0.0;
	int leaf_samples = 0;
	int max_leaf_depth = 0;
};

// Keep bounded value arithmetic inside the model's [-1, 1] convention.
float clamp_unit(float value) { return std::clamp(value, -1.0F, 1.0F); }

// Convert steady-clock elapsed time to seconds for deadlines and reporting.
double seconds_since(Clock::time_point start) {
	return std::chrono::duration<double>(Clock::now() - start).count();
}

// Centralize optional deadline checks so a zero movetime means no time cap.
bool deadline_reached(const std::optional<Clock::time_point> &deadline) {
	return deadline.has_value() && Clock::now() >= *deadline;
}

// Remove temporary reservations made while assembling one neural evaluation batch.
void clear_virtual(const std::vector<Node *> &path) {
	for (std::size_t index = 1; index < path.size(); ++index) {
		path[index]->virtual_visits = std::max(0, path[index]->virtual_visits - 1);
	}
}

// Back up a leaf value and negate it at every ply because side to move alternates.
void backpropagate(const std::vector<Node *> &path, float value) {
	for (auto iterator = path.rbegin(); iterator != path.rend(); ++iterator) {
		(*iterator)->visits += 1;
		(*iterator)->value_sum += value;
		value = -value;
	}
}

// Detect terminal leaves and optionally expose their exact side-to-move outcome.
bool is_terminal(const chess::Board &board, float *value = nullptr) {
	if (!game_is_over(board)) {
		return false;
	}
	if (value != nullptr) {
		*value = terminal_value_side_to_move(board);
	}
	return true;
}

} // namespace

struct Searcher::Impl {
	// Move the model once to its inference device and sanitize non-negative virtual loss.
	Impl(Model source_model, torch::Device source_device, SearchOptions source_options)
		: model(std::move(source_model)), device(std::move(source_device)),
		  options(source_options) {
		if (!model) {
			throw std::invalid_argument("Gadus search requires a model");
		}
		options.virtual_loss = std::max(0.0, options.virtual_loss);
		model->to(device);
		model->eval();
	}

	// Evaluate independent boards in one LibTorch batch and copy P/V outputs to host memory.
	Evaluation evaluate(const std::vector<chess::Board> &boards) {
		if (boards.empty()) {
			return {};
		}
		torch::InferenceMode guard;
		auto states = encode_boards(boards).to(device, true);
		auto [logits, raw_values] = model->forward(states);
		auto probabilities = torch::softmax(logits, 1).to(torch::kCPU).contiguous();
		auto values = raw_values.reshape({-1}).to(torch::kCPU).contiguous();

		Evaluation output;
		output.policies.resize(boards.size(), std::vector<float>(kActionSize));
		output.values.resize(boards.size());
		auto probability_rows = probabilities.accessor<float, 2>();
		auto value_rows = values.accessor<float, 1>();
		for (std::size_t row = 0; row < boards.size(); ++row) {
			std::copy_n(&probability_rows[static_cast<std::int64_t>(row)][0], kActionSize,
						output.policies[row].begin());
			output.values[row] = value_rows[static_cast<std::int64_t>(row)];
		}
		return output;
	}

	// Create one child per legal action using the masked, normalized network policy.
	void expand(Node *node, const chess::Board &board, const std::vector<float> &policy) {
		if (!node->children.empty()) {
			return;
		}
		const auto legal_policy = normalize_legal_policy(policy, board);
		for (const auto &move : legal_moves(board)) {
			node->children.push_back(
				std::make_unique<Node>(legal_policy[move_to_index(move)], move));
		}
	}

	// Sum priors already explored under a parent for FPU reduction.
	float visited_policy_mass(const Node *parent) const {
		float mass = 0.0F;
		for (const auto &child : parent->children) {
			if (child->visits > 0) {
				mass += std::max(0.0F, child->prior);
			}
		}
		return mass;
	}

	// Estimate an unvisited edge as parent Q minus uncertainty proportional to explored prior mass.
	float fpu(const Node *parent) const {
		const float parent_q = parent->visits > 0 ? parent->q() : 0.0F;
		return clamp_unit(parent_q - static_cast<float>(std::max(0.0, options.fpu_reduction)) *
										 std::sqrt(visited_policy_mass(parent)));
	}

	// Increase exploration logarithmically with parent visits: c_init + factor*log((N+base+1)/base).
	double scheduled_c_puct(const Node *parent) const {
		const double visits = std::max(0, parent->visits + parent->virtual_visits);
		const double base = std::max(1.0, options.c_puct_base);
		const double growth =
			std::max(0.0, options.c_puct_factor) * std::log((visits + base + 1.0) / base);
		return std::max(0.0, options.c_puct + growth);
	}

	// Score an edge with PUCT: Q + c_puct*P*sqrt(N_parent)/(1+N_child) - virtual loss.
	double selection_score(const Node *parent, const Node *child) const {
		const double exploitation = child->visits > 0 ? -child->q() : fpu(parent);
		const int child_visits = child->visits + child->virtual_visits;
		const double exploration = scheduled_c_puct(parent) * child->prior *
								   std::sqrt(parent->visits + parent->virtual_visits + 1.0) /
								   (1.0 + child_visits);
		return exploitation + exploration - options.virtual_loss * child->virtual_visits;
	}

	// Choose the maximum PUCT edge with deterministic prior and UCI tie breaks.
	Node *select_child(Node *parent) const {
		if (parent->children.empty()) {
			return nullptr;
		}
		return std::max_element(parent->children.begin(), parent->children.end(),
								[&](const auto &left, const auto &right) {
									const double left_score = selection_score(parent, left.get());
									const double right_score = selection_score(parent, right.get());
									if (left_score != right_score) {
										return left_score < right_score;
									}
									if (left->prior != right->prior) {
										return left->prior < right->prior;
									}
									return move_uci(left->move) < move_uci(right->move);
								})
			->get();
	}

	// Descend to an unexpanded or terminal node while reserving the path for batching.
	SelectedLeaf select_leaf(std::size_t state_index, TreeState &state) const {
		SelectedLeaf selected;
		selected.state_index = state_index;
		selected.board = state.board;
		selected.leaf = state.root.get();
		selected.path.push_back(selected.leaf);
		while (!selected.leaf->children.empty()) {
			selected.leaf = select_child(selected.leaf);
			selected.leaf->virtual_visits += 1;
			selected.board.makeMove(selected.leaf->move);
			selected.path.push_back(selected.leaf);
			if (game_is_over(selected.board)) {
				break;
			}
		}
		const int depth = static_cast<int>(selected.path.size()) - 1;
		state.total_leaf_depth += depth;
		state.leaf_samples += 1;
		state.max_leaf_depth = std::max(state.max_leaf_depth, depth);
		return selected;
	}

	// Combine normalized visit entropy, top-two visit proximity, and top-two Q proximity.
	double uncertainty(const Node *root) const {
		if (root->children.size() <= 1) {
			return 0.0;
		}
		double total = 0.0;
		for (const auto &child : root->children) {
			total += child->visits;
		}
		if (total <= 0.0) {
			for (const auto &child : root->children) {
				total += std::max(0.0F, child->prior);
			}
		}
		double entropy = 0.0;
		for (const auto &child : root->children) {
			const double weight = root->visits > 0 ? child->visits : std::max(0.0F, child->prior);
			const double probability = weight / std::max(1e-12, total);
			if (probability > 0.0) {
				entropy -= probability * std::log(probability);
			}
		}
		entropy /= std::max(1e-12, std::log(static_cast<double>(root->children.size())));

		std::vector<const Node *> ordered;
		ordered.reserve(root->children.size());
		for (const auto &child : root->children) {
			ordered.push_back(child.get());
		}
		std::sort(ordered.begin(), ordered.end(), [](const Node *left, const Node *right) {
			return std::pair(left->visits, left->prior) > std::pair(right->visits, right->prior);
		});
		const double first = ordered[0]->visits;
		const double second = ordered[1]->visits;
		const double visit_uncertainty =
			1.0 - std::abs(first - second) / std::max(1.0, first + second);
		const double q_uncertainty =
			1.0 - std::min(1.0, std::abs(-ordered[0]->q() + ordered[1]->q()) / 0.5);
		return std::clamp(0.5 * entropy + 0.35 * visit_uncertainty + 0.15 * q_uncertainty, 0.0,
						  1.0);
	}

	// Establish the mandatory simulation floor before uncertainty can extend the budget.
	int minimum_simulations() const {
		const int cap = std::max(0, options.mcts_sims);
		if (cap == 0) {
			return 0;
		}
		const int configured = options.mcts_min_sims > 0
								   ? options.mcts_min_sims
								   : std::max(std::max(1, options.mcts_batch_size), cap / 4);
		return std::max(1, std::min(cap, configured));
	}

	// Interpolate from minimum to the hard cap using the current root uncertainty.
	int dynamic_target(const Node *root, int minimum) const {
		const int cap = std::max(0, options.mcts_sims);
		const int desired =
			minimum + static_cast<int>(std::ceil(uncertainty(root) * std::max(0, cap - minimum)));
		return std::max(minimum, std::min(cap, desired));
	}

	// Convert root visits to legal move probabilities; priors keep zero-visit moves representable.
	std::vector<float> root_policy(const TreeState &state) const {
		std::vector<float> policy(kActionSize, 0.0F);
		for (const auto &child : state.root->children) {
			policy[move_to_index(child->move)] = child->visits + child->prior;
		}
		return normalize_legal_policy(policy, state.board);
	}

	// Apply optional post-search ranking rules without modifying priors or the MCTS tree.
	void apply_decision_components(const chess::Board &board, float root_value,
								   std::vector<float> &scores, std::unordered_set<int> &repetitions,
								   std::unordered_set<int> &mates) const {
		if (options.instant_mate_first) {
			int selected = -1;
			float selected_score = -std::numeric_limits<float>::infinity();
			for (const auto &move : legal_moves(board)) {
				auto probe = board;
				probe.makeMove(move);
				if (probe.isGameOver().first != chess::GameResultReason::CHECKMATE) {
					continue;
				}
				const int index = move_to_index(move);
				mates.insert(index);
				if (scores[index] > selected_score) {
					selected = index;
					selected_score = scores[index];
				}
			}
			if (selected >= 0) {
				scores[selected] = 1.0F;
			}
		}

		const float deduction =
			static_cast<float>(std::clamp(options.repetition_policy_penalty, 0.0, 1.0) *
							   std::clamp(static_cast<double>(root_value), 0.0, 1.0));
		if (deduction <= 0.0F) {
			return;
		}
		for (const auto &move : legal_moves(board)) {
			auto probe = board;
			probe.makeMove(move);
			bool repetition = probe.isRepetition(2);
			if (!repetition) {
				for (const auto &reply : legal_moves(probe)) {
					auto response = probe;
					response.makeMove(reply);
					if (response.isRepetition(2)) {
						repetition = true;
						break;
					}
				}
			}
			if (repetition) {
				const int index = move_to_index(move);
				scores[index] = std::max(0.0F, scores[index] - deduction);
				repetitions.insert(index);
			}
		}
	}

	// Assemble the final ranked move list and diagnostics from one completed tree.
	SearchResult make_result(TreeState &state, Clock::time_point start) const {
		SearchResult result;
		result.policy = options.type == SearchType::Closed || options.mcts_sims <= 0
							? normalize_legal_policy(state.network_policy, state.board)
							: root_policy(state);
		result.decision_scores = result.policy;
		result.value = state.root->visits > 0 ? state.root->q() : state.network_value;
		result.sims_completed = state.sims_completed;
		result.dynamic_target = options.type == SearchType::Closed ? 0 : state.dynamic_target;
		result.expanded_nodes = state.expanded_nodes;
		result.nn_batches = state.nn_batches;
		result.uncertainty =
			options.type == SearchType::Closed ? 0.0 : uncertainty(state.root.get());
		result.elapsed_ms = seconds_since(start) * 1000.0;

		std::unordered_set<int> repetitions;
		std::unordered_set<int> mates;
		apply_decision_components(state.board, result.value, result.decision_scores, repetitions,
								  mates);

		auto moves = legal_moves(state.board);
		std::sort(
			moves.begin(), moves.end(), [&](const chess::Move &left, const chess::Move &right) {
				const int left_index = move_to_index(left);
				const int right_index = move_to_index(right);
				if (result.decision_scores[left_index] != result.decision_scores[right_index]) {
					return result.decision_scores[left_index] > result.decision_scores[right_index];
				}
				if (result.policy[left_index] != result.policy[right_index]) {
					return result.policy[left_index] > result.policy[right_index];
				}
				return move_uci(left) > move_uci(right);
			});
		if (moves.empty()) {
			throw std::runtime_error("game is already over");
		}
		result.move = moves.front();

		const int row_count = std::min<int>(std::max(1, options.root_topn), moves.size());
		for (int row = 0; row < row_count; ++row) {
			const auto move = moves[row];
			const int action = move_to_index(move);
			RootMove root_move;
			root_move.move = move;
			root_move.probability = result.policy[action];
			root_move.decision_score = result.decision_scores[action];
			root_move.repetition_penalized = repetitions.contains(action);
			root_move.instant_mate = mates.contains(action);
			for (const auto &child : state.root->children) {
				if (child->move == move) {
					root_move.prior = child->prior;
					root_move.visits = child->visits;
					root_move.q = child->visits > 0 ? -child->q() : fpu(state.root.get());
					break;
				}
			}
			result.root.push_back(root_move);
		}
		return result;
	}

	// Search many independent roots while sharing neural leaf batches across games.
	std::vector<SearchResult> search_many(const std::vector<chess::Board> &boards,
									   const SearchProgressCallback &progress = {},
									   int progress_interval_ms = 0) {
		if (boards.empty()) {
			return {};
		}
		for (const auto &board : boards) {
			if (game_is_over(board)) {
				throw std::runtime_error("game is already over");
			}
		}
		const auto start = Clock::now();
		std::optional<Clock::time_point> deadline;
		if (options.movetime_ms > 0.0) {
			deadline = start + std::chrono::duration_cast<Clock::duration>(
								   std::chrono::duration<double, std::milli>(options.movetime_ms));
		}

		std::vector<TreeState> states;
		states.reserve(boards.size());
		for (const auto &board : boards) {
			TreeState state;
			state.board = board;
			states.push_back(std::move(state));
		}
		const auto roots = evaluate(boards);
		const int minimum = minimum_simulations();
		for (std::size_t index = 0; index < states.size(); ++index) {
			states[index].network_policy = roots.policies[index];
			states[index].network_value = roots.values[index];
			states[index].nn_batches = 1;
			states[index].dynamic_target = minimum;
			expand(states[index].root.get(), states[index].board, roots.policies[index]);
			states[index].expanded_nodes = 1;
		}
		auto next_progress = start;
		if (progress && states.size() == 1) {
			progress(make_result(states[0], start));
			next_progress = Clock::now() +
				std::chrono::milliseconds(std::max(1, progress_interval_ms));
		}

		if (options.type == SearchType::OnlyMcts && options.mcts_sims > 0) {
			const int batch_size = std::max(1, options.mcts_batch_size);
			while (!deadline_reached(deadline)) {
				bool active = false;
				bool progressed = false;
				std::vector<SelectedLeaf> selected;
				for (std::size_t state_index = 0; state_index < states.size(); ++state_index) {
					auto &state = states[state_index];
					if (state.sims_completed >= options.mcts_sims ||
						state.sims_completed >= state.dynamic_target) {
						continue;
					}
					active = true;
					const int wanted =
						std::min({batch_size, options.mcts_sims - state.sims_completed,
								  state.dynamic_target - state.sims_completed});
					std::unordered_set<Node *> selected_nodes;
					for (int attempt = 0, accepted = 0;
						 accepted < wanted && attempt < std::max(wanted * 5, wanted + 8);
						 ++attempt) {
						if (deadline_reached(deadline)) {
							break;
						}
						auto leaf = select_leaf(state_index, state);
						float terminal = 0.0F;
						if (is_terminal(leaf.board, &terminal)) {
							clear_virtual(leaf.path);
							backpropagate(leaf.path, terminal);
							state.sims_completed += 1;
							progressed = true;
							continue;
						}
						if (!selected_nodes.insert(leaf.leaf).second) {
							clear_virtual(leaf.path);
							continue;
						}
						selected.push_back(std::move(leaf));
						++accepted;
					}
				}

				for (std::size_t begin = 0; begin < selected.size(); begin += batch_size) {
					const auto end = std::min(selected.size(), begin + batch_size);
					std::vector<chess::Board> leaf_boards;
					for (std::size_t index = begin; index < end; ++index) {
						leaf_boards.push_back(selected[index].board);
					}
					const auto evaluation = evaluate(leaf_boards);
					std::unordered_set<std::size_t> evaluated_states;
					for (std::size_t index = begin; index < end; ++index) {
						auto &leaf = selected[index];
						auto &state = states[leaf.state_index];
						const std::size_t row = index - begin;
						if (leaf.leaf->children.empty()) {
							expand(leaf.leaf, leaf.board, evaluation.policies[row]);
							state.expanded_nodes += 1;
						}
						clear_virtual(leaf.path);
						backpropagate(leaf.path, evaluation.values[row]);
						state.sims_completed += 1;
						evaluated_states.insert(leaf.state_index);
						progressed = true;
					}
					for (const auto state_index : evaluated_states) {
						states[state_index].nn_batches += 1;
					}
				}

				for (auto &state : states) {
					if (state.sims_completed >= minimum) {
						state.dynamic_target = dynamic_target(state.root.get(), minimum);
					}
				}
				if (progress && states.size() == 1 && progress_interval_ms > 0 &&
					Clock::now() >= next_progress) {
					progress(make_result(states[0], start));
					next_progress = Clock::now() +
						std::chrono::milliseconds(progress_interval_ms);
				}
				if (!active || !progressed) {
					break;
				}
			}
		}

		std::vector<SearchResult> results;
		results.reserve(states.size());
		for (auto &state : states) {
			results.push_back(make_result(state, start));
		}
		return results;
	}

	Model model;
	torch::Device device;
	SearchOptions options;
};

// Construct the public value-type wrapper around the shared implementation.
Searcher::Searcher(Model model, torch::Device device, SearchOptions options)
	: impl_(std::make_shared<Impl>(std::move(model), std::move(device), options)) {}

// Search one position and expose timed snapshots to interactive front ends.
SearchResult Searcher::search(const chess::Board &board,
							  const SearchProgressCallback &progress, int progress_interval_ms) {
	return impl_->search_many({board}, progress, progress_interval_ms)[0];
}

// Search a batch without progress callbacks, as used by arena and FCPI.
std::vector<SearchResult> Searcher::search_many(const std::vector<chess::Board> &boards) {
	return impl_->search_many(boards);
}

// Convert the command-line search mode to the strongly typed enum.
SearchType parse_search_type(const std::string &value) {
	if (value == "closed") {
		return SearchType::Closed;
	}
	if (value == "only-mcts") {
		return SearchType::OnlyMcts;
	}
	throw std::invalid_argument("search-type must be closed or only-mcts");
}

// Convert a search mode back to its stable external spelling.
std::string search_type_name(SearchType value) {
	return value == SearchType::Closed ? "closed" : "only-mcts";
}

} // namespace gadus
