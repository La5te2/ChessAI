// Implements Melano K=2 anchored latent PUCT; search.cpp is its CLI front end.

#include "melano/search.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace melano {

namespace {

using Clock = std::chrono::steady_clock;

struct Evaluation {
	std::vector<std::vector<float>> policies;
	std::vector<float> values;
	std::vector<std::vector<float>> advantages;
	std::vector<torch::Tensor> latents;
};

struct Node {
	// Create an edge/node pair carrying P, A, and the derived Q prior V(s)+A(s,a).
	explicit Node(float initial_prior = 0.0F, chess::Move incoming = chess::Move::NO_MOVE,
				  float initial_advantage = 0.0F, float initial_q_prior = 0.0F)
		: prior(initial_prior), move(incoming), advantage(initial_advantage),
		  q_prior(initial_q_prior) {}

	// Return the empirical value from this node's side-to-move perspective.
	float q() const { return visits > 0 ? value_sum / static_cast<float>(visits) : 0.0F; }

	float prior = 0.0F;
	chess::Move move;
	float advantage = 0.0F;
	float q_prior = 0.0F;
	int visits = 0;
	float value_sum = 0.0F;
	int virtual_visits = 0;
	// K=2 anchors retain exact E(s) only at even tree depths.
	torch::Tensor anchor_latent;
	std::vector<std::unique_ptr<Node>> children;
};

struct SelectedLeaf {
	std::size_t state_index = 0;
	Node *leaf = nullptr;
	chess::Board board;
	std::vector<Node *> path;
	int depth = 0;
};

struct TreeState {
	chess::Board board;
	std::unique_ptr<Node> root = std::make_unique<Node>();
	std::vector<float> network_policy;
	std::vector<float> network_advantages;
	float network_value = 0.0F;
	int sims_completed = 0;
	int dynamic_target = 0;
	int expanded_nodes = 0;
	int nn_batches = 0;
	int exact_evaluations = 0;
	int latent_evaluations = 0;
	double total_leaf_depth = 0.0;
	int leaf_samples = 0;
	int max_leaf_depth = 0;
};

// Keep V, A-derived Q, and backed-up values inside the model's [-1, 1] convention.
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
			throw std::invalid_argument("Melano search requires a model");
		}
		options.virtual_loss = std::max(0.0, options.virtual_loss);
		validate_compute_precision(options.precision, device);
		model->to(device);
		model->eval();
	}

	// Gather only legal P/A entries before crossing the device boundary.
	Evaluation collect(ModelOutput prediction, const std::vector<chess::Board> &boards,
					   const torch::Tensor &latents = {}) {
		std::vector<std::vector<int>> legal_actions(boards.size());
		std::size_t legal_width = 1;
		for (std::size_t row = 0; row < boards.size(); ++row) {
			for (const auto &move : legal_moves(boards[row])) {
				legal_actions[row].push_back(move_to_index(move));
			}
			legal_width = std::max(legal_width, legal_actions[row].size());
		}

		const bool pin_memory = device.is_cuda();
		auto index_options =
			torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
		auto mask_options =
			torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU);
		if (pin_memory) {
			index_options = index_options.pinned_memory(true);
			mask_options = mask_options.pinned_memory(true);
		}
		auto legal_indices =
			torch::zeros({static_cast<std::int64_t>(boards.size()),
						  static_cast<std::int64_t>(legal_width)},
						 index_options);
		auto legal_mask =
			torch::zeros({static_cast<std::int64_t>(boards.size()),
						  static_cast<std::int64_t>(legal_width)},
						 mask_options);
		auto index_rows = legal_indices.accessor<std::int64_t, 2>();
		auto mask_rows = legal_mask.accessor<bool, 2>();
		for (std::size_t row = 0; row < legal_actions.size(); ++row) {
			for (std::size_t column = 0; column < legal_actions[row].size(); ++column) {
				index_rows[static_cast<std::int64_t>(row)]
						  [static_cast<std::int64_t>(column)] = legal_actions[row][column];
				mask_rows[static_cast<std::int64_t>(row)]
						 [static_cast<std::int64_t>(column)] = true;
			}
		}
		auto device_indices = legal_indices.to(device, true);
		auto device_mask = legal_mask.to(device, true);
		auto compact_logits = prediction.policy.to(torch::kFloat32).gather(1, device_indices);
		compact_logits = compact_logits.masked_fill(
			~device_mask, -std::numeric_limits<float>::infinity());
		auto probabilities =
			torch::softmax(compact_logits, 1).to(torch::kCPU).contiguous();
		auto values = prediction.value.reshape({-1})
						  .to(torch::kFloat32)
						  .to(torch::kCPU)
						  .contiguous();
		auto advantages = prediction.advantages.to(torch::kFloat32)
							  .gather(1, device_indices)
							  .to(torch::kCPU)
							  .contiguous();
		const auto rows = static_cast<std::size_t>(values.size(0));

		Evaluation output;
		output.policies.resize(rows, std::vector<float>(kActionSize, 0.0F));
		output.values.resize(rows);
		output.advantages.resize(rows, std::vector<float>(kActionSize, 0.0F));
		if (latents.defined()) {
			output.latents.reserve(rows);
		}
		auto probability_rows = probabilities.accessor<float, 2>();
		auto value_rows = values.accessor<float, 1>();
		auto advantage_rows = advantages.accessor<float, 2>();
		for (std::size_t row = 0; row < rows; ++row) {
			output.values[row] = value_rows[static_cast<std::int64_t>(row)];
			for (std::size_t column = 0; column < legal_actions[row].size(); ++column) {
				const int action = legal_actions[row][column];
				output.policies[row][action] =
					probability_rows[static_cast<std::int64_t>(row)]
									[static_cast<std::int64_t>(column)];
				output.advantages[row][action] =
					advantage_rows[static_cast<std::int64_t>(row)]
								  [static_cast<std::int64_t>(column)];
			}
			if (latents.defined()) {
				output.latents.push_back(
					latents.index({static_cast<std::int64_t>(row)}).contiguous());
			}
		}
		return output;
	}

	// Encode exact boards and retain their geometry-aware latents as K=2 anchors.
	Evaluation evaluate_exact(const std::vector<chess::Board> &boards) {
		if (boards.empty()) {
			return {};
		}
		torch::InferenceMode guard;
		const bool pin_memory = device.is_cuda();
		auto states = encode_boards(boards, pin_memory).to(device, true);
		torch::Tensor latents;
		ModelOutput prediction;
		{
			AutocastGuard autocast(options.precision, device);
			latents = model->encode(states);
			prediction = model->predict(latents);
		}
		return collect(std::move(prediction), boards, latents);
	}

	// Predict odd-depth leaves from their exact even-depth parent anchors.
	Evaluation evaluate_latent(const std::vector<torch::Tensor> &parents,
							   const std::vector<std::int64_t> &actions,
							   const std::vector<chess::Board> &boards) {
		if (parents.empty()) {
			return {};
		}
		if (parents.size() != actions.size()) {
			throw std::runtime_error("Melano latent evaluation batch is misaligned");
		}
		torch::InferenceMode guard;
		auto parent_batch = torch::stack(parents);
		auto action_batch =
			torch::tensor(actions, torch::TensorOptions().dtype(torch::kInt64).device(device));
		ModelOutput prediction;
		{
			AutocastGuard autocast(options.precision, device);
			auto successors = model->transition(parent_batch, action_batch);
			prediction = model->predict(successors);
		}
		return collect(std::move(prediction), boards);
	}

	// Expand legal edges and derive each current-player Q prior as clamp(V(s)+A(s,a)).
	void expand(Node *node, const chess::Board &board, const std::vector<float> &policy,
				const std::vector<float> &advantages, float parent_value) {
		if (!node->children.empty()) {
			return;
		}
		const auto legal_policy = normalize_legal_policy(policy, board);
		for (const auto &move : legal_moves(board)) {
			const int action = move_to_index(move);
			const float advantage = advantages[action];
			const float q_prior = clamp_unit(parent_value + advantage);
			node->children.push_back(
				std::make_unique<Node>(legal_policy[action], move, advantage, q_prior));
		}
	}

	// Sum priors already explored under a parent for first-play uncertainty reduction.
	float visited_policy_mass(const Node *parent) const {
		float mass = 0.0F;
		for (const auto &child : parent->children) {
			if (child->visits > 0) {
				mass += std::max(0.0F, child->prior);
			}
		}
		return mass;
	}

	// Blend one A-derived pseudo-visit with empirical child returns, both in parent perspective.
	float edge_value(const Node *parent, const Node *child) const {
		if (child->visits > 0) {
			return clamp_unit((static_cast<float>(child->visits) * -child->q() + child->q_prior) /
							  static_cast<float>(child->visits + 1));
		}
		return clamp_unit(child->q_prior -
			static_cast<float>(std::max(0.0, options.fpu_reduction)) *
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

	// Score an edge with PUCT using its pseudo-visit-adjusted exploitation value.
	double selection_score(const Node *parent, const Node *child) const {
		const double exploitation = edge_value(parent, child);
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
		selected.depth = depth;
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

	// Assemble final ranking plus P/V/A and pseudo-visit diagnostics for one tree.
	SearchResult make_result(TreeState &state, Clock::time_point start) const {
		SearchResult result;
		result.policy = options.type == SearchType::Closed || options.mcts_sims <= 0
							? normalize_legal_policy(state.network_policy, state.board)
							: root_policy(state);
		result.advantages = state.network_advantages;
		result.decision_scores = result.policy;
		result.value = state.root->visits > 0 ? state.root->q() : state.network_value;
		result.sims_completed = state.sims_completed;
		result.dynamic_target = options.type == SearchType::Closed ? 0 : state.dynamic_target;
		result.expanded_nodes = state.expanded_nodes;
		result.nn_batches = state.nn_batches;
		result.exact_evaluations = state.exact_evaluations;
		result.latent_evaluations = state.latent_evaluations;
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

		const auto row_count =
			std::min(moves.size(), static_cast<std::size_t>(std::max(1, options.root_topn)));
		for (std::size_t row = 0; row < row_count; ++row) {
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
					root_move.q = edge_value(state.root.get(), child.get());
					root_move.advantage = child->advantage;
					root_move.q_prior = child->q_prior;
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
		const auto roots = evaluate_exact(boards);
		const int minimum = minimum_simulations();
		for (std::size_t index = 0; index < states.size(); ++index) {
			states[index].network_policy = roots.policies[index];
			states[index].network_advantages = roots.advantages[index];
			states[index].network_value = roots.values[index];
			states[index].nn_batches = 1;
			states[index].exact_evaluations = 1;
			states[index].dynamic_target = minimum;
			states[index].root->anchor_latent = roots.latents[index];
			expand(states[index].root.get(), states[index].board, roots.policies[index],
				   roots.advantages[index], roots.values[index]);
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
					std::vector<std::size_t> exact_rows;
					std::vector<std::size_t> latent_rows;
					for (std::size_t index = begin; index < end; ++index) {
						if (selected[index].depth % 2 == 0) {
							exact_rows.push_back(index);
						} else {
							latent_rows.push_back(index);
						}
					}

					auto apply_evaluation = [&](std::size_t index, const Evaluation &evaluation,
												std::size_t row, bool exact) {
						auto &leaf = selected[index];
						auto &state = states[leaf.state_index];
						if (exact) {
							leaf.leaf->anchor_latent = evaluation.latents[row];
							state.exact_evaluations += 1;
						} else {
							state.latent_evaluations += 1;
						}
						if (leaf.leaf->children.empty()) {
							expand(leaf.leaf, leaf.board, evaluation.policies[row],
								   evaluation.advantages[row], evaluation.values[row]);
							state.expanded_nodes += 1;
						}
						clear_virtual(leaf.path);
						backpropagate(leaf.path, evaluation.values[row]);
						state.sims_completed += 1;
						progressed = true;
					};

					if (!exact_rows.empty()) {
						std::vector<chess::Board> leaf_boards;
						leaf_boards.reserve(exact_rows.size());
						for (const auto index : exact_rows) {
							leaf_boards.push_back(selected[index].board);
						}
						const auto evaluation = evaluate_exact(leaf_boards);
						std::unordered_set<std::size_t> evaluated_states;
						for (std::size_t row = 0; row < exact_rows.size(); ++row) {
							const auto index = exact_rows[row];
							apply_evaluation(index, evaluation, row, true);
							evaluated_states.insert(selected[index].state_index);
						}
						for (const auto state_index : evaluated_states) {
							states[state_index].nn_batches += 1;
						}
					}

					if (!latent_rows.empty()) {
						std::vector<torch::Tensor> parent_latents;
						std::vector<std::int64_t> actions;
						std::vector<chess::Board> leaf_boards;
						parent_latents.reserve(latent_rows.size());
						actions.reserve(latent_rows.size());
						leaf_boards.reserve(latent_rows.size());
						for (const auto index : latent_rows) {
							const auto &leaf = selected[index];
							if (leaf.path.size() < 2) {
								throw std::runtime_error(
									"odd-depth Melano leaf has no parent anchor");
							}
							const auto *parent = leaf.path[leaf.path.size() - 2];
							if (!parent->anchor_latent.defined()) {
								throw std::runtime_error(
									"odd-depth Melano leaf is missing its K=2 parent anchor");
							}
							parent_latents.push_back(parent->anchor_latent);
							actions.push_back(move_to_index(leaf.leaf->move));
							leaf_boards.push_back(leaf.board);
						}
						const auto evaluation =
							evaluate_latent(parent_latents, actions, leaf_boards);
						std::unordered_set<std::size_t> evaluated_states;
						for (std::size_t row = 0; row < latent_rows.size(); ++row) {
							const auto index = latent_rows[row];
							apply_evaluation(index, evaluation, row, false);
							evaluated_states.insert(selected[index].state_index);
						}
						for (const auto state_index : evaluated_states) {
							states[state_index].nn_batches += 1;
						}
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

} // namespace melano
