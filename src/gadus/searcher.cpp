// Implements Gadus batched PUCT; search.cpp only supplies the command-line front end.

#include "gadus/search.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <list>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <torch/utils.h>

namespace gadus {

namespace {

using Clock = std::chrono::steady_clock;

struct EvaluationRow {
	std::vector<chess::Move> legal_moves;
	std::vector<int> legal_indices;
	std::vector<float> legal_policy;
	float value = 0.0F;
	bool reused = false;
};

struct PackedStateHash {
	std::size_t operator()(const PackedState &state) const noexcept {
		std::size_t hash = sizeof(std::size_t) == 8 ? 1469598103934665603ULL : 2166136261U;
		const std::size_t prime = sizeof(std::size_t) == 8 ? 1099511628211ULL : 16777619U;
		for (const auto byte : state) {
			hash ^= byte;
			hash *= prime;
		}
		return hash;
	}
};

class EvaluationCache {
	struct Entry {
		EvaluationRow evaluation;
		std::list<PackedState>::iterator recency;
		std::size_t bytes = 0;
	};

	using Entries = std::unordered_map<PackedState, Entry, PackedStateHash>;

	public:
	/// Creates a cache with an approximate byte ceiling; the maximum size disables eviction.
	explicit EvaluationCache(
		std::size_t capacity_bytes = std::numeric_limits<std::size_t>::max())
		: capacity_bytes_(capacity_bytes) {}

	/// Copies a cached network result and marks the entry as recently used.
	bool get(const PackedState &state, EvaluationRow &output) {
		const auto found = entries_.find(state);
		if (found == entries_.end()) {
			return false;
		}
		recency_.splice(recency_.end(), recency_, found->second.recency);
		output = found->second.evaluation;
		return true;
	}

	/// Inserts or replaces one compact Policy/Value row and evicts least-recently-used entries.
	void put(const PackedState &state, const EvaluationRow &evaluation) {
		if (capacity_bytes_ == 0) {
			return;
		}
		const std::size_t bytes = entry_bytes(evaluation);
		if (bytes > capacity_bytes_) {
			return;
		}
		if (const auto found = entries_.find(state); found != entries_.end()) {
			used_bytes_ -= found->second.bytes;
			recency_.erase(found->second.recency);
			entries_.erase(found);
		}
		while (!recency_.empty() && used_bytes_ + bytes > capacity_bytes_) {
			const auto oldest = entries_.find(recency_.front());
			if (oldest != entries_.end()) {
				used_bytes_ -= oldest->second.bytes;
				entries_.erase(oldest);
			}
			recency_.pop_front();
		}
		recency_.push_back(state);
		auto iterator = std::prev(recency_.end());
		entries_.emplace(state, Entry{evaluation, iterator, bytes});
		used_bytes_ += bytes;
	}

	/// Changes the byte ceiling and evicts old entries until the cache fits.
	void set_capacity(std::size_t capacity_bytes) {
		capacity_bytes_ = capacity_bytes;
		while (!recency_.empty() && used_bytes_ > capacity_bytes_) {
			const auto oldest = entries_.find(recency_.front());
			if (oldest != entries_.end()) {
				used_bytes_ -= oldest->second.bytes;
				entries_.erase(oldest);
			}
			recency_.pop_front();
		}
	}

	/// Clears all retained rows without changing the configured byte ceiling.
	void clear() {
		entries_.clear();
		recency_.clear();
		used_bytes_ = 0;
	}

	private:
	// Include vector storage and conservative container overhead in the memory budget.
	static std::size_t entry_bytes(const EvaluationRow &evaluation) {
		return sizeof(PackedState) * 2 + sizeof(Entry) + 64 +
			   evaluation.legal_moves.capacity() * sizeof(chess::Move) +
			   evaluation.legal_indices.capacity() * sizeof(int) +
			   evaluation.legal_policy.capacity() * sizeof(float);
	}

	std::size_t capacity_bytes_;
	std::size_t used_bytes_ = 0;
	std::list<PackedState> recency_;
	Entries entries_;
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
	std::vector<Node> children;
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
	EvaluationRow network;
	float network_value = 0.0F;
	int sims_completed = 0;
	int dynamic_target = 0;
	int expanded_nodes = 0;
	int nn_batches = 0;
	int nn_evaluations = 0;
	int evaluation_reuses = 0;
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
	for (std::size_t index = 0; index < path.size(); ++index) {
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
		set_options(source_options);
		model->to(device);
		model->eval();
		model->fuse_for_inference();
		auto parameters = model->parameters();
		for (auto &parameter : parameters) {
			if (parameter.dim() == 4) {
				parameter.set_data(
					parameter.contiguous(torch::MemoryFormat::ChannelsLast));
			}
		}
	}

	// Apply mutable search controls without rebuilding the model or discarding compatible cache rows.
	void set_options(SearchOptions source_options) {
		validate_compute_precision(source_options.precision, device);
		source_options.virtual_loss = std::max(0.0, source_options.virtual_loss);
		source_options.evaluation_cache_mb =
			std::clamp(source_options.evaluation_cache_mb, 0, 65536);
		if (device.is_cpu() && source_options.cpu_threads > 0) {
			torch::set_num_threads(source_options.cpu_threads);
		}
		active_cpu_threads = device.is_cpu() ? torch::get_num_threads() : 0;
		persistent_evaluation_cache.set_capacity(
			static_cast<std::size_t>(source_options.evaluation_cache_mb) * 1024 * 1024);
		options = source_options;
	}

	// Evaluate a frozen batch and retain legal moves so tree expansion does not generate them again.
	std::vector<EvaluationRow> evaluate_rows(const std::vector<chess::Board> &boards,
											 EvaluationCache *cache = nullptr) {
		if (boards.empty()) {
			return {};
		}
		std::vector<EvaluationRow> output(boards.size());
		std::vector<PackedState> pending_states;
		std::vector<EvaluationRow> pending;
		std::vector<std::size_t> row_to_pending(boards.size());
		std::vector<bool> row_cached(boards.size(), false);
		std::vector<bool> row_reused(boards.size(), false);
		std::unordered_map<PackedState, std::size_t, PackedStateHash> pending_lookup;
		pending_states.reserve(boards.size());
		pending.reserve(boards.size());
		pending_lookup.reserve(boards.size() * 2);
		for (std::size_t row = 0; row < boards.size(); ++row) {
			const auto state = encode_state(boards[row]);
			if (cache != nullptr) {
				if (cache->get(state, output[row])) {
					output[row].reused = true;
					row_cached[row] = true;
					continue;
				}
			}
			const auto [found, inserted] =
				pending_lookup.emplace(state, pending.size());
			row_to_pending[row] = found->second;
			if (!inserted) {
				row_reused[row] = true;
				continue;
			}
			pending_states.push_back(state);
			pending.emplace_back();
			pending.back().legal_moves = legal_moves(boards[row]);
			pending.back().legal_indices.reserve(pending.back().legal_moves.size());
			for (const auto &move : pending.back().legal_moves) {
				pending.back().legal_indices.push_back(move_to_index(move));
			}
		}
		if (pending.empty()) {
			return output;
		}

		std::size_t legal_width = 1;
		for (const auto &evaluation : pending) {
			legal_width = std::max(legal_width, evaluation.legal_indices.size());
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
			torch::zeros({static_cast<std::int64_t>(pending.size()),
						  static_cast<std::int64_t>(legal_width)},
						 index_options);
		auto legal_mask =
			torch::zeros({static_cast<std::int64_t>(pending.size()),
						  static_cast<std::int64_t>(legal_width)},
						 mask_options);
		auto index_rows = legal_indices.accessor<std::int64_t, 2>();
		auto mask_rows = legal_mask.accessor<bool, 2>();
		for (std::size_t row = 0; row < pending.size(); ++row) {
			for (std::size_t column = 0; column < pending[row].legal_indices.size(); ++column) {
				index_rows[static_cast<std::int64_t>(row)]
						  [static_cast<std::int64_t>(column)] =
					pending[row].legal_indices[column];
				mask_rows[static_cast<std::int64_t>(row)]
						 [static_cast<std::int64_t>(column)] = true;
			}
		}

		torch::InferenceMode guard;
		auto states = decode_states_device(
			reinterpret_cast<const std::uint8_t *>(pending_states.data()),
			static_cast<std::int64_t>(pending_states.size()), device);
		states = states.contiguous(torch::MemoryFormat::ChannelsLast);
		auto device_indices = legal_indices.to(device, true);
		auto device_mask = legal_mask.to(device, true);
		torch::Tensor logits;
		torch::Tensor raw_values;
		{
			AutocastGuard autocast(options.precision, device);
			std::tie(logits, raw_values) = model->forward_legal(states, device_indices);
		}
		auto compact_logits = logits.to(torch::kFloat32);
		compact_logits = compact_logits.masked_fill(
			~device_mask, -std::numeric_limits<float>::infinity());
		auto probabilities =
			torch::softmax(compact_logits, 1).to(torch::kCPU).contiguous();
		auto values = raw_values.reshape({-1})
						  .to(torch::kFloat32)
						  .to(torch::kCPU)
						  .contiguous();

		auto probability_rows = probabilities.accessor<float, 2>();
		auto value_rows = values.accessor<float, 1>();
		for (std::size_t row = 0; row < pending.size(); ++row) {
			pending[row].legal_policy.resize(pending[row].legal_indices.size());
			for (std::size_t column = 0; column < pending[row].legal_indices.size(); ++column) {
				pending[row].legal_policy[column] =
					probability_rows[static_cast<std::int64_t>(row)]
									[static_cast<std::int64_t>(column)];
			}
			pending[row].value = value_rows[static_cast<std::int64_t>(row)];
			if (cache != nullptr) {
				auto cached = pending[row];
				cached.reused = false;
				cache->put(pending_states[row], cached);
			}
		}
		for (std::size_t row = 0; row < output.size(); ++row) {
			if (!row_cached[row]) {
				output[row] = pending[row_to_pending[row]];
				output[row].reused = row_reused[row];
			}
		}
		return output;
	}

	// Remove move objects from the public frozen-model result used by FCPI generation.
	std::vector<ClosedEvaluation> evaluate_closed(const std::vector<chess::Board> &boards) {
		auto rows = evaluate_rows(boards);
		std::vector<ClosedEvaluation> output(rows.size());
		for (std::size_t row = 0; row < rows.size(); ++row) {
			output[row].legal_indices = std::move(rows[row].legal_indices);
			output[row].legal_policy = std::move(rows[row].legal_policy);
			output[row].value = rows[row].value;
		}
		return output;
	}

	// Normalize an already masked legal policy without allocating the 4672-action vector.
	std::vector<float> normalize_compact_policy(const EvaluationRow &evaluation) const {
		std::vector<float> normalized(evaluation.legal_policy.size(), 0.0F);
		if (normalized.empty()) {
			return normalized;
		}
		double total = 0.0;
		for (std::size_t index = 0; index < normalized.size(); ++index) {
			normalized[index] = std::max(0.0F, evaluation.legal_policy[index]);
			total += normalized[index];
		}
		if (total <= 0.0) {
			std::fill(normalized.begin(), normalized.end(),
					  1.0F / static_cast<float>(normalized.size()));
		} else {
			for (auto &value : normalized) {
				value = static_cast<float>(value / total);
			}
		}
		return normalized;
	}

	// Materialize a public dense policy only when a final or progress result requires it.
	std::vector<float> dense_policy(const EvaluationRow &evaluation) const {
		std::vector<float> dense(kActionSize, 0.0F);
		const auto normalized = normalize_compact_policy(evaluation);
		for (std::size_t index = 0; index < evaluation.legal_indices.size(); ++index) {
			dense[evaluation.legal_indices[index]] = normalized[index];
		}
		return dense;
	}

	// Create one child per legal action directly from the compact network evaluation.
	void expand(Node *node, const EvaluationRow &evaluation) {
		if (!node->children.empty()) {
			return;
		}
		if (evaluation.legal_moves.size() != evaluation.legal_policy.size()) {
			throw std::runtime_error("legal move and policy widths differ");
		}
		const auto normalized = normalize_compact_policy(evaluation);
		node->children.reserve(evaluation.legal_moves.size());
		for (std::size_t index = 0; index < evaluation.legal_moves.size(); ++index) {
			node->children.emplace_back(normalized[index], evaluation.legal_moves[index]);
		}
	}

	// Sum priors already explored under a parent for FPU reduction.
	float visited_policy_mass(const Node *parent) const {
		float mass = 0.0F;
		for (const auto &child : parent->children) {
			if (child.visits > 0) {
				mass += std::max(0.0F, child.prior);
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

	// Return the edge value in the parent side-to-move perspective.
	float edge_value(const Node *parent, const Node *child) const {
		return child->visits > 0 ? -child->q() : fpu(parent);
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
		const double exploitation = edge_value(parent, child);
		const int child_visits = child->visits + child->virtual_visits;
		const double exploration = scheduled_c_puct(parent) * child->prior *
								   std::sqrt(parent->visits + parent->virtual_visits + 1.0) /
								   (1.0 + child_visits);
		return exploitation + exploration - options.virtual_loss * child->virtual_visits;
	}

	// Break equal PUCT scores by policy prior, then by the edge value in the parent perspective.
	Node *select_child(Node *parent) const {
		if (parent->children.empty()) {
			return nullptr;
		}
		auto selected = std::max_element(parent->children.begin(), parent->children.end(),
									 [&] (const Node &left, const Node &right) {
									const double left_score = selection_score(parent, &left);
									const double right_score = selection_score(parent, &right);
									if (left_score != right_score) {
										return left_score < right_score;
									}
									if (left.prior != right.prior) {
										return left.prior < right.prior;
									}
									return edge_value(parent, &left) <
										   edge_value(parent, &right);
								});
		return &*selected;
	}

	// Descend to an unexpanded or terminal node while reserving the path for batching.
	SelectedLeaf select_leaf(std::size_t state_index, TreeState &state) const {
		SelectedLeaf selected;
		selected.state_index = state_index;
		selected.board = state.board;
		selected.leaf = state.root.get();
		selected.path.reserve(16);
		selected.path.push_back(selected.leaf);
		selected.leaf->virtual_visits += 1;
		while (!selected.leaf->children.empty()) {
			selected.leaf = select_child(selected.leaf);
			selected.leaf->virtual_visits += 1;
			selected.board.makeMove(selected.leaf->move);
			selected.path.push_back(selected.leaf);
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
			total += child.visits;
		}
		if (total <= 0.0) {
			for (const auto &child : root->children) {
				total += std::max(0.0F, child.prior);
			}
		}
		double entropy = 0.0;
		for (const auto &child : root->children) {
			const double weight = root->visits > 0 ? child.visits : std::max(0.0F, child.prior);
			const double probability = weight / std::max(1e-12, total);
			if (probability > 0.0) {
				entropy -= probability * std::log(probability);
			}
		}
		entropy /= std::max(1e-12, std::log(static_cast<double>(root->children.size())));

		std::vector<const Node *> ordered;
		ordered.reserve(root->children.size());
		for (const auto &child : root->children) {
			ordered.push_back(&child);
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
			policy[move_to_index(child.move)] = child.visits + child.prior;
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
							? dense_policy(state.network)
							: root_policy(state);
		result.decision_scores = result.policy;
		result.value = state.root->visits > 0 ? state.root->q() : state.network_value;
		result.sims_completed = state.sims_completed;
		result.dynamic_target = options.type == SearchType::Closed ? 0 : state.dynamic_target;
		result.expanded_nodes = state.expanded_nodes;
		result.nn_batches = state.nn_batches;
		result.nn_evaluations = state.nn_evaluations;
		result.evaluation_reuses = state.evaluation_reuses;
		result.cpu_threads = active_cpu_threads;
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
				if (child.move == move) {
					root_move.prior = child.prior;
					root_move.visits = child.visits;
					root_move.q = edge_value(state.root.get(), &child);
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
									   int progress_interval_ms = 0,
									   const SearchCancelCallback &cancel = {}) {
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
		EvaluationCache local_evaluation_cache;
		auto *evaluation_cache = options.evaluation_cache_mb > 0
								 ? &persistent_evaluation_cache
								 : &local_evaluation_cache;
		auto roots = evaluate_rows(boards, evaluation_cache);
		const int minimum = minimum_simulations();
		for (std::size_t index = 0; index < states.size(); ++index) {
			states[index].network_value = roots[index].value;
			states[index].nn_batches = roots[index].reused ? 0 : 1;
			states[index].nn_evaluations = roots[index].reused ? 0 : 1;
			states[index].evaluation_reuses = roots[index].reused ? 1 : 0;
			states[index].dynamic_target = minimum;
			expand(states[index].root.get(), roots[index]);
			states[index].network = std::move(roots[index]);
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
			while (!deadline_reached(deadline) && !(cancel && cancel())) {
				bool active = false;
				bool progressed = false;
				std::vector<SelectedLeaf> selected;
				selected.reserve(states.size() * static_cast<std::size_t>(batch_size));
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
					selected_nodes.reserve(static_cast<std::size_t>(wanted) * 2);
					for (int attempt = 0, accepted = 0;
						 accepted < wanted && attempt < std::max(wanted * 5, wanted + 8);
						 ++attempt) {
						if (deadline_reached(deadline) || (cancel && cancel())) {
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
					leaf_boards.reserve(end - begin);
					for (std::size_t index = begin; index < end; ++index) {
						leaf_boards.push_back(std::move(selected[index].board));
					}
					auto evaluation = evaluate_rows(leaf_boards, evaluation_cache);
					std::unordered_set<std::size_t> evaluated_states;
					evaluated_states.reserve(end - begin);
					for (std::size_t index = begin; index < end; ++index) {
						auto &leaf = selected[index];
						auto &state = states[leaf.state_index];
						const std::size_t row = index - begin;
						if (leaf.leaf->children.empty()) {
							expand(leaf.leaf, evaluation[row]);
							state.expanded_nodes += 1;
						}
						if (evaluation[row].reused) {
							state.evaluation_reuses += 1;
						} else {
							state.nn_evaluations += 1;
							evaluated_states.insert(leaf.state_index);
						}
						clear_virtual(leaf.path);
						backpropagate(leaf.path, evaluation[row].value);
						state.sims_completed += 1;
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
	EvaluationCache persistent_evaluation_cache{0};
	int active_cpu_threads = 0;
};

// Construct the public value-type wrapper around the shared implementation.
Searcher::Searcher(Model model, torch::Device device, SearchOptions options)
	: impl_(std::make_shared<Impl>(std::move(model), std::move(device), options)) {}

// Search one position and expose timed snapshots to interactive front ends.
SearchResult Searcher::search(const chess::Board &board,
							  const SearchProgressCallback &progress, int progress_interval_ms,
							  const SearchCancelCallback &cancel) {
	return impl_->search_many({board}, progress, progress_interval_ms, cancel)[0];
}

// Search a batch without progress callbacks, as used by arena and FCPI.
std::vector<SearchResult> Searcher::search_many(const std::vector<chess::Board> &boards) {
	return impl_->search_many(boards);
}

// Return the compact frozen-model contract used by FCPI generation.
std::vector<ClosedEvaluation>
Searcher::evaluate_closed_many(const std::vector<chess::Board> &boards) {
	return impl_->evaluate_closed(boards);
}

// Update search controls while preserving network evaluations that remain model-compatible.
void Searcher::set_options(SearchOptions options) { impl_->set_options(options); }

// Clear the cross-search network cache without changing its configured capacity.
void Searcher::clear_evaluation_cache() { impl_->persistent_evaluation_cache.clear(); }

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
