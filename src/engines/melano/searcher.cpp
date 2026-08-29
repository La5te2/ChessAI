// Implements Melano exact-state batched PUCT; search.cpp is its CLI front end.

#include "melano/search.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <list>
#include <memory>
#include <stdexcept>
#include <torch/utils.h>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace melano {

namespace {

std::size_t power_of_two_bucket(std::size_t value) {
	std::size_t bucket = 1;
	while (bucket < value) {
		bucket *= 2;
	}
	return bucket;
}

using Clock = std::chrono::steady_clock;

struct CompactEvaluation {
	std::vector<chess::Move> legal_moves;
	std::vector<int> legal_indices;
	std::vector<float> legal_policy;
	float value = 0.0F;
	bool reused = false;
	std::uint64_t cache_id = 0;
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

struct TrajectoryHeat {
	std::uint64_t id = 0;
	double heat = 0.0;
	int depth = 0;
};

// Stores exact compact evaluations and combines ordinary recency with predicted trajectory locality.
class TrajectoryLruCache {
	struct Entry {
		CompactEvaluation evaluation;
		std::list<PackedState>::iterator recency;
		std::vector<std::uint64_t> children;
		std::uint64_t id = 0;
		std::uint64_t heat_generation = 0;
		double trajectory_heat = 0.0;
		std::size_t bytes = 0;
	};

	using Entries = std::unordered_map<PackedState, Entry, PackedStateHash>;

public:
	explicit TrajectoryLruCache(std::size_t capacity_bytes = std::numeric_limits<std::size_t>::max()) : capacity_bytes_(capacity_bytes) {}

	// Return one exact network evaluation and refresh its ordinary LRU position.
	bool get(const PackedState &state, CompactEvaluation &output) {
		const auto found = entries_.find(state);
		if (found == entries_.end()) {
			return false;
		}
		touch(found->second);
		output = found->second.evaluation;
		output.cache_id = found->second.id;
		return true;
	}

	// Insert one evaluation and return its identity inside this cache instance.
	std::uint64_t put(const PackedState &state, const CompactEvaluation &evaluation) {
		if (capacity_bytes_ == 0) {
			return 0;
		}
		if (const auto found = entries_.find(state); found != entries_.end()) {
			touch(found->second);
			return found->second.id;
		}
		const std::size_t bytes = entry_bytes(evaluation);
		if (bytes > capacity_bytes_) {
			return 0;
		}
		recency_.push_back(state);
		auto recency = std::prev(recency_.end());
		auto stored = evaluation;
		stored.cache_id = 0;
		const auto id = next_id_++;
		auto [entry, inserted] = entries_.emplace(state, Entry{std::move(stored), recency, {}, id, 0, 0.0, bytes});
		if (!inserted) {
			recency_.erase(recency);
			throw std::logic_error("evaluation cache insertion failed");
		}
		entries_by_id_.emplace(id, &entry->second);
		used_bytes_ += bytes;
		evict_to_capacity();
		return entries_by_id_.contains(id) ? id : 0;
	}

	// Record graph locality without making repeatedly linked ancestors artificially recent.
	void link(std::uint64_t parent_id, std::uint64_t child_id) {
		if (parent_id == 0 || child_id == 0 || parent_id == child_id) {
			return;
		}
		const auto parent_found = entries_by_id_.find(parent_id);
		const auto child_found = entries_by_id_.find(child_id);
		if (parent_found == entries_by_id_.end() || child_found == entries_by_id_.end()) {
			return;
		}
		auto &parent = *parent_found->second;
		parent.children.erase(
		    std::remove_if(parent.children.begin(), parent.children.end(), [&](std::uint64_t id) { return !entries_by_id_.contains(id); }), parent.children.end());
		if (std::find(parent.children.begin(), parent.children.end(), child_id) != parent.children.end()) {
			return;
		}
		const auto previous_capacity = parent.children.capacity();
		parent.children.push_back(child_id);
		const auto added_bytes = (parent.children.capacity() - previous_capacity) * sizeof(std::uint64_t);
		parent.bytes += added_bytes;
		used_bytes_ += added_bytes;
		evict_to_capacity();
	}

	// Record visit-derived heat for the completed search tree and refresh its recency.
	void promote_trajectory_heat(std::vector<TrajectoryHeat> trajectory) {
		if (trajectory.empty()) {
			return;
		}
		const auto generation = next_heat_generation_++;
		trajectory.erase(
		    std::remove_if(trajectory.begin(), trajectory.end(), [&](const TrajectoryHeat &item) { return item.id == 0 || !entries_by_id_.contains(item.id); }), trajectory.end());
		for (const auto &item : trajectory) {
			auto &entry = *entries_by_id_.at(item.id);
			entry.heat_generation = generation;
			entry.trajectory_heat = item.heat;
		}
		std::sort(trajectory.begin(), trajectory.end(), [](const auto &left, const auto &right) {
			if (left.heat != right.heat) {
				return left.heat < right.heat;
			}
			if (left.depth != right.depth) {
				return left.depth > right.depth;
			}
			return left.id < right.id;
		});
		for (const auto &item : trajectory) {
			if (const auto entry = entries_by_id_.find(item.id); entry != entries_by_id_.end()) {
				touch(*entry->second);
			}
		}
	}

	// Condition retained trajectory heat on the roots that actually begin the next search.
	void promote_trajectory_neighborhoods(const std::vector<PackedState> &roots) {
		struct FrontierItem {
			std::uint64_t id = 0;
			int depth = 0;
		};
		std::unordered_map<std::uint64_t, TrajectoryHeat> neighborhood;
		for (const auto &root : roots) {
			const auto found = entries_.find(root);
			if (found == entries_.end() || found->second.heat_generation == 0) {
				continue;
			}
			const auto generation = found->second.heat_generation;
			std::vector<FrontierItem> frontier{{found->second.id, 0}};
			std::unordered_set<std::uint64_t> visited;
			for (std::size_t cursor = 0; cursor < frontier.size(); ++cursor) {
				const auto item = frontier[cursor];
				if (!visited.insert(item.id).second) {
					continue;
				}
				const auto entry = entries_by_id_.find(item.id);
				if (entry == entries_by_id_.end() || entry->second->heat_generation != generation) {
					continue;
				}
				auto &aggregate = neighborhood[item.id];
				if (aggregate.id == 0) {
					aggregate.id = item.id;
					aggregate.depth = item.depth;
				} else {
					aggregate.depth = std::min(aggregate.depth, item.depth);
				}
				aggregate.heat += entry->second->trajectory_heat;
				for (const auto child_id : entry->second->children) {
					if (entries_by_id_.contains(child_id)) {
						frontier.push_back({child_id, item.depth + 1});
					}
				}
			}
		}
		std::vector<TrajectoryHeat> ordered;
		ordered.reserve(neighborhood.size());
		for (const auto &[id, item] : neighborhood) {
			ordered.push_back(item);
		}
		std::sort(ordered.begin(), ordered.end(), [](const auto &left, const auto &right) {
			if (left.heat != right.heat) {
				return left.heat < right.heat;
			}
			if (left.depth != right.depth) {
				return left.depth > right.depth;
			}
			return left.id < right.id;
		});
		for (const auto &item : ordered) {
			if (const auto entry = entries_by_id_.find(item.id); entry != entries_by_id_.end()) {
				touch(*entry->second);
			}
		}
	}

	void set_capacity(std::size_t capacity_bytes) {
		capacity_bytes_ = capacity_bytes;
		evict_to_capacity();
	}

	void clear() {
		entries_.clear();
		entries_by_id_.clear();
		recency_.clear();
		used_bytes_ = 0;
		next_id_ = 1;
		next_heat_generation_ = 1;
	}

private:
	static std::size_t entry_bytes(const CompactEvaluation &evaluation) {
		return sizeof(PackedState) * 2 + sizeof(Entry) + 64 + evaluation.legal_moves.capacity() * sizeof(chess::Move) + evaluation.legal_indices.capacity() * sizeof(int) +
		    evaluation.legal_policy.capacity() * sizeof(float);
	}

	void touch(Entry &entry) { recency_.splice(recency_.end(), recency_, entry.recency); }

	void erase_entry(Entries::iterator entry) {
		used_bytes_ -= entry->second.bytes;
		entries_by_id_.erase(entry->second.id);
		recency_.erase(entry->second.recency);
		entries_.erase(entry);
	}

	void evict_to_capacity() {
		while (!recency_.empty() && used_bytes_ > capacity_bytes_) {
			const auto oldest = entries_.find(recency_.front());
			if (oldest == entries_.end()) {
				recency_.pop_front();
				continue;
			}
			erase_entry(oldest);
		}
	}

	std::size_t capacity_bytes_;
	std::size_t used_bytes_ = 0;
	std::uint64_t next_id_ = 1;
	std::uint64_t next_heat_generation_ = 1;
	std::list<PackedState> recency_;
	Entries entries_;
	std::unordered_map<std::uint64_t, Entry *> entries_by_id_;
};

struct Node {
	// Create an edge/node pair with its Policy prior and incoming legal move.
	explicit Node(float initial_prior = 0.0F, chess::Move incoming = chess::Move::NO_MOVE) : prior(initial_prior), move(incoming) {}

	// Return the empirical value from this node's side-to-move perspective.
	float q() const { return visits > 0 ? value_sum / static_cast<float>(visits) : 0.0F; }

	float prior = 0.0F;
	chess::Move move;
	int visits = 0;
	float value_sum = 0.0F;
	int virtual_visits = 0;
	float visited_prior_mass = 0.0F;
	std::uint64_t evaluation_id = 0;
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
	CompactEvaluation network;
	float network_value = 0.0F;
	int sims_completed = 0;
	int dynamic_target = 0;
	int expanded_nodes = 0;
	int nn_batches = 0;
	int nn_evaluations = 0;
	int evaluation_reuses = 0;
};

// Keep bounded value arithmetic inside the model's [-1, 1] convention.
float clamp_unit(float value) {
	return std::clamp(value, -1.0F, 1.0F);
}

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
	for (std::size_t index = path.size(); index-- > 0;) {
		auto *node = path[index];
		if (index > 0 && node->visits == 0) {
			path[index - 1]->visited_prior_mass += std::max(0.0F, node->prior);
		}
		node->visits += 1;
		node->value_sum += value;
		value = -value;
	}
}

// Aggregate each cached state's visit share within the completed root tree.
void collect_trajectory_heat(const Node *node, double denominator, int depth, std::unordered_map<std::uint64_t, TrajectoryHeat> &output) {
	if (node == nullptr || node->visits <= 0 || denominator <= 0.0) {
		return;
	}
	if (node->evaluation_id != 0) {
		auto &item = output[node->evaluation_id];
		if (item.id == 0) {
			item.id = node->evaluation_id;
			item.depth = depth;
		} else {
			item.depth = std::min(item.depth, depth);
		}
		item.heat += static_cast<double>(node->visits) / denominator;
	}
	for (const auto &child : node->children) {
		if (child.visits > 0) {
			collect_trajectory_heat(&child, denominator, depth + 1, output);
		}
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
	    : model(std::move(source_model)), device(std::move(source_device)), options(source_options) {
		if (!model) {
			throw std::invalid_argument("Melano search requires a model");
		}
		set_options(source_options);
		model->to(device);
		model->eval();
		model->fuse_for_inference();
	}

	// Apply mutable search controls without rebuilding the model or its compatible cache rows.
	void set_options(SearchOptions source_options) {
		validate_compute_precision(source_options.precision, device);
		source_options.virtual_loss = std::max(0.0, source_options.virtual_loss);
		source_options.evaluation_cache_mb = std::clamp(source_options.evaluation_cache_mb, 0, 65536);
		if (device.is_cpu() && source_options.cpu_threads > 0) {
			torch::set_num_threads(source_options.cpu_threads);
		}
		active_cpu_threads = device.is_cpu() ? torch::get_num_threads() : 0;
		persistent_evaluation_cache.set_capacity(static_cast<std::size_t>(source_options.evaluation_cache_mb) * 1024 * 1024);
		options = source_options;
	}

	// Evaluate unique uncached states and map each compact result back to its request row.
	std::vector<CompactEvaluation> evaluate_compact(const std::vector<chess::Board> &boards, TrajectoryLruCache *cache = nullptr) {
		if (boards.empty()) {
			return {};
		}
		std::vector<CompactEvaluation> output(boards.size());
		std::vector<PackedState> pending_states;
		std::vector<chess::Board> pending_boards;
		std::vector<CompactEvaluation> pending;
		std::vector<std::size_t> row_to_pending(boards.size());
		std::vector<bool> row_cached(boards.size(), false);
		std::vector<bool> row_reused(boards.size(), false);
		std::unordered_map<PackedState, std::size_t, PackedStateHash> pending_lookup;
		pending_states.reserve(boards.size());
		pending_boards.reserve(boards.size());
		pending.reserve(boards.size());
		pending_lookup.reserve(boards.size() * 2);
		for (std::size_t row = 0; row < boards.size(); ++row) {
			const auto state = encode_state(boards[row]);
			if (cache != nullptr && cache->get(state, output[row])) {
				output[row].reused = true;
				row_cached[row] = true;
				continue;
			}
			const auto [found, inserted] = pending_lookup.emplace(state, pending.size());
			row_to_pending[row] = found->second;
			if (!inserted) {
				row_reused[row] = true;
				continue;
			}
			pending_states.push_back(state);
			pending_boards.push_back(boards[row]);
			pending.emplace_back();
			pending.back().legal_moves = legal_moves(boards[row]);
			pending.back().legal_indices.reserve(pending.back().legal_moves.size());
			for (const auto &move : pending.back().legal_moves) {
				pending.back().legal_indices.push_back(move_to_index(move, boards[row].sideToMove()));
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
		const auto evaluation_width = pin_memory ? power_of_two_bucket(legal_width) : legal_width;
		auto index_options = torch::TensorOptions().dtype(torch::kInt16).device(torch::kCPU);
		if (pin_memory) {
			index_options = index_options.pinned_memory(true);
		}
		auto legal_indices = torch::full(
		    {static_cast<std::int64_t>(pending.size()), static_cast<std::int64_t>(evaluation_width)}, -1, index_options);
		auto index_rows = legal_indices.accessor<std::int16_t, 2>();
		for (std::size_t row = 0; row < pending.size(); ++row) {
			for (std::size_t column = 0; column < pending[row].legal_indices.size(); ++column) {
				index_rows[static_cast<std::int64_t>(row)][static_cast<std::int64_t>(column)] =
				    static_cast<std::int16_t>(pending[row].legal_indices[column]);
			}
		}

		torch::InferenceMode guard;
		auto states = encode_boards(pending_boards, pin_memory);
		const auto device_states = states.to(device, true);
		const auto device_indices = legal_indices.to(device, true);
		torch::Tensor logits;
		torch::Tensor raw_values;
		{
			AutocastGuard autocast(options.precision, device);
			std::tie(logits, raw_values) = model->forward_legal(device_states, device_indices.clamp_min(0));
		}
		auto probabilities = torch::softmax(
		    logits.to(torch::kFloat32).masked_fill(device_indices < 0, -std::numeric_limits<float>::infinity()), 1);
		probabilities = probabilities.to(torch::kCPU).contiguous();
		auto values = raw_values.reshape({-1}).to(torch::kFloat32).to(torch::kCPU).contiguous();

		auto probability_rows = probabilities.accessor<float, 2>();
		auto value_rows = values.accessor<float, 1>();
		for (std::size_t row = 0; row < pending.size(); ++row) {
			pending[row].legal_policy.resize(pending[row].legal_indices.size());
			for (std::size_t column = 0; column < pending[row].legal_indices.size(); ++column) {
				pending[row].legal_policy[column] = probability_rows[static_cast<std::int64_t>(row)][static_cast<std::int64_t>(column)];
			}
			pending[row].value = value_rows[static_cast<std::int64_t>(row)];
			if (cache != nullptr) {
				auto cached = pending[row];
				cached.reused = false;
				pending[row].cache_id = cache->put(pending_states[row], cached);
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

	// Materialize the public action-space Policy only for final or progress results.
	std::vector<float> dense_policy(const CompactEvaluation &evaluation) const {
		std::vector<float> policy(kActionSize, 0.0F);
		for (std::size_t index = 0; index < evaluation.legal_indices.size(); ++index) {
			policy[evaluation.legal_indices[index]] = evaluation.legal_policy[index];
		}
		return policy;
	}

	// Create one child per legal action directly from the compact network Policy.
	void expand(Node *node, const CompactEvaluation &evaluation) {
		if (!node->children.empty()) {
			return;
		}
		if (evaluation.legal_moves.size() != evaluation.legal_policy.size()) {
			throw std::runtime_error("legal move and policy widths differ");
		}
		node->children.reserve(evaluation.legal_moves.size());
		for (std::size_t index = 0; index < evaluation.legal_moves.size(); ++index) {
			node->children.emplace_back(evaluation.legal_policy[index], evaluation.legal_moves[index]);
		}
		node->evaluation_id = evaluation.cache_id;
	}

	// Sum priors already explored under a parent for first-play uncertainty reduction.
	float visited_policy_mass(const Node *parent) const { return parent->visited_prior_mass; }

	// Estimate an unvisited edge as parent Q minus uncertainty proportional to explored prior mass.
	float fpu(const Node *parent) const {
		const float parent_q = parent->visits > 0 ? parent->q() : 0.0F;
		return clamp_unit(parent_q - static_cast<float>(std::max(0.0, options.fpu_reduction)) * std::sqrt(visited_policy_mass(parent)));
	}

	// Return the edge value in the parent side-to-move perspective.
	float edge_value(const Node *parent, const Node *child) const { return child->visits > 0 ? -child->q() : fpu(parent); }

	// Increase exploration logarithmically with parent visits: c_init +
	// factor*log((N+base+1)/base).
	double scheduled_c_puct(const Node *parent) const {
		const double visits = std::max(0, parent->visits + parent->virtual_visits);
		const double base = std::max(1.0, options.c_puct_base);
		const double growth = std::max(0.0, options.c_puct_factor) * std::log((visits + base + 1.0) / base);
		return std::max(0.0, options.c_puct + growth);
	}

	// Score an edge with PUCT: Q + c_puct*P*sqrt(N_parent)/(1+N_child) - virtual loss.
	double selection_score(const Node *parent, const Node *child) const {
		const double exploitation = edge_value(parent, child);
		const int child_visits = child->visits + child->virtual_visits;
		const double exploration = scheduled_c_puct(parent) * child->prior * std::sqrt(parent->visits + parent->virtual_visits + 1.0) / (1.0 + child_visits);
		return exploitation + exploration - options.virtual_loss * child->virtual_visits;
	}

	// Break equal PUCT scores by Policy prior, then by the edge value in the parent perspective.
	Node *select_child(Node *parent) const {
		if (parent->children.empty()) {
			return nullptr;
		}
		return &*std::max_element(parent->children.begin(), parent->children.end(), [&](const auto &left, const auto &right) {
			const double left_score = selection_score(parent, &left);
			const double right_score = selection_score(parent, &right);
			if (left_score != right_score) {
				return left_score < right_score;
			}
			if (left.prior != right.prior) {
				return left.prior < right.prior;
			}
			return edge_value(parent, &left) < edge_value(parent, &right);
		});
	}

	// Descend to an unexpanded or terminal node while reserving the path for batching.
	SelectedLeaf select_leaf(std::size_t state_index, TreeState &state) const {
		SelectedLeaf selected;
		selected.state_index = state_index;
		selected.board = state.board;
		selected.leaf = state.root.get();
		selected.path.push_back(selected.leaf);
		selected.leaf->virtual_visits += 1;
		while (!selected.leaf->children.empty()) {
			selected.leaf = select_child(selected.leaf);
			selected.leaf->virtual_visits += 1;
			selected.board.makeMove(selected.leaf->move);
			selected.path.push_back(selected.leaf);
			if (game_is_over(selected.board)) {
				break;
			}
		}
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
		std::sort(
		    ordered.begin(), ordered.end(), [](const Node *left, const Node *right) { return std::pair(left->visits, left->prior) > std::pair(right->visits, right->prior); });
		const double first = ordered[0]->visits;
		const double second = ordered[1]->visits;
		const double visit_uncertainty = 1.0 - std::abs(first - second) / std::max(1.0, first + second);
		const double q_uncertainty = 1.0 - std::min(1.0, std::abs(-ordered[0]->q() + ordered[1]->q()) / 0.5);
		return std::clamp(0.5 * entropy + 0.35 * visit_uncertainty + 0.15 * q_uncertainty, 0.0, 1.0);
	}

	// Establish the mandatory simulation floor before uncertainty can extend the budget.
	int minimum_simulations() const {
		const int cap = std::max(0, options.mcts_sims);
		if (cap == 0) {
			return 0;
		}
		const int configured = options.mcts_min_sims > 0 ? options.mcts_min_sims : std::max(std::max(1, options.mcts_batch_size), cap / 4);
		return std::max(1, std::min(cap, configured));
	}

	// Interpolate from minimum to the hard cap using the current root uncertainty.
	int dynamic_target(const Node *root, int minimum) const {
		const int cap = std::max(0, options.mcts_sims);
		const int desired = minimum + static_cast<int>(std::ceil(uncertainty(root) * std::max(0, cap - minimum)));
		return std::max(minimum, std::min(cap, desired));
	}

	// Track the cost of one uncached neural evaluation for deadline-aware batching.
	void observe_evaluation(const std::vector<CompactEvaluation> &rows, Clock::time_point started) {
		const auto fresh = std::count_if(rows.begin(), rows.end(), [](const CompactEvaluation &row) { return !row.reused; });
		if (fresh != 1) {
			return;
		}
		const double sample = std::max(0.05, std::chrono::duration<double, std::milli>(Clock::now() - started).count());
		single_evaluation_ms = single_evaluation_ms <= 0.0 ? sample : std::max(sample, 0.8 * single_evaluation_ms + 0.2 * sample);
	}

	// Reserve enough time to finish an in-flight neural call and shrink the final batches.
	int deadline_batch_size(const std::optional<Clock::time_point> &deadline, int configured) const {
		configured = std::max(1, configured);
		if (!deadline.has_value()) {
			return configured;
		}
		const double remaining = std::chrono::duration<double, std::milli>(*deadline - Clock::now()).count();
		const double row_budget = std::max(1.0, 1.25 * std::max(1.0, single_evaluation_ms));
		const int rows = static_cast<int>(std::floor((remaining - 2.0) / row_budget));
		return std::clamp(rows, 0, configured);
	}

	// Convert root visits to legal move probabilities; priors keep zero-visit moves representable.
	std::vector<float> root_policy(const TreeState &state) const {
		std::vector<float> policy(kActionSize, 0.0F);
		float total = 0.0F;
		for (const auto &child : state.root->children) {
			const float weight = child.visits + child.prior;
			policy[move_to_index(child.move, state.board.sideToMove())] = weight;
			total += weight;
		}
		if (total > 0.0F) {
			for (const auto &child : state.root->children) {
				policy[move_to_index(child.move, state.board.sideToMove())] /= total;
			}
		}
		return policy;
	}

	// Apply optional post-search ranking rules without modifying priors or the MCTS tree.
	void apply_decision_components(
	    const chess::Board &board, float root_value, std::vector<float> &scores, std::unordered_set<int> &repetitions, std::unordered_set<int> &mates) const {
		if (options.instant_mate_first) {
			int selected = -1;
			float selected_score = -std::numeric_limits<float>::infinity();
			for (const auto &move : legal_moves(board)) {
				auto probe = board;
				probe.makeMove(move);
				if (probe.isGameOver().first != chess::GameResultReason::CHECKMATE) {
					continue;
				}
				const int index = move_to_index(move, board.sideToMove());
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

		const float deduction = static_cast<float>(std::clamp(options.repetition_policy_penalty, 0.0, 1.0) * std::clamp(static_cast<double>(root_value), 0.0, 1.0));
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
				const int index = move_to_index(move, board.sideToMove());
				scores[index] = std::max(0.0F, scores[index] - deduction);
				repetitions.insert(index);
			}
		}
	}

	// Assemble the final ranked move list and diagnostics from one completed tree.
	SearchResult make_result(TreeState &state, Clock::time_point start) const {
		SearchResult result;
		result.policy = options.mcts_sims <= 0 ? dense_policy(state.network) : root_policy(state);
		result.decision_scores = result.policy;
		result.value = state.root->visits > 0 ? state.root->q() : state.network_value;
		result.sims_completed = state.sims_completed;
		result.dynamic_target = options.mcts_sims <= 0 ? 0 : state.dynamic_target;
		result.expanded_nodes = state.expanded_nodes;
		result.nn_batches = state.nn_batches;
		result.nn_evaluations = state.nn_evaluations;
		result.evaluation_reuses = state.evaluation_reuses;
		result.cpu_threads = active_cpu_threads;
		result.uncertainty = options.mcts_sims <= 0 ? 0.0 : uncertainty(state.root.get());
		result.elapsed_ms = seconds_since(start) * 1000.0;

		std::unordered_set<int> repetitions;
		std::unordered_set<int> mates;
		apply_decision_components(state.board, result.value, result.decision_scores, repetitions, mates);

		auto moves = legal_moves(state.board);
		std::sort(moves.begin(), moves.end(), [&](const chess::Move &left, const chess::Move &right) {
			const int left_index = move_to_index(left, state.board.sideToMove());
			const int right_index = move_to_index(right, state.board.sideToMove());
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

		const auto row_count = std::min(moves.size(), static_cast<std::size_t>(std::max(1, options.root_topn)));
		for (std::size_t row = 0; row < row_count; ++row) {
			const auto move = moves[row];
			const int action = move_to_index(move, state.board.sideToMove());
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
	std::vector<SearchResult> search_many(
	    const std::vector<chess::Board> &boards, const SearchProgressCallback &progress = {}, int progress_interval_ms = 0, const SearchCancelCallback &cancel = {}) {
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
			deadline = start + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double, std::milli>(options.movetime_ms));
		}

		std::vector<TreeState> states;
		states.reserve(boards.size());
		for (const auto &board : boards) {
			TreeState state;
			state.board = board;
			states.push_back(std::move(state));
		}
		TrajectoryLruCache local_evaluation_cache;
		auto *evaluation_cache = options.evaluation_cache_mb > 0 ? &persistent_evaluation_cache : &local_evaluation_cache;
		if (options.evaluation_cache_mb > 0) {
			std::vector<PackedState> root_states;
			root_states.reserve(boards.size());
			for (const auto &board : boards) {
				root_states.push_back(encode_state(board));
			}
			persistent_evaluation_cache.promote_trajectory_neighborhoods(root_states);
		}
		const auto root_evaluation_started = Clock::now();
		auto roots = evaluate_compact(boards, evaluation_cache);
		observe_evaluation(roots, root_evaluation_started);
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
			next_progress = Clock::now() + std::chrono::milliseconds(std::max(1, progress_interval_ms));
		}

		const int simulation_limit = options.unbounded_simulations ? std::numeric_limits<int>::max() : std::max(0, options.mcts_sims);
		if (options.mcts_sims > 0) {
			const int configured_batch_size = std::max(1, options.mcts_batch_size);
			while (!deadline_reached(deadline) && !(cancel && cancel())) {
				const int batch_size = deadline_batch_size(deadline, configured_batch_size);
				if (batch_size == 0) {
					break;
				}
				bool active = false;
				bool progressed = false;
				std::vector<SelectedLeaf> selected;
				selected.reserve(states.size() * static_cast<std::size_t>(batch_size));
				for (std::size_t state_index = 0; state_index < states.size(); ++state_index) {
					auto &state = states[state_index];
					if (state.sims_completed >= simulation_limit || (!options.unbounded_simulations && state.sims_completed >= state.dynamic_target)) {
						continue;
					}
					active = true;
					int wanted = std::min(batch_size, simulation_limit - state.sims_completed);
					if (!options.unbounded_simulations) {
						wanted = std::min(wanted, state.dynamic_target - state.sims_completed);
					}
					std::unordered_set<Node *> selected_nodes;
					selected_nodes.reserve(static_cast<std::size_t>(wanted) * 2);
					for (int attempt = 0, scheduled = 0; scheduled < wanted && attempt < std::max(wanted * 5, wanted + 8); ++attempt) {
						if (deadline_reached(deadline) || (cancel && cancel())) {
							break;
						}
						auto leaf = select_leaf(state_index, state);
						float terminal = 0.0F;
						if (is_terminal(leaf.board, &terminal)) {
							clear_virtual(leaf.path);
							backpropagate(leaf.path, terminal);
							state.sims_completed += 1;
							++scheduled;
							progressed = true;
							continue;
						}
						if (!selected_nodes.insert(leaf.leaf).second) {
							clear_virtual(leaf.path);
							continue;
						}
						selected.push_back(std::move(leaf));
						++scheduled;
					}
				}

				std::size_t begin = 0;
				while (begin < selected.size()) {
					const int allowed = deadline_batch_size(deadline, batch_size);
					if (allowed == 0 || (cancel && cancel())) {
						for (std::size_t index = begin; index < selected.size(); ++index) {
							clear_virtual(selected[index].path);
						}
						break;
					}
					const auto end = std::min(selected.size(), begin + static_cast<std::size_t>(allowed));
					std::vector<chess::Board> leaf_boards;
					leaf_boards.reserve(end - begin);
					for (std::size_t index = begin; index < end; ++index) {
						leaf_boards.push_back(std::move(selected[index].board));
					}
					const auto evaluation_started = Clock::now();
					auto evaluation = evaluate_compact(leaf_boards, evaluation_cache);
					observe_evaluation(evaluation, evaluation_started);
					std::unordered_set<std::size_t> evaluated_states;
					evaluated_states.reserve(end - begin);
					for (std::size_t index = begin; index < end; ++index) {
						auto &leaf = selected[index];
						auto &state = states[leaf.state_index];
						const std::size_t row = index - begin;
						if (options.evaluation_cache_mb > 0 && leaf.path.size() > 1) {
							persistent_evaluation_cache.link(leaf.path[leaf.path.size() - 2]->evaluation_id, evaluation[row].cache_id);
						}
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
					begin = end;
				}

				for (auto &state : states) {
					if (state.sims_completed >= minimum) {
						state.dynamic_target = dynamic_target(state.root.get(), minimum);
					}
				}
				if (progress && states.size() == 1 && progress_interval_ms > 0 && Clock::now() >= next_progress) {
					progress(make_result(states[0], start));
					next_progress = Clock::now() + std::chrono::milliseconds(progress_interval_ms);
				}
				if (!active || !progressed) {
					break;
				}
			}
		}

		std::vector<SearchResult> results;
		results.reserve(states.size());
		std::unordered_map<std::uint64_t, TrajectoryHeat> trajectory_heat;
		for (auto &state : states) {
			results.push_back(make_result(state, start));
			if (options.evaluation_cache_mb > 0 && state.root->visits > 0) {
				collect_trajectory_heat(state.root.get(), static_cast<double>(state.root->visits), 0, trajectory_heat);
			}
		}
		if (options.evaluation_cache_mb > 0 && !trajectory_heat.empty()) {
			std::vector<TrajectoryHeat> ordered_heat;
			ordered_heat.reserve(trajectory_heat.size());
			for (const auto &[id, item] : trajectory_heat) {
				ordered_heat.push_back(item);
			}
			persistent_evaluation_cache.promote_trajectory_heat(std::move(ordered_heat));
		}
		return results;
	}

	Model model;
	torch::Device device;
	SearchOptions options;
	TrajectoryLruCache persistent_evaluation_cache{0};
	int active_cpu_threads = 0;
	double single_evaluation_ms = 0.0;
};

// Construct the public value-type wrapper around the shared implementation.
Searcher::Searcher(Model model, torch::Device device, SearchOptions options) : impl_(std::make_shared<Impl>(std::move(model), std::move(device), options)) {
}

// Search one position and expose timed snapshots to interactive front ends.
SearchResult Searcher::search(const chess::Board &board, const SearchProgressCallback &progress, int progress_interval_ms, const SearchCancelCallback &cancel) {
	return impl_->search_many({board}, progress, progress_interval_ms, cancel)[0];
}

// Search independent positions together so their leaf evaluations share neural batches.
std::vector<SearchResult> Searcher::search_many(const std::vector<chess::Board> &boards) {
	return impl_->search_many(boards);
}

// Apply search controls while preserving compatible network evaluations.
void Searcher::set_options(SearchOptions options) {
	impl_->set_options(options);
}

// Clear retained cross-search evaluations without changing capacity.
void Searcher::clear_evaluation_cache() {
	impl_->persistent_evaluation_cache.clear();
}

} // namespace melano
