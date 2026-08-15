// Implements Gadus batched PUCT MCTS with root coverage and adaptive internal widening.

#include "gadus/search.hpp"
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

namespace gadus {

namespace {

using Clock = std::chrono::steady_clock;

struct EvaluationRow {
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

// Stores exact network rows and combines ordinary recency with predicted trajectory locality.
class TrajectoryLruCache {
	struct Entry {
		EvaluationRow evaluation;
		std::list<PackedState>::iterator recency;
		std::vector<std::uint64_t> children;
		std::uint64_t id = 0;
		std::uint64_t heat_generation = 0;
		double trajectory_heat = 0.0;
		std::size_t bytes = 0;
	};

	using Entries = std::unordered_map<PackedState, Entry, PackedStateHash>;

	public:
	/// Creates a cache with an approximate byte ceiling; the maximum size disables eviction.
	explicit TrajectoryLruCache(
		std::size_t capacity_bytes = std::numeric_limits<std::size_t>::max())
		: capacity_bytes_(capacity_bytes) {}

	/// Copies a cached network result and marks the entry as recently used.
	bool get(const PackedState &state, EvaluationRow &output) {
		const auto found = entries_.find(state);
		if (found == entries_.end()) {
			return false;
		}
		touch(found->second);
		output = found->second.evaluation;
		output.cache_id = found->second.id;
		return true;
	}

	/// Inserts one compact Policy/Value row and returns its stable cache-local identity.
	std::uint64_t put(const PackedState &state, const EvaluationRow &evaluation) {
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
		auto iterator = std::prev(recency_.end());
		auto stored = evaluation;
		stored.cache_id = 0;
		const auto id = next_id_++;
		auto [inserted, success] = entries_.emplace(
			state, Entry{std::move(stored), iterator, {}, id, 0, 0.0, bytes});
		if (!success) {
			recency_.erase(iterator);
			throw std::logic_error("evaluation cache insertion failed");
		}
		entries_by_id_.emplace(id, &inserted->second);
		used_bytes_ += bytes;
		evict_to_capacity();
		return entries_by_id_.contains(id) ? id : 0;
	}

	/// Records one evaluated parent-child transition without changing ordinary LRU recency.
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
			std::remove_if(parent.children.begin(), parent.children.end(),
						   [&](std::uint64_t id) { return !entries_by_id_.contains(id); }),
			parent.children.end());
		if (std::find(parent.children.begin(), parent.children.end(), child_id) !=
			parent.children.end()) {
			return;
		}
		const auto previous_capacity = parent.children.capacity();
		parent.children.push_back(child_id);
		const auto added_bytes =
			(parent.children.capacity() - previous_capacity) * sizeof(std::uint64_t);
		parent.bytes += added_bytes;
		used_bytes_ += added_bytes;
		evict_to_capacity();
	}

	/// Records visit-derived heat for the completed search tree and refreshes its recency.
	void promote_trajectory_heat(std::vector<TrajectoryHeat> trajectory) {
		if (trajectory.empty()) {
			return;
		}
		const auto generation = next_heat_generation_++;
		trajectory.erase(
			std::remove_if(trajectory.begin(), trajectory.end(), [&](const TrajectoryHeat &item) {
				return item.id == 0 || !entries_by_id_.contains(item.id);
			}),
			trajectory.end());
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

	/// Conditions retained trajectory heat on the roots that actually begin the next search.
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
				if (entry == entries_by_id_.end() ||
					entry->second->heat_generation != generation) {
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

	/// Changes the byte ceiling and evicts old entries until the cache fits.
	void set_capacity(std::size_t capacity_bytes) {
		capacity_bytes_ = capacity_bytes;
		evict_to_capacity();
	}

	/// Clears all retained rows without changing the configured byte ceiling.
	void clear() {
		entries_.clear();
		entries_by_id_.clear();
		recency_.clear();
		used_bytes_ = 0;
		next_id_ = 1;
		next_heat_generation_ = 1;
	}

	private:
	// Include vector storage and conservative container overhead in the memory budget.
	static std::size_t entry_bytes(const EvaluationRow &evaluation) {
		return sizeof(PackedState) * 2 + sizeof(Entry) + 64 +
			   evaluation.legal_moves.capacity() * sizeof(chess::Move) +
			   evaluation.legal_indices.capacity() * sizeof(int) +
			   evaluation.legal_policy.capacity() * sizeof(float);
	}

	// Move one live entry to the newest end without changing its cached value.
	void touch(Entry &entry) { recency_.splice(recency_.end(), recency_, entry.recency); }

	// Remove one map entry and its LRU node while leaving stale graph ids harmless.
	void erase_entry(Entries::iterator entry) {
		used_bytes_ -= entry->second.bytes;
		entries_by_id_.erase(entry->second.id);
		recency_.erase(entry->second.recency);
		entries_.erase(entry);
	}

	// Enforce the approximate byte ceiling after insertions, links, or capacity changes.
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
	std::size_t active_children = 0;
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
	EvaluationRow network;
	float network_value = 0.0F;
	int sims_completed = 0;
	int expanded_nodes = 0;
	int nn_batches = 0;
	int nn_evaluations = 0;
	int evaluation_reuses = 0;
	double total_leaf_depth = 0.0;
	int leaf_samples = 0;
	int max_leaf_depth = 0;
	int root_visit_floor = 0;
	bool root_prior_fixed = false;
	double root_prior_exponent = 1.0;
	double root_prior_normalizer = 1.0;
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

// Aggregate each cached state's visit share within the completed root tree.
void collect_trajectory_heat(const Node *node, double denominator, int depth,
							 std::unordered_map<std::uint64_t, TrajectoryHeat> &output) {
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
				parameter.set_data(parameter.contiguous(torch::MemoryFormat::ChannelsLast));
			}
		}
	}

	// Apply mutable search controls without rebuilding the model or discarding compatible cache
	// rows.
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

	// Evaluate a frozen batch and retain legal moves so tree expansion does not generate them
	// again.
	std::vector<EvaluationRow> evaluate_rows(const std::vector<chess::Board> &boards,
											 TrajectoryLruCache *cache = nullptr) {
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
			const auto [found, inserted] = pending_lookup.emplace(state, pending.size());
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
		auto index_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
		auto mask_options = torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU);
		if (pin_memory) {
			index_options = index_options.pinned_memory(true);
			mask_options = mask_options.pinned_memory(true);
		}
		auto legal_indices = torch::zeros(
			{static_cast<std::int64_t>(pending.size()), static_cast<std::int64_t>(legal_width)},
			index_options);
		auto legal_mask = torch::zeros(
			{static_cast<std::int64_t>(pending.size()), static_cast<std::int64_t>(legal_width)},
			mask_options);
		auto index_rows = legal_indices.accessor<std::int64_t, 2>();
		auto mask_rows = legal_mask.accessor<bool, 2>();
		for (std::size_t row = 0; row < pending.size(); ++row) {
			for (std::size_t column = 0; column < pending[row].legal_indices.size(); ++column) {
				index_rows[static_cast<std::int64_t>(row)][static_cast<std::int64_t>(column)] =
					pending[row].legal_indices[column];
				mask_rows[static_cast<std::int64_t>(row)][static_cast<std::int64_t>(column)] = true;
			}
		}

		torch::InferenceMode guard;
		auto states =
			decode_states_device(reinterpret_cast<const std::uint8_t *>(pending_states.data()),
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
		compact_logits =
			compact_logits.masked_fill(~device_mask, -std::numeric_limits<float>::infinity());
		auto probabilities = torch::softmax(compact_logits, 1).to(torch::kCPU).contiguous();
		auto values = raw_values.reshape({-1}).to(torch::kFloat32).to(torch::kCPU).contiguous();

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
		node->evaluation_id = evaluation.cache_id;
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
		std::stable_sort(node->children.begin(), node->children.end(),
			[](const Node &left, const Node &right) { return left.prior > right.prior; });
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

	// Increase exploration logarithmically with parent visits: c_init +
	// factor*log((N+base+1)/base).
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

	// Compare two children by PUCT score, policy prior, and parent-perspective value.
	bool child_precedes(const Node *parent, const Node *candidate, const Node *selected) const {
		const double candidate_score = selection_score(parent, candidate);
		const double selected_score = selection_score(parent, selected);
		if (candidate_score != selected_score) {
			return candidate_score > selected_score;
		}
		if (candidate->prior != selected->prior) {
			return candidate->prior > selected->prior;
		}
		return edge_value(parent, candidate) > edge_value(parent, selected);
	}

	// Give fully opened internal nodes a sublinear per-action evidence floor.
	int internal_verification_floor(const Node *parent) const {
		const int actions = static_cast<int>(parent->children.size());
		const int visits = std::max(0, parent->visits);
		if (actions <= 0 || parent->active_children < parent->children.size()) {
			return 0;
		}
		const double denominator =
			static_cast<double>(actions) * std::log(std::exp(1.0) + visits);
		return std::max(1, static_cast<int>(std::floor(visits / denominator)));
	}

	// Open internal actions, verify a fully opened node, then continue with PUCT.
	Node *select_child(Node *parent, double opening_exponent) const {
		const int augmented_visits = std::max(0, parent->visits + parent->virtual_visits);
		const auto requested = static_cast<std::size_t>(std::ceil(std::pow(
			static_cast<double>(augmented_visits) + 1.0, opening_exponent)));
		parent->active_children = std::max(
			parent->active_children, std::min(parent->children.size(), std::max<std::size_t>(1, requested)));
		for (std::size_t index = 0; index < parent->active_children; ++index) {
			auto &child = parent->children[index];
			if (child.visits + child.virtual_visits == 0) {
				return &child;
			}
		}
		const int verification_floor = internal_verification_floor(parent);
		Node *under_verified = nullptr;
		int largest_deficit = 0;
		for (std::size_t index = 0; index < parent->active_children; ++index) {
			auto &child = parent->children[index];
			const int deficit =
				verification_floor - (child.visits + child.virtual_visits);
			if (deficit <= 0) {
				continue;
			}
			if (under_verified == nullptr || deficit > largest_deficit ||
				(deficit == largest_deficit &&
				 child_precedes(parent, &child, under_verified))) {
				under_verified = &child;
				largest_deficit = deficit;
			}
		}
		if (under_verified != nullptr) {
			return under_verified;
		}
		Node *selected = nullptr;
		for (std::size_t index = 0; index < parent->active_children; ++index) {
			auto &child = parent->children[index];
			if (selected == nullptr || child_precedes(parent, &child, selected)) {
				selected = &child;
			}
		}
		return selected;
	}

	// Derive the fair visit floor from the total budget and complete legal width.
	int root_fair_visit_floor(const TreeState &state) const {
		const int actions = static_cast<int>(state.root->children.size());
		const int budget = std::max(0, options.mcts_sims);
		if (actions <= 0 || budget <= 0) {
			return 0;
		}
		const double denominator =
			static_cast<double>(actions) * std::log(std::exp(1.0) + budget);
		return std::max(1, static_cast<int>(std::floor(budget / denominator)));
	}

	// Report whether every legal root action has completed its fair-stage visits.
	bool root_fair_complete(const TreeState &state) const {
		for (const auto &child : state.root->children) {
			if (child.visits < state.root_visit_floor) {
				return false;
			}
		}
		return true;
	}

	// Fix 1/alpha from fair-stage rank agreement between original Policy and empirical Q.
	void fix_root_prior(TreeState &state) const {
		if (state.root_prior_fixed) {
			return;
		}
		double concordant = 0.0;
		double discordant = 0.0;
		double prior_ties = 0.0;
		double value_ties = 0.0;
		for (std::size_t left = 0; left < state.root->children.size(); ++left) {
			const auto &left_child = state.root->children[left];
			for (std::size_t right = left + 1; right < state.root->children.size(); ++right) {
				const auto &right_child = state.root->children[right];
				const int prior_order = (left_child.prior > right_child.prior) -
									(left_child.prior < right_child.prior);
				const float left_value = -left_child.q();
				const float right_value = -right_child.q();
				const int value_order =
					(left_value > right_value) - (left_value < right_value);
				if (prior_order == 0 && value_order != 0) {
					prior_ties += 1.0;
				} else if (prior_order != 0 && value_order == 0) {
					value_ties += 1.0;
				} else if (prior_order != 0 && value_order != 0) {
					if (prior_order == value_order) {
						concordant += 1.0;
					} else {
						discordant += 1.0;
					}
				}
			}
		}
		const double ranked_pairs = concordant + discordant;
		const double denominator =
			std::sqrt((ranked_pairs + prior_ties) * (ranked_pairs + value_ties));
		const double tau = denominator > 0.0
			? std::clamp((concordant - discordant) / denominator, -1.0, 1.0)
			: 0.0;
		state.root_prior_exponent = 0.5 * (1.0 + tau);
		state.root_prior_normalizer = 0.0;
		for (const auto &child : state.root->children) {
			const double prior = std::max(0.0F, child.prior);
			state.root_prior_normalizer += state.root_prior_exponent == 0.0
				? 1.0
				: std::pow(prior, state.root_prior_exponent);
		}
		if (state.root_prior_normalizer <= 0.0) {
			state.root_prior_exponent = 0.0;
			state.root_prior_normalizer = static_cast<double>(state.root->children.size());
		}
		state.root_prior_fixed = true;
	}

	// Widen at least by a square root and accelerate when fair Q opposes Policy ordering.
	double internal_opening_exponent(const TreeState &state) const {
		return std::max(0.5, 1.0 - state.root_prior_exponent);
	}

	// Return one legal root action's fixed power-tempered prior.
	double root_tempered_prior(const TreeState &state, const Node &child) const {
		const double numerator = state.root_prior_exponent == 0.0
			? 1.0
			: std::pow(std::max(0.0F, child.prior), state.root_prior_exponent);
		return numerator / state.root_prior_normalizer;
	}

	// Score one fair-evaluated root edge with its fixed power-tempered prior.
	double root_selection_score(const TreeState &state, const Node *child, double prior) const {
		const auto *root = state.root.get();
		const int child_visits = child->visits + child->virtual_visits;
		const double exploration = scheduled_c_puct(root) * prior *
								   std::sqrt(root->visits + root->virtual_visits + 1.0) /
								   (1.0 + child_visits);
		return -child->q() + exploration - options.virtual_loss * child->virtual_visits;
	}

	// Fill the fair floor on every legal root action, then allocate the remainder with fixed PUCT.
	Node *select_root_action(TreeState &state) const {
		Node *selected = nullptr;
		int largest_deficit = 0;
		for (auto &child : state.root->children) {
			const int augmented_visits = child.visits + child.virtual_visits;
			const int deficit = state.root_visit_floor - augmented_visits;
			if (deficit <= 0) {
				continue;
			}
			if (selected == nullptr || deficit > largest_deficit ||
				(deficit == largest_deficit &&
				 child_precedes(state.root.get(), &child, selected))) {
				selected = &child;
				largest_deficit = deficit;
			}
		}
		if (selected != nullptr) {
			return selected;
		}
		if (!root_fair_complete(state)) {
			return nullptr;
		}

		fix_root_prior(state);
		double selected_score = -std::numeric_limits<double>::infinity();
		double selected_prior = 0.0;
		for (auto &child : state.root->children) {
			const double prior = root_tempered_prior(state, child);
			const double score = root_selection_score(state, &child, prior);
			const float value = -child.q();
			const float selected_value = selected != nullptr ? -selected->q() : 0.0F;
			if (selected == nullptr || score > selected_score ||
				(score == selected_score && prior > selected_prior) ||
				(score == selected_score && prior == selected_prior && value > selected_value)) {
				selected = &child;
				selected_score = score;
				selected_prior = prior;
			}
		}
		return selected;
	}

	// Enter one selected root action, then use PUCT below the root until reaching a leaf.
	SelectedLeaf select_leaf(std::size_t state_index, TreeState &state, Node *root_action) const {
		SelectedLeaf selected;
		selected.state_index = state_index;
		selected.board = state.board;
		selected.leaf = state.root.get();
		selected.path.reserve(16);
		selected.path.push_back(selected.leaf);
		selected.leaf->virtual_visits += 1;
		if (root_action != nullptr) {
			selected.leaf = root_action;
			selected.leaf->virtual_visits += 1;
			selected.board.makeMove(selected.leaf->move);
			selected.path.push_back(selected.leaf);
		}
		const double opening_exponent = internal_opening_exponent(state);
		while (!selected.leaf->children.empty()) {
			selected.leaf = select_child(selected.leaf, opening_exponent);
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

	// Track the cost of one uncached neural evaluation for deadline-aware batching.
	void observe_evaluation(const std::vector<EvaluationRow> &rows, Clock::time_point started) {
		const auto fresh = std::count_if(rows.begin(), rows.end(),
			[](const EvaluationRow &row) { return !row.reused; });
		if (fresh != 1) {
			return;
		}
		const double sample = std::max(0.05,
			std::chrono::duration<double, std::milli>(Clock::now() - started).count());
		single_evaluation_ms = single_evaluation_ms <= 0.0
			? sample
			: std::max(sample, 0.8 * single_evaluation_ms + 0.2 * sample);
	}

	// Reserve enough time to finish an in-flight neural call and shrink the final batches.
	int deadline_batch_size(const std::optional<Clock::time_point> &deadline,
							int configured) const {
		configured = std::max(1, configured);
		if (!deadline.has_value()) {
			return configured;
		}
		const double remaining =
			std::chrono::duration<double, std::milli>(*deadline - Clock::now()).count();
		const double row_budget = std::max(1.0, 1.25 * std::max(1.0, single_evaluation_ms));
		const int rows = static_cast<int>(std::floor((remaining - 2.0) / row_budget));
		return std::clamp(rows, 0, configured);
	}

	// Convert all legal root visits and original priors to move probabilities.
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
		result.expanded_nodes = state.expanded_nodes;
		result.nn_batches = state.nn_batches;
		result.nn_evaluations = state.nn_evaluations;
		result.evaluation_reuses = state.evaluation_reuses;
		result.cpu_threads = active_cpu_threads;
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
		TrajectoryLruCache local_evaluation_cache;
		auto *evaluation_cache = options.evaluation_cache_mb > 0 ? &persistent_evaluation_cache
																 : &local_evaluation_cache;
		if (options.evaluation_cache_mb > 0) {
			std::vector<PackedState> root_states;
			root_states.reserve(boards.size());
			for (const auto &board : boards) {
				root_states.push_back(encode_state(board));
			}
			persistent_evaluation_cache.promote_trajectory_neighborhoods(root_states);
		}
		const auto root_evaluation_started = Clock::now();
		auto roots = evaluate_rows(boards, evaluation_cache);
		observe_evaluation(roots, root_evaluation_started);
		for (std::size_t index = 0; index < states.size(); ++index) {
			states[index].network_value = roots[index].value;
			states[index].nn_batches = roots[index].reused ? 0 : 1;
			states[index].nn_evaluations = roots[index].reused ? 0 : 1;
			states[index].evaluation_reuses = roots[index].reused ? 1 : 0;
			expand(states[index].root.get(), roots[index]);
			states[index].root_visit_floor = root_fair_visit_floor(states[index]);
			states[index].network = std::move(roots[index]);
			states[index].expanded_nodes = 1;
		}
		auto next_progress = start;
		if (progress && states.size() == 1) {
			progress(make_result(states[0], start));
			next_progress =
				Clock::now() + std::chrono::milliseconds(std::max(1, progress_interval_ms));
		}

		const int simulation_limit = options.unbounded_simulations
			? std::numeric_limits<int>::max()
			: std::max(0, options.mcts_sims);
		if (options.type == SearchType::Open && options.mcts_sims > 0) {
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
					if (state.sims_completed >= simulation_limit) {
						continue;
					}
					active = true;
					const int wanted =
						std::min(batch_size, simulation_limit - state.sims_completed);
					std::unordered_set<Node *> selected_nodes;
					selected_nodes.reserve(static_cast<std::size_t>(wanted) * 2);
					for (int attempt = 0, scheduled = 0;
						 scheduled < wanted && attempt < std::max(wanted * 5, wanted + 8);
						 ++attempt) {
						if (deadline_reached(deadline) || (cancel && cancel())) {
							break;
						}
						auto *root_action = select_root_action(state);
						if (root_action == nullptr)
							break;
						auto leaf = select_leaf(state_index, state, root_action);
						float terminal = 0.0F;
						if (is_terminal(leaf.board, &terminal)) {
							clear_virtual(leaf.path);
							backpropagate(leaf.path, terminal);
							state.sims_completed += 1;
							scheduled += 1;
							progressed = true;
							continue;
						}
						if (!selected_nodes.insert(leaf.leaf).second) {
							clear_virtual(leaf.path);
							// Exact rollback makes the deterministic next attempt select the same leaf.
							break;
						}
						selected.push_back(std::move(leaf));
						++scheduled;
					}
				}

				std::size_t begin = 0;
				while (begin < selected.size()) {
					const int allowed = deadline_batch_size(deadline, batch_size);
					if (allowed == 0 || (cancel && cancel())) {
						for (std::size_t index = selected.size(); index-- > begin;) {
							clear_virtual(selected[index].path);
						}
						break;
					}
					const auto end = std::min(
						selected.size(), begin + static_cast<std::size_t>(allowed));
					std::vector<chess::Board> leaf_boards;
					leaf_boards.reserve(end - begin);
					for (std::size_t index = begin; index < end; ++index) {
						leaf_boards.push_back(std::move(selected[index].board));
					}
					const auto evaluation_started = Clock::now();
					auto evaluation = evaluate_rows(leaf_boards, evaluation_cache);
					observe_evaluation(evaluation, evaluation_started);
					std::unordered_set<std::size_t> evaluated_states;
					evaluated_states.reserve(end - begin);
					for (std::size_t index = begin; index < end; ++index) {
						auto &leaf = selected[index];
						auto &state = states[leaf.state_index];
						const std::size_t row = index - begin;
						if (options.evaluation_cache_mb > 0 && leaf.path.size() > 1) {
							persistent_evaluation_cache.link(
								leaf.path[leaf.path.size() - 2]->evaluation_id,
								evaluation[row].cache_id);
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

				if (progress && states.size() == 1 && progress_interval_ms > 0 &&
					Clock::now() >= next_progress) {
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
			if (options.evaluation_cache_mb <= 0) {
				continue;
			}
			if (state.root->visits > 0) {
				collect_trajectory_heat(state.root.get(), static_cast<double>(state.root->visits), 0,
								trajectory_heat);
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
Searcher::Searcher(Model model, torch::Device device, SearchOptions options)
	: impl_(std::make_shared<Impl>(std::move(model), std::move(device), options)) {}

// Search one position and expose timed snapshots to interactive front ends.
SearchResult Searcher::search(const chess::Board &board, const SearchProgressCallback &progress,
							  int progress_interval_ms, const SearchCancelCallback &cancel) {
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
	if (value == "open") {
		return SearchType::Open;
	}
	throw std::invalid_argument("search-type must be closed or open");
}

// Convert a search mode back to its stable external spelling.
std::string search_type_name(SearchType value) {
	return value == SearchType::Closed ? "closed" : "open";
}

} // namespace gadus
