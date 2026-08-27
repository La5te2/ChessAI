#pragma once

// Melano direct-policy and exact-state batched PUCT search API.

#include "melano/game.hpp"
#include "melano/model.hpp"
#include "melano/precision.hpp"
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace melano {

struct SearchOptions {
	ComputePrecision precision = ComputePrecision::Fp32;
	int cpu_threads = 0;
	int mcts_sims = 100;
	int mcts_min_sims = 0;
	int mcts_batch_size = 32;
	bool unbounded_simulations = false;
	double movetime_ms = 0.0;
	double c_puct = 0.5;
	double c_puct_base = 19652.0;
	double c_puct_factor = 1.0;
	double fpu_reduction = 0.15;
	double virtual_loss = 0.0;
	double repetition_policy_penalty = 0.0;
	bool instant_mate_first = false;
	int evaluation_cache_mb = 0;
	int root_topn = 10;
};

struct RootMove {
	chess::Move move;
	float probability = 0.0F;
	float decision_score = 0.0F;
	float prior = 0.0F;
	float q = 0.0F;
	int visits = 0;
	bool repetition_penalized = false;
	bool instant_mate = false;
};

struct SearchResult {
	chess::Move move;
	std::vector<float> policy;
	std::vector<float> decision_scores;
	float value = 0.0F;
	int sims_completed = 0;
	int dynamic_target = 0;
	int expanded_nodes = 0;
	int nn_batches = 0;
	int nn_evaluations = 0;
	int evaluation_reuses = 0;
	int cpu_threads = 0;
	double uncertainty = 0.0;
	double elapsed_ms = 0.0;
	std::vector<RootMove> root;
};

using SearchProgressCallback = std::function<void(const SearchResult &)>;
using SearchCancelCallback = std::function<bool()>;

class Searcher {
public:
	/// Owns a Melano model in inference mode and applies the configured search controls.
	Searcher(Model model, torch::Device device, SearchOptions options);
	/// Searches one position and optionally emits periodic snapshots for interactive clients.
	SearchResult search(const chess::Board &board, const SearchProgressCallback &progress = {}, int progress_interval_ms = 0, const SearchCancelCallback &cancel = {});
	/// Searches independent positions together so leaf evaluations share neural batches.
	std::vector<SearchResult> search_many(const std::vector<chess::Board> &boards);
	/// Applies search controls while retaining network evaluations compatible with the model.
	void set_options(SearchOptions options);
	/// Clears cross-search network evaluations without changing the configured capacity.
	void clear_evaluation_cache();

private:
	struct Impl;
	std::shared_ptr<Impl> impl_;
};

} // namespace melano
