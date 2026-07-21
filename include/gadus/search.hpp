#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "gadus/game.hpp"
#include "gadus/model.hpp"

namespace gadus {

enum class SearchType { Closed, OnlyMcts };

struct SearchOptions {
	SearchType type = SearchType::OnlyMcts;
	int mcts_sims = 100;
	int mcts_min_sims = 0;
	int mcts_batch_size = 32;
	double movetime_ms = 0.0;
	double c_puct = 0.5;
	double c_puct_base = 19652.0;
	double c_puct_factor = 1.0;
	double fpu_reduction = 0.15;
	double virtual_loss = 0.0;
	double repetition_policy_penalty = 0.0;
	bool instant_mate_first = false;
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
	double uncertainty = 0.0;
	double elapsed_ms = 0.0;
	std::vector<RootMove> root;
};

class Searcher {
	public:
	Searcher(Model model, torch::Device device, SearchOptions options);
	SearchResult search(const chess::Board &board);
	std::vector<SearchResult> search_many(const std::vector<chess::Board> &boards);

	private:
	struct Impl;
	std::shared_ptr<Impl> impl_;
};

SearchType parse_search_type(const std::string &value);
std::string search_type_name(SearchType value);

} // namespace gadus
