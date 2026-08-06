// Implements deterministic Value-guided best-first minimax without Policy or MCTS statistics.

#include "eleginus/search.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <stdexcept>

namespace eleginus {

namespace {

struct Edge {
	chess::Move move{chess::Move::NO_MOVE};
	int child = -1;
};

struct Node {
	chess::Board board;
	FloatAccumulator value_accumulator;
	float static_value = 0.5F;
	float backed_value = 0.5F;
	int parent = -1;
	int depth = 0;
	int subtree_nodes = 1;
	bool expanded = false;
	std::vector<Edge> children;
};

struct Candidate {
	float priority = 0.0F;
	int node = -1;
	bool operator<(const Candidate &other) const noexcept { return priority < other.priority; }
};

float frontier_priority(const Node &node, const SearchOptions &options) {
	const float uncertainty = 1.0F - 2.0F * std::abs(node.static_value - 0.5F);
	return -options.depth_penalty * static_cast<float>(node.depth) +
		   options.uncertainty_weight * uncertainty;
}

} // namespace

Searcher::Searcher(const CpuValue &value, SearchOptions options)
	: value_(&value), options_(options) {
	if (options_.expansions <= 0 || options_.max_depth <= 0)
		throw std::invalid_argument("Eleginus BFM search limits must be positive");
}

SearchResult Searcher::search(const chess::Board &board) const {
	SearchResult result;
	if (game_is_over(board)) {
		result.value = terminal_score_side_to_move(board);
		return result;
	}

	Node root;
	root.board = board;
	root.value_accumulator = value_->refresh(board);
	root.static_value = value_->evaluate(root.value_accumulator);
	root.backed_value = root.static_value;

	std::vector<Node> nodes;
	nodes.reserve(static_cast<std::size_t>(options_.expansions) * 32 + 1);
	nodes.push_back(std::move(root));
	std::priority_queue<Candidate> frontier;

	auto propagate = [&](int node_index) {
		for (int current = node_index; current >= 0;) {
			auto &node = nodes[static_cast<std::size_t>(current)];
			if (!node.children.empty()) {
				float best = 0.0F;
				for (const auto &edge : node.children) {
					best = std::max(best,
						1.0F - nodes[static_cast<std::size_t>(edge.child)].backed_value);
				}
				node.backed_value = best;
			}
			current = node.parent;
		}
	};

	auto expand = [&](int node_index) {
		if (nodes[static_cast<std::size_t>(node_index)].expanded ||
			nodes[static_cast<std::size_t>(node_index)].depth >= options_.max_depth)
			return;
		const auto moves = legal_moves(nodes[static_cast<std::size_t>(node_index)].board);
		if (moves.empty()) {
			nodes[static_cast<std::size_t>(node_index)].expanded = true;
			return;
		}
		nodes[static_cast<std::size_t>(node_index)].children.reserve(moves.size());
		for (const auto &move : moves) {
			const auto before = nodes[static_cast<std::size_t>(node_index)].board;
			auto after = before;
			after.makeMove(move);
			Node child;
			child.board = std::move(after);
			child.value_accumulator = value_->update(
				nodes[static_cast<std::size_t>(node_index)].value_accumulator, before, child.board);
			child.static_value = game_is_over(child.board)
								 ? terminal_score_side_to_move(child.board)
								 : value_->evaluate(child.value_accumulator);
			child.backed_value = child.static_value;
			child.parent = node_index;
			child.depth = nodes[static_cast<std::size_t>(node_index)].depth + 1;
			const int child_index = static_cast<int>(nodes.size());
			nodes.push_back(std::move(child));
			nodes[static_cast<std::size_t>(node_index)].children.push_back({move, child_index});
			if (!game_is_over(nodes[static_cast<std::size_t>(child_index)].board) &&
				nodes[static_cast<std::size_t>(child_index)].depth < options_.max_depth) {
				frontier.push({frontier_priority(
					nodes[static_cast<std::size_t>(child_index)], options_), child_index});
			}
		}
		nodes[static_cast<std::size_t>(node_index)].expanded = true;
		for (int current = node_index; current >= 0;
			 current = nodes[static_cast<std::size_t>(current)].parent) {
			nodes[static_cast<std::size_t>(current)].subtree_nodes += static_cast<int>(moves.size());
		}
		result.evaluated_nodes += static_cast<int>(moves.size());
		++result.expanded_nodes;
		propagate(node_index);
	};

	expand(0);
	while (result.expanded_nodes < options_.expansions && !frontier.empty()) {
		const auto candidate = frontier.top();
		frontier.pop();
		if (!nodes[static_cast<std::size_t>(candidate.node)].expanded)
			expand(candidate.node);
	}

	result.value = nodes.front().backed_value;
	result.root.reserve(nodes.front().children.size());
	for (const auto &edge : nodes.front().children) {
		const auto &child = nodes[static_cast<std::size_t>(edge.child)];
		result.root.push_back({edge.move, 1.0F - child.backed_value, child.subtree_nodes});
	}
	const auto best = std::max_element(
		result.root.begin(), result.root.end(),
		[](const RootMove &left, const RootMove &right) { return left.value < right.value; });
	if (best != result.root.end())
		result.move = best->move;
	return result;
}

} // namespace eleginus
