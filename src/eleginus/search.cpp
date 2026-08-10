// Implements Policy-ordered principal-variation search over incremental Value states.

#include "eleginus/search.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace eleginus {

namespace {

inline constexpr int kMateScore = 30000;
inline constexpr int kMateThreshold = 29000;
inline constexpr int kMaximumStaticScore = 25000;
inline constexpr int kInfinity = 32000;
inline constexpr float kValueCentipawnScale = 150.0F;

struct Position {
	chess::Board board;
	FloatAccumulator policy_accumulator;
	FloatAccumulator value_accumulator;
};

struct OrderedMove {
	chess::Move move{chess::Move::NO_MOVE};
	float order = 0.0F;
};

struct IterationResult {
	chess::Move move{chess::Move::NO_MOVE};
	int score_cp = -kInfinity;
	std::vector<RootMove> root;
};

struct SearchInterrupted final {};

enum class Bound : std::uint8_t {
	exact,
	lower,
	upper,
};

struct TranspositionEntry {
	std::uint64_t key = 0;
	int depth = -1;
	int score = 0;
	Bound bound = Bound::upper;
	chess::Move best_move{chess::Move::NO_MOVE};
	bool occupied = false;
};

std::uint64_t mix_key(std::uint64_t value) noexcept {
	value ^= value >> 30;
	value *= 0xbf58476d1ce4e5b9ULL;
	value ^= value >> 27;
	value *= 0x94d049bb133111ebULL;
	return value ^ (value >> 31);
}

std::uint64_t transposition_key(const chess::Board &board) noexcept {
	const std::uint64_t repetition = board.isRepetition(1) ? 1U : 0U;
	const std::uint64_t rule_state =
		(static_cast<std::uint64_t>(board.halfMoveClock()) << 1U) | repetition;
	return board.hash() ^ mix_key(rule_state + 1U);
}

int score_to_table(int score, int ply) noexcept {
	if (score >= kMateThreshold)
		return score + ply;
	if (score <= -kMateThreshold)
		return score - ply;
	return score;
}

int score_from_table(int score, int ply) noexcept {
	if (score >= kMateThreshold)
		return score - ply;
	if (score <= -kMateThreshold)
		return score + ply;
	return score;
}

class TranspositionTable {
	public:
	explicit TranspositionTable(std::size_t megabytes, std::size_t shards = 1) {
		const std::size_t bytes =
			std::max<std::size_t>(sizeof(TranspositionEntry),
				(std::max<std::size_t>(1, megabytes) * 1024U * 1024U) /
					std::max<std::size_t>(1, shards));
		entries_.resize(std::max<std::size_t>(1, bytes / sizeof(TranspositionEntry)));
	}

	const TranspositionEntry *probe(std::uint64_t key) const noexcept {
		const auto &entry = entries_[key % entries_.size()];
		return entry.occupied && entry.key == key ? &entry : nullptr;
	}

	void store(std::uint64_t key, int depth, int score, Bound bound,
			   const chess::Move &best_move) noexcept {
		auto &entry = entries_[key % entries_.size()];
		if (!entry.occupied || entry.key != key || depth >= entry.depth || bound == Bound::exact) {
			entry = TranspositionEntry{key, depth, score, bound, best_move, true};
		}
	}

	private:
	std::vector<TranspositionEntry> entries_;
};

bool is_no_move(const chess::Move &move) noexcept {
	return move.move() == chess::Move::NO_MOVE;
}

std::optional<int> rule_terminal_score(const chess::Board &board, int ply) {
	if (board.isHalfMoveDraw()) {
		const auto outcome = board.getHalfMoveDrawType();
		return outcome.second == chess::GameResult::LOSE ? -kMateScore + ply : 0;
	}
	if (board.isInsufficientMaterial() || board.isRepetition())
		return 0;
	return std::nullopt;
}

int empty_move_score(const chess::Board &board, int ply) noexcept {
	return board.inCheck() ? -kMateScore + ply : 0;
}

int piece_value(chess::PieceType piece) noexcept {
	if (piece == chess::PieceType::PAWN)
		return 100;
	if (piece == chess::PieceType::KNIGHT)
		return 320;
	if (piece == chess::PieceType::BISHOP)
		return 330;
	if (piece == chess::PieceType::ROOK)
		return 500;
	if (piece == chess::PieceType::QUEEN)
		return 900;
	return 0;
}

int tactical_order(const chess::Board &board, const chess::Move &move) noexcept {
	int score = 0;
	if (board.isCapture(move)) {
		const auto victim = board.getCapturing<chess::PieceType>(move);
		const auto attacker = board.at(move.from()).type();
		score += 16 * piece_value(victim) - piece_value(attacker);
	}
	if (move.typeOf() == chess::Move::PROMOTION)
		score += 2000 + piece_value(move.promotionType());
	return score;
}

int late_move_reduction(int depth, std::size_t move_index) noexcept {
	int reduction = 1;
	if (depth >= 6)
		++reduction;
	if (move_index >= 8)
		++reduction;
	return std::min(reduction, depth - 2);
}

class PvsContext {
	public:
	PvsContext(const CpuPolicy &policy, const CpuValue &value, SearchOptions options,
			   SearchCancelCallback cancel, std::atomic_uint64_t *shared_nodes = nullptr,
			   std::size_t hash_shards = 1)
		: policy_(policy), value_(value), options_(std::move(options)), cancel_(std::move(cancel)),
		  table_(options_.hash_mb, hash_shards), shared_nodes_(shared_nodes) {}

	void check_stop(bool force = false) const {
		const std::uint64_t visited =
			shared_nodes_ == nullptr ? nodes : shared_nodes_->load(std::memory_order_relaxed);
		if (options_.node_limit > 0 && visited >= options_.node_limit)
			throw SearchInterrupted{};
		if (cancel_ && (force || (nodes & 255U) == 0U) && cancel_())
			throw SearchInterrupted{};
	}

	void visit_node() {
		++nodes;
		if (shared_nodes_ != nullptr)
			shared_nodes_->fetch_add(1, std::memory_order_relaxed);
		check_stop();
	}

	std::vector<OrderedMove> ordered_moves(const Position &position,
									   std::vector<chess::Move> moves, bool use_policy,
									   const chess::Move &preferred = chess::Move::NO_MOVE) {
		if (moves.empty())
			return {};
		std::vector<OrderedMove> ordered;
		ordered.reserve(moves.size());
		if (use_policy) {
			const auto priors = policy_.evaluate(position.policy_accumulator, moves);
			for (std::size_t index = 0; index < moves.size(); ++index)
				ordered.push_back({moves[index], priors[index]});
		} else {
			for (const auto &move : moves)
				ordered.push_back({move, static_cast<float>(tactical_order(position.board, move))});
		}
		std::sort(ordered.begin(), ordered.end(), [](const OrderedMove &left,
													  const OrderedMove &right) {
			if (left.order != right.order)
				return left.order > right.order;
			return move_uci(left.move) < move_uci(right.move);
		});
		if (!is_no_move(preferred)) {
			const auto found = std::find_if(ordered.begin(), ordered.end(),
				[&](const OrderedMove &candidate) { return candidate.move == preferred; });
			if (found != ordered.end())
				std::rotate(ordered.begin(), found, found + 1);
		}
		return ordered;
	}

	Position play(const Position &position, const chess::Move &move) const {
		Position child;
		child.board = position.board;
		child.board.makeMove(move);
		child.policy_accumulator = policy_.update(
			position.policy_accumulator, position.board, child.board);
		child.value_accumulator = value_.update(
			position.value_accumulator, position.board, child.board);
		return child;
	}

	int evaluate(const Position &position) {
		++evaluated_nodes;
		return std::clamp(static_cast<int>(std::lround(
			value_.evaluate(position.value_accumulator) * kValueCentipawnScale)),
			-kMaximumStaticScore, kMaximumStaticScore);
	}

	int quiescence(const Position &position, int ply, int remaining_depth,
				   int alpha, int beta) {
		visit_node();
		selective_depth = std::max(selective_depth, ply);
		if (const auto score = rule_terminal_score(position.board, ply))
			return *score;

		const bool in_check = position.board.inCheck();
		int best = -kInfinity;
		int stand_pat = -kInfinity;
		if (!in_check) {
			stand_pat = evaluate(position);
			best = stand_pat;
			if (stand_pat >= beta)
				return stand_pat;
			alpha = std::max(alpha, stand_pat);
			if (remaining_depth <= 0)
				return stand_pat;
		} else if (remaining_depth <= -4) {
			return evaluate(position);
		}

		auto moves = legal_moves(position.board);
		if (moves.empty())
			return empty_move_score(position.board, ply);
		if (!in_check) {
			moves.erase(std::remove_if(moves.begin(), moves.end(), [&](const chess::Move &move) {
				return !position.board.isCapture(move) && move.typeOf() != chess::Move::PROMOTION;
			}), moves.end());
			if (moves.empty())
				return stand_pat;
		}

		const auto ordered = ordered_moves(position, std::move(moves), false);
		for (const auto &candidate : ordered) {
			if (!in_check && candidate.move.typeOf() != chess::Move::PROMOTION &&
				position.board.givesCheck(candidate.move) == chess::CheckType::NO_CHECK) {
				const int optimistic_gain = piece_value(
					position.board.getCapturing<chess::PieceType>(candidate.move));
				if (stand_pat + optimistic_gain + 120 < alpha)
					continue;
			}
			auto child = play(position, candidate.move);
			const int score = -quiescence(child, ply + 1, remaining_depth - 1,
				-beta, -alpha);
			best = std::max(best, score);
			if (score >= beta)
				return score;
			alpha = std::max(alpha, score);
		}
		return best;
	}

	int pvs(const Position &position, int depth, int ply, int alpha, int beta) {
		if (depth <= 0)
			return quiescence(position, ply, options_.quiescence_depth, alpha, beta);
		visit_node();
		selective_depth = std::max(selective_depth, ply);
		if (const auto score = rule_terminal_score(position.board, ply))
			return *score;

		const std::uint64_t key = transposition_key(position.board);
		chess::Move preferred{chess::Move::NO_MOVE};
		if (const auto *entry = table_.probe(key)) {
			preferred = entry->best_move;
			if (entry->depth >= depth) {
				const int cached = score_from_table(entry->score, ply);
				if (entry->bound == Bound::exact)
					return cached;
				if (entry->bound == Bound::lower)
					alpha = std::max(alpha, cached);
				else
					beta = std::min(beta, cached);
				if (alpha >= beta)
					return cached;
			}
		}
		const int window_alpha = alpha;
		const int window_beta = beta;

		auto moves = legal_moves(position.board);
		if (moves.empty())
			return empty_move_score(position.board, ply);
		const bool in_check = position.board.inCheck();
		const auto ordered = ordered_moves(position, std::move(moves), true, preferred);
		int best = -kInfinity;
		chess::Move best_move{chess::Move::NO_MOVE};
		for (std::size_t move_index = 0; move_index < ordered.size(); ++move_index) {
			const auto &candidate = ordered[move_index];
			const bool first = move_index == 0;
			const bool quiet = !position.board.isCapture(candidate.move) &&
				candidate.move.typeOf() != chess::Move::PROMOTION;
			const bool checking = position.board.givesCheck(candidate.move) != chess::CheckType::NO_CHECK;
			auto child = play(position, candidate.move);
			int score = 0;
			if (first) {
				score = -pvs(child, depth - 1, ply + 1, -beta, -alpha);
			} else {
				const bool reduce = depth >= 3 && move_index >= 3 && quiet && !checking &&
					!in_check && candidate.move != preferred;
				if (reduce) {
					const int reduction = late_move_reduction(depth, move_index);
					score = -pvs(child, depth - 1 - reduction, ply + 1, -alpha - 1, -alpha);
					if (score > alpha)
						score = -pvs(child, depth - 1, ply + 1, -alpha - 1, -alpha);
				} else {
					score = -pvs(child, depth - 1, ply + 1, -alpha - 1, -alpha);
				}
				if (score > alpha && score < beta)
					score = -pvs(child, depth - 1, ply + 1, -beta, -alpha);
			}
			if (score > best) {
				best = score;
				best_move = candidate.move;
			}
			alpha = std::max(alpha, score);
			if (alpha >= beta)
				break;
		}

		Bound bound = Bound::exact;
		if (best <= window_alpha)
			bound = Bound::upper;
		else if (best >= window_beta)
			bound = Bound::lower;
		table_.store(key, depth, score_to_table(best, ply), bound, best_move);
		if (!is_no_move(best_move))
			ordering_hints_[position.board.hash()] = best_move;
		return best;
	}

	IterationResult search_root(const Position &root, int depth, int alpha, int beta) {
		IterationResult result;
		auto moves = legal_moves(root.board);
		if (moves.empty()) {
			result.score_cp = empty_move_score(root.board, 0);
			return result;
		}
		chess::Move preferred{chess::Move::NO_MOVE};
		const auto hint = ordering_hints_.find(root.board.hash());
		if (hint != ordering_hints_.end())
			preferred = hint->second;
		const auto ordered = ordered_moves(root, std::move(moves), true, preferred);
		result.root.reserve(ordered.size());
		const bool exact_root_lines = options_.multipv > 1;
		for (std::size_t move_index = 0; move_index < ordered.size(); ++move_index) {
			check_stop(true);
			const auto &candidate = ordered[move_index];
			const auto nodes_before = nodes;
			auto child = play(root, candidate.move);
			int score = 0;
			bool exact_score = exact_root_lines || move_index == 0;
			if (exact_root_lines) {
				score = -pvs(child, depth - 1, 1, -kInfinity, kInfinity);
			} else if (move_index == 0) {
				score = -pvs(child, depth - 1, 1, -beta, -alpha);
			} else {
				score = -pvs(child, depth - 1, 1, -alpha - 1, -alpha);
				if (score > alpha && score < beta) {
					score = -pvs(child, depth - 1, 1, -beta, -alpha);
					exact_score = true;
				}
			}
			result.root.push_back(
				{candidate.move, candidate.order, score, nodes - nodes_before, exact_score});
			if (score > result.score_cp) {
				result.score_cp = score;
				result.move = candidate.move;
			}
			if (!exact_root_lines)
				alpha = std::max(alpha, score);
			if (!exact_root_lines && alpha >= beta)
				break;
		}
		if (!is_no_move(result.move)) {
			ordering_hints_[root.board.hash()] = result.move;
			table_.store(transposition_key(root.board), depth,
				score_to_table(result.score_cp, 0), Bound::exact, result.move);
		}
		std::sort(result.root.begin(), result.root.end(), [](const RootMove &left,
													 const RootMove &right) {
			if (left.score_cp != right.score_cp)
				return left.score_cp > right.score_cp;
			if (left.exact_score != right.exact_score)
				return left.exact_score;
			if (left.prior != right.prior)
				return left.prior > right.prior;
			return move_uci(left.move) < move_uci(right.move);
		});
		return result;
	}

	IterationResult search_root_parallel(const Position &root, int depth, int alpha, int beta,
										std::vector<std::unique_ptr<PvsContext>> &contexts,
										std::atomic_bool &parallel_stop) {
		IterationResult result;
		auto moves = legal_moves(root.board);
		if (moves.empty()) {
			result.score_cp = empty_move_score(root.board, 0);
			return result;
		}
		chess::Move preferred{chess::Move::NO_MOVE};
		const auto hint = ordering_hints_.find(root.board.hash());
		if (hint != ordering_hints_.end())
			preferred = hint->second;
		const auto ordered = ordered_moves(root, std::move(moves), true, preferred);
		std::vector<std::optional<RootMove>> rows(ordered.size());
		const bool exact_root_lines = options_.multipv > 1;

		check_stop(true);
		const auto first_nodes = nodes;
		auto first_child = play(root, ordered.front().move);
		const int first_score = -pvs(first_child, depth - 1, 1, -beta, -alpha);
		rows.front() = RootMove{
			ordered.front().move, ordered.front().order, first_score, nodes - first_nodes, true};
		std::atomic_int shared_alpha{std::max(alpha, first_score)};
		std::atomic_size_t next_move{1};
		std::atomic_bool interrupted{false};
		std::mutex error_mutex;
		std::exception_ptr error;

		auto search_remaining = [&](PvsContext &context) {
			try {
				while (!parallel_stop.load(std::memory_order_relaxed)) {
					const std::size_t move_index =
						next_move.fetch_add(1, std::memory_order_relaxed);
					if (move_index >= ordered.size())
						break;
					context.check_stop(true);
					const auto nodes_before = context.nodes;
					auto child = context.play(root, ordered[move_index].move);
					int score = 0;
					bool exact_score = exact_root_lines;
					if (exact_root_lines) {
						score = -context.pvs(child, depth - 1, 1, -kInfinity, kInfinity);
					} else {
						int search_alpha = shared_alpha.load(std::memory_order_relaxed);
						score = -context.pvs(
							child, depth - 1, 1, -search_alpha - 1, -search_alpha);
						search_alpha = shared_alpha.load(std::memory_order_relaxed);
						if (score > search_alpha) {
							score = -context.pvs(child, depth - 1, 1, -beta, -search_alpha);
							exact_score = score > search_alpha;
							int observed = shared_alpha.load(std::memory_order_relaxed);
							while (score > observed &&
								   !shared_alpha.compare_exchange_weak(observed, score,
									   std::memory_order_relaxed)) {
							}
						}
					}
					rows[move_index] = RootMove{ordered[move_index].move,
						ordered[move_index].order, score, context.nodes - nodes_before, exact_score};
				}
			} catch (const SearchInterrupted &) {
				interrupted = true;
				parallel_stop = true;
			} catch (...) {
				{
					std::lock_guard lock(error_mutex);
					if (!error)
						error = std::current_exception();
				}
				parallel_stop = true;
			}
		};

		std::vector<std::thread> workers;
		workers.reserve(contexts.size() - 1);
		for (std::size_t index = 1; index < contexts.size(); ++index)
			workers.emplace_back(search_remaining, std::ref(*contexts[index]));
		search_remaining(*contexts.front());
		for (auto &worker : workers)
			worker.join();
		if (error)
			std::rethrow_exception(error);
		if (interrupted || parallel_stop.load(std::memory_order_relaxed))
			throw SearchInterrupted{};

		result.root.reserve(rows.size());
		for (const auto &row : rows) {
			if (row)
				result.root.push_back(*row);
		}
		result.move = ordered.front().move;
		result.score_cp = first_score;
		for (const auto &row : result.root) {
			if (row.exact_score && row.score_cp > result.score_cp) {
				result.move = row.move;
				result.score_cp = row.score_cp;
			}
		}
		for (auto &context : contexts) {
			context->ordering_hints_[root.board.hash()] = result.move;
			context->table_.store(transposition_key(root.board), depth,
				score_to_table(result.score_cp, 0), Bound::exact, result.move);
		}
		std::sort(result.root.begin(), result.root.end(), [](const RootMove &left,
														 const RootMove &right) {
			if (left.score_cp != right.score_cp)
				return left.score_cp > right.score_cp;
			if (left.exact_score != right.exact_score)
				return left.exact_score;
			if (left.prior != right.prior)
				return left.prior > right.prior;
			return move_uci(left.move) < move_uci(right.move);
		});
		return result;
	}

	const CpuPolicy &policy_;
	const CpuValue &value_;
	SearchOptions options_;
	SearchCancelCallback cancel_;
	TranspositionTable table_;
	std::unordered_map<std::uint64_t, chess::Move> ordering_hints_;
	std::atomic_uint64_t *shared_nodes_ = nullptr;
	std::uint64_t nodes = 0;
	std::uint64_t evaluated_nodes = 0;
	int selective_depth = 0;
};

} // namespace

Searcher::Searcher(const CpuPolicy &policy, const CpuValue &value, SearchOptions options)
	: policy_(&policy), value_(&value), options_(options) {
	if (options_.depth <= 0 || options_.depth > 64 ||
		options_.quiescence_depth < 0 || options_.quiescence_depth > 32 ||
		options_.hash_mb == 0 || options_.hash_mb > 4096 ||
		options_.threads <= 0 || options_.threads > 256 ||
		options_.multipv <= 0 || options_.multipv > 256) {
		throw std::invalid_argument("Eleginus PVS options are outside the supported range");
	}
}

SearchResult Searcher::search(const chess::Board &board,
						  const SearchProgressCallback &progress,
						  const SearchCancelCallback &cancel) const {
	SearchResult result;
	if (const auto score = rule_terminal_score(board, 0)) {
		result.score_cp = *score;
		return result;
	}
	if (const auto moves = legal_moves(board); moves.empty()) {
		result.score_cp = empty_move_score(board, 0);
		return result;
	}

	const auto root_moves = legal_moves(board);
	Position root{board, policy_->refresh(board), value_->refresh(board)};
	const std::size_t worker_count = std::min<std::size_t>(
		static_cast<std::size_t>(options_.threads), std::max<std::size_t>(1, root_moves.size()));
	std::atomic_uint64_t shared_nodes{0};
	std::atomic_bool parallel_stop{false};
	SearchCancelCallback combined_cancel = [&] {
		return parallel_stop.load(std::memory_order_relaxed) || (cancel && cancel());
	};
	std::vector<std::unique_ptr<PvsContext>> contexts;
	contexts.reserve(worker_count);
	for (std::size_t index = 0; index < worker_count; ++index) {
		contexts.push_back(std::make_unique<PvsContext>(
			*policy_, *value_, options_, combined_cancel, &shared_nodes, worker_count));
	}
	IterationResult completed;
	for (int depth = 1; depth <= options_.depth; ++depth) {
		if (cancel && cancel())
			break;
		parallel_stop = false;
		try {
			completed = worker_count == 1
				? contexts.front()->search_root(root, depth, -kInfinity, kInfinity)
				: contexts.front()->search_root_parallel(
					root, depth, -kInfinity, kInfinity, contexts, parallel_stop);
		} catch (const SearchInterrupted &) {
			break;
		}
		result.move = completed.move;
		result.score_cp = completed.score_cp;
		result.depth = depth;
		result.root = completed.root;
		result.nodes = shared_nodes.load(std::memory_order_relaxed);
		result.evaluated_nodes = 0;
		result.selective_depth = 0;
		for (const auto &context : contexts) {
			result.evaluated_nodes += context->evaluated_nodes;
			result.selective_depth = std::max(result.selective_depth, context->selective_depth);
		}
		if (progress)
			progress(result);
	}
	result.nodes = shared_nodes.load(std::memory_order_relaxed);
	result.evaluated_nodes = 0;
	result.selective_depth = 0;
	for (const auto &context : contexts) {
		result.evaluated_nodes += context->evaluated_nodes;
		result.selective_depth = std::max(result.selective_depth, context->selective_depth);
	}
	return result;
}

} // namespace eleginus
