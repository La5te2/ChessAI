// Implements principal-variation search over incremental Value states.

#include "eleginus/search.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
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
	ValueAccumulator value_accumulator;
};

struct OrderedMove {
	chess::Move move{chess::Move::NO_MOVE};
	float order = 0.0F;
};

inline constexpr std::size_t kMaximumLegalMoves = 256;

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
	std::int16_t score = 0;
	std::uint16_t move = chess::Move::NO_MOVE;
	std::uint8_t depth = 0;
	std::uint8_t generation = 0;
	std::uint8_t flags = 0;
	std::uint8_t padding = 0;

	static constexpr std::uint8_t kOccupied = 1;
	static constexpr std::uint8_t kBoundShift = 1;

	bool occupied() const noexcept { return (flags & kOccupied) != 0; }
	Bound bound() const noexcept {
		return static_cast<Bound>((flags >> kBoundShift) & 3U);
	}
	chess::Move best_move() const noexcept { return chess::Move(move); }

	void assign(std::uint64_t new_key, int new_depth, int new_score, Bound new_bound,
				const chess::Move &new_move, std::uint8_t new_generation) noexcept {
		key = new_key;
		score = static_cast<std::int16_t>(std::clamp(new_score,
			static_cast<int>(std::numeric_limits<std::int16_t>::min()),
			static_cast<int>(std::numeric_limits<std::int16_t>::max())));
		move = new_move.move();
		depth = static_cast<std::uint8_t>(std::clamp(new_depth, 0, 255));
		generation = new_generation;
		flags = static_cast<std::uint8_t>(kOccupied |
			(static_cast<std::uint8_t>(new_bound) << kBoundShift));
	}
};

static_assert(sizeof(TranspositionEntry) == 16);

struct TranspositionCluster {
	std::array<TranspositionEntry, 4> entries{};
};

static_assert(sizeof(TranspositionCluster) == 64);

std::uint64_t mix_key(std::uint64_t value) noexcept {
	value ^= value >> 30;
	value *= 0xbf58476d1ce4e5b9ULL;
	value ^= value >> 27;
	value *= 0x94d049bb133111ebULL;
	return value ^ (value >> 31);
}

int history_index(const chess::Move &move) noexcept {
	return move.from().index() * kBoardSquares + move.to().index();
}

std::array<char, 5> move_lexical_key(const chess::Move &move) noexcept {
	const int from = move.from().index();
	const int to = move.to().index();
	char promotion = 0;
	if (move.typeOf() == chess::Move::PROMOTION) {
		if (move.promotionType() == chess::PieceType::BISHOP)
			promotion = 'b';
		else if (move.promotionType() == chess::PieceType::KNIGHT)
			promotion = 'n';
		else if (move.promotionType() == chess::PieceType::QUEEN)
			promotion = 'q';
		else if (move.promotionType() == chess::PieceType::ROOK)
			promotion = 'r';
	}
	return {
		static_cast<char>('a' + from % 8), static_cast<char>('1' + from / 8),
		static_cast<char>('a' + to % 8), static_cast<char>('1' + to / 8), promotion,
	};
}

chess::Movelist generate_legal_moves(const chess::Board &board) {
	chess::Movelist moves;
	chess::movegen::legalmoves(moves, board);
	return moves;
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
			std::max<std::size_t>(sizeof(TranspositionCluster),
				(std::max<std::size_t>(1, megabytes) * 1024U * 1024U) /
					std::max<std::size_t>(1, shards));
		clusters_.resize(std::max<std::size_t>(1, bytes / sizeof(TranspositionCluster)));
	}

	const TranspositionEntry *probe(std::uint64_t key) const noexcept {
		const auto &cluster = clusters_[key % clusters_.size()];
		for (const auto &entry : cluster.entries) {
			if (entry.occupied() && entry.key == key)
				return &entry;
		}
		return nullptr;
	}

	void advance_generation() noexcept { ++generation_; }

	void store(std::uint64_t key, int depth, int score, Bound bound,
			   const chess::Move &best_move) noexcept {
		auto &cluster = clusters_[key % clusters_.size()];
		for (auto &entry : cluster.entries) {
			if (entry.occupied() && entry.key == key) {
				entry.generation = generation_;
				if (depth >= entry.depth || bound == Bound::exact)
					entry.assign(key, depth, score, bound, best_move, generation_);
				return;
			}
		}
		for (auto &entry : cluster.entries) {
			if (!entry.occupied()) {
				entry.assign(key, depth, score, bound, best_move, generation_);
				return;
			}
		}
		auto *replacement = &cluster.entries.front();
		auto replacement_quality = replacement_score(*replacement);
		for (auto &entry : cluster.entries) {
			const int quality = replacement_score(entry);
			if (quality < replacement_quality) {
				replacement = &entry;
				replacement_quality = quality;
			}
		}
		replacement->assign(key, depth, score, bound, best_move, generation_);
	}

	private:
	int replacement_score(const TranspositionEntry &entry) const noexcept {
		const auto age = static_cast<std::uint8_t>(generation_ - entry.generation);
		return static_cast<int>(entry.depth) + (entry.bound() == Bound::exact ? 8 : 0) -
			static_cast<int>(age) * 16;
	}

	std::vector<TranspositionCluster> clusters_;
	std::uint8_t generation_ = 0;
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

int late_move_reduction(int depth, std::size_t quiet_move_index) noexcept {
	int reduction = 1;
	if (depth >= 6)
		++reduction;
	if (quiet_move_index >= 8)
		++reduction;
	return std::min(reduction, depth - 2);
}

void update_history_value(int &history, int bonus) noexcept {
	constexpr int limit = 16000;
	bonus = std::clamp(bonus, -limit, limit);
	history += bonus - history * std::abs(bonus) / limit;
}

class PvsContext {
	public:
	PvsContext(const CpuValue &value, SearchOptions options,
			   SearchCancelCallback cancel, std::atomic_uint64_t *shared_nodes = nullptr,
			   std::size_t hash_shards = 1)
		: value_(value), options_(std::move(options)), cancel_(std::move(cancel)),
		  table_(options_.hash_mb, hash_shards), shared_nodes_(shared_nodes) {}

	void advance_generation() noexcept { table_.advance_generation(); }

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
									   const chess::Movelist &moves, int ply,
									   const chess::Move &preferred = chess::Move::NO_MOVE,
									   bool tactical_only = false) {
		if (moves.empty())
			return {};
		std::vector<OrderedMove> ordered;
		ordered.reserve(moves.size());
		const auto side = static_cast<std::size_t>(position.board.sideToMove());
		for (const auto &move : moves) {
			if (tactical_only && !position.board.isCapture(move) &&
				move.typeOf() != chess::Move::PROMOTION)
				continue;
			int score = tactical_order(position.board, move) * 1024;
			if (!position.board.isCapture(move) && move.typeOf() != chess::Move::PROMOTION) {
				score += history_[side][static_cast<std::size_t>(history_index(move))];
				if (ply >= 0 && ply < static_cast<int>(killers_.size())) {
					if (move == killers_[static_cast<std::size_t>(ply)][0])
						score += 900000;
					else if (move == killers_[static_cast<std::size_t>(ply)][1])
						score += 800000;
				}
			}
			ordered.push_back({move, static_cast<float>(score)});
		}
		std::sort(ordered.begin(), ordered.end(), [](const OrderedMove &left,
													  const OrderedMove &right) {
			if (left.order != right.order)
				return left.order > right.order;
			return move_lexical_key(left.move) < move_lexical_key(right.move);
		});
		if (!is_no_move(preferred)) {
			const auto found = std::find_if(ordered.begin(), ordered.end(),
				[&](const OrderedMove &candidate) { return candidate.move == preferred; });
			if (found != ordered.end())
				std::rotate(ordered.begin(), found, found + 1);
		}
		return ordered;
	}

	class ScopedMove {
		public:
		ScopedMove(const CpuValue &value, Position &position, chess::Move move)
			: value_(value), position_(position), move_(move) {
			position_.board.makeMove(move_);
			try {
				undo_ = value_.apply(position_.value_accumulator, position_.board);
				active_ = true;
			} catch (...) {
				position_.board.unmakeMove(move_);
				throw;
			}
		}

		~ScopedMove() {
			if (active_) {
				value_.undo(position_.value_accumulator, undo_);
				position_.board.unmakeMove(move_);
			}
		}

		ScopedMove(const ScopedMove &) = delete;
		ScopedMove &operator=(const ScopedMove &) = delete;

		private:
		const CpuValue &value_;
		Position &position_;
		chess::Move move_;
		ValueUndoState undo_{};
		bool active_ = false;
	};

	int evaluate(const Position &position) {
		++evaluated_nodes;
		return std::clamp(static_cast<int>(std::lround(
			value_.evaluate(position.value_accumulator) * kValueCentipawnScale)),
			-kMaximumStaticScore, kMaximumStaticScore);
	}

	int quiescence(Position &position, int ply, int remaining_depth,
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

		const auto moves = generate_legal_moves(position.board);
		if (moves.empty())
			return empty_move_score(position.board, ply);
		const auto ordered = ordered_moves(
			position, moves, ply, chess::Move::NO_MOVE, !in_check);
		if (ordered.empty())
			return stand_pat;
		for (const auto &candidate : ordered) {
			if (!in_check && candidate.move.typeOf() != chess::Move::PROMOTION &&
				position.board.givesCheck(candidate.move) == chess::CheckType::NO_CHECK) {
				const int optimistic_gain = piece_value(
					position.board.getCapturing<chess::PieceType>(candidate.move));
				if (stand_pat + optimistic_gain + 120 < alpha)
					continue;
			}
			ScopedMove child(value_, position, candidate.move);
			const int score = -quiescence(position, ply + 1, remaining_depth - 1,
				-beta, -alpha);
			best = std::max(best, score);
			if (score >= beta)
				return score;
			alpha = std::max(alpha, score);
		}
		return best;
	}

	int pvs(Position &position, int depth, int ply, int alpha, int beta) {
		if (depth <= 0)
			return quiescence(position, ply, options_.quiescence_depth, alpha, beta);
		visit_node();
		selective_depth = std::max(selective_depth, ply);
		if (const auto score = rule_terminal_score(position.board, ply))
			return *score;

		const std::uint64_t key = transposition_key(position.board);
		chess::Move preferred{chess::Move::NO_MOVE};
		if (const auto *entry = table_.probe(key)) {
			preferred = entry->best_move();
			if (entry->depth >= depth) {
				const int cached = score_from_table(entry->score, ply);
				if (entry->bound() == Bound::exact)
					return cached;
				if (entry->bound() == Bound::lower)
					alpha = std::max(alpha, cached);
				else
					beta = std::min(beta, cached);
				if (alpha >= beta)
					return cached;
			}
		}
		const int window_alpha = alpha;
		const int window_beta = beta;

		const auto moves = generate_legal_moves(position.board);
		if (moves.empty())
			return empty_move_score(position.board, ply);
		const bool in_check = position.board.inCheck();
		const auto ordered = ordered_moves(position, moves, ply, preferred);
		int best = -kInfinity;
		chess::Move best_move{chess::Move::NO_MOVE};
		std::size_t quiet_move_index = 0;
		std::array<chess::Move, kMaximumLegalMoves> failed_quiet_moves{};
		std::size_t failed_quiet_count = 0;
		for (std::size_t move_index = 0; move_index < ordered.size(); ++move_index) {
			const auto &candidate = ordered[move_index];
			const bool first = move_index == 0;
			const bool quiet = !position.board.isCapture(candidate.move) &&
				candidate.move.typeOf() != chess::Move::PROMOTION;
			const bool checking = position.board.givesCheck(candidate.move) != chess::CheckType::NO_CHECK;
			const auto moving_side = static_cast<std::size_t>(position.board.sideToMove());
			const std::size_t current_quiet_index = quiet_move_index;
			if (quiet)
				++quiet_move_index;
			ScopedMove child(value_, position, candidate.move);
			int score = 0;
			if (first) {
				score = -pvs(position, depth - 1, ply + 1, -beta, -alpha);
			} else {
				const bool reduce = depth >= 3 && current_quiet_index >= 3 && quiet && !checking &&
					!in_check && candidate.move != preferred;
				if (reduce) {
					const int reduction = late_move_reduction(depth, current_quiet_index);
					score = -pvs(position, depth - 1 - reduction, ply + 1, -alpha - 1, -alpha);
					if (score > alpha)
						score = -pvs(position, depth - 1, ply + 1, -alpha - 1, -alpha);
				} else {
					score = -pvs(position, depth - 1, ply + 1, -alpha - 1, -alpha);
				}
				if (score > alpha && score < beta)
					score = -pvs(position, depth - 1, ply + 1, -beta, -alpha);
			}
			if (score > best) {
				best = score;
				best_move = candidate.move;
			}
			alpha = std::max(alpha, score);
			if (alpha >= beta) {
				if (quiet) {
					const int bonus = depth * depth;
					auto &history = history_[moving_side]
						[static_cast<std::size_t>(history_index(candidate.move))];
					update_history_value(history, bonus);
					const int malus = -std::max(1, bonus / 2);
					for (std::size_t index = 0; index < failed_quiet_count; ++index) {
						const auto &failed = failed_quiet_moves[index];
						auto &failed_history = history_[moving_side]
							[static_cast<std::size_t>(history_index(failed))];
						update_history_value(failed_history, malus);
					}
					if (ply < static_cast<int>(killers_.size()) &&
						candidate.move != killers_[static_cast<std::size_t>(ply)][0]) {
						killers_[static_cast<std::size_t>(ply)][1] =
							killers_[static_cast<std::size_t>(ply)][0];
						killers_[static_cast<std::size_t>(ply)][0] = candidate.move;
					}
				}
				break;
			}
			if (quiet && failed_quiet_count < failed_quiet_moves.size())
				failed_quiet_moves[failed_quiet_count++] = candidate.move;
		}

		Bound bound = Bound::exact;
		if (best <= window_alpha)
			bound = Bound::upper;
		else if (best >= window_beta)
			bound = Bound::lower;
		table_.store(key, depth, score_to_table(best, ply), bound, best_move);
		return best;
	}

	IterationResult search_root(Position &root, int depth, int alpha, int beta) {
		IterationResult result;
		const auto moves = generate_legal_moves(root.board);
		if (moves.empty()) {
			result.score_cp = empty_move_score(root.board, 0);
			return result;
		}
		chess::Move preferred{chess::Move::NO_MOVE};
		if (const auto *entry = table_.probe(transposition_key(root.board)))
			preferred = entry->best_move();
		const auto ordered = ordered_moves(root, moves, 0, preferred);
		result.root.reserve(ordered.size());
		const bool exact_root_lines = options_.multipv > 1;
		for (std::size_t move_index = 0; move_index < ordered.size(); ++move_index) {
			check_stop(true);
			const auto &candidate = ordered[move_index];
			const auto nodes_before = nodes;
			ScopedMove child(value_, root, candidate.move);
			int score = 0;
			bool exact_score = exact_root_lines || move_index == 0;
			if (exact_root_lines) {
				score = -pvs(root, depth - 1, 1, -kInfinity, kInfinity);
			} else if (move_index == 0) {
				score = -pvs(root, depth - 1, 1, -beta, -alpha);
			} else {
				score = -pvs(root, depth - 1, 1, -alpha - 1, -alpha);
				if (score > alpha && score < beta) {
					score = -pvs(root, depth - 1, 1, -beta, -alpha);
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
			table_.store(transposition_key(root.board), depth,
				score_to_table(result.score_cp, 0), Bound::exact, result.move);
		}
		std::sort(result.root.begin(), result.root.end(), [](const RootMove &left,
													 const RootMove &right) {
			if (left.score_cp != right.score_cp)
				return left.score_cp > right.score_cp;
			if (left.exact_score != right.exact_score)
				return left.exact_score;
			if (left.order != right.order)
				return left.order > right.order;
			return move_lexical_key(left.move) < move_lexical_key(right.move);
		});
		return result;
	}

	IterationResult search_root_parallel(Position &root, int depth, int alpha, int beta,
										std::vector<std::unique_ptr<PvsContext>> &contexts,
										std::atomic_bool &parallel_stop) {
		IterationResult result;
		const auto moves = generate_legal_moves(root.board);
		if (moves.empty()) {
			result.score_cp = empty_move_score(root.board, 0);
			return result;
		}
		chess::Move preferred{chess::Move::NO_MOVE};
		if (const auto *entry = table_.probe(transposition_key(root.board)))
			preferred = entry->best_move();
		const auto ordered = ordered_moves(root, moves, 0, preferred);
		std::vector<std::optional<RootMove>> rows(ordered.size());
		const bool exact_root_lines = options_.multipv > 1;

		check_stop(true);
		const auto first_nodes = nodes;
		int first_score = 0;
		{
			ScopedMove first_child(value_, root, ordered.front().move);
			first_score = -pvs(root, depth - 1, 1, -beta, -alpha);
		}
		rows.front() = RootMove{
			ordered.front().move, ordered.front().order, first_score, nodes - first_nodes, true};
		std::atomic_int shared_alpha{std::max(alpha, first_score)};
		std::atomic_size_t next_move{1};
		std::atomic_bool interrupted{false};
		std::mutex error_mutex;
		std::exception_ptr error;

		auto search_remaining = [&](PvsContext &context) {
			try {
				Position workspace = root;
				while (!parallel_stop.load(std::memory_order_relaxed)) {
					const std::size_t move_index =
						next_move.fetch_add(1, std::memory_order_relaxed);
					if (move_index >= ordered.size())
						break;
					context.check_stop(true);
					const auto nodes_before = context.nodes;
					ScopedMove child(context.value_, workspace, ordered[move_index].move);
					int score = 0;
					bool exact_score = exact_root_lines;
					if (exact_root_lines) {
						score = -context.pvs(workspace, depth - 1, 1, -kInfinity, kInfinity);
					} else {
						int search_alpha = shared_alpha.load(std::memory_order_relaxed);
						score = -context.pvs(
							workspace, depth - 1, 1, -search_alpha - 1, -search_alpha);
						search_alpha = shared_alpha.load(std::memory_order_relaxed);
						if (score > search_alpha) {
							score = -context.pvs(workspace, depth - 1, 1, -beta, -search_alpha);
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
			context->table_.store(transposition_key(root.board), depth,
				score_to_table(result.score_cp, 0), Bound::exact, result.move);
		}
		std::sort(result.root.begin(), result.root.end(), [](const RootMove &left,
														 const RootMove &right) {
			if (left.score_cp != right.score_cp)
				return left.score_cp > right.score_cp;
			if (left.exact_score != right.exact_score)
				return left.exact_score;
			if (left.order != right.order)
				return left.order > right.order;
			return move_lexical_key(left.move) < move_lexical_key(right.move);
		});
		return result;
	}

	const CpuValue &value_;
	SearchOptions options_;
	SearchCancelCallback cancel_;
	TranspositionTable table_;
	std::array<std::array<int, kBoardSquares * kBoardSquares>, 2> history_{};
	std::array<std::array<chess::Move, 2>, 64> killers_{};
	std::atomic_uint64_t *shared_nodes_ = nullptr;
	std::uint64_t nodes = 0;
	std::uint64_t evaluated_nodes = 0;
	int selective_depth = 0;
};

} // namespace

Searcher::Searcher(const CpuValue &value, SearchOptions options)
	: value_(&value), options_(options) {
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
	const auto root_moves = generate_legal_moves(board);
	if (root_moves.empty()) {
		result.score_cp = empty_move_score(board, 0);
		return result;
	}
	Position root{board, value_->refresh(board)};
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
			*value_, options_, combined_cancel, &shared_nodes, worker_count));
	}
	IterationResult completed;
	for (int depth = 1; depth <= options_.depth; ++depth) {
		if (cancel && cancel())
			break;
		for (auto &context : contexts)
			context->advance_generation();
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
