#include "eleginus/game.hpp"
#include "eleginus/search.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace eleginus {

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kMateScore = 30000;
constexpr int kMateThreshold = 29000;
constexpr int kInfinity = 32000;
constexpr int kMaximumStaticScore = 25000;
constexpr std::size_t kMaximumLegalMoves = 256;

enum class Bound : std::uint8_t { exact, lower, upper };

struct Entry {
	std::uint64_t key = 0;
	std::int16_t score = 0;
	std::uint16_t move = chess::Move::NO_MOVE;
	std::uint8_t depth = 0;
	std::uint8_t generation = 0;
	std::uint8_t flags = 0;
	std::uint8_t padding = 0;

	bool occupied() const noexcept { return (flags & 1U) != 0; }
	Bound bound() const noexcept { return static_cast<Bound>((flags >> 1U) & 3U); }
	chess::Move best_move() const noexcept { return chess::Move(move); }

	void assign(std::uint64_t new_key, int new_depth, int new_score, Bound new_bound, chess::Move new_move,
		std::uint8_t new_generation) noexcept {
		key = new_key;
		score = static_cast<std::int16_t>(std::clamp(new_score, -32768, 32767));
		move = new_move.move();
		depth = static_cast<std::uint8_t>(std::clamp(new_depth, 0, 255));
		generation = new_generation;
		flags = static_cast<std::uint8_t>(1U | (static_cast<std::uint8_t>(new_bound) << 1U));
	}
};

static_assert(sizeof(Entry) == 16);

struct Cluster {
	std::array<Entry, 4> entries{};
};

static_assert(sizeof(Cluster) == 64);

class Table {
public:
	explicit Table(std::size_t megabytes) {
		const auto bytes = std::max<std::size_t>(sizeof(Cluster), std::max<std::size_t>(1, megabytes) * 1024U * 1024U);
		clusters_.resize(std::max<std::size_t>(1, bytes / sizeof(Cluster)));
	}

	const Entry *probe(std::uint64_t key) const noexcept {
		for (const auto &entry : clusters_[key % clusters_.size()].entries) {
			if (entry.occupied() && entry.key == key) {
				return &entry;
			}
		}
		return nullptr;
	}

	void advance() noexcept { ++generation_; }

	void store(std::uint64_t key, int depth, int score, Bound bound, chess::Move move) noexcept {
		auto &cluster = clusters_[key % clusters_.size()];
		for (auto &entry : cluster.entries) {
			if (entry.occupied() && entry.key == key) {
				if (depth >= entry.depth || bound == Bound::exact) {
					entry.assign(key, depth, score, bound, move, generation_);
				}
				return;
			}
		}
		for (auto &entry : cluster.entries) {
			if (!entry.occupied()) {
				entry.assign(key, depth, score, bound, move, generation_);
				return;
			}
		}
		auto *replacement = &cluster.entries.front();
		for (auto &entry : cluster.entries) {
			if (quality(entry) < quality(*replacement)) {
				replacement = &entry;
			}
		}
		replacement->assign(key, depth, score, bound, move, generation_);
	}

private:
	int quality(const Entry &entry) const noexcept {
		const int age = static_cast<std::uint8_t>(generation_ - entry.generation);
		return entry.depth + (entry.bound() == Bound::exact ? 8 : 0) - 16 * age;
	}

	std::vector<Cluster> clusters_;
	std::uint8_t generation_ = 0;
};

} // namespace

class SearchState {
public:
	explicit SearchState(std::size_t hash_mb) : table(hash_mb) {}

	Table table;
	std::array<std::array<int, 64 * 64>, 2> history{};
	std::array<std::array<chess::Move, 2>, 64> killers{};
};

namespace {

struct Interrupted final {};

std::uint64_t mix(std::uint64_t value) noexcept {
	value ^= value >> 30;
	value *= 0xbf58476d1ce4e5b9ULL;
	value ^= value >> 27;
	value *= 0x94d049bb133111ebULL;
	return value ^ (value >> 31);
}

std::uint64_t table_key(const chess::Board &board) noexcept {
	const std::uint64_t rule = (static_cast<std::uint64_t>(board.halfMoveClock()) << 1U) | (board.isRepetition(1) ? 1U : 0U);
	return board.hash() ^ mix(rule + 1U);
}

int score_to_table(int score, int ply) noexcept {
	if (score >= kMateThreshold) {
		return score + ply;
	}
	if (score <= -kMateThreshold) {
		return score - ply;
	}
	return score;
}

int score_from_table(int score, int ply) noexcept {
	if (score >= kMateThreshold) {
		return score - ply;
	}
	if (score <= -kMateThreshold) {
		return score + ply;
	}
	return score;
}

std::optional<int> terminal_score(const chess::Board &board, int ply) {
	if (board.isHalfMoveDraw()) {
		return board.getHalfMoveDrawType().second == chess::GameResult::LOSE ? -kMateScore + ply : 0;
	}
	if (board.isInsufficientMaterial() || board.isRepetition()) {
		return 0;
	}
	return std::nullopt;
}

int empty_score(const chess::Board &board, int ply) noexcept {
	return board.inCheck() ? -kMateScore + ply : 0;
}

int piece_value(chess::PieceType piece) noexcept {
	constexpr std::array<int, 6> values{{100, 320, 330, 500, 900, 0}};
	const int index = static_cast<int>(piece.internal());
	return index >= 0 && index < 6 ? values[static_cast<std::size_t>(index)] : 0;
}

int tactical_order(const chess::Board &board, chess::Move move) noexcept {
	int score = 0;
	if (board.isCapture(move)) {
		score += 16 * piece_value(board.getCapturing<chess::PieceType>(move)) - piece_value(board.at(move.from()).type());
	}
	if (move.typeOf() == chess::Move::PROMOTION) {
		score += 2000 + piece_value(move.promotionType());
	}
	return score;
}

int history_index(chess::Move move) noexcept {
	return move.from().index() * 64 + move.to().index();
}

void update_history(int &history, int bonus) noexcept {
	constexpr int limit = 16000;
	bonus = std::clamp(bonus, -limit, limit);
	history += bonus - history * std::abs(bonus) / limit;
}

int reduction_for(int depth, std::size_t quiet_index) noexcept {
	int reduction = 1 + (depth >= 6 ? 1 : 0) + (quiet_index >= 8 ? 1 : 0);
	return std::min(reduction, depth - 2);
}

struct OrderedMove {
	chess::Move move{chess::Move::NO_MOVE};
	int order = 0;
};

struct Iteration {
	chess::Move move{chess::Move::NO_MOVE};
	int score = -kInfinity;
	std::vector<RootMove> root;
};

class Context {
public:
	Context(const Model &model, const SearchOptions &options, SearchState &state, SearchCancel cancel)
		: model_(model), options_(options), state_(state), cancel_(std::move(cancel)), started_(Clock::now()) {}

	void advance() noexcept { state_.table.advance(); }
	std::uint64_t elapsed_ms() const noexcept {
		return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - started_).count());
	}

	void check_stop(bool force = false) const {
		if (options_.node_limit > 0 && nodes >= options_.node_limit) {
			throw Interrupted{};
		}
		if (options_.movetime_ms > 0 && (force || (nodes & 255U) == 0U) &&
			std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - started_).count() >= options_.movetime_ms) {
			throw Interrupted{};
		}
		if (cancel_ && (force || (nodes & 255U) == 0U) && cancel_()) {
			throw Interrupted{};
		}
	}

	std::vector<OrderedMove> ordered(const chess::Board &board, const chess::Movelist &moves, int ply,
		chess::Move preferred = chess::Move(chess::Move::NO_MOVE), bool tactical_only = false) {
		std::vector<OrderedMove> output;
		output.reserve(moves.size());
		const auto side = static_cast<std::size_t>(board.sideToMove());
		for (const auto move : moves) {
			const bool tactical = board.isCapture(move) || move.typeOf() == chess::Move::PROMOTION;
			if (tactical_only && !tactical) {
				continue;
			}
			int score = tactical_order(board, move) * 1024;
			if (!tactical) {
					score += state_.history[side][static_cast<std::size_t>(history_index(move))];
					if (ply >= 0 && ply < static_cast<int>(state_.killers.size())) {
						if (move == state_.killers[static_cast<std::size_t>(ply)][0]) {
							score += 900000;
						} else if (move == state_.killers[static_cast<std::size_t>(ply)][1]) {
						score += 800000;
					}
				}
			}
			if (move == preferred) {
				score += 2000000;
			}
			output.push_back({move, score});
		}
		std::sort(output.begin(), output.end(), [](const OrderedMove &left, const OrderedMove &right) {
			return left.order != right.order ? left.order > right.order : left.move.move() < right.move.move();
		});
		return output;
	}

	int evaluate(const chess::Board &board) {
		++evaluated_nodes;
		return std::clamp(model_.centipawns(board), -kMaximumStaticScore, kMaximumStaticScore);
	}

	int quiescence(chess::Board &board, int ply, int remaining, int alpha, int beta) {
		visit(ply);
		if (const auto terminal = terminal_score(board, ply)) {
			return *terminal;
		}
		const bool in_check = board.inCheck();
		const bool mandatory_move_probe = !in_check && remaining == options_.quiescence_depth &&
			board.occ().count() <= FeatureMap::kTransitionPieceLimit;
		int stand_pat = -kInfinity;
		int best = -kInfinity;
		if (!in_check) {
			stand_pat = evaluate(board);
			best = mandatory_move_probe ? -kInfinity : stand_pat;
			if ((!mandatory_move_probe && stand_pat >= beta) || remaining <= 0) {
				return stand_pat;
			}
			if (!mandatory_move_probe) {
				alpha = std::max(alpha, stand_pat);
			}
		} else if (remaining <= -4) {
			return evaluate(board);
		}
		chess::Movelist moves;
		chess::movegen::legalmoves(moves, board);
		if (moves.empty()) {
			return empty_score(board, ply);
		}
		const auto candidates = ordered(board, moves, ply, chess::Move(chess::Move::NO_MOVE), !in_check && !mandatory_move_probe);
		if (candidates.empty()) {
			return stand_pat;
		}
		for (const auto &candidate : candidates) {
			if (!in_check && !mandatory_move_probe && candidate.move.typeOf() != chess::Move::PROMOTION &&
				board.givesCheck(candidate.move) == chess::CheckType::NO_CHECK) {
				const int gain = piece_value(board.getCapturing<chess::PieceType>(candidate.move));
				if (stand_pat + gain + 120 < alpha) {
					continue;
				}
			}
			board.makeMove(candidate.move);
			const int score = -quiescence(board, ply + 1, remaining - 1, -beta, -alpha);
			board.unmakeMove(candidate.move);
			best = std::max(best, score);
			if (score >= beta) {
				return score;
			}
			alpha = std::max(alpha, score);
		}
		return best;
	}

	int pvs(chess::Board &board, int depth, int ply, int alpha, int beta) {
		if (depth <= 0) {
			return quiescence(board, ply, options_.quiescence_depth, alpha, beta);
		}
		visit(ply);
		if (const auto terminal = terminal_score(board, ply)) {
			return *terminal;
		}

		const auto key = table_key(board);
		chess::Move preferred(chess::Move::NO_MOVE);
		if (const auto *entry = state_.table.probe(key)) {
			preferred = entry->best_move();
			if (entry->depth >= depth) {
				const int cached = score_from_table(entry->score, ply);
				if (entry->bound() == Bound::exact) {
					return cached;
				}
				if (entry->bound() == Bound::lower) {
					alpha = std::max(alpha, cached);
				} else {
					beta = std::min(beta, cached);
				}
				if (alpha >= beta) {
					return cached;
				}
			}
		}
		const int original_alpha = alpha;
		const int original_beta = beta;

		chess::Movelist moves;
		chess::movegen::legalmoves(moves, board);
		if (moves.empty()) {
			return empty_score(board, ply);
		}
		const bool in_check = board.inCheck();
		const bool transition_sensitive = board.occ().count() <= FeatureMap::kTransitionPieceLimit;
		const auto candidates = ordered(board, moves, ply, preferred);
		int best = -kInfinity;
		chess::Move best_move(chess::Move::NO_MOVE);
		std::size_t quiet_index = 0;
		std::array<chess::Move, kMaximumLegalMoves> failed_quiet{};
		std::size_t failed_count = 0;
		for (std::size_t index = 0; index < candidates.size(); ++index) {
			const auto move = candidates[index].move;
			const bool quiet = !board.isCapture(move) && move.typeOf() != chess::Move::PROMOTION;
			const bool checking = board.givesCheck(move) != chess::CheckType::NO_CHECK;
			const auto moving_side = static_cast<std::size_t>(board.sideToMove());
			const std::size_t current_quiet = quiet_index;
			quiet_index += quiet ? 1U : 0U;
			board.makeMove(move);
			int score;
			if (index == 0) {
				score = -pvs(board, depth - 1, ply + 1, -beta, -alpha);
			} else {
				const bool reduce = depth >= 3 && current_quiet >= 3 && quiet && !checking && !in_check &&
					!transition_sensitive && move != preferred;
				if (reduce) {
					score = -pvs(board, depth - 1 - reduction_for(depth, current_quiet), ply + 1, -alpha - 1, -alpha);
					if (score > alpha) {
						score = -pvs(board, depth - 1, ply + 1, -alpha - 1, -alpha);
					}
				} else {
					score = -pvs(board, depth - 1, ply + 1, -alpha - 1, -alpha);
				}
				if (score > alpha && score < beta) {
					score = -pvs(board, depth - 1, ply + 1, -beta, -alpha);
				}
			}
			board.unmakeMove(move);
			if (score > best) {
				best = score;
				best_move = move;
			}
			alpha = std::max(alpha, score);
			if (alpha >= beta) {
				if (quiet) {
					const int bonus = depth * depth;
					update_history(state_.history[moving_side][static_cast<std::size_t>(history_index(move))], bonus);
					for (std::size_t failed = 0; failed < failed_count; ++failed) {
						update_history(state_.history[moving_side][static_cast<std::size_t>(history_index(failed_quiet[failed]))], -std::max(1, bonus / 2));
					}
					if (ply < static_cast<int>(state_.killers.size()) && move != state_.killers[static_cast<std::size_t>(ply)][0]) {
						state_.killers[static_cast<std::size_t>(ply)][1] = state_.killers[static_cast<std::size_t>(ply)][0];
						state_.killers[static_cast<std::size_t>(ply)][0] = move;
					}
				}
				break;
			}
			if (quiet && failed_count < failed_quiet.size()) {
				failed_quiet[failed_count++] = move;
			}
		}

		Bound bound = Bound::exact;
		if (best <= original_alpha) {
			bound = Bound::upper;
		} else if (best >= original_beta) {
			bound = Bound::lower;
		}
		state_.table.store(key, depth, score_to_table(best, ply), bound, best_move);
		return best;
	}

	Iteration root(chess::Board &board, int depth) {
		Iteration result;
		chess::Movelist moves;
		chess::movegen::legalmoves(moves, board);
		if (moves.empty()) {
			result.score = empty_score(board, 0);
			return result;
		}
		chess::Move preferred(chess::Move::NO_MOVE);
		if (const auto *entry = state_.table.probe(table_key(board))) {
			preferred = entry->best_move();
		}
		const auto candidates = ordered(board, moves, 0, preferred);
		int alpha = -kInfinity;
		const bool exact_lines = options_.multipv > 1;
		for (std::size_t index = 0; index < candidates.size(); ++index) {
			check_stop(true);
			const auto before = nodes;
			const auto move = candidates[index].move;
			board.makeMove(move);
			int score;
			if (exact_lines || index == 0) {
				score = -pvs(board, depth - 1, 1, -kInfinity, exact_lines ? kInfinity : -alpha);
			} else {
				score = -pvs(board, depth - 1, 1, -alpha - 1, -alpha);
				if (score > alpha) {
					score = -pvs(board, depth - 1, 1, -kInfinity, -alpha);
				}
			}
			board.unmakeMove(move);
			result.root.push_back({move, score, nodes - before});
			if (score > result.score) {
				result.score = score;
				result.move = move;
			}
			if (!exact_lines) {
				alpha = std::max(alpha, score);
			}
		}
		std::sort(result.root.begin(), result.root.end(), [](const RootMove &left, const RootMove &right) {
			return left.score_cp != right.score_cp ? left.score_cp > right.score_cp : left.move.move() < right.move.move();
		});
		state_.table.store(table_key(board), depth, score_to_table(result.score, 0), Bound::exact, result.move);
		return result;
	}

	std::uint64_t nodes = 0;
	std::uint64_t evaluated_nodes = 0;
	int selective_depth = 0;

private:
	void visit(int ply) {
		++nodes;
		selective_depth = std::max(selective_depth, ply);
		check_stop();
	}

	const Model &model_;
	const SearchOptions &options_;
	SearchState &state_;
	SearchCancel cancel_;
	Clock::time_point started_;
};

} // namespace

Searcher::Searcher(const Model &model, SearchOptions options) : model_(&model), options_(options) {
	if (options.depth <= 0 || options.depth > 64 || options.quiescence_depth < 0 || options.quiescence_depth > 32 ||
		options.hash_mb == 0 || options.hash_mb > 4096 || options.multipv <= 0 || options.multipv > 256) {
		throw std::invalid_argument("Eleginus search options are outside the supported range");
	}
	state_ = std::make_unique<SearchState>(options_.hash_mb);
}

Searcher::~Searcher() = default;
Searcher::Searcher(Searcher &&) noexcept = default;
Searcher &Searcher::operator=(Searcher &&) noexcept = default;

SearchResult Searcher::search(const chess::Board &board, const SearchProgress &progress, const SearchCancel &cancel) {
	SearchResult result;
	if (const auto terminal = terminal_score(board, 0)) {
		result.score_cp = *terminal;
		return result;
	}
	if (legal_moves(board).empty()) {
		result.score_cp = empty_score(board, 0);
		return result;
	}
	chess::Board root = board;
	Context context(*model_, options_, *state_, cancel);
	for (int depth = 1; depth <= options_.depth; ++depth) {
		context.advance();
		try {
			const auto iteration = context.root(root, depth);
			result.move = iteration.move;
			result.score_cp = iteration.score;
			result.depth = depth;
			result.root = iteration.root;
			result.nodes = context.nodes;
			result.evaluated_nodes = context.evaluated_nodes;
			result.selective_depth = context.selective_depth;
			result.elapsed_ms = context.elapsed_ms();
			if (progress) {
				progress(result);
			}
		} catch (const Interrupted &) {
			break;
		}
	}
	result.nodes = context.nodes;
	result.evaluated_nodes = context.evaluated_nodes;
	result.selective_depth = context.selective_depth;
	result.elapsed_ms = context.elapsed_ms();
	return result;
}

} // namespace eleginus
