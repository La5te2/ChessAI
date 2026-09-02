#include "eleginus/game.hpp"
#include "eleginus/search.hpp"
#include <algorithm>
#include <array>
#include <bit>
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
		constexpr std::size_t kSequentialMultiPVLimit = 8;
		constexpr int kTransitionPieceLimit = 10;
		constexpr std::array<int, 6> kPieceValues{100, 320, 330, 500, 900, 0};

		int toCp(float h) {
			if (!std::isfinite(h))
				throw std::runtime_error("nonfinite Eleginus evaluation");
			return static_cast<int>(std::lround(std::clamp(400.0F * h, -25000.0F, 25000.0F)));
		}

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
			chess::Move bestMove() const noexcept { return chess::Move(move); }

			void assign(std::uint64_t newKey, int newDepth, int newScore, Bound newBound, chess::Move newMove, std::uint8_t newGeneration) noexcept {
				key = newKey;
				score = static_cast<std::int16_t>(std::clamp(newScore, -32768, 32767));
				move = newMove.move();
				depth = static_cast<std::uint8_t>(std::clamp(newDepth, 0, 255));
				generation = newGeneration;
				flags = static_cast<std::uint8_t>(1U | (static_cast<std::uint8_t>(newBound) << 1U));
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
				clusters.resize(std::bit_floor(std::max<std::size_t>(1, bytes / sizeof(Cluster))));
				mask = clusters.size() - 1;
			}

			const Entry *probe(std::uint64_t key) const noexcept {
				for (const auto &entry : clusters[key & mask].entries) {
					if (entry.occupied() && entry.key == key) {
						return &entry;
					}
				}
				return nullptr;
			}

			void advance() noexcept { ++generation; }

			void store(std::uint64_t key, int depth, int score, Bound bound, chess::Move move) noexcept {
				auto &cluster = clusters[key & mask];
				for (auto &entry : cluster.entries) {
					if (entry.occupied() && entry.key == key) {
						if (depth >= entry.depth || bound == Bound::exact) {
							entry.assign(key, depth, score, bound, move, generation);
						}
						return;
					}
				}
				for (auto &entry : cluster.entries) {
					if (!entry.occupied()) {
						entry.assign(key, depth, score, bound, move, generation);
						return;
					}
				}
				auto *replacement = &cluster.entries.front();
				for (auto &entry : cluster.entries) {
					if (quality(entry) < quality(*replacement)) {
						replacement = &entry;
					}
				}
				replacement->assign(key, depth, score, bound, move, generation);
			}

		private:
			int quality(const Entry &entry) const noexcept {
				const int age = static_cast<std::uint8_t>(generation - entry.generation);
				return entry.depth + (entry.bound() == Bound::exact ? 8 : 0) - 16 * age;
			}

			std::vector<Cluster> clusters;
			std::size_t mask = 0;
			std::uint8_t generation = 0;
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

		std::uint64_t tableKey(const chess::Board &board) noexcept {
			const std::uint64_t rule = (static_cast<std::uint64_t>(board.halfMoveClock()) << 1U) | (board.isRepetition(1) ? 1U : 0U);
			return board.hash() ^ mix(rule + 1U);
		}

		int scoreToTable(int score, int ply) noexcept {
			if (score >= kMateThreshold) {
				return score + ply;
			}
			if (score <= -kMateThreshold) {
				return score - ply;
			}
			return score;
		}

		int scoreFromTable(int score, int ply) noexcept {
			if (score >= kMateThreshold) {
				return score - ply;
			}
			if (score <= -kMateThreshold) {
				return score + ply;
			}
			return score;
		}

		std::optional<int> terminalScore(const chess::Board &board, int ply, bool repetition = true) {
			if (board.isHalfMoveDraw()) {
				return board.getHalfMoveDrawType().second == chess::GameResult::LOSE ? -kMateScore + ply : 0;
			}
			if (board.isInsufficientMaterial() || (repetition && board.isRepetition())) {
				return 0;
			}
			return std::nullopt;
		}

		int emptyScore(const chess::Board &board, int ply) noexcept {
			return board.inCheck() ? -kMateScore + ply : 0;
		}

		int pieceValue(chess::PieceType piece) noexcept {
			const int index = static_cast<int>(piece.internal());
			return index >= 0 && index < 6 ? kPieceValues[static_cast<std::size_t>(index)] : 0;
		}

		// Least-valuable legal recaptures on one square. SEE is a selective-search heuristic, not a proven bound.
		int see(const chess::Board &board, chess::Move move) {
			if (move.typeOf() == chess::Move::CASTLING)
				return 0;
			using Bits = std::uint64_t;
			std::array<std::array<Bits, 6>, 2> pieces;
			for (int c = 0; c < 2; ++c)
				for (int t = 0; t < 6; ++t)
					pieces[c][t] = board.pieces(chess::PieceType(static_cast<chess::PieceType::underlying>(t)), chess::Color(c)).getBits();
			int side = static_cast<int>(board.sideToMove());
			const int from = move.from().index(), to = move.to().index();
			const Bits target = 1ULL << to;
			const int moving = static_cast<int>(board.at(move.from()).type().internal());
			int victim = move.typeOf() == chess::Move::PROMOTION ? static_cast<int>(move.promotionType().internal()) : moving;
			std::array<int, 64> gain;
			gain[0] = kPieceValues[victim] - kPieceValues[moving];
			Bits occupied = board.occ().getBits() & ~(1ULL << from);
			if (board.isCapture(move)) {
				const int captured = static_cast<int>(board.getCapturing<chess::PieceType>(move).internal());
				const Bits square = 1ULL << (move.typeOf() == chess::Move::ENPASSANT ? to ^ 8 : to);
				pieces[side ^ 1][captured] &= ~square;
				occupied &= ~square;
				gain[0] += kPieceValues[captured];
			}
			pieces[side][moving] &= ~(1ULL << from);
			pieces[side][victim] |= target;
			occupied |= target;
			const auto attackers = [&](int square, int color) {
				const auto q = chess::Square(square);
				const auto occ = chess::Bitboard(occupied);
				const auto &p = pieces[color];
				return (chess::attacks::pawn(chess::Color(color ^ 1), q).getBits() & p[0]) | (chess::attacks::knight(q).getBits() & p[1]) |
				    (chess::attacks::bishop(q, occ).getBits() & (p[2] | p[4])) | (chess::attacks::rook(q, occ).getBits() & (p[3] | p[4])) |
				    (chess::attacks::king(q).getBits() & p[5]);
			};
			int depth = 0;
			while (victim != 5) {
				side ^= 1;
				const auto candidates = attackers(to, side);
				bool found = false;
				for (int type = 0; type < 6 && !found; ++type) {
					auto remaining = candidates & pieces[side][type];
					while (remaining) {
						const Bits source = remaining & (~remaining + 1);
						remaining &= remaining - 1;
						const int placed = type == 0 && (to / 8 == 0 || to / 8 == 7) ? 4 : type;
						pieces[side][type] &= ~source;
						pieces[side][placed] |= target;
						pieces[side ^ 1][victim] &= ~target;
						occupied &= ~source;
						// Reject pinned recapturers and king captures onto an attacked square.
						if (attackers(std::countr_zero(pieces[side][5]), side ^ 1)) {
							pieces[side][type] |= source;
							pieces[side][placed] &= ~target;
							pieces[side ^ 1][victim] |= target;
							occupied |= source;
							continue;
						}
						++depth;
						gain[depth] = kPieceValues[victim] + kPieceValues[placed] - kPieceValues[type] - gain[depth - 1];
						victim = placed;
						found = true;
						break;
					}
				}
				if (!found)
					break;
			}
			while (depth > 0) {
				gain[depth - 1] = -std::max(-gain[depth - 1], gain[depth]);
				--depth;
			}
			return gain[0];
		}

		bool mateBounds(int ply, int &alpha, int &beta) noexcept {
			alpha = std::max(alpha, -kMateScore + ply);
			beta = std::min(beta, kMateScore - ply - 1);
			return alpha >= beta;
		}

		int tacticalOrder(const chess::Board &board, chess::Move move) noexcept {
			int score = 0;
			if (board.isCapture(move)) {
				score += 16 * pieceValue(board.getCapturing<chess::PieceType>(move)) - pieceValue(board.at(move.from()).type());
			}
			if (move.typeOf() == chess::Move::PROMOTION) {
				score += 2000 + pieceValue(move.promotionType());
			}
			return score;
		}

		int historyIndex(chess::Move move) noexcept {
			return move.from().index() * 64 + move.to().index();
		}

		void updateHistory(int &history, int bonus) noexcept {
			constexpr int limit = 16000;
			bonus = std::clamp(bonus, -limit, limit);
			history += bonus - history * std::abs(bonus) / limit;
		}

		int reductionFor(int depth, std::size_t index, bool pv, bool improving, int history) noexcept {
			static const auto reductions = [] {
				std::array<std::array<int, kMaximumLegalMoves>, 65> table{};
				for (int d = 1; d <= 64; ++d)
					for (std::size_t m = 1; m < kMaximumLegalMoves; ++m)
						table[d][m] = static_cast<int>(std::log(d) * std::log(m + 1) / 2.0);
				return table;
			}();
			const int reduction = reductions[std::min(depth, 64)][std::min(index, kMaximumLegalMoves - 1)] + !pv + !improving - history / 4000;
			return std::clamp(reduction, 1, depth - 2);
		}

		struct OrderedMove {
			chess::Move move{chess::Move::NO_MOVE};
			int order = 0;
			int gain = 0;
		};

		class MovePicker {
		public:
			void add(chess::Move move, int order, int gain = 0) { items[count++] = {move, order, gain}; }
			int gain() const { return items[cursor - 1].gain; }
			chess::Move next() {
				if (cursor == count)
					return chess::Move(chess::Move::NO_MOVE);
				auto best = cursor;
				for (auto i = cursor + 1; i < count; ++i) {
					const auto &a = items[i], &b = items[best];
					if (a.order > b.order || (a.order == b.order && a.move.move() < b.move.move()))
						best = i;
				}
				std::swap(items[cursor], items[best]);
				return items[cursor++].move;
			}

		private:
			std::array<OrderedMove, kMaximumLegalMoves> items;
			std::size_t cursor = 0, count = 0;
		};

		void tacticalMoves(chess::Movelist &moves, const chess::Board &board) {
			chess::movegen::legalmoves<chess::movegen::MoveGenType::CAPTURE>(moves, board);
			const auto rank = board.sideToMove() == chess::Color::WHITE ? 0x00FF000000000000ULL : 0x000000000000FF00ULL;
			if (!(board.pieces(chess::PieceType::PAWN, board.sideToMove()).getBits() & rank))
				return;
			// CAPTURE excludes non-capturing promotions, including underpromotions.
			chess::Movelist pawns;
			chess::movegen::legalmoves<chess::movegen::MoveGenType::QUIET>(pawns, board, chess::PieceGenType::PAWN);
			for (const auto move : pawns)
				if (move.typeOf() == chess::Move::PROMOTION)
					moves.add(move);
		}

		struct Iteration {
			chess::Move move{chess::Move::NO_MOVE};
			int score = -kInfinity;
			std::vector<RootMove> root;
		};

		class Context {
		public:
			Context(const Model &model, const SearchOptions &options, SearchState &state, SearchCancel cancel)
			    : net(model), opts(options), state(state), cancelled(std::move(cancel)), started(Clock::now()) {}

			void advance() noexcept { state.table.advance(); }
			std::uint64_t elapsed_ms() const noexcept { return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - started).count()); }

			void checkStop(bool force = false) const {
				if (opts.node_limit > 0 && nodes >= opts.node_limit) {
					throw Interrupted{};
				}
				if (opts.movetime_ms > 0 && (force || (nodes & 255U) == 0U) &&
				    std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - started).count() >= opts.movetime_ms) {
					throw Interrupted{};
				}
				if (cancelled && (force || (nodes & 255U) == 0U) && cancelled()) {
					throw Interrupted{};
				}
			}

			MovePicker ordered(const chess::Board &board, const chess::Movelist &moves, int ply, chess::Move preferred = chess::Move(chess::Move::NO_MOVE)) {
				MovePicker output;
				const auto side = static_cast<std::size_t>(board.sideToMove());
				for (const auto move : moves) {
					if (move == preferred) {
						output.add(move, std::numeric_limits<int>::max());
						continue;
					}
					const bool tactical = board.isCapture(move) || move.typeOf() == chess::Move::PROMOTION;
					int score = 0, gain = 0;
					if (tactical) {
						gain = see(board, move);
						score = (gain >= 0 ? 2000000 : -2000000) + 32 * gain + tacticalOrder(board, move);
					} else {
						score += state.history[side][static_cast<std::size_t>(historyIndex(move))];
						if (ply >= 0 && ply < static_cast<int>(state.killers.size())) {
							if (move == state.killers[static_cast<std::size_t>(ply)][0]) {
								score += 900000;
							} else if (move == state.killers[static_cast<std::size_t>(ply)][1]) {
								score += 800000;
							}
						}
					}
					output.add(move, score, gain);
				}
				return output;
			}

			int evaluate(const chess::Board &board) {
				const auto key = board.hash();
				auto &entry = evals[key & (evals.size() - 1)];
				if (entry.valid && entry.key == key)
					return entry.score;
				++evaluated_nodes;
				const int score = std::clamp(toCp(net.score(board)), -kMaximumStaticScore, kMaximumStaticScore);
				entry = {key, score, true};
				return score;
			}

			int quiescence(chess::Board &board, int ply, int remaining, int alpha, int beta) {
				visit(ply);
				if (const auto terminal = terminalScore(board, ply, !nullSearch)) {
					return *terminal;
				}
				if (mateBounds(ply, alpha, beta))
					return alpha;
				const bool inCheck = board.inCheck();
				const bool pawnEnding = !board.hasNonPawnMaterial(chess::Color::WHITE) && !board.hasNonPawnMaterial(chess::Color::BLACK);
				const bool probeMoves = !inCheck && remaining == opts.quiescence_depth && pawnEnding;
				int standPat = -kInfinity;
				int best = -kInfinity;
				if (!inCheck) {
					standPat = evaluate(board);
					best = probeMoves ? -kInfinity : standPat;
					if ((!probeMoves && standPat >= beta) || remaining <= 0) {
						return chess::movegen::anylegalmoves(board) ? standPat : 0;
					}
					if (!probeMoves) {
						alpha = std::max(alpha, standPat);
					}
				} else if (remaining <= -4) {
					return chess::movegen::anylegalmoves(board) ? evaluate(board) : -kMateScore + ply;
				}
				chess::Movelist moves;
				if (inCheck || probeMoves)
					chess::movegen::legalmoves(moves, board);
				else
					tacticalMoves(moves, board);
				if (moves.empty()) {
					if (inCheck || probeMoves)
						return emptyScore(board, ply);
					return chess::movegen::anylegalmoves(board) ? standPat : 0;
				}
				auto candidates = ordered(board, moves, ply);
				for (auto move = candidates.next(); move.move() != chess::Move::NO_MOVE; move = candidates.next()) {
					if (!inCheck && !probeMoves && move.typeOf() != chess::Move::PROMOTION && board.givesCheck(move) == chess::CheckType::NO_CHECK) {
						const int gain = pieceValue(board.getCapturing<chess::PieceType>(move));
						if (standPat + gain + 120 < alpha || candidates.gain() < 0) {
							continue;
						}
					}
					board.makeMove(move);
					const int score = -quiescence(board, ply + 1, remaining - 1, -beta, -alpha);
					board.unmakeMove(move);
					if (score > best) {
						best = score;
					}
					if (score >= beta) {
						return score;
					}
					alpha = std::max(alpha, score);
				}
				return best;
			}

			int pvs(chess::Board &board, int depth, int ply, int alpha, int beta) {
				if (depth <= 0)
					return quiescence(board, ply, opts.quiescence_depth, alpha, beta);
				visit(ply);
				if (const auto terminal = terminalScore(board, ply, !nullSearch)) {
					return *terminal;
				}
				if (mateBounds(ply, alpha, beta))
					return alpha;

				const bool pv = beta - alpha > 1;
				const auto key = tableKey(board);
				chess::Move preferred(chess::Move::NO_MOVE);
				// Artificial null lines neither read nor publish ordinary transposition bounds.
				if (const auto *entry = nullSearch ? nullptr : state.table.probe(key)) {
					preferred = entry->bestMove();
					if (entry->depth >= depth) {
						const int cached = scoreFromTable(entry->score, ply);
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
				const int originalAlpha = alpha;
				const int originalBeta = beta;

				const bool inCheck = board.inCheck();
				const bool sparse = board.occ().count() <= kTransitionPieceLimit;
				const bool pawnEnding = !board.hasNonPawnMaterial(chess::Color::WHITE) && !board.hasNonPawnMaterial(chess::Color::BLACK);
				const int staticScore = inCheck ? -kInfinity : evaluate(board);
				const bool improving = ply >= 2 && staticScores[ply - 2] != -kInfinity && staticScore > staticScores[ply - 2];
				staticScores[ply] = staticScore;
				// Forward pruning is selective: never use it on the PV, in check, near mate bounds or in sparse endings.
				const bool prune = !pv && !inCheck && !pawnEnding && board.halfMoveClock() < 90 && std::abs(beta) < kMateThreshold;
				if (prune && ply >= nullBan) {
					const int margin = (improving ? 70 : 100) * depth;
					if (depth <= 6 && staticScore - margin >= beta) {
						return chess::movegen::anylegalmoves(board) ? staticScore - margin : 0;
					}
					if (!nullSearch && !sparse && depth >= 3 && staticScore >= beta && board.hasNonPawnMaterial(board.sideToMove())) {
						const int reduction = std::min(depth, 3 + depth / 4 + std::min(3, (staticScore - beta) / 200));
						nullSearch = true;
						board.makeNullMove();
						const int score = -pvs(board, depth - reduction, ply + 1, -beta, -beta + 1);
						board.unmakeNullMove();
						nullSearch = false;
						if (score >= beta && score < kMateThreshold) {
							if (depth < 10)
								return chess::movegen::anylegalmoves(board) ? score : 0;
							// Verify deep null cutoffs using real moves with null pruning disabled over this subtree.
							const int saved = nullBan;
							nullBan = ply + depth;
							const int verified = pvs(board, depth - reduction, ply, beta - 1, beta);
							nullBan = saved;
							staticScores[ply] = staticScore;
							if (verified >= beta)
								return verified;
						}
					}
				}
				chess::Movelist moves;
				chess::movegen::legalmoves(moves, board);
				if (moves.empty())
					return emptyScore(board, ply);
				auto candidates = ordered(board, moves, ply, preferred);
				int best = -kInfinity;
				chess::Move bestMove(chess::Move::NO_MOVE);
				std::size_t quietIndex = 0;
				std::array<chess::Move, kMaximumLegalMoves> failedQuiet{};
				std::size_t failedCount = 0;
				std::size_t index = 0;
				for (auto move = candidates.next(); move.move() != chess::Move::NO_MOVE; move = candidates.next(), ++index) {
					const bool quiet = !board.isCapture(move) && move.typeOf() != chess::Move::PROMOTION;
					const bool checking = board.givesCheck(move) != chess::CheckType::NO_CHECK;
					const auto movingSide = static_cast<std::size_t>(board.sideToMove());
					const std::size_t currentQuiet = quietIndex;
					quietIndex += quiet ? 1U : 0U;
					const int history = state.history[movingSide][static_cast<std::size_t>(historyIndex(move))];
					const bool killer = ply < static_cast<int>(state.killers.size()) && (move == state.killers[ply][0] || move == state.killers[ply][1]);
					const bool advancedPawn = board.at(move.from()).type() == chess::PieceType::PAWN && (movingSide == 0 ? move.to().index() / 8 >= 5 : move.to().index() / 8 <= 2);
					const bool lateQuiet = index > 0 && quiet && !checking && !killer && !advancedPawn && move != preferred && move.typeOf() != chess::Move::CASTLING;
					if (prune && lateQuiet && best > -kMateThreshold) {
						if (depth <= 4 && currentQuiet >= static_cast<std::size_t>(3 + depth * depth + (improving ? depth * depth : 0)))
							continue;
						if (depth <= 3 && staticScore + 100 + 100 * depth <= alpha && history <= 0)
							continue;
					}
					board.makeMove(move);
					int score;
					if (index == 0) {
						score = -pvs(board, depth - 1, ply + 1, -beta, -alpha);
					} else {
						const bool reduce = depth >= 3 && currentQuiet >= 2 && lateQuiet && !inCheck && !pawnEnding;
						if (reduce) {
							score = -pvs(board, depth - 1 - reductionFor(depth, index, pv, improving, history), ply + 1, -alpha - 1, -alpha);
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
						bestMove = move;
					}
					alpha = std::max(alpha, score);
					if (alpha >= beta) {
						if (quiet) {
							const int bonus = depth * depth;
							updateHistory(state.history[movingSide][static_cast<std::size_t>(historyIndex(move))], bonus);
							for (std::size_t failed = 0; failed < failedCount; ++failed) {
								updateHistory(state.history[movingSide][static_cast<std::size_t>(historyIndex(failedQuiet[failed]))], -std::max(1, bonus / 2));
							}
							if (ply < static_cast<int>(state.killers.size()) && move != state.killers[static_cast<std::size_t>(ply)][0]) {
								state.killers[static_cast<std::size_t>(ply)][1] = state.killers[static_cast<std::size_t>(ply)][0];
								state.killers[static_cast<std::size_t>(ply)][0] = move;
							}
						}
						break;
					}
					if (quiet && failedCount < failedQuiet.size()) {
						failedQuiet[failedCount++] = move;
					}
				}
				Bound bound = Bound::exact;
				if (best <= originalAlpha) {
					bound = Bound::upper;
				} else if (best >= originalBeta) {
					bound = Bound::lower;
				}
				if (!nullSearch)
					state.table.store(key, depth, scoreToTable(best, ply), bound, bestMove);
				return best;
			}

			Iteration rootLines(chess::Board &board, int depth, const chess::Movelist &moves, chess::Move preferred, std::size_t lineCount) {
				Iteration result;
				std::vector<chess::Move> selected;
				selected.reserve(lineCount);
				result.root.reserve(lineCount);
				for (std::size_t line = 0; line < lineCount; ++line) {
					RootMove best;
					best.score_cp = -kInfinity;
					auto candidates = ordered(board, moves, 0, preferred);
					std::size_t index = 0;
					for (auto move = candidates.next(); move.move() != chess::Move::NO_MOVE; move = candidates.next()) {
						if (std::find(selected.begin(), selected.end(), move) != selected.end())
							continue;
						checkStop(true);
						const auto before = nodes;
						board.makeMove(move);
						int score;
						if (index++ == 0) {
							score = -pvs(board, depth - 1, 1, -kInfinity, kInfinity);
						} else {
							score = -pvs(board, depth - 1, 1, -best.score_cp - 1, -best.score_cp);
							if (score > best.score_cp)
								score = -pvs(board, depth - 1, 1, -kInfinity, -best.score_cp);
						}
						board.unmakeMove(move);
						if (score > best.score_cp)
							best = {move, score, nodes - before};
					}
					selected.push_back(best.move);
					result.root.push_back(best);
					preferred = chess::Move(chess::Move::NO_MOVE);
				}
				std::sort(result.root.begin(), result.root.end(), [](const RootMove &left, const RootMove &right) {
					return left.score_cp != right.score_cp ? left.score_cp > right.score_cp : left.move.move() < right.move.move();
				});
				result.move = result.root.front().move;
				result.score = result.root.front().score_cp;
				state.table.store(tableKey(board), depth, scoreToTable(result.score, 0), Bound::exact, result.move);
				return result;
			}

			Iteration root(chess::Board &board, int depth, int alpha = -kInfinity, int beta = kInfinity) {
				Iteration result;
				staticScores.fill(-kInfinity);
				const bool multipleLines = opts.multipv > 1;
				if (multipleLines) {
					alpha = -kInfinity;
					beta = kInfinity;
				}
				const int originalAlpha = alpha, originalBeta = beta;
				chess::Movelist moves;
				chess::movegen::legalmoves(moves, board);
				if (moves.empty()) {
					result.score = emptyScore(board, 0);
					return result;
				}
				const auto lineCount = std::min<std::size_t>(static_cast<std::size_t>(opts.multipv), moves.size());
				chess::Move preferred(chess::Move::NO_MOVE);
				if (const auto *entry = state.table.probe(tableKey(board))) {
					preferred = entry->bestMove();
				}
				if (multipleLines && lineCount < moves.size() && lineCount <= kSequentialMultiPVLimit)
					return rootLines(board, depth, moves, preferred, lineCount);
				auto candidates = ordered(board, moves, 0, preferred);
				result.root.reserve(moves.size());
				std::size_t index = 0;
				for (auto move = candidates.next(); move.move() != chess::Move::NO_MOVE; move = candidates.next(), ++index) {
					checkStop(true);
					const auto before = nodes;
					board.makeMove(move);
					int score;
					if (multipleLines || index == 0) {
						score = -pvs(board, depth - 1, 1, -beta, -alpha);
					} else {
						score = -pvs(board, depth - 1, 1, -alpha - 1, -alpha);
						if (score > alpha && score < beta) {
							score = -pvs(board, depth - 1, 1, -beta, -alpha);
						}
					}
					board.unmakeMove(move);
					result.root.push_back({move, score, nodes - before});
					if (score > result.score) {
						result.score = score;
						result.move = move;
					}
					if (!multipleLines) {
						alpha = std::max(alpha, score);
						if (alpha >= beta)
							break;
					}
				}
				std::sort(result.root.begin(), result.root.end(), [&](const RootMove &left, const RootMove &right) {
					if (left.score_cp != right.score_cp)
						return left.score_cp > right.score_cp;
					// A tied upper bound from a null-window search must not replace the actual PV move.
					if ((left.move == result.move) != (right.move == result.move))
						return left.move == result.move;
					return left.move.move() < right.move.move();
				});
				const auto bound = result.score <= originalAlpha ? Bound::upper : result.score >= originalBeta ? Bound::lower : Bound::exact;
				state.table.store(tableKey(board), depth, scoreToTable(result.score, 0), bound, result.move);
				return result;
			}

			Iteration deepen(chess::Board &board, int depth, int previous) {
				if (depth < 4 || opts.multipv > 1 || std::abs(previous) >= kMateThreshold)
					return root(board, depth);
				int margin = 32;
				int alpha = std::max(-kInfinity, previous - margin), beta = std::min(kInfinity, previous + margin);
				for (;;) {
					auto result = root(board, depth, alpha, beta);
					// Widen failed bounds without reducing depth; only a completed window is published.
					if (result.score <= alpha && alpha > -kInfinity)
						alpha = std::max(-kInfinity, result.score - margin);
					else if (result.score >= beta && beta < kInfinity)
						beta = std::min(kInfinity, result.score + margin);
					else
						return result;
					margin = std::min(kInfinity, 2 * margin);
				}
			}

			std::uint64_t nodes = 0;
			std::uint64_t evaluated_nodes = 0;
			int selective_depth = 0;

		private:
			struct EvalEntry {
				std::uint64_t key = 0;
				int score = 0;
				bool valid = false;
			};
			void visit(int ply) {
				++nodes;
				selective_depth = std::max(selective_depth, ply);
				checkStop();
			}

			const Model &net;
			// Only static evaluation is cached: repetition and fifty-move adjudication stay in search.
			std::vector<EvalEntry> evals{65536};
			const SearchOptions &opts;
			SearchState &state;
			SearchCancel cancelled;
			Clock::time_point started;
			std::array<int, 128> staticScores{};
			bool nullSearch = false;
			int nullBan = 0;
		};

	} // namespace

	Searcher::Searcher(const Model &model, SearchOptions options) : net(&model), opts(options) {
		if (options.depth <= 0 || options.depth > 64 || options.quiescence_depth < 0 || options.quiescence_depth > 32 || options.hash_mb == 0 || options.hash_mb > 4096 ||
		    options.multipv <= 0 || options.multipv > 256) {
			throw std::invalid_argument("Eleginus search options are outside the supported range");
		}
		state = std::make_unique<SearchState>(opts.hash_mb);
	}

	Searcher::~Searcher() = default;
	Searcher::Searcher(Searcher &&) noexcept = default;
	Searcher &Searcher::operator=(Searcher &&) noexcept = default;

	SearchResult Searcher::search(const chess::Board &board, const SearchProgress &progress, const SearchCancel &cancel) {
		SearchResult result;
		if (const auto terminal = terminalScore(board, 0)) {
			result.score_cp = *terminal;
			return result;
		}
		if (legalmoves(board).empty()) {
			result.score_cp = emptyScore(board, 0);
			return result;
		}
		chess::Board root = board;
		Context context(*net, opts, *state, cancel);
		for (int depth = 1; depth <= opts.depth; ++depth) {
			context.advance();
			try {
				const auto iteration = context.deepen(root, depth, result.score_cp);
				result.move = iteration.move;
				result.score_cp = iteration.score;
				result.depth = depth;
				result.root = iteration.root;
				result.nodes = context.nodes;
				result.evaluated_nodes = context.evaluated_nodes;
				result.selective_depth = context.selective_depth;
				result.elapsed_ms = context.elapsed_ms();
				if (opts.collect_leaf) {
					chess::Board leaf = root;
					for (int ply = 0; ply < depth; ++ply) {
						const auto *entry = state->table.probe(tableKey(leaf));
						if (!entry || entry->bestMove().move() == chess::Move::NO_MOVE)
							break;
						chess::Movelist moves;
						chess::movegen::legalmoves(moves, leaf);
						if (std::find(moves.begin(), moves.end(), entry->bestMove()) == moves.end())
							break;
						leaf.makeMove(entry->bestMove());
					}
					net->extract(leaf, result.leaf);
				}
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
