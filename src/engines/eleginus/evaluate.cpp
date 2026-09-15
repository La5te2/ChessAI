#include "eleginus/evaluate.hpp"
#include "eleginus/formula.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace eleginus {
	namespace evaluation {
		// ------------------------------ Fixed parameters --------------------------------
		// Compile-time formula and score-adjustment coefficients.

		struct FormulaWeights {
			float base = 0.0F;
			std::array<float, 5> material{};
		};

		struct AdjustmentWeights {
			float pressureCenter = 0.0F;
			float pressureWidth = 1.0F;
			std::array<float, 7> winnability{};
			std::array<float, 5> scaling{};
		};

		namespace fixed {
			#include "weights.inl"
		}

		// -------------------------- King-pressure response -----------------------------
		// Precomputed sigmoid response used by king-pressure formulas.

		// Fixed-point sigmoid lookup for king-attack counts from 0 through 64.
		class SigmoidCurve {
		public:
			static constexpr int scale = 4096;

			// Build all lookup entries from the sigmoid center and width.
			SigmoidCurve(float center, float width) {
				for (std::size_t x = 0; x < values.size(); ++x) {
					const float y = 1.0F / (1.0F + std::exp(-(static_cast<float>(x) - center) / width));
					values[x] = static_cast<std::int32_t>(std::lround(scale * y));
				}
			}

			// Clamp an attack count to the table range and return its fixed-point response.
			std::int32_t operator()(std::int64_t value) const noexcept {
				const auto last = static_cast<std::int64_t>(values.size() - 1);
				return values[static_cast<std::size_t>(std::clamp(value, std::int64_t{0}, last))];
			}

		private:
			std::array<std::int32_t, 65> values{};
		};

		// -------------------------- Board input preparation ----------------------------
		// Encode board atoms, iterate occupied squares and derive material coordinates.

		// Encode the twelve piece bitboards, side to move and castling rights.
		std::array<Word, atomCount> inputs(const chess::Board &board) {
			std::array<Word, atomCount> in{};
			const std::array<Word, 2> byColor{board.us(chess::Color::WHITE).getBits(), board.us(chess::Color::BLACK).getBits()};
			for (int type = 0; type < 6; ++type) {
				const auto pieces = board.pieces(chess::PieceType(static_cast<chess::PieceType::underlying>(type))).getBits();
				in[pieceAtomIndex(0, type)] = pieces & byColor[0];
				in[pieceAtomIndex(1, type)] = pieces & byColor[1];
			}
			for (int color = 0; color < 2; ++color) {
				for (int wing = 0; wing < 2; ++wing) {
					const auto flank = wing == 0 ? chess::Board::CastlingRights::Side::KING_SIDE : chess::Board::CastlingRights::Side::QUEEN_SIDE;
					if (board.castlingRights().has(static_cast<chess::Color>(color), flank)) in[atomIndex(Atom::CR)] |= 1ULL << (2 * color + wing);
				}
			}
			in[atomIndex(Atom::STM)] = static_cast<Word>(board.sideToMove());
			return in;
		}

		// Iterate the occupied square indices of a bitboard.
		struct Squares {
			Word mask;
			struct Iterator {
				Word mask;
				int operator*() const { return std::countr_zero(mask); }
				Iterator &operator++() {
					mask &= mask - 1;
					return *this;
				}
				bool operator!=(const Iterator &other) const { return mask != other.mask; }
			};
			Iterator begin() const { return {mask}; }
			Iterator end() const { return {0}; }
		};

		// Count one piece type across both colors from the encoded board inputs.
		unsigned pieceCount(const std::array<Word, atomCount> &in, std::size_t type) noexcept {
			return std::popcount(in[pieceAtomIndex(0, type)] | in[pieceAtomIndex(1, type)]);
		}

		// Normalize total piece counts relative to their initial counts.
		std::array<float, 5> materialCoordinates(const std::array<Word, atomCount> &in) noexcept {
			constexpr std::array<float, 5> initial{{16.0F, 4.0F, 4.0F, 4.0F, 2.0F}};
			std::array<float, 5> coordinate{};
			for (std::size_t type = 0; type < coordinate.size(); ++type) {
				coordinate[type] = static_cast<float>(pieceCount(in, type)) / initial[type] - 1.0F;
			}
			return coordinate;
		}

		// --------------------------- Formula result handling -----------------------------
		// Accumulate formula signals with their effective coefficients.

		// Accumulate emitted formula signals directly into a score.
		class ScoreAccumulator {
		public:
			explicit ScoreAccumulator(const std::array<float, formulaCount> &coefficients) : coefficients(coefficients) {}
			// Add one weighted formula signal.
			void add(std::uint32_t index, std::int32_t signal) {
				total = std::fma(coefficients[index], static_cast<float>(signal), total);
			}
			float value() const noexcept { return total; }
			int direction() const noexcept { return (total > 0.0F) - (total < 0.0F); }
			// Apply the winnability adjustment while preserving the score sign.
			void winnable(float value) noexcept {
				if (total > 0.0F) total = std::max(0.0F, total + value);
				else if (total < 0.0F) total = std::min(0.0F, total - value);
			}
			void scale(float value) noexcept { total *= value; }

		private:
			const std::array<float, formulaCount> &coefficients;
			float total = 0.0F;
		};

		// -------------------------- Formula evaluation runtime ---------------------------
		// Board facts, caches and operations used by formula definitions.

		// Prepare shared board facts and send formula results to the score accumulator.
		class Runtime : public FormulaAtoms<Runtime> {
		public:
			using FormulaAtoms<Runtime>::BB;
			using FormulaAtoms<Runtime>::NUM;
			using FormulaAtoms<Runtime>::OR;
			using FormulaAtoms<Runtime>::PCS;
			using FormulaAtoms<Runtime>::REL;

			struct Sum {
				std::int64_t total = 0;
				void add(InterSignal value) { total += number(value.bits); }
			};
			// Select or compute the pawn and attack data for this board.
			Runtime(const chess::Board &board, const std::array<Word, atomCount> &inputs, ScoreAccumulator &score, const AdjustmentWeights &adjustments,
				const SigmoidCurve &curve)
				: in(inputs), occupied(board.occ().getBits()), score(score), adjustments(adjustments), curve(curve) {
				// Reuse pawn structure data keyed by both pawn bitboards.
				thread_local std::array<Pawns, 1024> table{};
				const Word hash = in[atomIndex(Atom::WP)] * 0x9e3779b97f4a7c15ULL ^ std::rotl(in[atomIndex(Atom::BP)] * 0xbf58476d1ce4e5b9ULL, 29);
				pawnCache = &table[(hash ^ (hash >> 32)) & (table.size() - 1)];
				if (!pawnCache->valid || pawnCache->white != in[atomIndex(Atom::WP)] || pawnCache->black != in[atomIndex(Atom::BP)]) {
					pawnCache->white = in[atomIndex(Atom::WP)];
					pawnCache->black = in[atomIndex(Atom::BP)];
					for (unsigned color = 0; color < 2; ++color) {
						const Word pawns = in[6 * color];
						const Word east = pawns & 0x7F7F7F7F7F7F7F7FULL;
						const Word west = pawns & 0xFEFEFEFEFEFEFEFEULL;
						pawnCache->attacks[color] = color == 0 ? (east << 9) | (west << 7) : (east >> 7) | (west >> 9);
						Word north = pawns;
						Word south = pawns;
						for (unsigned distance : {8U, 16U, 32U}) {
							north |= north << distance;
							south |= south >> distance;
						}
						pawnCache->span[color] = color == 0 ? north : south;
						pawnCache->files[color] = north | south;
					}
					for (unsigned color = 0; color < 2; ++color) {
						const Word span = pawnCache->span[color ^ 1U];
						const Word east = (span & 0x7F7F7F7F7F7F7F7FULL) << 1;
						const Word west = (span & 0xFEFEFEFEFEFEFEFEULL) >> 1;
						pawnCache->passed[color] = in[6 * color] & ~(span | east | west);
						const Word pawns = in[6 * color];
						const Word adjacent = ((pawnCache->files[color] & 0x7F7F7F7F7F7F7F7FULL) << 1)
							| ((pawnCache->files[color] & 0xFEFEFEFEFEFEFEFEULL) >> 1);
						auto &shape = pawnCache->shape[color];
						shape = {};
						shape.doubled = std::popcount(pawns & (pawns << 8));
						shape.isolated = std::popcount(pawns & ~adjacent);
						const Word phalanx = (pawns & 0x7F7F7F7F7F7F7F7FULL) << 1;
						const Word connected = ((pawnCache->passed[color] & 0x7F7F7F7F7F7F7F7FULL) << 1)
							| ((pawnCache->passed[color] & 0xFEFEFEFEFEFEFEFEULL) >> 1);
						Word pieces = pawns;
						while (pieces) {
							const int square = std::countr_zero(pieces);
							pieces &= pieces - 1;
							const Word bit = 1ULL << square;
							const int rank = color == 0 ? square / 8 : 7 - square / 8;
							if (rank < 1 || rank > 6) continue;
							const auto bucket = static_cast<std::size_t>(rank - 1);
							shape.phalanx[bucket] += (phalanx & bit) != 0;
							if (rank >= 2) shape.defended[bucket] += (pawnCache->attacks[color] & bit) != 0;
							if ((pawnCache->passed[color] & bit) == 0) continue;
							++shape.passed[bucket];
							shape.supported[bucket] += (pawnCache->attacks[color] & bit) != 0;
							shape.connected[bucket] += (connected & bit) != 0;
						}
					}
					for (auto &king : pawnCache->kings) {
						king.valid = false;
					}
					pawnCache->valid = true;
				}
				// Build pawn and king attack maps from their piece bitboards.
				for (unsigned color = 0; color < 2; ++color) {
					pawnKingAttacks[color][0] = pawnCache->attacks[color];
					Word kings = in[6 * color + 5];
					while (kings) {
						const int square = std::countr_zero(kings);
						kings &= kings - 1;
						pawnKingAttacks[color][1] |= chess::attacks::king(chess::Square(square)).getBits();
					}
				}
				thread_local Attacks attacks;
				thread_local Rays rays;
				attackCache = &attacks;
				rayCache = &rays;
				const Word changed = attackCache->occupied ^ occupied;
				for (unsigned color = 0; color < 2; ++color) {
					Word all = 0, twice = 0;
					for (unsigned type = 0; type < 6; ++type) {
						const auto i = 6 * color + type;
						auto &map = attackCache->maps[color][type];
						auto &overlap = attackCache->twice[color][type];
						const bool slider = type >= 2 && type <= 4;
						// Recompute a slider map when a piece or a square on its previous ray changes.
						if (!attackCache->valid || attackCache->pieces[i] != in[i] || (slider && (changed & map))) {
							if (type == 0) {
								const Word east = in[i] & 0x7F7F7F7F7F7F7F7FULL;
								const Word west = in[i] & 0xFEFEFEFEFEFEFEFEULL;
								const Word left = color == 0 ? east << 9 : east >> 7;
								const Word right = color == 0 ? west << 7 : west >> 9;
								map = pawnKingAttacks[color][0];
								overlap = left & right;
							} else if (type == 5) {
								map = pawnKingAttacks[color][1];
								overlap = 0;
							} else {
								map = overlap = 0;
								auto pieces = in[i];
								while (pieces) {
									const int square = std::countr_zero(pieces);
									pieces &= pieces - 1;
									const auto a = attackFrom(NUM(color), type, NUM(square)).bits;
									attackCache->from[color][type][square] = a;
									overlap |= map & a;
									map |= a;
								}
							}
						}
						twice |= overlap | (all & map);
						all |= map;
						attackCache->pieces[i] = in[i];
					}
					attackCache->maps[color][6] = all;
					attackCache->maps[color][7] = twice;
				}
				attackCache->occupied = occupied;
				attackCache->valid = true;
			}

			// ------------------------------ Shared board facts -------------------------------
			// Accessors for cached attacks, pawn structure, mobility and king-area measurements.
			InterSignal occ() const { return BB(occupied); }
			unsigned roleIndex(InterSignal role) const { return color(role); }
			Squares squares(InterSignal role, int type) const { return {REL(role, PCS(role, type).bits).bits}; }
			Squares locations(InterSignal set) const { return {set.bits}; }

			// Fill each file forward from every set square.
			InterSignal fill(InterSignal x, InterSignal role) const {
				for (unsigned d : {8U, 16U, 32U}) {
					x.bits |= color(role) == 0 ? x.bits << d : x.bits >> d;
				}
				return x;
			}

			// Generate attacks from one square and reuse unchanged bishop and rook rays.
			InterSignal attackFrom(InterSignal role, int type, InterSignal square) {
				const int s = static_cast<int>(number(square.bits));
				if (type == 0) return BB(chess::attacks::pawn(static_cast<chess::Color>(color(role)), chess::Square(s)).getBits());
				if (type == 1) return BB(chess::attacks::knight(chess::Square(s)).getBits());
				if (type == 5) return BB(chess::attacks::king(chess::Square(s)).getBits());
				if (type == 4) return OR(attackFrom(role, 2, square), attackFrom(role, 3, square));
				auto &ray = rayCache->rays[type - 2][s];
				if (ray.map == 0 || ((ray.occupied ^ occupied) & ray.map)) {
					const auto source = chess::Square(s);
					const auto blockers = chess::Bitboard(occupied);
					ray.map = type == 2 ? chess::attacks::bishop(source, blockers).getBits() : chess::attacks::rook(source, blockers).getBits();
				}
				ray.occupied = occupied;
				return BB(ray.map);
			}
			// Return the cached attacks of one knight, bishop, rook or queen.
			InterSignal pieceAttack(InterSignal role, int type, int square) const {
				return BB(attackCache->from[color(role)][type][static_cast<std::size_t>(square)]);
			}

			// Return one piece-type attack union or the complete attack union of a side.
			InterSignal attacks(InterSignal role, int type = 6) const {
				if (type == 0 || type == 5) return BB(pawnKingAttacks[color(role)][type == 0 ? 0 : 1]);
				return BB(attackCache->maps[color(role)][type]);
			}
			// Return cached aggregate attack and pawn-structure data.
			InterSignal doubleAttacks(InterSignal role) const { return BB(attackCache->maps[color(role)][7]); }
			InterSignal pawnFiles(InterSignal role) const { return BB(pawnCache->files[color(role)]); }
			InterSignal passedPawns(InterSignal role) const { return BB(pawnCache->passed[color(role)]); }
			const auto &pawnStructure(InterSignal role) const { return pawnCache->shape[color(role)]; }
			// Count the pieces of one side that attack a square.
			InterSignal attackCount(InterSignal role, InterSignal square) {
				const auto reverse = NUM(color(role) ^ 1U);
				int total = std::popcount(PCS(role, 0).bits & attackFrom(reverse, 0, square).bits);
				for (int type = 1; type < 6; ++type) {
					total += std::popcount(PCS(role, type).bits & attackFrom(role, type, square).bits);
				}
				return NUM(total);
			}

			// Compute and cache shelter, pawn-storm and open-file values around one king.
			void prepareKingPawns(InterSignal role) {
				auto &entry = pawnCache->kings[color(role)];
				const Word king = PCS(role, 5).bits;
				if (!entry.valid || entry.king != king) {
					entry.king = king;
					entry.values.fill(0);
					entry.shelter.fill(0);
					entry.blocked.fill(0);
					entry.storm.fill(0);
					const unsigned side = color(role);
					const Word f = REL(role, in[6 * side] & ~attackCache->maps[side ^ 1][0]).bits;
					const Word e = REL(role, in[6 * (side ^ 1)]).bits;
					const Word pawns = in[atomIndex(Atom::WP)] | in[atomIndex(Atom::BP)];
					for (int square : squares(role, 5)) {
						const int rank = square / 8, file = square % 8;
						const Word front = rank == 7 ? 0 : ~((1ULL << (8 * (rank + 1))) - 1);
						// Accumulate nearest-pawn distances and open-file count for each neighboring file.
						for (int x = std::max(0, file - 1); x <= std::min(7, file + 1); ++x) {
							const Word mask = 0x0101010101010101ULL << x;
							entry.values[6] += (pawns & mask) == 0;
							const std::array<Word, 2> nearest{f & mask & front, e & mask & front};
							for (unsigned i = 0; i < nearest.size(); ++i) {
								if (!nearest[i]) continue;
								const int distance = std::countr_zero(nearest[i]) / 8 - rank;
								if (distance <= 3) ++entry.values[2 * (distance - 1) + i];
							}
						}

						const int center = std::clamp(file, 1, 6);
						const Word ahead = ~((1ULL << (8 * rank)) - 1);
						for (int x = center - 1; x <= center + 1; ++x) {
							const Word mask = 0x0101010101010101ULL << x;
							const Word own = f & mask & ahead;
							const Word enemy = e & mask & ahead;
							const int ownRank = own ? std::min(6, std::countr_zero(own) / 8) : 0;
							const int enemyRank = enemy ? std::min(6, std::countr_zero(enemy) / 8) : 0;
							const int edge = std::min(x, 7 - x);
							++entry.shelter[7 * edge + ownRank];
							if (ownRank != 0 && ownRank == enemyRank - 1) ++entry.blocked[enemyRank];
							else ++entry.storm[7 * edge + enemyRank];
						}
					}
					entry.valid = true;
				}
			}

			// Return a nearest-pawn distance count or the king-neighborhood open-file count.
			InterSignal kingPawn(InterSignal role, unsigned slot) {
				const auto &entry = pawnCache->kings[color(role)];
				return NUM(entry.values[slot]);
			}
			// Return one king-shelter bucket.
			InterSignal shelter(InterSignal role, unsigned slot) {
				return NUM(pawnCache->kings[color(role)].shelter[slot]);
			}
			// Return one blocked pawn-storm bucket.
			InterSignal blockedStorm(InterSignal role, unsigned slot) {
				return NUM(pawnCache->kings[color(role)].blocked[slot]);
			}
			// Return one unblocked pawn-storm bucket.
			InterSignal storm(InterSignal role, unsigned slot) {
				return NUM(pawnCache->kings[color(role)].storm[slot]);
			}

			// Count pieces by primary and secondary mobility and cache the resulting histograms.
			const auto &mobility(InterSignal role, int type, InterSignal area, InterSignal guard) {
				thread_local std::array<std::array<Mobility, 4>, 2> table{};
				auto &entry = table[color(role)][type - 1];
				const Word pieces = PCS(role, type).bits;
				// Recompute when pieces, target areas or occupied squares on previous attack maps change.
				const Word changed = (entry.area ^ area.bits) | (entry.guard ^ guard.bits) | (type >= 2 ? entry.occupied ^ occupied : 0);
				if (!entry.valid || entry.pieces != pieces || (changed & entry.reach)) {
					entry.counts.fill(0);
					entry.secondary.fill(0);
					for (int square : locations(BB(pieces))) {
						const Word map = pieceAttack(role, type, square).bits;
						++entry.counts[std::popcount(map & area.bits)];
						++entry.secondary[std::popcount(map & guard.bits)];
					}
					entry.pieces = pieces;
					entry.reach = attacks(role, type).bits;
					entry.valid = true;
				}
				entry.area = area.bits;
				entry.guard = guard.bits;
				entry.occupied = occupied;
				return entry;
			}

			// ---------------------- Formula results and score adjustments --------------------
			// Emit formula roots and apply king-pressure, winnability and scaling operations.

			// Convert a widened sum to an intermediate signal.
			InterSignal sum(const Sum &terms) const { return NUM(terms.total); }

			// Map a king-attack count through the fixed sigmoid response.
			InterSignal SIG(InterSignal value) const { return NUM(curve(number(value.bits))); }
			// Apply the winnability coefficients to the favored side's score magnitude.
			void WIN(InterSignal pawns, InterSignal symmetric, InterSignal asymmetric, InterSignal pawnEnding, InterSignal strongPawns,
				InterSignal oppositePassers) {
				const auto &params = adjustments.winnability;
				const float value = params[6] + params[0] * static_cast<float>(number(pawns.bits)) +
					params[1] * static_cast<float>(number(symmetric.bits)) + params[2] * static_cast<float>(number(asymmetric.bits)) +
					params[3] * static_cast<float>(number(pawnEnding.bits)) + params[4] * static_cast<float>(number(strongPawns.bits)) +
					params[5] * static_cast<float>(number(oppositePassers.bits));
				score.winnable(value);
			}
			// Scale the score for pawnless and opposite-bishop endings.
			void SCALE(InterSignal thinPawnless, InterSignal pureOpposite, InterSignal mixedOpposite, InterSignal strongPawns,
				InterSignal strongPassers) {
				const auto &params = adjustments.scaling;
				float factor = 1.0F;
				if (number(thinPawnless.bits) != 0) factor = std::min(factor, params[0]);
				if (number(pureOpposite.bits) != 0) {
					factor = std::min(factor, params[1] + params[4] * static_cast<float>(number(strongPawns.bits)) +
						params[3] * static_cast<float>(number(strongPassers.bits)));
				} else if (number(mixedOpposite.bits) != 0) {
					factor = std::min(factor, params[2] + params[3] * static_cast<float>(number(strongPassers.bits)));
				}
				score.scale(std::clamp(factor, 0.0F, 1.0F));
			}
			// Emit one formula signal and advance its parameter index.
			void root(InterSignal signal) {
				const auto value = number(signal.bits);
				if (value != 0) score.add(index, static_cast<std::int32_t>(value));
				++index;
			}
			// Verify that formula execution consumed every formula coefficient.
			void complete() const {
				if (index != formulaCount) throw std::logic_error("formula and weight counts differ");
			}
			// Return the sign of the accumulated score.
			InterSignal direction() const { return NUM(score.direction()); }
			// Advance over zero-valued entries in a fixed formula table.
			void skip(unsigned n) { index += n; }

		private:
			// --------------------------- Runtime caches and state ----------------------------
			// Cached pawn, attack, ray and mobility data shared across board evaluations.
			struct KingPawn {
				Word king = 0;
				std::array<int, 7> values{}; // Friendly/enemy pawns at distances 1-3, then open files.
				std::array<int, 28> shelter{};
				std::array<int, 7> blocked{};
				std::array<int, 28> storm{};
				bool valid = false;
			};
			struct PawnStructure {
				std::array<int, 6> passed{};
				std::array<int, 6> supported{};
				std::array<int, 6> connected{};
				std::array<int, 6> phalanx{};
				std::array<int, 6> defended{};
				int doubled = 0;
				int isolated = 0;
			};
			struct Pawns {
				Word white = 0, black = 0;
				std::array<Word, 2> attacks{};
				std::array<Word, 2> span{};
				std::array<Word, 2> files{};
				std::array<Word, 2> passed{};
				std::array<PawnStructure, 2> shape{};
				std::array<KingPawn, 2> kings{};
				bool valid = false;
			};
			struct Mobility {
				Word pieces = 0, occupied = 0, area = 0, guard = 0, reach = 0;
				std::array<int, 28> counts{};
				std::array<int, 28> secondary{};
				bool valid = false;
			};
			struct Ray {
				Word occupied = 0, map = 0;
			};
			struct Attacks {
				std::array<Word, 12> pieces{};
				Word occupied = 0;
				std::array<std::array<Word, 8>, 2> maps{};
				std::array<std::array<Word, 6>, 2> twice{};
				std::array<std::array<std::array<Word, 64>, 6>, 2> from{};
				bool valid = false;
			};
			struct Rays {
				std::array<std::array<Ray, 64>, 2> rays{};
			};

			friend class FormulaAtoms<Runtime>;
			// Return one encoded board atom.
			Word input(Atom atom) const noexcept { return in[atomIndex(atom)]; }
			// Convert a role signal to its color index.
			static unsigned color(InterSignal role) { return static_cast<unsigned>(number(role.bits)); }
			Pawns *pawnCache;
			Attacks *attackCache;
			Rays *rayCache;
			std::array<Word, atomCount> in;
			std::array<std::array<Word, 2>, 2> pawnKingAttacks{};
			Word occupied;
			ScoreAccumulator &score;
			const AdjustmentWeights &adjustments;
			const SigmoidCurve &curve;
			unsigned index = 0;
		};

		// ----------------------------- Formula execution ---------------------------------
		// Fixed formula groups and intermediate values shared within one board evaluation.

		// Evaluate the complete formula set.
		class Formulas {
			using Sum = Runtime::Sum;
			using SideSignals = std::array<std::optional<InterSignal>, 2>;

		public:
			// Initialize side constants and occupied squares.
			explicit Formulas(Runtime runtime) : b(std::move(runtime)) {
				z = b.NUM(0);
				o = b.NUM(1);
				us = b.NUM(0);
				them = b.NUM(1);
				occ = b.occ();
			}

			// Evaluate formula groups in parameter order and verify the final count.
			void execute() {
				tempo();
				material();
				pst();
				bishopPair();
				pawns();
				mobility();
				pieces();
				threats();
				kings();
				endgames();
				b.complete();
			}

		private:
			// ---------------------- Formula bodies and shared helpers ------------------------
			// Formula declarations and intermediate calculations reused by multiple groups.

			// Declare each formula group as an inline member.
			#define FORMULA(name) void name()
			#include "formula.inl"
			#undef FORMULA

			// Cache one side-specific intermediate signal for this board evaluation.
			template <class F> InterSignal shared(SideSignals &signals, InterSignal role, F &&make) {
				auto &value = signals[b.roleIndex(role)];
				if (!value) value = make();
				return *value;
			}

			// Return one file or rank mask.
			static constexpr std::uint64_t fileMask(int file) noexcept {
				return 0x0101010101010101ULL << file;
			}
			static constexpr std::uint64_t rankMask(int rank) noexcept {
				return 0xFFULL << (8 * rank);
			}

			// Return the Chebyshev-distance ring around a square.
			static std::uint64_t ringMask(int square, int distance) noexcept {
				static const auto masks = [] {
					std::array<std::array<Word, 8>, 64> table{};
					for (int s = 0; s < 64; ++s) {
						for (int t = 0; t < 64; ++t) {
							const int d = std::max(std::abs(t % 8 - s % 8), std::abs(t / 8 - s / 8));
							table[s][d] |= 1ULL << t;
						}
					}
					return table;
				}();
				return masks[square][distance];
			}

			// Return all squares diagonally aligned with a square.
			static std::uint64_t diagonalMask(int square) noexcept {
				static const auto masks = [] {
					std::array<Word, 64> table{};
					for (int s = 0; s < 64; ++s) {
						for (int t = 0; t < 64; ++t) {
							const int df = std::abs(t % 8 - s % 8), dr = std::abs(t / 8 - s / 8);
							if (df && df == dr) table[s] |= 1ULL << t;
						}
					}
					return table;
				}();
				return masks[square];
			}

			// Subtract the second role-relative measurement from the first.
			InterSignal diff(InterSignal own, InterSignal enemy) { return b.SUB(own, enemy); }

			// Return and cache all occupied squares of one side.
			InterSignal own(InterSignal role) {
				return shared(occupancyBySide, role, [&] {
					InterSignal result = b.PCS(role, 0);
					for (int type = 1; type < 6; ++type) {
						result = b.OR(result, b.PCS(role, type));
					}
					return result;
				});
			}

			// Expose cached pawn facts under formula-level names.
			InterSignal pawnAttacks(InterSignal role) {
				return b.attacks(role, 0);
			}

			InterSignal files(InterSignal role) {
				return b.pawnFiles(role);
			}

			InterSignal passedPawns(InterSignal role) { return b.passedPawns(role); }

			// Return squares controlled by enemy pawns or controlled twice only by the enemy.
			InterSignal strongSquares(InterSignal role, InterSignal opponent) {
				return shared(
					strongMaps, role, [&] { return b.OR(pawnAttacks(opponent), b.AND(b.doubleAttacks(opponent), b.NOT(b.doubleAttacks(role)))); });
			}

			// Emit one formula signal.
			void F(InterSignal signal) { b.root(signal); }

			// Return the mobility area after removing blocked pawns, undeveloped pawns and enemy pawn attacks.
			InterSignal mobilityArea(InterSignal role, InterSignal opponent) {
				return shared(mobilityAreas, role, [&] {
					const auto pawns = b.PCS(role, 0);
					const auto blocked = b.AND(pawns, b.SH(occ, role, 1));
					const auto early = b.AND(pawns, b.REL(role, rankMask(1) | rankMask(2)));
					return b.NOT(b.OR(blocked, b.OR(early, pawnAttacks(opponent))));
				});
			}

			// Exclude enemy minor-piece control from rook mobility and enemy rook control from queen mobility.
			InterSignal secondaryArea(InterSignal role, InterSignal opponent, int type) {
				if (type < 3) return b.BB(0);
				InterSignal unsafe = b.OR(b.attacks(opponent, 1), b.attacks(opponent, 2));
				if (type == 4) unsafe = b.OR(unsafe, b.attacks(opponent, 3));
				return b.AND(mobilityArea(role, opponent), b.NOT(unsafe));
			}

			// Count queens sharing a file with each rook.
			InterSignal rookLine(InterSignal role, InterSignal opponent) {
				Sum terms;
				for (int square : b.squares(role, 3)) {
					const auto present = b.ANY(b.AND(b.PCS(role, 3), b.REL(role, 1ULL << square)));
					const auto queens = b.OR(b.PCS(role, 4), b.PCS(opponent, 4));
					terms.add(b.MUL(present, b.POP(b.AND(queens, b.BB(fileMask(square % 8))))));
				}
				return b.sum(terms);
			}

			// Count enemy pawns on the diagonals of each bishop.
			InterSignal bishopXray(InterSignal role, InterSignal opponent) {
				Sum terms;
				for (int square : b.squares(role, 2)) {
					const auto present = b.ANY(b.AND(b.PCS(role, 2), b.REL(role, 1ULL << square)));
					terms.add(b.MUL(present, b.POP(b.AND(b.PCS(opponent, 0), b.REL(role, diagonalMask(square))))));
				}
				return b.sum(terms);
			}

			// Count moves by one piece type to squares from which that type attacks the enemy king.
			InterSignal potentialChecks(InterSignal attacker, InterSignal defender, int type) {
				if (type == 0) return z;
				Sum terms;
				for (int square : b.locations(b.PCS(defender, 5))) {
					const auto geometry = b.attackFrom(attacker, type, b.NUM(square));
					for (int source : b.locations(b.PCS(attacker, type))) {
						const auto moves = b.pieceAttack(attacker, type, source);
						terms.add(b.POP(b.AND(moves, geometry)));
					}
				}
				return b.sum(terms);
			}

			// Count unoccupied king destinations outside enemy control.
			InterSignal escapes(InterSignal defender, InterSignal attacker) {
				return b.POP(b.AND(b.attacks(defender, 5), b.AND(b.NOT(own(defender)), b.NOT(b.attacks(attacker)))));
			}

			// Return the friendly-pawn or enemy-pawn count at one king-relative distance.
			InterSignal kingPawns(InterSignal defender, int distance, bool friendly) {
				return b.kingPawn(defender, 2 * (distance - 1) + !friendly);
			}

			// Return the pawnless-file count around the king.
			InterSignal kingOpenFiles(InterSignal role) {
				return b.kingPawn(role, 6);
			}

			// Return and cache the king's distance-two ring or four-file flank.
			InterSignal kingRegion(InterSignal role, bool flank) {
				return shared(flank ? kingFlanks : kingRings, role, [&] {
					InterSignal mask = b.BB(0);
					for (int square : b.squares(role, 5)) {
						Word region = ringMask(square, 2);
						if (flank) {
							constexpr std::array<int, 8> first{{0, 0, 0, 2, 2, 4, 4, 4}};
							region = 0;
							for (int file = first[static_cast<std::size_t>(square % 8)]; file < first[static_cast<std::size_t>(square % 8)] + 4; ++file) {
								region |= fileMask(file);
							}
							region &= rankMask(0) | rankMask(1) | rankMask(2) | rankMask(3) | rankMask(4);
						}
						mask = b.OR(mask, b.REL(role, region));
					}
					return mask;
				});
			}

			// Count set squares inside one king flank.
			InterSignal flank(InterSignal map, InterSignal defender) {
				return b.POP(b.AND(map, kingRegion(defender, true)));
			}

			// Sum non-pawn material in 3/3/5/9 units.
			InterSignal nonPawnMaterial(InterSignal role) {
				InterSignal total = z;
				constexpr std::array<int, 5> value{{0, 3, 3, 5, 9}};
				for (int type = 1; type <= 4; ++type) {
					total = b.ADD(total, b.MUL(b.NUM(value[static_cast<std::size_t>(type)]), b.POP(b.PCS(role, type))));
				}
				return total;
			}

			Runtime b;
			SideSignals occupancyBySide, strongMaps, mobilityAreas, pawnPushAttacks, kingRings, kingFlanks;
			InterSignal z{};
			InterSignal o{};
			InterSignal us{};
			InterSignal them{};
			InterSignal occ{};
		};

		// Calculate one material-adjusted formula coefficient.
		float coefficient(const FormulaWeights &weights, const std::array<float, 5> &material) noexcept {
			float value = weights.base;
			for (std::size_t type = 0; type < material.size(); ++type) value = std::fma(weights.material[type], material[type], value);
			return value;
		}

		// ---------------------------- Prepared evaluation state ----------------------------
		// Prepare fixed coefficients, material-dependent formula indices and the king-pressure table.
		class State {
		public:
			State() : curve(fixed::formulaGlobals.pressureCenter, fixed::formulaGlobals.pressureWidth) {
				for (std::size_t i = 0; i < formulaCount; ++i) {
					base[i] = fixed::formulaWeights[i].base;
					const auto &material = fixed::formulaWeights[i].material;
					if (std::any_of(material.begin(), material.end(), [](float value) { return value != 0.0F; })) {
						materialFormulas.push_back(static_cast<std::uint16_t>(i));
					}
				}
			}

			// Return formula coefficients for the board's material signature. The per-thread cache keys five
			// piece counts packed into six-bit fields and recomputes material-dependent coefficients on a miss.
			const std::array<float, formulaCount> &coefficients(const std::array<Word, atomCount> &in) const noexcept {
				if (materialFormulas.empty()) return base;
				struct Entry {
					std::uint32_t signature = 0;
					std::array<float, formulaCount> values{};
					bool valid = false;
				};
				thread_local std::array<Entry, 16> cache{};
				std::uint32_t signature = 0;
				for (std::size_t type = 0; type < 5; ++type) {
					signature |= pieceCount(in, type) << (6 * type);
				}
				auto &entry = cache[(signature * 0x9e3779b9U) >> 28];
				if (!entry.valid || entry.signature != signature) {
					const auto coordinate = materialCoordinates(in);
					entry.values = base;
					for (std::uint16_t index : materialFormulas) {
						entry.values[index] = coefficient(fixed::formulaWeights[index], coordinate);
					}
					entry.signature = signature;
					entry.valid = true;
				}
				return entry.values;
			}

			std::array<float, formulaCount> base{};
			std::vector<std::uint16_t> materialFormulas;
			SigmoidCurve curve;
		};

		const State &evaluationState() {
			static const State value;
			return value;
		}
	} // namespace evaluation

	// Evaluate a board by accumulating each emitted formula signal directly into its score.
	float evaluate(const chess::Board &board) {
		const auto &prepared = evaluation::evaluationState();
		const auto in = evaluation::inputs(board);
		evaluation::ScoreAccumulator score(prepared.coefficients(in));
		evaluation::Formulas(evaluation::Runtime(board, in, score, evaluation::fixed::formulaGlobals, prepared.curve)).execute();
		return score.value();
	}

} // namespace eleginus
