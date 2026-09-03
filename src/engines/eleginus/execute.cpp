#include "eleginus/formula.hpp"
#include "eleginus/atom.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>

namespace eleginus {
	namespace {
		using Word = std::uint64_t;

		std::int64_t number(Word x) noexcept {
			return std::bit_cast<std::int64_t>(x);
		}
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

		std::uint64_t flip(Word x) noexcept {
			x = ((x & 0x00FF00FF00FF00FFULL) << 8) | ((x >> 8) & 0x00FF00FF00FF00FFULL);
			x = ((x & 0x0000FFFF0000FFFFULL) << 16) | ((x >> 16) & 0x0000FFFF0000FFFFULL);
			return (x << 32) | (x >> 32);
		}

		// Sparse square iteration preserves the fixed coordinate order while visiting occupied squares only.
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

		// A signal is the single value representation used throughout the formula algebra.
		struct Signal {
			Word bits = 0;
			Signal() = default;
			explicit Signal(Word x) : bits(x) {}
		};
		// These aliases identify the producing layer.
		using AtomSignal = Signal;
		using InterSignal = Signal;

		class ScoreOutput {
		public:
			explicit ScoreOutput(std::span<const float> values) : values(values) {}

			void put(std::uint32_t index, std::int32_t score) {
				total = std::fma(values[index], static_cast<float>(score), total);
			}

			float finish() const noexcept { return total; }
			int direction() const noexcept { return (total > 0.0F) - (total < 0.0F); }

		private:
			std::span<const float> values;
			float total = 0.0F;
		};

		template <class Output> class Runtime {
		public:
			struct Sum {
				std::int64_t total = 0;
				void add(InterSignal value) { total += number(value.bits); }
			};
			Runtime(const chess::Board &board, Output &out) : in(inputs(board)), occupied(board.occ().getBits()), output(out) {
				// Pure pawn facts share one cache entry and verify both complete pawn bitboards on every hit.
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
				// Pawn and king attacks depend only on their piece bitboards.
				for (unsigned color = 0; color < 2; ++color) {
					fixedAttacks[color][0] = pawnCache->attacks[color];
					Word kings = in[6 * color + 5];
					while (kings) {
						const int square = std::countr_zero(kings);
						kings &= kings - 1;
						fixedAttacks[color][1] |= chess::attacks::king(chess::Square(square)).getBits();
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
						// Slider maps include the first blocker; changes hidden behind it cannot affect a ray.
						if (!attackCache->valid || attackCache->pieces[i] != in[i] || (slider && (changed & map))) {
							if (type == 0) {
								const Word east = in[i] & 0x7F7F7F7F7F7F7F7FULL;
								const Word west = in[i] & 0xFEFEFEFEFEFEFEFEULL;
								const Word left = color == 0 ? east << 9 : east >> 7;
								const Word right = color == 0 ? west << 7 : west >> 9;
								map = fixedAttacks[color][0];
								overlap = left & right;
							} else if (type == 5) {
								map = fixedAttacks[color][1];
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
			// NUM constructs a signed integer signal; BB constructs a raw bitboard signal.
			AtomSignal NUM(std::int64_t x) const { return AtomSignal(static_cast<Word>(x)); }
			AtomSignal BB(Word x) const { return AtomSignal(x); }

			// ATOM reads one irreducible board input.
			AtomSignal ATOM(Atom atom) const { return AtomSignal(in[atomIndex(atom)]); }

			// ADD, SUB, MUL, ABS and MIN form the integer arithmetic primitives.
			AtomSignal ADD(InterSignal a, InterSignal b) const { return NUM(number(a.bits) + number(b.bits)); }
			AtomSignal SUB(InterSignal a, InterSignal b) const { return NUM(number(a.bits) - number(b.bits)); }
			AtomSignal MUL(InterSignal a, InterSignal b) const { return NUM(number(a.bits) * number(b.bits)); }
			AtomSignal ABS(InterSignal a) const { return NUM(std::abs(number(a.bits))); }
			AtomSignal MIN(InterSignal a, InterSignal b) const { return NUM(std::min(number(a.bits), number(b.bits))); }

			// LAND, LOR and LNOT map logical results to zero and one.
			AtomSignal LAND(InterSignal a, InterSignal b) const { return NUM(number(a.bits) != 0 && number(b.bits) != 0); }
			AtomSignal LOR(InterSignal a, InterSignal b) const { return NUM(number(a.bits) != 0 || number(b.bits) != 0); }
			AtomSignal LNOT(InterSignal a) const { return NUM(number(a.bits) == 0); }

			// EQ, GT, LT, LE and GE compare integer signals and return zero or one.
			AtomSignal EQ(InterSignal a, InterSignal b) const { return NUM(a.bits == b.bits); }
			AtomSignal GT(InterSignal a, InterSignal b) const { return NUM(number(a.bits) > number(b.bits)); }
			AtomSignal LT(InterSignal a, InterSignal b) const { return NUM(number(a.bits) < number(b.bits)); }
			AtomSignal LE(InterSignal a, InterSignal b) const { return NUM(number(a.bits) <= number(b.bits)); }
			AtomSignal GE(InterSignal a, InterSignal b) const { return NUM(number(a.bits) >= number(b.bits)); }

			// AND, OR and NOT combine bitboards; POP and ANY reduce them to integers.
			AtomSignal AND(InterSignal a, InterSignal b) const { return BB(a.bits & b.bits); }
			AtomSignal OR(InterSignal a, InterSignal b) const { return BB(a.bits | b.bits); }
			AtomSignal NOT(InterSignal a) const { return BB(~a.bits); }
			AtomSignal POP(InterSignal a) const { return NUM(std::popcount(a.bits)); }
			AtomSignal ANY(InterSignal a) const { return NUM(a.bits != 0); }

			// PCS selects a piece set; REL and SQ normalize coordinates; CR exposes rule state.
			AtomSignal PCS(InterSignal role, int type) const { return BB(in[pieceAtomIndex(color(role), static_cast<std::size_t>(type))]); }
			AtomSignal REL(InterSignal role, Word mask) const { return BB(color(role) == 0 ? mask : flip(mask)); }
			AtomSignal SQ(InterSignal role, int square) const { return NUM(square ^ (color(role) == 0 ? 0 : 56)); }
			AtomSignal CR(InterSignal role, int wing) const { return NUM((in[atomIndex(Atom::CR)] & (1ULL << (2 * color(role) + wing))) != 0); }

			// SH performs one role-relative step: forward, backward, east, west and the four diagonals.
			AtomSignal SH(InterSignal x, InterSignal role, int direction) const {
				const Word east = x.bits & 0x7F7F7F7F7F7F7F7FULL, west = x.bits & 0xFEFEFEFEFEFEFEFEULL;
				const bool white = color(role) == 0;
				switch (direction) {
					case 0: return BB(white ? x.bits << 8 : x.bits >> 8);
					case 1: return BB(white ? x.bits >> 8 : x.bits << 8);
					case 2: return BB(east << 1);
					case 3: return BB(west >> 1);
					case 4: return BB(white ? east << 9 : east >> 7);
					case 5: return BB(white ? west << 7 : west >> 9);
					case 6: return BB(white ? east >> 7 : east << 9);
					case 7: return BB(white ? west >> 9 : west << 7);
					default: throw std::logic_error("invalid shift direction");
				}
			}

			// Formula traversal and derived shared calculations follow the primitive set.
			InterSignal occ() const { return BB(occupied); }
			unsigned roleIndex(InterSignal role) const { return color(role); }
			Squares squares(InterSignal role, int type) const { return {REL(role, PCS(role, type).bits).bits}; }
			Squares locations(InterSignal set) const { return {set.bits}; }

			InterSignal fill(InterSignal x, InterSignal role) const {
				for (unsigned d : {8U, 16U, 32U}) {
					x.bits |= color(role) == 0 ? x.bits << d : x.bits >> d;
				}
				return x;
			}

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
			InterSignal pieceAttack(InterSignal role, int type, int square) const {
				return BB(attackCache->from[color(role)][type][static_cast<std::size_t>(square)]);
			}

			InterSignal attacks(InterSignal role, int type = 6) const {
				if (type == 0 || type == 5) return BB(fixedAttacks[color(role)][type == 0 ? 0 : 1]);
				return BB(attackCache->maps[color(role)][type]);
			}
			InterSignal doubleAttacks(InterSignal role) const { return BB(attackCache->maps[color(role)][7]); }
			InterSignal pawnFiles(InterSignal role) const { return BB(pawnCache->files[color(role)]); }
			InterSignal passedPawns(InterSignal role) const { return BB(pawnCache->passed[color(role)]); }
			const auto &pawnStructure(InterSignal role) const { return pawnCache->shape[color(role)]; }
			InterSignal attackCount(InterSignal role, InterSignal square) {
				const auto reverse = NUM(color(role) ^ 1U);
				int total = std::popcount(PCS(role, 0).bits & attackFrom(reverse, 0, square).bits);
				for (int type = 1; type < 6; ++type) {
					total += std::popcount(PCS(role, type).bits & attackFrom(role, type, square).bits);
				}
				return NUM(total);
			}

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
						// One file visit supplies all shelter/storm distances and the open-file signal.
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

			InterSignal kingPawn(InterSignal role, unsigned slot) {
				const auto &entry = pawnCache->kings[color(role)];
				return NUM(entry.values[slot]);
			}
			InterSignal shelter(InterSignal role, unsigned slot) {
				return NUM(pawnCache->kings[color(role)].shelter[slot]);
			}
			InterSignal blockedStorm(InterSignal role, unsigned slot) {
				return NUM(pawnCache->kings[color(role)].blocked[slot]);
			}
			InterSignal storm(InterSignal role, unsigned slot) {
				return NUM(pawnCache->kings[color(role)].storm[slot]);
			}

			const auto &mobility(InterSignal role, int type, InterSignal area, InterSignal guard) {
				thread_local std::array<std::array<Mobility, 4>, 2> table{};
				auto &entry = table[color(role)][type - 1];
				const Word pieces = PCS(role, type).bits;
				// Only changes on a previous attack ray can change a stationary piece's count.
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

			InterSignal sum(const Sum &terms) const { return NUM(terms.total); }
			void root(InterSignal score) {
				const auto value = number(score.bits);
				if (value != 0) output.put(index, static_cast<std::int32_t>(value));
				++index;
			}
			InterSignal direction() const { return NUM(output.direction()); }
			void skip(unsigned n) { index += n; }

		private:
			struct KingPawn {
				Word king = 0;
				std::array<int, 7> values{}; // Shelter/storm at three distances, then open king files.
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
			static unsigned color(InterSignal role) { return static_cast<unsigned>(number(role.bits)); }
			Pawns *pawnCache;
			Attacks *attackCache;
			Rays *rayCache;
			std::array<Word, atomCount> in;
			std::array<std::array<Word, 2>, 2> fixedAttacks{};
			Word occupied;
			Output &output;
			unsigned index = 0;
		};

		template <class B> class Formulas {
			using IS = InterSignal;
			using Sum = typename B::Sum;
			using Pair = std::array<std::optional<IS>, 2>;

		#define ELEGINUS_FORMULAS
		#define FORMULA(name) void name()
		#include "formula.inl"
		#undef FORMULA
		#undef ELEGINUS_FORMULAS

		public:
			explicit Formulas(B runtime) : b(std::move(runtime)) {
				z = b.NUM(0);
				o = b.NUM(1);
				us = b.NUM(0);
				them = b.NUM(1);
				occ = b.occ();
				phase = phaseUnits();
			}

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
		}
			static std::span<const float> weights() noexcept {
				return formulaWeights;
			}

		private:
			template <class F> IS shared(Pair &pair, IS role, F &&make) {
				auto &value = pair[b.roleIndex(role)];
				if (!value) value = make();
				return *value;
			}

			static constexpr std::uint64_t fileMask(int file) noexcept {
				return 0x0101010101010101ULL << file;
			}
			static constexpr std::uint64_t rankMask(int rank) noexcept {
				return 0xFFULL << (8 * rank);
			}

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

			IS diff(IS own, IS enemy) { return b.SUB(own, enemy); }

			IS own(IS role) {
				return shared(owned, role, [&] {
					IS result = b.PCS(role, 0);
					for (int type = 1; type < 6; ++type) {
						result = b.OR(result, b.PCS(role, type));
					}
					return result;
				});
			}

			IS pawnAttacks(IS role) {
				return b.attacks(role, 0);
			}

			IS files(IS role) {
				return b.pawnFiles(role);
			}

			IS passedPawns(IS role, IS opponent) {
				(void)opponent;
				return b.passedPawns(role);
			}

			IS strongSquares(IS role, IS opponent) {
				return shared(
					strongMap, role, [&] { return b.OR(pawnAttacks(opponent), b.AND(b.doubleAttacks(opponent), b.NOT(b.doubleAttacks(role)))); });
			}

			IS phaseUnits() {
				IS total = z;
				constexpr std::array<int, 6> units{{0, 1, 1, 2, 4, 0}};
				for (const auto role : {us, them}) {
					for (int type = 1; type <= 4; ++type) {
						total = b.ADD(total, b.MUL(b.NUM(units[static_cast<std::size_t>(type)]), b.POP(b.PCS(role, type))));
					}
				}
				return b.MIN(total, b.NUM(24));
			}

			void F(IS signal) { b.root(signal); }

			IS mobilityArea(IS role, IS opponent) {
				return shared(mobilityAreaCache, role, [&] {
					const auto pawns = b.PCS(role, 0);
					const auto blocked = b.AND(pawns, b.SH(occ, role, 1));
					const auto early = b.AND(pawns, b.REL(role, rankMask(1) | rankMask(2)));
					return b.NOT(b.OR(blocked, b.OR(early, pawnAttacks(opponent))));
				});
			}

			IS secondaryArea(IS role, IS opponent, int type) {
				if (type < 3) return b.BB(0);
				IS unsafe = b.OR(b.attacks(opponent, 1), b.attacks(opponent, 2));
				if (type == 4) unsafe = b.OR(unsafe, b.attacks(opponent, 3));
				return b.AND(mobilityArea(role, opponent), b.NOT(unsafe));
			}

			IS rookLine(IS role, IS opponent) {
				Sum terms;
				for (int square : b.squares(role, 3)) {
					const auto present = b.ANY(b.AND(b.PCS(role, 3), b.REL(role, 1ULL << square)));
					const auto queens = b.OR(b.PCS(role, 4), b.PCS(opponent, 4));
					terms.add(b.MUL(present, b.POP(b.AND(queens, b.BB(fileMask(square % 8))))));
				}
				return b.sum(terms);
			}

			IS bishopXray(IS role, IS opponent) {
				Sum terms;
				for (int square : b.squares(role, 2)) {
					const auto present = b.ANY(b.AND(b.PCS(role, 2), b.REL(role, 1ULL << square)));
					terms.add(b.MUL(present, b.POP(b.AND(b.PCS(opponent, 0), b.REL(role, diagonalMask(square))))));
				}
				return b.sum(terms);
			}

			IS potentialChecks(IS attacker, IS defender, int type) {
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

			IS escapes(IS defender, IS attacker) {
				return b.POP(b.AND(b.attacks(defender, 5), b.AND(b.NOT(own(defender)), b.NOT(b.attacks(attacker)))));
			}

			IS kingPawns(IS defender, IS attacker, int distance, bool friendly) {
				(void)attacker;
				return b.kingPawn(defender, 2 * (distance - 1) + !friendly);
			}

			IS kingOpenFiles(IS role) {
				return b.kingPawn(role, 6);
			}

			IS kingRegion(IS role, bool flank) {
				return shared(flank ? kingFlank : kingRing, role, [&] {
					IS mask = b.BB(0);
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

			IS flank(IS map, IS defender) {
				return b.POP(b.AND(map, kingRegion(defender, true)));
			}

			IS nonPawnMaterial(IS role) {
				IS total = z;
				constexpr std::array<int, 5> value{{0, 3, 3, 5, 9}};
				for (int type = 1; type <= 4; ++type) {
					total = b.ADD(total, b.MUL(b.NUM(value[static_cast<std::size_t>(type)]), b.POP(b.PCS(role, type))));
				}
				return total;
			}

			B b;
			Pair owned, strongMap, mobilityAreaCache, pushAttack, kingRing, kingFlank;
			IS z{};
			IS o{};
			IS us{};
			IS them{};
			IS occ{};
			IS phase{};
		};

	} // namespace

	float FormulaSet::score(const chess::Board &board) {
		ScoreOutput sink(Formulas<Runtime<ScoreOutput>>::weights());
		Formulas(Runtime(board, sink)).execute();
		return sink.finish();
	}

} // namespace eleginus
