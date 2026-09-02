#include "eleginus/formula.hpp"
#include "eleginus/atom.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <optional>
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
				in[static_cast<std::size_t>(type)] = pieces & byColor[0];
				in[static_cast<std::size_t>(6 + type)] = pieces & byColor[1];
			}
			for (int color = 0; color < 2; ++color) {
				for (int wing = 0; wing < 2; ++wing) {
					const auto flank = wing == 0 ? chess::Board::CastlingRights::Side::KING_SIDE : chess::Board::CastlingRights::Side::QUEEN_SIDE;
					if (board.castlingRights().has(static_cast<chess::Color>(color), flank))
						in[atomIndex(Atom::CR)] |= 1ULL << (2 * color + wing);
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

	} // namespace

	namespace {

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

		// Fixed formulas execute directly on scalar and bitboard values.
		struct Signal {
			Word bits = 0;
			Signal() = default;
			explicit Signal(Word x) : bits(x) {}
		};

		struct Features {
			static constexpr bool conditions = true;
			std::vector<Feature> &values;
			void put(std::uint32_t index, std::int32_t score, std::int32_t condition) { values.push_back({static_cast<std::uint16_t>(index), score, condition}); }
		};

		struct Weighted {
			static constexpr bool conditions = false;
			std::span<const float> weights;
			float value = 0.0F;
			void put(std::uint32_t index, std::int32_t score, std::int32_t) { value += static_cast<float>(score) * weights[index]; }
		};

		struct Graybox {
			static constexpr bool conditions = true;
			struct Storage {
				std::array<std::int32_t, kFormulaCount> score{}, condition{};
				std::array<std::uint32_t, kFormulaCount> stamp{};
				std::uint32_t generation = 0;
			};
			std::span<const float> weights;
			Storage &storage;
			float value = 0.0F;
			void put(std::uint32_t index, std::int32_t score, std::int32_t condition) {
				value += static_cast<float>(score) * weights[index];
				storage.score[index] = score;
				storage.condition[index] = condition;
				storage.stamp[index] = storage.generation;
			}
			std::int32_t score(std::uint16_t index) const { return storage.stamp[index] == storage.generation ? storage.score[index] : 0; }
			std::int32_t condition(std::uint16_t index) const { return storage.stamp[index] == storage.generation ? storage.condition[index] : 0; }
		};

		template <class Output> class Runtime {
		public:
			using Value = Signal;
			struct Sum {
				std::int64_t total = 0;
				void add(Value value) { total += number(value.bits); }
			};
			static constexpr bool conditions = Output::conditions;

			Runtime(const chess::Board &board, Output &out) : in(inputs(board)), occupied(board.occ().getBits()), output(out) {
				// Verify both full pawn bitboards on a hit; non-pawn pieces and turn are not dependencies.
				thread_local std::array<Pawns, 1024> table{};
				const Word hash = in[atomIndex(Atom::WP)] * 0x9e3779b97f4a7c15ULL ^ std::rotl(in[atomIndex(Atom::BP)] * 0xbf58476d1ce4e5b9ULL, 29);
				pawnCache = &table[(hash ^ (hash >> 32)) & (table.size() - 1)];
				if (!pawnCache->valid || pawnCache->white != in[atomIndex(Atom::WP)] || pawnCache->black != in[atomIndex(Atom::BP)]) {
					pawnCache->white = in[atomIndex(Atom::WP)];
					pawnCache->black = in[atomIndex(Atom::BP)];
					pawnCache->shapeReady = false;
					for (auto &king : pawnCache->kings)
						king.valid = false;
					pawnCache->valid = true;
				}
				thread_local Attacks attacks;
				thread_local Rays rays;
				attackCache = &attacks;
				rayCache = &rays;
				const Word changed = attacks.occupied ^ occupied;
				for (unsigned color = 0; color < 2; ++color) {
					Word all = 0, twice = 0;
					for (unsigned type = 0; type < 6; ++type) {
						const auto i = 6 * color + type;
						auto &map = attacks.maps[color][type];
						auto &overlap = attacks.twice[color][type];
						const bool slider = type >= 2 && type <= 4;
						// Slider maps include the first blocker; changes hidden behind it cannot affect a ray.
						if (!attacks.valid || attacks.pieces[i] != in[i] || (slider && (changed & map))) {
							if (type == 0) {
								const Word east = in[i] & 0x7F7F7F7F7F7F7F7FULL;
								const Word west = in[i] & 0xFEFEFEFEFEFEFEFEULL;
								const Word left = color == 0 ? east << 9 : east >> 7;
								const Word right = color == 0 ? west << 7 : west >> 9;
								map = left | right;
								overlap = left & right;
							} else {
								map = overlap = 0;
								auto pieces = in[i];
								while (pieces) {
									const int square = std::countr_zero(pieces);
									pieces &= pieces - 1;
									const auto a = attackFrom(NUM(color), type, NUM(square)).bits;
									overlap |= map & a;
									map |= a;
								}
							}
						}
						twice |= overlap | (all & map);
						all |= map;
						attacks.pieces[i] = in[i];
					}
					attacks.maps[color][6] = all;
					attacks.maps[color][7] = twice;
				}
				attacks.occupied = occupied;
				attacks.valid = true;
			}
			// NUM constructs a signed integer signal; BB constructs a raw bitboard signal.
			Value NUM(std::int64_t x) const { return Value(static_cast<Word>(x)); }
			Value BB(Word x) const { return Value(x); }

			// ATOM reads one irreducible board input.
			Value ATOM(Atom atom) const { return Value(in[atomIndex(atom)]); }

			// ADD, SUB, MUL, ABS and MIN form the integer arithmetic primitives.
			Value ADD(Value a, Value b) const { return NUM(number(a.bits) + number(b.bits)); }
			Value SUB(Value a, Value b) const { return NUM(number(a.bits) - number(b.bits)); }
			Value MUL(Value a, Value b) const { return NUM(number(a.bits) * number(b.bits)); }
			Value ABS(Value a) const { return NUM(std::abs(number(a.bits))); }
			Value MIN(Value a, Value b) const { return NUM(std::min(number(a.bits), number(b.bits))); }

			// LAND, LOR and LNOT map logical results to zero and one.
			Value LAND(Value a, Value b) const { return NUM(number(a.bits) != 0 && number(b.bits) != 0); }
			Value LOR(Value a, Value b) const { return NUM(number(a.bits) != 0 || number(b.bits) != 0); }
			Value LNOT(Value a) const { return NUM(number(a.bits) == 0); }

			// EQ, GT, LT, LE and GE compare integer signals and return zero or one.
			Value EQ(Value a, Value b) const { return NUM(a.bits == b.bits); }
			Value GT(Value a, Value b) const { return NUM(number(a.bits) > number(b.bits)); }
			Value LT(Value a, Value b) const { return NUM(number(a.bits) < number(b.bits)); }
			Value LE(Value a, Value b) const { return NUM(number(a.bits) <= number(b.bits)); }
			Value GE(Value a, Value b) const { return NUM(number(a.bits) >= number(b.bits)); }

			// AND, OR and NOT combine bitboards; POP and ANY reduce them to integers.
			Value AND(Value a, Value b) const { return BB(a.bits & b.bits); }
			Value OR(Value a, Value b) const { return BB(a.bits | b.bits); }
			Value NOT(Value a) const { return BB(~a.bits); }
			Value POP(Value a) const { return NUM(std::popcount(a.bits)); }
			Value ANY(Value a) const { return NUM(a.bits != 0); }

			// PCS selects a piece set; REL and SQ normalize coordinates; CR and OCC expose rule and occupancy state.
			Value PCS(Value role, int type) const { return BB(in[6 * color(role) + type]); }
			Value REL(Value role, Word mask) const { return BB(color(role) == 0 ? mask : flip(mask)); }
			Value SQ(Value role, int square) const { return NUM(square ^ (color(role) == 0 ? 0 : 56)); }
			Value CR(Value role, int wing) const { return NUM((in[atomIndex(Atom::CR)] & (1ULL << (2 * color(role) + wing))) != 0); }
			Value OCC() const { return BB(occupied); }

			// SH performs one role-relative step: forward, backward, east, west and the four diagonals.
			Value SH(Value x, Value role, int direction) const {
				const Word east = x.bits & 0x7F7F7F7F7F7F7F7FULL, west = x.bits & 0xFEFEFEFEFEFEFEFEULL;
				const bool white = color(role) == 0;
				switch (direction) {
				case 0:
					return BB(white ? x.bits << 8 : x.bits >> 8);
				case 1:
					return BB(white ? x.bits >> 8 : x.bits << 8);
				case 2:
					return BB(east << 1);
				case 3:
					return BB(west >> 1);
				case 4:
					return BB(white ? east << 9 : east >> 7);
				case 5:
					return BB(white ? west << 7 : west >> 9);
				case 6:
					return BB(white ? east >> 7 : east << 9);
				case 7:
					return BB(white ? west >> 9 : west << 7);
				default:
					throw std::logic_error("invalid shift direction");
				}
			}

			// Formula traversal and derived shared calculations follow the primitive set.
			unsigned roleIndex(Value role) const { return color(role); }
			Squares squares(Value role, int type) const { return {REL(role, PCS(role, type).bits).bits}; }
			Squares locations(Value set) const { return {set.bits}; }

			Value fill(Value x, Value role) const {
				for (unsigned d : {8U, 16U, 32U})
					x.bits |= color(role) == 0 ? x.bits << d : x.bits >> d;
				return x;
			}

			Value attackFrom(Value role, int type, Value square) {
				const int s = static_cast<int>(number(square.bits));
				if (type == 0)
					return BB(chess::attacks::pawn(static_cast<chess::Color>(color(role)), chess::Square(s)).getBits());
				if (type == 1)
					return BB(chess::attacks::knight(chess::Square(s)).getBits());
				if (type == 5)
					return BB(chess::attacks::king(chess::Square(s)).getBits());
				if (type == 4)
					return OR(attackFrom(role, 2, square), attackFrom(role, 3, square));
				auto &ray = rayCache->rays[type - 2][s];
				if (ray.map == 0 || ((ray.occupied ^ occupied) & ray.map))
					ray.map = (type == 2 ? chess::attacks::bishop(chess::Square(s), chess::Bitboard(occupied)) : chess::attacks::rook(chess::Square(s), chess::Bitboard(occupied)))
					              .getBits();
				ray.occupied = occupied;
				return BB(ray.map);
			}

			Value attacks(Value role, int type = 6) const { return BB(attackCache->maps[color(role)][type]); }
			Value doubleAttacks(Value role) const { return BB(attackCache->maps[color(role)][7]); }

			std::array<int, 2> pawnShape(Value role) {
				if (!pawnCache->shapeReady) {
					for (unsigned side = 0; side < 2; ++side) {
						int doubled = 0, islands = 0;
						bool previous = false;
						for (int file = 0; file < 8; ++file) {
							const int n = std::popcount(in[6 * side] & (0x0101010101010101ULL << file));
							doubled += std::max(0, n - 1);
							islands += n != 0 && !previous;
							previous = n != 0;
						}
						pawnCache->shape[side] = {doubled, islands};
					}
					pawnCache->shapeReady = true;
				}
				return pawnCache->shape[color(role)];
			}

			Value kingPawn(Value role, unsigned slot) {
				auto &entry = pawnCache->kings[color(role)];
				const Word king = PCS(role, 5).bits;
				if (!entry.valid || entry.king != king) {
					entry.king = king;
					entry.values.fill(0);
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
								if (!nearest[i])
									continue;
								const int distance = std::countr_zero(nearest[i]) / 8 - rank;
								if (distance <= 3)
									++entry.values[2 * (distance - 1) + i];
							}
						}
					}
					entry.valid = true;
				}
				return NUM(entry.values[slot]);
			}

			const auto &mobility(Value role, int type, Value area, Value guard) {
				thread_local std::array<std::array<Mobility, 4>, 2> table{};
				auto &entry = table[color(role)][type - 1];
				const Word pieces = PCS(role, type).bits;
				// Only changes on a previous attack ray can change a stationary piece's count.
				const Word changed = (entry.area ^ area.bits) | (entry.guard ^ guard.bits) | (type >= 2 ? entry.occupied ^ occupied : 0);
				if (!entry.valid || entry.pieces != pieces || (changed & entry.reach)) {
					entry.counts.fill(0);
					entry.secondary = 0;
					for (int square : locations(BB(pieces))) {
						const Word map = attackFrom(role, type, NUM(square)).bits;
						++entry.counts[std::popcount(map & area.bits)];
						entry.secondary += std::popcount(map & guard.bits);
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

			Value sum(const Sum &terms) const { return NUM(terms.total); }
			void root(Value score, Value condition) {
				const auto s = number(score.bits), c = number(condition.bits);
				if (s != 0 || c != 0)
					output.put(index, static_cast<std::int32_t>(s), static_cast<std::int32_t>(c));
				++index;
			}
			void skip(unsigned n) { index += n; }
			void finish() const {}

		private:
			struct KingPawn {
				Word king = 0;
				std::array<int, 7> values{}; // Shelter/storm at three distances, then open king files.
				bool valid = false;
			};
			struct Pawns {
				Word white = 0, black = 0;
				std::array<std::array<int, 2>, 2> shape{}; // Doubled pawns and islands, by absolute color.
				std::array<KingPawn, 2> kings{};
				bool valid = false, shapeReady = false;
			};
			struct Mobility {
				Word pieces = 0, occupied = 0, area = 0, guard = 0, reach = 0;
				std::array<int, 28> counts{};
				int secondary = 0;
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
				bool valid = false;
			};
			struct Rays {
				std::array<std::array<Ray, 64>, 2> rays{};
			};
			static unsigned color(Value role) { return static_cast<unsigned>(number(role.bits)); }
			Pawns *pawnCache;
			Attacks *attackCache;
			Rays *rayCache;
			std::array<Word, atomCount> in;
			Word occupied;
			Output &output;
			unsigned index = 0;
		};

		template <class B> class Formulas {
			using IS = typename B::Value;
			using Sum = typename B::Sum;
			using Pair = std::array<std::optional<IS>, 2>;

			struct Difference {
				IS score, own, enemy;
				operator IS() const { return score; }
			};

		public:
			explicit Formulas(B runtime) : b(std::move(runtime)) {
				z = b.NUM(0);
				o = b.NUM(1);
				us = b.NUM(0);
				them = b.NUM(1);
				occ = b.OCC();
				phase = phaseUnits();
			}

			auto execute() {
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
				return b.finish();
			}

		private:
			template <class F> IS shared(Pair &pair, IS role, F &&make) {
				auto &value = pair[b.roleIndex(role)];
				if (!value)
					value = make();
				return *value;
			}

			static constexpr std::uint64_t fileMask(int file) noexcept { return 0x0101010101010101ULL << file; }
			static constexpr std::uint64_t rankMask(int rank) noexcept { return 0xFFULL << (8 * rank); }

			static std::uint64_t ringMask(int square, int distance) noexcept {
				static const auto masks = [] {
					std::array<std::array<Word, 8>, 64> table{};
					for (int s = 0; s < 64; ++s)
						for (int t = 0; t < 64; ++t) {
							const int d = std::max(std::abs(t % 8 - s % 8), std::abs(t / 8 - s / 8));
							table[s][d] |= 1ULL << t;
						}
					return table;
				}();
				return masks[square][distance];
			}

			static std::uint64_t diagonalMask(int square) noexcept {
				static const auto masks = [] {
					std::array<Word, 64> table{};
					for (int s = 0; s < 64; ++s)
						for (int t = 0; t < 64; ++t) {
							const int df = std::abs(t % 8 - s % 8), dr = std::abs(t / 8 - s / 8);
							if (df && df == dr)
								table[s] |= 1ULL << t;
						}
					return table;
				}();
				return masks[square];
			}

			Difference diff(IS own, IS enemy) { return {b.SUB(own, enemy), own, enemy}; }

			IS own(IS role) {
				return shared(owned, role, [&] {
					IS result = b.PCS(role, 0);
					for (int type = 1; type < 6; ++type)
						result = b.OR(result, b.PCS(role, type));
					return result;
				});
			}

			IS pawnAttacks(IS role) {
				return shared(pawnAttack, role, [&] { return b.OR(b.SH(b.PCS(role, 0), role, 4), b.SH(b.PCS(role, 0), role, 5)); });
			}

			IS files(IS set, IS role, IS opponent) { return b.OR(b.fill(set, role), b.fill(set, opponent)); }

			IS passedPawns(IS role, IS opponent) {
				return shared(passedMap, role, [&] {
					const auto span = b.fill(b.PCS(opponent, 0), opponent);
					const auto stops = b.OR(span, b.OR(b.SH(span, role, 2), b.SH(span, role, 3)));
					return b.AND(b.PCS(role, 0), b.NOT(stops));
				});
			}

			IS strongSquares(IS role, IS opponent) {
				return shared(strongMap, role, [&] { return b.OR(pawnAttacks(opponent), b.AND(b.doubleAttacks(opponent), b.NOT(b.doubleAttacks(role)))); });
			}

			IS phaseUnits() {
				IS total = z;
				constexpr std::array<int, 6> units{{0, 1, 1, 2, 4, 0}};
				for (const auto role : {us, them})
					for (int type = 1; type <= 4; ++type)
						total = b.ADD(total, b.MUL(b.NUM(units[static_cast<std::size_t>(type)]), b.POP(b.PCS(role, type))));
				return b.MIN(total, b.NUM(24));
			}

			void F(Difference signal) {
				if constexpr (B::conditions)
					b.root(signal.score, b.ADD(signal.own, signal.enemy));
				else
					b.root(signal.score, z);
			}

			void F(IS signal) {
				if constexpr (B::conditions)
					b.root(signal, b.ABS(signal));
				else
					b.root(signal, z);
			}

			IS mobilityArea(IS role, IS opponent) {
				return shared(mobilityAreaCache, role, [&] {
					const auto pawns = b.PCS(role, 0);
					const auto blocked = b.AND(pawns, b.SH(occ, role, 1));
					const auto early = b.AND(pawns, b.REL(role, rankMask(1) | rankMask(2)));
					return b.AND(b.NOT(own(role)), b.NOT(b.OR(blocked, b.OR(early, pawnAttacks(opponent)))));
				});
			}

			IS secondaryArea(IS role, IS opponent, int type) {
				if (type < 3)
					return b.BB(0);
				IS unsafe = b.OR(pawnAttacks(opponent), b.OR(b.attacks(opponent, 1), b.attacks(opponent, 2)));
				if (type == 4)
					unsafe = b.OR(unsafe, b.attacks(opponent, 3));
				return b.AND(b.NOT(own(role)), b.NOT(unsafe));
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

			IS ringAttacks(IS attacker, IS defender, int type, int distance) {
				const auto attack = b.attacks(attacker, type);
				const auto region = distance == 1 ? b.attacks(defender, 5) : kingRegion(defender, false);
				return b.POP(b.AND(attack, region));
			}

			IS potentialChecks(IS attacker, IS defender, int type) {
				Sum terms;
				for (int square : b.squares(defender, 5)) {
					const auto king = b.ANY(b.AND(b.PCS(defender, 5), b.REL(defender, 1ULL << square)));
					const auto geometry = b.attackFrom(type == 0 ? defender : attacker, type, b.SQ(defender, square));
					const auto destinations = b.AND(geometry, b.AND(b.attacks(attacker, type), b.NOT(own(attacker))));
					terms.add(b.MUL(king, b.POP(destinations)));
				}
				return b.sum(terms);
			}

			IS escapes(IS defender, IS attacker) { return b.POP(b.AND(b.attacks(defender, 5), b.AND(b.NOT(own(defender)), b.NOT(b.attacks(attacker))))); }

			IS kingPawns(IS defender, IS attacker, int distance, bool friendly) {
				(void)attacker;
				return b.kingPawn(defender, 2 * (distance - 1) + !friendly);
			}

			IS kingOpenFiles(IS role) { return b.kingPawn(role, 6); }

			IS kingRegion(IS role, bool flank) {
				return shared(flank ? kingFlank : kingRing, role, [&] {
					IS mask = b.BB(0);
					for (int square : b.squares(role, 5)) {
						Word region = ringMask(square, 2);
						if (flank) {
							region = 0;
							for (int file = std::max(0, square % 8 - 1); file <= std::min(7, square % 8 + 1); ++file)
								region |= fileMask(file);
						}
						mask = b.OR(mask, b.REL(role, region));
					}
					return mask;
				});
			}

			IS flank(IS map, IS defender) { return b.POP(b.AND(map, kingRegion(defender, true))); }

			IS nonPawnMaterial(IS role) {
				IS total = z;
				constexpr std::array<int, 5> value{{0, 3, 3, 5, 9}};
				for (int type = 1; type <= 4; ++type)
					total = b.ADD(total, b.MUL(b.NUM(value[static_cast<std::size_t>(type)]), b.POP(b.PCS(role, type))));
				return total;
			}

#define ELEGINUS_FORMULAS
#define FORMULA(name) void name()
#include "formula.inl"
#undef FORMULA
#undef ELEGINUS_FORMULAS

			B b;
			Pair owned, pawnAttack, passedMap, strongMap, pawnFile, mobilityAreaCache, pushAttack, kingRing, kingFlank;
			IS z{};
			IS o{};
			IS us{};
			IS them{};
			IS occ{};
			IS phase{};
		};

	} // namespace

	void detail::extract(const chess::Board &board, std::vector<Feature> &out) {
		out.clear();
		if (out.capacity() < kFormulaCount)
			out.reserve(kFormulaCount);
		Features sink{out};
		Formulas<Runtime<Features>>(Runtime(board, sink)).execute();
	}

	float detail::score(const chess::Board &board, std::span<const float> weights) {
		if (weights.size() != kFormulaCount)
			throw std::invalid_argument("formula weight count does not match the fixed formula set");
		Weighted sink{weights};
		Formulas<Runtime<Weighted>>(Runtime(board, sink)).execute();
		return static_cast<float>(sink.value);
	}

	float detail::score(
	    const chess::Board &board, std::span<const float> base, std::span<const std::uint16_t> rows, std::span<const std::uint16_t> conditions, std::span<const float> relations) {
		if (base.size() != kFormulaCount || rows.size() != conditions.size() || rows.size() != relations.size())
			throw std::invalid_argument("graybox coordinates do not match the fixed formula set");
		thread_local Graybox::Storage storage;
		if (++storage.generation == 0) {
			storage.stamp.fill(0);
			++storage.generation;
		}
		Graybox sink{base, storage};
		Formulas<Runtime<Graybox>>(Runtime(board, sink)).execute();
		for (std::size_t i = 0; i < relations.size(); ++i)
			sink.value += relations[i] * static_cast<float>(sink.score(rows[i])) * static_cast<float>(sink.condition(conditions[i]));
		return sink.value;
	}

} // namespace eleginus

namespace eleginus {

	void FormulaSet::evaluate(const chess::Board &board, std::vector<Feature> &out) {
		detail::extract(board, out);
	}

	float FormulaSet::evaluate(const chess::Board &board, std::span<const float> weights) {
		return detail::score(board, weights);
	}

	float FormulaSet::evaluate(
	    const chess::Board &board, std::span<const float> base, std::span<const std::uint16_t> rows, std::span<const std::uint16_t> conditions, std::span<const float> relations) {
		return detail::score(board, base, rows, conditions, relations);
	}

} // namespace eleginus
