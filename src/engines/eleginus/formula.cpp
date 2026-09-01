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
		std::int64_t number(Word x) noexcept { return std::bit_cast<std::int64_t>(x); }
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
			void put(std::uint32_t index, std::int32_t score, std::int32_t condition) {
				values.push_back({static_cast<std::uint16_t>(index), score, condition});
			}
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
				void push_back(Value v) { total += number(v.bits); }
			};
			static constexpr bool conditions = Output::conditions;

			Runtime(const chess::Board &board, Output &out) : in(inputs(board)), occupied(board.occ().getBits()), output(out) {
				// Verify both full pawn bitboards on a hit; non-pawn pieces and turn are not dependencies.
				thread_local std::array<Pawns, 1024> table{};
				const Word hash = in[atomIndex(Atom::WP)] * 0x9e3779b97f4a7c15ULL ^
				    std::rotl(in[atomIndex(Atom::BP)] * 0xbf58476d1ce4e5b9ULL, 29);
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
			Value NUM(std::int64_t x) const { return Value(static_cast<Word>(x)); }
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
			Value BB(Word x) const { return Value(x); }
			Value ATOM(Atom atom) const { return Value(in[atomIndex(atom)]); }
			Value ADD(Value a, Value b) const { return NUM(number(a.bits) + number(b.bits)); }
			Value SUB(Value a, Value b) const { return NUM(number(a.bits) - number(b.bits)); }
			Value MUL(Value a, Value b) const { return NUM(number(a.bits) * number(b.bits)); }
			Value LAND(Value a, Value b) const { return NUM(number(a.bits) != 0 && number(b.bits) != 0); }
			Value LOR(Value a, Value b) const { return NUM(number(a.bits) != 0 || number(b.bits) != 0); }
			Value LNOT(Value a) const { return NUM(number(a.bits) == 0); }
			Value EQ(Value a, Value b) const { return NUM(a.bits == b.bits); }
			Value DIV(Value a, Value b) const { return NUM(number(b.bits) != 0 ? number(a.bits) / number(b.bits) : 0); }
			Value ABS(Value a) const { return NUM(std::abs(number(a.bits))); }
			Value GT(Value a, Value b) const { return NUM(number(a.bits) > number(b.bits)); }
			Value LT(Value a, Value b) const { return NUM(number(a.bits) < number(b.bits)); }
			Value LE(Value a, Value b) const { return NUM(number(a.bits) <= number(b.bits)); }
			Value GE(Value a, Value b) const { return NUM(number(a.bits) >= number(b.bits)); }
			Value MAX(Value a, Value b) const { return NUM(std::max(number(a.bits), number(b.bits))); }
			Value MIN(Value a, Value b) const { return NUM(std::min(number(a.bits), number(b.bits))); }
			Value AND(Value a, Value b) const { return BB(a.bits & b.bits); }
			Value OR(Value a, Value b) const { return BB(a.bits | b.bits); }
			Value NOT(Value a) const { return BB(~a.bits); }
			Value POP(Value a) const { return NUM(std::popcount(a.bits)); }
			Value ANY(Value a) const { return NUM(a.bits != 0); }
			unsigned side(Value role) const { return color(role); }
			Value PCS(Value role, int type) const { return BB(in[6 * color(role) + type]); }
			Value REL(Value role, Word mask) const { return BB(color(role) == 0 ? mask : flip(mask)); }
			Value sq(Value role, int square) const { return NUM(square ^ (color(role) == 0 ? 0 : 56)); }
			Value RIGHT(Value role, int side) const {
				return NUM((in[atomIndex(Atom::CR)] & (1ULL << (2 * color(role) + side))) != 0);
			}
			Value OCC() const { return BB(occupied); }
			Squares squares(Value role, int type) const { return {REL(role, PCS(role, type).bits).bits}; }
			Squares locations(Value set) const { return {set.bits}; }
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
			Value attacks2(Value role) const { return BB(attackCache->maps[color(role)][7]); }
			Value sum(const Sum &terms) const { return NUM(terms.total); }
			void root(Value d, Value t) {
				const auto di = number(d.bits), ti = number(t.bits);
				if (di != 0 || ti != 0)
					output.put(index, static_cast<std::int32_t>(di), static_cast<std::int32_t>(ti));
				++index;
			}
			void skip(unsigned n) {
				index += n;
			}
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
			static unsigned color(Value role) {
				return static_cast<unsigned>(number(role.bits));
			}
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
				emitMaterial();
				emitPst();
				emitBishopPair();
				emitPawns();
				emitMobility();
				emitPieces();
				emitThreats();
				emitKings();
				emitEndgames();
				return b.finish();
			}

		private:
			template <class F> IS shared(Pair &pair, IS role, F &&make) {
				auto &value = pair[b.side(role)];
				if (!value)
					value = make();
				return *value;
			}

			bool has(int type) const { return b.PCS(us, type).bits != 0 || b.PCS(them, type).bits != 0; }
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

			IS CNT(IS set) { return b.POP(set); }
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
				return shared(strongMap, role, [&] { return b.OR(pawnAttacks(opponent), b.AND(b.attacks2(opponent), b.NOT(b.attacks2(role)))); });
			}

			IS passerKingDistance(IS kingrole, IS passed, int distance) {
				Sum terms;
				for (int square : b.locations(passed)) {
					const auto pawn = b.ANY(b.AND(passed, b.BB(1ULL << square)));
					const auto king = b.ANY(b.AND(b.PCS(kingrole, 5), b.BB(ringMask(square, distance))));
					terms.push_back(b.LAND(pawn, king));
				}
				return b.sum(terms);
			}

			IS phaseUnits() {
				IS total = z;
				constexpr std::array<int, 6> units{{0, 1, 1, 2, 4, 0}};
				for (const auto role : {us, them}) {
					for (int type = 1; type <= 4; ++type) {
						total = b.ADD(total, b.MUL(b.NUM(units[static_cast<std::size_t>(type)]), CNT(b.PCS(role, type))));
					}
				}
				return b.MIN(total, b.NUM(24));
			}

			void put(Difference signal) {
				if constexpr (B::conditions)
					b.root(signal.score, b.ADD(signal.own, signal.enemy));
				else
					b.root(signal.score, z);
			}

			void put(IS signal) {
				if constexpr (B::conditions)
					b.root(signal, b.ABS(signal));
				else
					b.root(signal, z);
			}

			void tempo() {
				const auto stm = b.ATOM(Atom::STM);
				// Tempo: +1 when White moves and -1 when Black moves.
				put(diff(b.EQ(stm, us), b.EQ(stm, them)));
			}

			void emitMaterial() {
				// Catalog entries material.pawn through material.queen.
				for (int type = 0; type < 5; ++type)
					put(diff(CNT(b.PCS(us, type)), CNT(b.PCS(them, type))));
			}

			void emitPst() {
				// Catalog entries pst.pawn.a1 through pst.king.h8.
				for (int type = 0; type < 6; ++type) {
					const auto friendly = b.REL(us, b.PCS(us, type).bits).bits;
					const auto enemy = b.REL(them, b.PCS(them, type).bits).bits;
					unsigned next = 0;
					// Visit occupied normalized squares in index order; opposing activations may cancel in the score but remain part of the condition.
					for (int square : b.locations(b.BB(friendly | enemy))) {
						b.skip(square - next);
						const auto mask = 1ULL << square;
						put(diff(b.NUM((friendly & mask) != 0), b.NUM((enemy & mask) != 0)));
						next = square + 1;
					}
					b.skip(64 - next);
				}
			}

			void emitBishopPair() {
				// Catalog entry bishopPair.
				const auto friendly = b.GE(CNT(b.PCS(us, 2)), b.NUM(2));
				const auto enemy = b.GE(CNT(b.PCS(them, 2)), b.NUM(2));
				put(diff(friendly, enemy));
			}

			void emitPawns() {
				// Catalog entries pawn.passed.rank2 through passer.enemyKing.distance7.
				const auto fp = b.PCS(us, 0);
				const auto ep = b.PCS(them, 0);
				const auto fpass = passedPawns(us, them);
				const auto epass = passedPawns(them, us);
				const auto fatt = pawnAttacks(us);
				const auto eatt = pawnAttacks(them);
				const auto ffiles = shared(pawnFile, us, [&] { return files(fp, us, them); });
				const auto efiles = shared(pawnFile, them, [&] { return files(ep, them, us); });
				const auto fneighbours = b.OR(b.SH(ffiles, us, 2), b.SH(ffiles, us, 3));
				const auto eneighbours = b.OR(b.SH(efiles, them, 2), b.SH(efiles, them, 3));
				std::array<std::array<int, 78>, 2> counts{};
				const auto collect = [&](unsigned side, IS role, IS opponent, IS pawns, IS passed, IS att, IS oppatt, IS neighbours) {
					auto &values = counts[side];
					const auto shape = b.pawnShape(role);
					values[60] += shape[0];
					values[61] += std::popcount(pawns.bits & ~neighbours.bits);
					values[62] += std::popcount(pawns.bits & ~att.bits & b.SH(oppatt, opponent, 0).bits);
					values[63] += shape[1];
					std::array<Word, 8> sets{};
					if (passed.bits) {
						const auto a = b.attacks(role), e = b.attacks(opponent);
						const auto wide = b.BB(e.bits | b.SH(e, role, 2).bits | b.SH(e, role, 3).bits);
						const auto control = b.BB((a.bits & ~e.bits) | (b.attacks2(role).bits & ~b.attacks2(opponent).bits));
						sets = {passed.bits, ~b.SH(b.fill(occ, opponent), opponent, 0).bits, ~b.SH(b.fill(e, opponent), opponent, 0).bits,
						    ~b.SH(b.fill(wide, opponent), opponent, 0).bits, b.SH(control, role, 1).bits, att.bits, b.SH(occ, opponent, 0).bits,
						    b.SH(passed, role, 2).bits | b.SH(passed, role, 3).bits};
					}
					const Word phalanx = b.SH(pawns, role, 2).bits;
					const bool white = number(role.bits) == 0;
					// Visit each pawn once for rank signals and both king-distance histograms.
					for (int square : b.locations(pawns)) {
						const Word bit = 1ULL << square;
						const int rank = white ? square / 8 : 7 - square / 8;
						if (rank >= 1 && rank <= 6) {
							const int start = 10 * (rank - 1);
							values[start + 8] += (phalanx & bit) != 0;
							values[start + 9] += (att.bits & bit) != 0;
							if (passed.bits & bit)
								for (int i = 0; i < 8; ++i)
									values[start + i] += (sets[i] & bit) != 0;
						}
						if (!(passed.bits & bit))
							continue;
						for (int k = 0; k < 2; ++k)
							for (int king : b.locations(b.PCS(k == 0 ? role : opponent, 5))) {
								const int d = std::max(std::abs(square / 8 - king / 8), std::abs(square % 8 - king % 8));
								if (d != 3)
									values[64 + 2 * (d < 3 ? d : d - 1) + k] += 1;
							}
					}
				};
				collect(0, us, them, fp, fpass, fatt, eatt, fneighbours);
				collect(1, them, us, ep, epass, eatt, fatt, eneighbours);
				for (std::size_t i = 0; i < counts[0].size(); ++i)
					put(diff(b.NUM(counts[0][i]), b.NUM(counts[1][i])));
			}

			IS sq(IS role, int square) { return b.sq(role, square); }

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
				if (type == 4) {
					unsafe = b.OR(unsafe, b.attacks(opponent, 3));
				}
				return b.AND(b.NOT(own(role)), b.NOT(unsafe));
			}
			void emitMobility() {
				// Catalog entries mobility.knight.0 through mobility.queen.safe.
				constexpr std::array<int, 4> max{{8, 13, 14, 27}};
				constexpr std::array<int, 4> reference{{4, 5, 7, 12}};
				for (int type = 1; type <= 4; ++type) {
					const auto &f = b.mobility(us, type, mobilityArea(us, them), secondaryArea(us, them, type));
					const auto &e = b.mobility(them, type, mobilityArea(them, us), secondaryArea(them, us, type));
					for (int bucket = 0; bucket <= max[type - 1]; ++bucket) {
						if (bucket == reference[type - 1])
							continue;
						put(diff(b.NUM(f.counts[bucket]), b.NUM(e.counts[bucket])));
					}
					if (type >= 3)
						put(diff(b.NUM(f.secondary), b.NUM(e.secondary)));
				}
			}

			IS rookLine(IS role, IS opponent) {
				Sum terms;
				for (int square : b.squares(role, 3)) {
					const auto present = b.ANY(b.AND(b.PCS(role, 3), b.REL(role, 1ULL << square)));
					const auto queens = b.OR(b.PCS(role, 4), b.PCS(opponent, 4));
					terms.push_back(b.MUL(present, CNT(b.AND(queens, b.BB(fileMask(square % 8))))));
				}
				return b.sum(terms);
			}

			IS bishopXray(IS role, IS opponent) {
				Sum terms;
				for (int square : b.squares(role, 2)) {
					const auto present = b.ANY(b.AND(b.PCS(role, 2), b.REL(role, 1ULL << square)));
					terms.push_back(b.MUL(present, CNT(b.AND(b.PCS(opponent, 0), b.REL(role, diagonalMask(square))))));
				}
				return b.sum(terms);
			}

			void emitPieces() {
				// Catalog entries piece.minorBehindPawn.knight through piece.space.
				for (int type = 1; type <= 2; ++type) {
					const auto friendly = CNT(b.AND(b.PCS(us, type), b.SH(b.PCS(us, 0), them, 0)));
					const auto enemy = CNT(b.AND(b.PCS(them, type), b.SH(b.PCS(them, 0), us, 0)));
					put(diff(friendly, enemy));
				}
				constexpr std::uint64_t light = 0x55AA55AA55AA55AAULL;
				const auto bishoppawns = [&](IS role) {
					const auto l = b.REL(role, light);
					const auto d = b.NOT(l);
					return b.ADD(b.MUL(CNT(b.AND(b.PCS(role, 2), l)), CNT(b.AND(b.PCS(role, 0), l))), b.MUL(CNT(b.AND(b.PCS(role, 2), d)), CNT(b.AND(b.PCS(role, 0), d))));
				};
				put(diff(bishoppawns(us), bishoppawns(them)));
				put(diff(CNT(b.AND(b.PCS(us, 2), b.NOT(pawnAttacks(us)))), CNT(b.AND(b.PCS(them, 2), b.NOT(pawnAttacks(them))))));
				const auto central = fileMask(2) | fileMask(3) | fileMask(4) | fileMask(5);
				const auto fblocked = b.AND(b.PCS(us, 0), b.AND(b.SH(occ, us, 1), b.REL(us, central & (rankMask(1) | rankMask(2)))));
				const auto eblocked = b.AND(b.PCS(them, 0), b.AND(b.SH(occ, them, 1), b.REL(them, central & (rankMask(1) | rankMask(2)))));
				put(diff(b.MUL(CNT(b.PCS(us, 2)), CNT(fblocked)), b.MUL(CNT(b.PCS(them, 2)), CNT(eblocked))));
				put(diff(bishopXray(us, them), bishopXray(them, us)));
				put(diff(CNT(b.AND(b.PCS(us, 3), b.REL(us, rankMask(6)))), CNT(b.AND(b.PCS(them, 3), b.REL(them, rankMask(6))))));
				put(diff(rookLine(us, them), rookLine(them, us)));

				{
					IS fopen = z;
					IS eopen = z;
					IS fsemi = z;
					IS esemi = z;
					for (int file = 0; file < 8; ++file) {
						const auto mask = b.BB(fileMask(file));
						const auto frooks = CNT(b.AND(b.PCS(us, 3), mask));
						const auto erooks = CNT(b.AND(b.PCS(them, 3), mask));
						const auto fp = b.ANY(b.AND(b.PCS(us, 0), mask));
						const auto ep = b.ANY(b.AND(b.PCS(them, 0), mask));
						fopen = b.ADD(fopen, b.MUL(frooks, b.LNOT(b.LOR(fp, ep))));
						eopen = b.ADD(eopen, b.MUL(erooks, b.LNOT(b.LOR(fp, ep))));
						fsemi = b.ADD(fsemi, b.MUL(frooks, b.LAND(b.LNOT(fp), ep)));
						esemi = b.ADD(esemi, b.MUL(erooks, b.LAND(b.LNOT(ep), fp)));
					}
					put(diff(fopen, eopen));
					put(diff(fsemi, esemi));
				}

				const auto outposts = [&](IS role, IS opponent, int type) {
					const auto ranks = b.REL(role, rankMask(3) | rankMask(4) | rankMask(5));
					const auto future = pawnAttacks(opponent);
					const auto viable = b.AND(ranks, b.AND(pawnAttacks(role), b.NOT(b.fill(future, opponent))));
					return CNT(b.AND(b.PCS(role, type), viable));
				};
				for (int type = 1; type <= 2; ++type) {
					put(diff(outposts(us, them, type), outposts(them, us, type)));
				}

				const auto restricted = [&](IS role, IS opponent) {
					const auto guarded = strongSquares(role, opponent);
					return CNT(b.AND(b.AND(b.attacks(role), b.attacks(opponent)), b.NOT(guarded)));
				};
				put(diff(restricted(us, them), restricted(them, us)));
				constexpr std::uint64_t center = 0x00003C3C3C000000ULL;
				const auto space = [&](IS role, IS opponent) { return CNT(b.AND(b.AND(b.attacks(role), b.REL(role, center)), b.NOT(pawnAttacks(opponent)))); };
				put(diff(space(us, them), space(them, us)));
			}

			void emitThreats() {
				// Catalog entries threat.pawn.hanging through threat.queenPressure.rook.queenAbsent.
				const auto evalside = [&](IS role, IS opponent, int victim) {
					const auto targets = b.PCS(opponent, victim);
					const auto guarded = strongSquares(role, opponent);
					const auto weak = b.AND(targets, b.NOT(guarded));
					const auto hanging = b.AND(targets, b.AND(b.attacks(role), b.OR(b.NOT(b.attacks(opponent)), b.attacks2(role))));
					const auto minor = b.AND(weak, b.OR(b.attacks(role, 1), b.attacks(role, 2)));
					const auto rook = b.AND(weak, b.attacks(role, 3));
					const auto pushatt = shared(pushAttack, role, [&] {
						const auto pushed = b.AND(b.SH(b.PCS(role, 0), role, 0), b.NOT(occ));
						return b.OR(b.SH(pushed, role, 4), b.SH(pushed, role, 5));
					});
					return std::array<IS, 5>{CNT(hanging), CNT(b.AND(targets, pawnAttacks(role))), CNT(minor), CNT(rook), CNT(b.AND(targets, pushatt))};
				};
				const auto kingthreat = [&](IS role, IS opponent) {
					const auto targets = own(opponent);
					const auto guarded = strongSquares(role, opponent);
					return b.ANY(b.AND(b.AND(targets, b.NOT(guarded)), b.attacks(role, 5)));
				};
				const auto queenpressure = [&](IS role, IS opponent, int type) {
					Sum terms;
					const auto guarded = strongSquares(role, opponent);
					for (int square : b.squares(opponent, 4)) {
						const auto queen = b.ANY(b.AND(b.PCS(opponent, 4), b.REL(opponent, 1ULL << square)));
						const auto sources = b.AND(b.PCS(role, type), b.attackFrom(role, type, sq(opponent, square)));
						terms.push_back(b.MUL(queen, CNT(b.AND(sources, b.NOT(guarded)))));
					}
					return b.sum(terms);
				};

				for (int victim = 0; victim < 5; ++victim) {
					{
						const auto friendly = evalside(us, them, victim);
						const auto enemy = evalside(them, us, victim);
						put(diff(friendly[0], enemy[0]));
						put(diff(friendly[1], enemy[1]));
						put(diff(friendly[2], enemy[2]));
						put(diff(friendly[3], enemy[3]));
						put(diff(friendly[4], enemy[4]));
					}
				}
				put(diff(kingthreat(us, them), kingthreat(them, us)));
				{
					for (int type = 1; type <= 3; ++type) {
						const auto friendly = queenpressure(us, them, type);
						const auto enemy = queenpressure(them, us, type);
						const auto fqueenless = b.EQ(CNT(b.PCS(us, 4)), z);
						const auto equeenless = b.EQ(CNT(b.PCS(them, 4)), z);
						put(diff(b.MUL(friendly, b.LNOT(fqueenless)), b.MUL(enemy, b.LNOT(equeenless))));
						put(diff(b.MUL(friendly, fqueenless), b.MUL(enemy, equeenless)));
					}
				}
			}

			IS ringAttacks(IS attacker, IS defender, int type, int distance) {
				// The king attack map is exactly its inner ring on a legal board.
				return distance == 1 ? CNT(b.AND(b.attacks(attacker, type), b.attacks(defender, 5)))
				                     : CNT(b.AND(b.attacks(attacker, type), kingRegion(defender, false)));
			}

			IS potentialChecks(IS attacker, IS defender, int type) {
				Sum terms;
				for (int square : b.squares(defender, 5)) {
					const auto king = b.ANY(b.AND(b.PCS(defender, 5), b.REL(defender, 1ULL << square)));
					// Reversing the defender's pawn direction gives the squares from which an attacking pawn checks the king.
					const auto geometry = b.attackFrom(type == 0 ? defender : attacker, type, sq(defender, square));
					const auto destinations = b.AND(geometry, b.AND(b.attacks(attacker, type), b.NOT(own(attacker))));
					terms.push_back(b.MUL(king, CNT(destinations)));
				}
				return b.sum(terms);
			}

			IS escapes(IS defender, IS attacker) {
				return CNT(b.AND(b.attacks(defender, 5), b.AND(b.NOT(own(defender)), b.NOT(b.attacks(attacker)))));
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
							region = 0;
							for (int file = std::max(0, square % 8 - 1); file <= std::min(7, square % 8 + 1); ++file)
								region |= fileMask(file);
						}
						mask = b.OR(mask, b.REL(role, region));
					}
					return mask;
				});
			}

			IS flank(IS map, IS defender) {
				return CNT(b.AND(map, kingRegion(defender, true)));
			}

			void emitKings() {
				// Catalog entries king.innerRing.pawn through king.castlingQueenSide.
				Sum friendly;
				Sum enemy;
				for (int type = 0; type < 5; ++type) {
					const auto fi = ringAttacks(us, them, type, 1);
					const auto ei = ringAttacks(them, us, type, 1);
					friendly.push_back(fi);
					enemy.push_back(ei);
					put(diff(fi, ei));
					put(diff(ringAttacks(us, them, type, 2), ringAttacks(them, us, type, 2)));
					put(diff(potentialChecks(us, them, type), potentialChecks(them, us, type)));
				}
				const auto fp = b.sum(friendly);
				const auto ep = b.sum(enemy);
				for (const int threshold : {2, 4, 6, 8}) {
					put(diff(b.GE(fp, b.NUM(threshold)), b.GE(ep, b.NUM(threshold))));
				}
				put(diff(escapes(us, them), escapes(them, us)));
				{
					for (int distance = 1; distance <= 3; ++distance) {
						put(diff(kingPawns(us, them, distance, true), kingPawns(them, us, distance, true)));
						put(diff(kingPawns(them, us, distance, false), kingPawns(us, them, distance, false)));
					}
				}
				put(diff(kingOpenFiles(us), kingOpenFiles(them)));
				put(diff(flank(b.attacks(us), us), flank(b.attacks(them), them)));
				put(diff(flank(b.attacks(us), them), flank(b.attacks(them), us)));
				put(diff(flank(b.attacks2(us), us), flank(b.attacks2(them), them)));
				put(diff(flank(b.attacks2(us), them), flank(b.attacks2(them), us)));
				const auto fblockedstorm = b.AND(b.PCS(us, 0), b.SH(b.PCS(them, 0), them, 0));
				const auto eblockedstorm = b.AND(b.PCS(them, 0), b.SH(b.PCS(us, 0), us, 0));
				put(diff(flank(fblockedstorm, us), flank(eblockedstorm, them)));
				const auto fnoqueen = b.EQ(CNT(b.PCS(us, 4)), z);
				const auto enoqueen = b.EQ(CNT(b.PCS(them, 4)), z);
				put(diff(b.MUL(fp, enoqueen), b.MUL(ep, fnoqueen)));
				for (int side = 0; side < 2; ++side) {
					const auto friendlyright = b.RIGHT(us, side);
					const auto enemyright = b.RIGHT(them, side);
					put(diff(friendlyright, enemyright));
				}
			}

			IS nonPawnMaterial(IS role) {
				IS total = z;
				constexpr std::array<int, 5> value{{0, 3, 3, 5, 9}};
				for (int type = 1; type <= 4; ++type) {
					total = b.ADD(total, b.MUL(b.NUM(value[static_cast<std::size_t>(type)]), CNT(b.PCS(role, type))));
				}
				return total;
			}

			void emitEndgames() {
				// Catalog entries endgame.oppositeBishops through endgame.oppositeBishopPassers.
				constexpr std::uint64_t light = 0x55AA55AA55AA55AAULL;
				const auto onefb = b.EQ(CNT(b.PCS(us, 2)), o);
				const auto oneeb = b.EQ(CNT(b.PCS(them, 2)), o);
				const auto opposite = b.LOR(b.LAND(b.ANY(b.AND(b.PCS(us, 2), b.BB(light))), b.ANY(b.AND(b.PCS(them, 2), b.NOT(b.BB(light))))),
				    b.LAND(b.ANY(b.AND(b.PCS(us, 2), b.NOT(b.BB(light)))), b.ANY(b.AND(b.PCS(them, 2), b.BB(light)))));
				const auto fpawns = CNT(b.PCS(us, 0));
				const auto epawns = CNT(b.PCS(them, 0));
				const auto material = b.ADD(diff(nonPawnMaterial(us), nonPawnMaterial(them)), diff(fpawns, epawns));
				const auto positive = b.GT(material, z);
				const auto negative = b.LT(material, z);
				const auto direction = diff(positive, negative);
				put(b.MUL(b.LAND(b.LAND(onefb, oneeb), opposite), direction));
				const auto fpawnless = b.EQ(fpawns, z);
				const auto epawnless = b.EQ(epawns, z);
				const auto thin = b.LE(b.ABS(diff(nonPawnMaterial(us), nonPawnMaterial(them))), o);
				put(diff(b.LAND(b.LAND(positive, fpawnless), thin), b.LAND(b.LAND(negative, epawnless), thin)));

				IS symmetric = z;
				IS asymmetric = z;
				for (int file = 0; file < 8; ++file) {
					const auto fp = b.ANY(b.AND(b.PCS(us, 0), b.BB(fileMask(file))));
					const auto ep = b.ANY(b.AND(b.PCS(them, 0), b.BB(fileMask(file))));
					symmetric = b.ADD(symmetric, b.LAND(fp, ep));
					asymmetric = b.ADD(asymmetric, b.LOR(b.LAND(fp, b.LNOT(ep)), b.LAND(ep, b.LNOT(fp))));
				}
				put(b.MUL(direction, b.ADD(fpawns, epawns)));
				put(b.MUL(direction, symmetric));
				put(b.MUL(direction, asymmetric));
				put(b.MUL(direction, b.EQ(phase, z)));
				const auto strongpawns = diff(b.MUL(positive, fpawns), b.MUL(negative, epawns));
				put(strongpawns);
				const auto strongpassers = diff(b.MUL(positive, CNT(passedPawns(us, them))), b.MUL(negative, CNT(passedPawns(them, us))));
				put(b.MUL(b.LAND(b.LAND(onefb, oneeb), opposite), strongpassers));
			}

			B b;
			Pair owned, pawnAttack, passedMap, strongMap, pawnFile, clearPath, safePath, wideAttack, stopPath, controlMap, mobilityAreaCache, pushAttack, kingRing, kingFlank;
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
		if (weights.size() != FormulaSet::fixed().size())
			throw std::invalid_argument("formula weight count does not match the fixed formula set");
		Weighted sink{weights};
		Formulas<Runtime<Weighted>>(Runtime(board, sink)).execute();
		return static_cast<float>(sink.value);
	}

	float detail::score(const chess::Board &board, std::span<const float> base, std::span<const std::uint16_t> rows,
	    std::span<const std::uint16_t> conditions, std::span<const float> relations) {
		if (base.size() != FormulaSet::fixed().size() || rows.size() != conditions.size() || rows.size() != relations.size())
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
	namespace {
		constexpr std::array<std::string_view, kFormulaCount> catalogNames{{
			"tempo",
			"material.pawn",
			"material.knight",
			"material.bishop",
			"material.rook",
			"material.queen",
			"pst.pawn.a1",
			"pst.pawn.b1",
			"pst.pawn.c1",
			"pst.pawn.d1",
			"pst.pawn.e1",
			"pst.pawn.f1",
			"pst.pawn.g1",
			"pst.pawn.h1",
			"pst.pawn.a2",
			"pst.pawn.b2",
			"pst.pawn.c2",
			"pst.pawn.d2",
			"pst.pawn.e2",
			"pst.pawn.f2",
			"pst.pawn.g2",
			"pst.pawn.h2",
			"pst.pawn.a3",
			"pst.pawn.b3",
			"pst.pawn.c3",
			"pst.pawn.d3",
			"pst.pawn.e3",
			"pst.pawn.f3",
			"pst.pawn.g3",
			"pst.pawn.h3",
			"pst.pawn.a4",
			"pst.pawn.b4",
			"pst.pawn.c4",
			"pst.pawn.d4",
			"pst.pawn.e4",
			"pst.pawn.f4",
			"pst.pawn.g4",
			"pst.pawn.h4",
			"pst.pawn.a5",
			"pst.pawn.b5",
			"pst.pawn.c5",
			"pst.pawn.d5",
			"pst.pawn.e5",
			"pst.pawn.f5",
			"pst.pawn.g5",
			"pst.pawn.h5",
			"pst.pawn.a6",
			"pst.pawn.b6",
			"pst.pawn.c6",
			"pst.pawn.d6",
			"pst.pawn.e6",
			"pst.pawn.f6",
			"pst.pawn.g6",
			"pst.pawn.h6",
			"pst.pawn.a7",
			"pst.pawn.b7",
			"pst.pawn.c7",
			"pst.pawn.d7",
			"pst.pawn.e7",
			"pst.pawn.f7",
			"pst.pawn.g7",
			"pst.pawn.h7",
			"pst.pawn.a8",
			"pst.pawn.b8",
			"pst.pawn.c8",
			"pst.pawn.d8",
			"pst.pawn.e8",
			"pst.pawn.f8",
			"pst.pawn.g8",
			"pst.pawn.h8",
			"pst.knight.a1",
			"pst.knight.b1",
			"pst.knight.c1",
			"pst.knight.d1",
			"pst.knight.e1",
			"pst.knight.f1",
			"pst.knight.g1",
			"pst.knight.h1",
			"pst.knight.a2",
			"pst.knight.b2",
			"pst.knight.c2",
			"pst.knight.d2",
			"pst.knight.e2",
			"pst.knight.f2",
			"pst.knight.g2",
			"pst.knight.h2",
			"pst.knight.a3",
			"pst.knight.b3",
			"pst.knight.c3",
			"pst.knight.d3",
			"pst.knight.e3",
			"pst.knight.f3",
			"pst.knight.g3",
			"pst.knight.h3",
			"pst.knight.a4",
			"pst.knight.b4",
			"pst.knight.c4",
			"pst.knight.d4",
			"pst.knight.e4",
			"pst.knight.f4",
			"pst.knight.g4",
			"pst.knight.h4",
			"pst.knight.a5",
			"pst.knight.b5",
			"pst.knight.c5",
			"pst.knight.d5",
			"pst.knight.e5",
			"pst.knight.f5",
			"pst.knight.g5",
			"pst.knight.h5",
			"pst.knight.a6",
			"pst.knight.b6",
			"pst.knight.c6",
			"pst.knight.d6",
			"pst.knight.e6",
			"pst.knight.f6",
			"pst.knight.g6",
			"pst.knight.h6",
			"pst.knight.a7",
			"pst.knight.b7",
			"pst.knight.c7",
			"pst.knight.d7",
			"pst.knight.e7",
			"pst.knight.f7",
			"pst.knight.g7",
			"pst.knight.h7",
			"pst.knight.a8",
			"pst.knight.b8",
			"pst.knight.c8",
			"pst.knight.d8",
			"pst.knight.e8",
			"pst.knight.f8",
			"pst.knight.g8",
			"pst.knight.h8",
			"pst.bishop.a1",
			"pst.bishop.b1",
			"pst.bishop.c1",
			"pst.bishop.d1",
			"pst.bishop.e1",
			"pst.bishop.f1",
			"pst.bishop.g1",
			"pst.bishop.h1",
			"pst.bishop.a2",
			"pst.bishop.b2",
			"pst.bishop.c2",
			"pst.bishop.d2",
			"pst.bishop.e2",
			"pst.bishop.f2",
			"pst.bishop.g2",
			"pst.bishop.h2",
			"pst.bishop.a3",
			"pst.bishop.b3",
			"pst.bishop.c3",
			"pst.bishop.d3",
			"pst.bishop.e3",
			"pst.bishop.f3",
			"pst.bishop.g3",
			"pst.bishop.h3",
			"pst.bishop.a4",
			"pst.bishop.b4",
			"pst.bishop.c4",
			"pst.bishop.d4",
			"pst.bishop.e4",
			"pst.bishop.f4",
			"pst.bishop.g4",
			"pst.bishop.h4",
			"pst.bishop.a5",
			"pst.bishop.b5",
			"pst.bishop.c5",
			"pst.bishop.d5",
			"pst.bishop.e5",
			"pst.bishop.f5",
			"pst.bishop.g5",
			"pst.bishop.h5",
			"pst.bishop.a6",
			"pst.bishop.b6",
			"pst.bishop.c6",
			"pst.bishop.d6",
			"pst.bishop.e6",
			"pst.bishop.f6",
			"pst.bishop.g6",
			"pst.bishop.h6",
			"pst.bishop.a7",
			"pst.bishop.b7",
			"pst.bishop.c7",
			"pst.bishop.d7",
			"pst.bishop.e7",
			"pst.bishop.f7",
			"pst.bishop.g7",
			"pst.bishop.h7",
			"pst.bishop.a8",
			"pst.bishop.b8",
			"pst.bishop.c8",
			"pst.bishop.d8",
			"pst.bishop.e8",
			"pst.bishop.f8",
			"pst.bishop.g8",
			"pst.bishop.h8",
			"pst.rook.a1",
			"pst.rook.b1",
			"pst.rook.c1",
			"pst.rook.d1",
			"pst.rook.e1",
			"pst.rook.f1",
			"pst.rook.g1",
			"pst.rook.h1",
			"pst.rook.a2",
			"pst.rook.b2",
			"pst.rook.c2",
			"pst.rook.d2",
			"pst.rook.e2",
			"pst.rook.f2",
			"pst.rook.g2",
			"pst.rook.h2",
			"pst.rook.a3",
			"pst.rook.b3",
			"pst.rook.c3",
			"pst.rook.d3",
			"pst.rook.e3",
			"pst.rook.f3",
			"pst.rook.g3",
			"pst.rook.h3",
			"pst.rook.a4",
			"pst.rook.b4",
			"pst.rook.c4",
			"pst.rook.d4",
			"pst.rook.e4",
			"pst.rook.f4",
			"pst.rook.g4",
			"pst.rook.h4",
			"pst.rook.a5",
			"pst.rook.b5",
			"pst.rook.c5",
			"pst.rook.d5",
			"pst.rook.e5",
			"pst.rook.f5",
			"pst.rook.g5",
			"pst.rook.h5",
			"pst.rook.a6",
			"pst.rook.b6",
			"pst.rook.c6",
			"pst.rook.d6",
			"pst.rook.e6",
			"pst.rook.f6",
			"pst.rook.g6",
			"pst.rook.h6",
			"pst.rook.a7",
			"pst.rook.b7",
			"pst.rook.c7",
			"pst.rook.d7",
			"pst.rook.e7",
			"pst.rook.f7",
			"pst.rook.g7",
			"pst.rook.h7",
			"pst.rook.a8",
			"pst.rook.b8",
			"pst.rook.c8",
			"pst.rook.d8",
			"pst.rook.e8",
			"pst.rook.f8",
			"pst.rook.g8",
			"pst.rook.h8",
			"pst.queen.a1",
			"pst.queen.b1",
			"pst.queen.c1",
			"pst.queen.d1",
			"pst.queen.e1",
			"pst.queen.f1",
			"pst.queen.g1",
			"pst.queen.h1",
			"pst.queen.a2",
			"pst.queen.b2",
			"pst.queen.c2",
			"pst.queen.d2",
			"pst.queen.e2",
			"pst.queen.f2",
			"pst.queen.g2",
			"pst.queen.h2",
			"pst.queen.a3",
			"pst.queen.b3",
			"pst.queen.c3",
			"pst.queen.d3",
			"pst.queen.e3",
			"pst.queen.f3",
			"pst.queen.g3",
			"pst.queen.h3",
			"pst.queen.a4",
			"pst.queen.b4",
			"pst.queen.c4",
			"pst.queen.d4",
			"pst.queen.e4",
			"pst.queen.f4",
			"pst.queen.g4",
			"pst.queen.h4",
			"pst.queen.a5",
			"pst.queen.b5",
			"pst.queen.c5",
			"pst.queen.d5",
			"pst.queen.e5",
			"pst.queen.f5",
			"pst.queen.g5",
			"pst.queen.h5",
			"pst.queen.a6",
			"pst.queen.b6",
			"pst.queen.c6",
			"pst.queen.d6",
			"pst.queen.e6",
			"pst.queen.f6",
			"pst.queen.g6",
			"pst.queen.h6",
			"pst.queen.a7",
			"pst.queen.b7",
			"pst.queen.c7",
			"pst.queen.d7",
			"pst.queen.e7",
			"pst.queen.f7",
			"pst.queen.g7",
			"pst.queen.h7",
			"pst.queen.a8",
			"pst.queen.b8",
			"pst.queen.c8",
			"pst.queen.d8",
			"pst.queen.e8",
			"pst.queen.f8",
			"pst.queen.g8",
			"pst.queen.h8",
			"pst.king.a1",
			"pst.king.b1",
			"pst.king.c1",
			"pst.king.d1",
			"pst.king.e1",
			"pst.king.f1",
			"pst.king.g1",
			"pst.king.h1",
			"pst.king.a2",
			"pst.king.b2",
			"pst.king.c2",
			"pst.king.d2",
			"pst.king.e2",
			"pst.king.f2",
			"pst.king.g2",
			"pst.king.h2",
			"pst.king.a3",
			"pst.king.b3",
			"pst.king.c3",
			"pst.king.d3",
			"pst.king.e3",
			"pst.king.f3",
			"pst.king.g3",
			"pst.king.h3",
			"pst.king.a4",
			"pst.king.b4",
			"pst.king.c4",
			"pst.king.d4",
			"pst.king.e4",
			"pst.king.f4",
			"pst.king.g4",
			"pst.king.h4",
			"pst.king.a5",
			"pst.king.b5",
			"pst.king.c5",
			"pst.king.d5",
			"pst.king.e5",
			"pst.king.f5",
			"pst.king.g5",
			"pst.king.h5",
			"pst.king.a6",
			"pst.king.b6",
			"pst.king.c6",
			"pst.king.d6",
			"pst.king.e6",
			"pst.king.f6",
			"pst.king.g6",
			"pst.king.h6",
			"pst.king.a7",
			"pst.king.b7",
			"pst.king.c7",
			"pst.king.d7",
			"pst.king.e7",
			"pst.king.f7",
			"pst.king.g7",
			"pst.king.h7",
			"pst.king.a8",
			"pst.king.b8",
			"pst.king.c8",
			"pst.king.d8",
			"pst.king.e8",
			"pst.king.f8",
			"pst.king.g8",
			"pst.king.h8",
			"bishopPair",
			"pawn.passed.rank2",
			"pawn.clear.rank2",
			"pawn.safe.rank2",
			"pawn.wideSafe.rank2",
			"pawn.controlledPush.rank2",
			"pawn.supported.rank2",
			"pawn.blocked.rank2",
			"pawn.connected.rank2",
			"pawn.phalanx.rank2",
			"pawn.pawnSupported.rank2",
			"pawn.passed.rank3",
			"pawn.clear.rank3",
			"pawn.safe.rank3",
			"pawn.wideSafe.rank3",
			"pawn.controlledPush.rank3",
			"pawn.supported.rank3",
			"pawn.blocked.rank3",
			"pawn.connected.rank3",
			"pawn.phalanx.rank3",
			"pawn.pawnSupported.rank3",
			"pawn.passed.rank4",
			"pawn.clear.rank4",
			"pawn.safe.rank4",
			"pawn.wideSafe.rank4",
			"pawn.controlledPush.rank4",
			"pawn.supported.rank4",
			"pawn.blocked.rank4",
			"pawn.connected.rank4",
			"pawn.phalanx.rank4",
			"pawn.pawnSupported.rank4",
			"pawn.passed.rank5",
			"pawn.clear.rank5",
			"pawn.safe.rank5",
			"pawn.wideSafe.rank5",
			"pawn.controlledPush.rank5",
			"pawn.supported.rank5",
			"pawn.blocked.rank5",
			"pawn.connected.rank5",
			"pawn.phalanx.rank5",
			"pawn.pawnSupported.rank5",
			"pawn.passed.rank6",
			"pawn.clear.rank6",
			"pawn.safe.rank6",
			"pawn.wideSafe.rank6",
			"pawn.controlledPush.rank6",
			"pawn.supported.rank6",
			"pawn.blocked.rank6",
			"pawn.connected.rank6",
			"pawn.phalanx.rank6",
			"pawn.pawnSupported.rank6",
			"pawn.passed.rank7",
			"pawn.clear.rank7",
			"pawn.safe.rank7",
			"pawn.wideSafe.rank7",
			"pawn.controlledPush.rank7",
			"pawn.supported.rank7",
			"pawn.blocked.rank7",
			"pawn.connected.rank7",
			"pawn.phalanx.rank7",
			"pawn.pawnSupported.rank7",
			"pawn.doubled",
			"pawn.isolated",
			"pawn.backward",
			"pawn.islands",
			"passer.friendlyKing.distance0",
			"passer.enemyKing.distance0",
			"passer.friendlyKing.distance1",
			"passer.enemyKing.distance1",
			"passer.friendlyKing.distance2",
			"passer.enemyKing.distance2",
			"passer.friendlyKing.distance4",
			"passer.enemyKing.distance4",
			"passer.friendlyKing.distance5",
			"passer.enemyKing.distance5",
			"passer.friendlyKing.distance6",
			"passer.enemyKing.distance6",
			"passer.friendlyKing.distance7",
			"passer.enemyKing.distance7",
			"mobility.knight.0",
			"mobility.knight.1",
			"mobility.knight.2",
			"mobility.knight.3",
			"mobility.knight.5",
			"mobility.knight.6",
			"mobility.knight.7",
			"mobility.knight.8",
			"mobility.bishop.0",
			"mobility.bishop.1",
			"mobility.bishop.2",
			"mobility.bishop.3",
			"mobility.bishop.4",
			"mobility.bishop.6",
			"mobility.bishop.7",
			"mobility.bishop.8",
			"mobility.bishop.9",
			"mobility.bishop.10",
			"mobility.bishop.11",
			"mobility.bishop.12",
			"mobility.bishop.13",
			"mobility.rook.0",
			"mobility.rook.1",
			"mobility.rook.2",
			"mobility.rook.3",
			"mobility.rook.4",
			"mobility.rook.5",
			"mobility.rook.6",
			"mobility.rook.8",
			"mobility.rook.9",
			"mobility.rook.10",
			"mobility.rook.11",
			"mobility.rook.12",
			"mobility.rook.13",
			"mobility.rook.14",
			"mobility.rook.safe",
			"mobility.queen.0",
			"mobility.queen.1",
			"mobility.queen.2",
			"mobility.queen.3",
			"mobility.queen.4",
			"mobility.queen.5",
			"mobility.queen.6",
			"mobility.queen.7",
			"mobility.queen.8",
			"mobility.queen.9",
			"mobility.queen.10",
			"mobility.queen.11",
			"mobility.queen.13",
			"mobility.queen.14",
			"mobility.queen.15",
			"mobility.queen.16",
			"mobility.queen.17",
			"mobility.queen.18",
			"mobility.queen.19",
			"mobility.queen.20",
			"mobility.queen.21",
			"mobility.queen.22",
			"mobility.queen.23",
			"mobility.queen.24",
			"mobility.queen.25",
			"mobility.queen.26",
			"mobility.queen.27",
			"mobility.queen.safe",
			"piece.minorBehindPawn.knight",
			"piece.minorBehindPawn.bishop",
			"piece.bishopPawnColor",
			"piece.unprotectedBishop",
			"piece.badBishop",
			"piece.bishopXrayPawn",
			"piece.rookSeventh",
			"piece.rookQueenFile",
			"piece.rookOpenFile",
			"piece.rookSemiOpenFile",
			"piece.outpostKnight",
			"piece.outpostBishop",
			"piece.restricted",
			"piece.space",
			"threat.pawn.hanging",
			"threat.pawn.pawn",
			"threat.pawn.minor",
			"threat.pawn.rook",
			"threat.pawn.pawnPush",
			"threat.knight.hanging",
			"threat.knight.pawn",
			"threat.knight.minor",
			"threat.knight.rook",
			"threat.knight.pawnPush",
			"threat.bishop.hanging",
			"threat.bishop.pawn",
			"threat.bishop.minor",
			"threat.bishop.rook",
			"threat.bishop.pawnPush",
			"threat.rook.hanging",
			"threat.rook.pawn",
			"threat.rook.minor",
			"threat.rook.rook",
			"threat.rook.pawnPush",
			"threat.queen.hanging",
			"threat.queen.pawn",
			"threat.queen.minor",
			"threat.queen.rook",
			"threat.queen.pawnPush",
			"threat.king",
			"threat.queenPressure.knight.queenPresent",
			"threat.queenPressure.knight.queenAbsent",
			"threat.queenPressure.bishop.queenPresent",
			"threat.queenPressure.bishop.queenAbsent",
			"threat.queenPressure.rook.queenPresent",
			"threat.queenPressure.rook.queenAbsent",
			"king.innerRing.pawn",
			"king.outerRing.pawn",
			"king.potentialCheck.pawn",
			"king.innerRing.knight",
			"king.outerRing.knight",
			"king.potentialCheck.knight",
			"king.innerRing.bishop",
			"king.outerRing.bishop",
			"king.potentialCheck.bishop",
			"king.innerRing.rook",
			"king.outerRing.rook",
			"king.potentialCheck.rook",
			"king.innerRing.queen",
			"king.outerRing.queen",
			"king.potentialCheck.queen",
			"king.pressureAtLeast2",
			"king.pressureAtLeast4",
			"king.pressureAtLeast6",
			"king.pressureAtLeast8",
			"king.escape",
			"king.shelter.distance1",
			"king.storm.distance1",
			"king.shelter.distance2",
			"king.storm.distance2",
			"king.shelter.distance3",
			"king.storm.distance3",
			"king.openFiles",
			"king.friendlyAttackFlank",
			"king.enemyAttackFlank",
			"king.friendlyDoubleAttackFlank",
			"king.enemyDoubleAttackFlank",
			"king.blockedStorm",
			"king.pressureWithoutQueen",
			"king.castlingKingSide",
			"king.castlingQueenSide",
			"endgame.oppositeBishops",
			"endgame.pawnlessThinAdvantage",
			"endgame.totalPawns",
			"endgame.symmetricPawnFiles",
			"endgame.asymmetricPawnFiles",
			"endgame.bareKings",
			"endgame.strongSidePawns",
			"endgame.oppositeBishopPassers",
		}};

		constexpr std::array<float, kFormulaCount> catalogInitial{{
			0.16F,
			0.22F,
			0.773F,
			0.828F,
			1.2365F,
			2.4515F,
			-0.013125F,
			-0.009375F,
			-0.005625F,
			-0.001875F,
			-0.001875F,
			-0.005625F,
			-0.009375F,
			-0.013125F,
			0.006875F,
			0.010625F,
			0.014375F,
			0.018125F,
			0.018125F,
			0.014375F,
			0.010625F,
			0.006875F,
			0.026875F,
			0.030625F,
			0.034375F,
			0.038125F,
			0.038125F,
			0.034375F,
			0.030625F,
			0.026875F,
			0.046875F,
			0.050625F,
			0.054375F,
			0.058125F,
			0.058125F,
			0.054375F,
			0.050625F,
			0.046875F,
			0.063125F,
			0.066875F,
			0.070625F,
			0.074375F,
			0.074375F,
			0.070625F,
			0.066875F,
			0.063125F,
			0.075625F,
			0.079375F,
			0.083125F,
			0.086875F,
			0.086875F,
			0.083125F,
			0.079375F,
			0.075625F,
			0.088125F,
			0.091875F,
			0.095625F,
			0.099375F,
			0.099375F,
			0.095625F,
			0.091875F,
			0.088125F,
			0.100625F,
			0.104375F,
			0.108125F,
			0.111875F,
			0.111875F,
			0.108125F,
			0.104375F,
			0.100625F,
			-0.07875F,
			-0.05625F,
			-0.03375F,
			-0.01125F,
			-0.01125F,
			-0.03375F,
			-0.05625F,
			-0.07875F,
			-0.05625F,
			-0.03375F,
			-0.01125F,
			0.01125F,
			0.01125F,
			-0.01125F,
			-0.03375F,
			-0.05625F,
			-0.03375F,
			-0.01125F,
			0.01125F,
			0.03375F,
			0.03375F,
			0.01125F,
			-0.01125F,
			-0.03375F,
			-0.01125F,
			0.01125F,
			0.03375F,
			0.05625F,
			0.05625F,
			0.03375F,
			0.01125F,
			-0.01125F,
			-0.01125F,
			0.01125F,
			0.03375F,
			0.05625F,
			0.05625F,
			0.03375F,
			0.01125F,
			-0.01125F,
			-0.03375F,
			-0.01125F,
			0.01125F,
			0.03375F,
			0.03375F,
			0.01125F,
			-0.01125F,
			-0.03375F,
			-0.05625F,
			-0.03375F,
			-0.01125F,
			0.01125F,
			0.01125F,
			-0.01125F,
			-0.03375F,
			-0.05625F,
			-0.07875F,
			-0.05625F,
			-0.03375F,
			-0.01125F,
			-0.01125F,
			-0.03375F,
			-0.05625F,
			-0.07875F,
			-0.0525F,
			-0.0375F,
			-0.0225F,
			-0.0075F,
			-0.0075F,
			-0.0225F,
			-0.0375F,
			-0.0525F,
			-0.0375F,
			-0.0225F,
			-0.0075F,
			0.0075F,
			0.0075F,
			-0.0075F,
			-0.0225F,
			-0.0375F,
			-0.0225F,
			-0.0075F,
			0.0075F,
			0.0225F,
			0.0225F,
			0.0075F,
			-0.0075F,
			-0.0225F,
			-0.0075F,
			0.0075F,
			0.0225F,
			0.0375F,
			0.0375F,
			0.0225F,
			0.0075F,
			-0.0075F,
			-0.0075F,
			0.0075F,
			0.0225F,
			0.0375F,
			0.0375F,
			0.0225F,
			0.0075F,
			-0.0075F,
			-0.0225F,
			-0.0075F,
			0.0075F,
			0.0225F,
			0.0225F,
			0.0075F,
			-0.0075F,
			-0.0225F,
			-0.0375F,
			-0.0225F,
			-0.0075F,
			0.0075F,
			0.0075F,
			-0.0075F,
			-0.0225F,
			-0.0375F,
			-0.0525F,
			-0.0375F,
			-0.0225F,
			-0.0075F,
			-0.0075F,
			-0.0225F,
			-0.0375F,
			-0.0525F,
			0.0F,
			0.0F,
			0.0F,
			0.0F,
			0.0F,
			0.0F,
			0.0F,
			0.0F,
			0.003125F,
			0.003125F,
			0.003125F,
			0.003125F,
			0.003125F,
			0.003125F,
			0.003125F,
			0.003125F,
			0.00625F,
			0.00625F,
			0.00625F,
			0.00625F,
			0.00625F,
			0.00625F,
			0.00625F,
			0.00625F,
			0.009375F,
			0.009375F,
			0.009375F,
			0.009375F,
			0.009375F,
			0.009375F,
			0.009375F,
			0.009375F,
			0.0125F,
			0.0125F,
			0.0125F,
			0.0125F,
			0.0125F,
			0.0125F,
			0.0125F,
			0.0125F,
			0.015625F,
			0.015625F,
			0.015625F,
			0.015625F,
			0.015625F,
			0.015625F,
			0.015625F,
			0.015625F,
			0.035F,
			0.035F,
			0.035F,
			0.035F,
			0.035F,
			0.035F,
			0.035F,
			0.035F,
			0.021875F,
			0.021875F,
			0.021875F,
			0.021875F,
			0.021875F,
			0.021875F,
			0.021875F,
			0.021875F,
			-0.02625F,
			-0.01875F,
			-0.01125F,
			-0.00375F,
			-0.00375F,
			-0.01125F,
			-0.01875F,
			-0.02625F,
			-0.01875F,
			-0.01125F,
			-0.00375F,
			0.00375F,
			0.00375F,
			-0.00375F,
			-0.01125F,
			-0.01875F,
			-0.01125F,
			-0.00375F,
			0.00375F,
			0.01125F,
			0.01125F,
			0.00375F,
			-0.00375F,
			-0.01125F,
			-0.00375F,
			0.00375F,
			0.01125F,
			0.01875F,
			0.01875F,
			0.01125F,
			0.00375F,
			-0.00375F,
			-0.00375F,
			0.00375F,
			0.01125F,
			0.01875F,
			0.01875F,
			0.01125F,
			0.00375F,
			-0.00375F,
			-0.01125F,
			-0.00375F,
			0.00375F,
			0.01125F,
			0.01125F,
			0.00375F,
			-0.00375F,
			-0.01125F,
			-0.01875F,
			-0.01125F,
			-0.00375F,
			0.00375F,
			0.00375F,
			-0.00375F,
			-0.01125F,
			-0.01875F,
			-0.02625F,
			-0.01875F,
			-0.01125F,
			-0.00375F,
			-0.00375F,
			-0.01125F,
			-0.01875F,
			-0.02625F,
			-0.02F,
			-0.01F,
			0.0F,
			0.01F,
			0.01F,
			0.0F,
			-0.01F,
			-0.02F,
			-0.03F,
			-0.02F,
			-0.01F,
			0.0F,
			0.0F,
			-0.01F,
			-0.02F,
			-0.03F,
			-0.025F,
			-0.015F,
			-0.005F,
			0.005F,
			0.005F,
			-0.005F,
			-0.015F,
			-0.025F,
			-0.02F,
			-0.01F,
			0.0F,
			0.01F,
			0.01F,
			0.0F,
			-0.01F,
			-0.02F,
			-0.025F,
			-0.015F,
			-0.005F,
			0.005F,
			0.005F,
			-0.005F,
			-0.015F,
			-0.025F,
			-0.04F,
			-0.03F,
			-0.02F,
			-0.01F,
			-0.01F,
			-0.02F,
			-0.03F,
			-0.04F,
			-0.055F,
			-0.045F,
			-0.035F,
			-0.025F,
			-0.025F,
			-0.035F,
			-0.045F,
			-0.055F,
			-0.07F,
			-0.06F,
			-0.05F,
			-0.04F,
			-0.04F,
			-0.05F,
			-0.06F,
			-0.07F,
			0.15F,
			0.0375F,
			0.01375F,
			0.01375F,
			0.01F,
			0.0125F,
			0.0175F,
			-0.01375F,
			0.00875F,
			0.0175F,
			0.0175F,
			0.065F,
			0.0225F,
			0.02375F,
			0.016875F,
			0.02F,
			0.02875F,
			-0.025F,
			0.015F,
			0.02625F,
			0.02375F,
			0.1475F,
			0.03125F,
			0.03375F,
			0.02375F,
			0.0275F,
			0.04F,
			-0.03625F,
			0.02125F,
			0.035F,
			0.03F,
			0.285F,
			0.04F,
			0.04375F,
			0.030625F,
			0.035F,
			0.05125F,
			-0.0475F,
			0.0275F,
			0.04375F,
			0.03625F,
			0.4775F,
			0.04875F,
			0.05375F,
			0.0375F,
			0.0425F,
			0.0625F,
			-0.05875F,
			0.03375F,
			0.0525F,
			0.0425F,
			0.725F,
			0.0575F,
			0.06375F,
			0.044375F,
			0.05F,
			0.07375F,
			-0.07F,
			0.04F,
			0.06125F,
			0.04875F,
			-0.035F,
			-0.0275F,
			-0.0325F,
			-0.02F,
			0.006F,
			-0.006F,
			0.004F,
			-0.004F,
			0.002F,
			-0.002F,
			-0.002F,
			0.002F,
			-0.004F,
			0.004F,
			-0.006F,
			0.006F,
			-0.008F,
			0.008F,
			-0.108F,
			-0.081F,
			-0.054F,
			-0.027F,
			0.027F,
			0.054F,
			0.081F,
			0.108F,
			-0.1125F,
			-0.09F,
			-0.0675F,
			-0.045F,
			-0.0225F,
			0.0225F,
			0.045F,
			0.0675F,
			0.09F,
			0.1125F,
			0.135F,
			0.1575F,
			0.18F,
			-0.1134F,
			-0.0972F,
			-0.081F,
			-0.0648F,
			-0.0486F,
			-0.0324F,
			-0.0162F,
			0.0162F,
			0.0324F,
			0.0486F,
			0.0648F,
			0.081F,
			0.0972F,
			0.1134F,
			0.01F,
			-0.0972F,
			-0.0891F,
			-0.081F,
			-0.0729F,
			-0.0648F,
			-0.0567F,
			-0.0486F,
			-0.0405F,
			-0.0324F,
			-0.0243F,
			-0.0162F,
			-0.0081F,
			0.0081F,
			0.0162F,
			0.0243F,
			0.0324F,
			0.0405F,
			0.0486F,
			0.0567F,
			0.0648F,
			0.0729F,
			0.081F,
			0.0891F,
			0.0972F,
			0.1053F,
			0.1134F,
			0.1215F,
			0.005F,
			0.03F,
			0.03F,
			-0.007F,
			-0.014F,
			-0.007F,
			0.015F,
			0.095F,
			0.065F,
			0.1F,
			0.065F,
			0.11F,
			0.11F,
			0.01F,
			0.008F,
			0.07F,
			0.063F,
			0.028F,
			0.0245F,
			0.021F,
			0.16F,
			0.144F,
			0.064F,
			0.056F,
			0.048F,
			0.16F,
			0.144F,
			0.064F,
			0.056F,
			0.048F,
			0.22F,
			0.198F,
			0.088F,
			0.077F,
			0.066F,
			0.3F,
			0.27F,
			0.12F,
			0.105F,
			0.09F,
			0.065F,
			0.0375F,
			0.055F,
			0.045F,
			0.066F,
			0.03F,
			0.044F,
			0.0125F,
			0.0045F,
			0.006F,
			0.025F,
			0.009F,
			0.012F,
			0.025F,
			0.009F,
			0.012F,
			0.03125F,
			0.01125F,
			0.015F,
			0.04375F,
			0.01575F,
			0.021F,
			0.0025F,
			0.005F,
			0.0075F,
			0.01F,
			0.015F,
			0.005625F,
			0.0045F,
			0.00375F,
			0.003F,
			0.001875F,
			0.0015F,
			-0.0275F,
			0.0015F,
			0.002F,
			0.003F,
			0.004F,
			0.01F,
			-0.005F,
			0.015F,
			0.01F,
			-0.05F,
			-0.15F,
			0.002F,
			-0.005F,
			0.007F,
			0.02F,
			0.006F,
			0.0125F,
		}};
	} // namespace

	const FormulaSet &FormulaSet::fixed() {
		static const FormulaSet formulas;
		return formulas;
	}

	std::span<const float> FormulaSet::initial() const noexcept { return catalogInitial; }
	std::span<const std::string_view> FormulaSet::names() const noexcept { return catalogNames; }

} // namespace eleginus
