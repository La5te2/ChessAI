#include "eleginus/formula.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <stdexcept>
#include <utility>
#ifdef ELEGINUS_COMPILE
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <unordered_map>
#else
#include "eleginus/catalog.hpp"
#include "eleginus/model.hpp"
#endif

namespace eleginus {
	// Regions name formula subgraphs, not additional atomic instructions.
	enum class Region : unsigned {
		pawns,
		rookfiles,
		knightMobility,
		bishopMobility,
		rookMobility,
		queenMobility,
		pawnThreat,
		knightThreat,
		bishopThreat,
		rookThreat,
		queenThreat,
		queenPressure,
		kingPawns,
		count
	};
	enum class Type : std::uint8_t { F64, U64, B1 };
#ifdef ELEGINUS_COMPILE
	using Id = std::uint32_t;
	enum class Op : std::uint8_t { IMM, LD, NOT, AND, OR, XOR, SHL, SHR, UADD, USUB, CLZ, CTZ, POP, ADD, SUB, MUL, DIV, NEG, ABS, MIN, MAX, EQ, LT, LE, GT, GE, SEL };
	struct Node {
		Op op = Op::IMM;
		Type type = Type::F64;
		std::uint16_t aux = 0;
		Id a = 0, b = 0, c = 0;
		std::uint64_t imm = 0;
		bool operator==(const Node &) const = default;
	};
	struct Root {
		Id node;
		float weight;
	};
	struct Graph {
		std::vector<Node> nodes_;
		std::vector<Root> roots_;
		std::vector<std::uint8_t> families_;
		std::array<unsigned, static_cast<unsigned>(Region::count)> sizes_{};
		void validate() const;
		void write(const char *path) const;
	};
#endif
#ifndef ELEGINUS_COMPILE
	// The inference receiver consumes each nonzero signal immediately.
	struct Projection {
		static constexpr bool compact = true;
		Evaluator &eval;
		void begin(const std::array<double, 4> &coords) { eval.begin(coords); }
		void put(std::uint32_t index, float value) { eval.accept(index, value); }
	};
#endif
	namespace {

		using Word = std::uint64_t;
		double real(Word x) noexcept {
			return std::bit_cast<double>(x);
		}
		Word word(double x) noexcept {
			return std::bit_cast<Word>(x);
		}

#ifdef ELEGINUS_COMPILE
		int arity(Op op) noexcept {
			switch (op) {
			case Op::IMM:
			case Op::LD:
				return 0;
			case Op::NOT:
			case Op::CLZ:
			case Op::CTZ:
			case Op::POP:
			case Op::NEG:
			case Op::ABS:
				return 1;
			case Op::AND:
			case Op::OR:
			case Op::XOR:
			case Op::SHL:
			case Op::SHR:
			case Op::UADD:
			case Op::USUB:
			case Op::ADD:
			case Op::SUB:
			case Op::MUL:
			case Op::DIV:
			case Op::MIN:
			case Op::MAX:
			case Op::EQ:
			case Op::LT:
			case Op::LE:
			case Op::GT:
			case Op::GE:
				return 2;
			case Op::SEL:
				return 3;
			}
			return -1;
		}

		bool commutes(Op op) noexcept {
			return op == Op::AND || op == Op::OR || op == Op::XOR || op == Op::UADD || op == Op::ADD || op == Op::MUL || op == Op::MIN || op == Op::MAX || op == Op::EQ;
		}

#endif
#ifndef ELEGINUS_COMPILE
		std::array<Word, 15> inputs(const chess::Board &board) {
			std::array<Word, 15> in{};
			for (int color = 0; color < 2; ++color) {
				for (int type = 0; type < 6; ++type) {
					in[static_cast<std::size_t>(color * 6 + type)] =
					    board.pieces(chess::PieceType(static_cast<chess::PieceType::underlying>(type)), static_cast<chess::Color>(color)).getBits();
				}
				for (int side = 0; side < 2; ++side) {
					const auto flank = side == 0 ? chess::Board::CastlingRights::Side::KING_SIDE : chess::Board::CastlingRights::Side::QUEEN_SIDE;
					if (board.castlingRights().has(static_cast<chess::Color>(color), flank))
						in[13] |= 1ULL << (2 * color + side);
				}
			}
			in[12] = word(static_cast<int>(board.sideToMove()));
			in[14] = board.enpassantSq().is_valid() ? 1ULL << board.enpassantSq().index() : 0;
			return in;
		}

#endif
		std::uint64_t flip(Word x) noexcept {
			x = ((x & 0x00FF00FF00FF00FFULL) << 8) | ((x >> 8) & 0x00FF00FF00FF00FFULL);
			x = ((x & 0x0000FFFF0000FFFFULL) << 16) | ((x >> 16) & 0x0000FFFF0000FFFFULL);
			return (x << 32) | (x >> 32);
		}

	} // namespace

#ifdef ELEGINUS_COMPILE
	// Assembly construction shares identical atomic expressions.
	class Asm {
	public:
		Id num(double x, Type type = Type::F64);
		Id bits(std::uint64_t x);
		Id input(unsigned slot);
		Id unary(Op op, Type type, Id a);
		Id binary(Op op, Type type, Id a, Id b);
		Id sel(Id condition, Id yes, Id no);
		void root(Id id, float weight);
		void family(unsigned group) { families_.push_back(static_cast<std::uint8_t>(group)); }
		template <class F> void group(Region region, bool, F &&f) {
			auto &size = sizes_[static_cast<unsigned>(region)];
			if (size)
				throw std::logic_error("duplicate formula region");
			const auto first = roots_.size();
			f();
			size = static_cast<unsigned>(roots_.size() - first);
		}
		Graph finish();

	protected:
		const Node &node(Id id) const { return nodes_[id]; }

	private:
		struct Hash {
			std::size_t operator()(const Node &n) const noexcept;
		};
		Id emit(Node n);
		std::vector<Node> nodes_;
		std::vector<Root> roots_;
		std::vector<std::uint8_t> families_;
		std::array<unsigned, static_cast<unsigned>(Region::count)> sizes_{};
		std::unordered_map<Node, Id, Hash> ids_;
	};

	std::size_t Asm::Hash::operator()(const Node &n) const noexcept {
		std::uint64_t h = n.imm ^ (static_cast<std::uint64_t>(n.op) << 56) ^ (static_cast<std::uint64_t>(n.type) << 48);
		for (const Id x : {n.a, n.b, n.c})
			h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
		return static_cast<std::size_t>(h);
	}

	Id Asm::emit(Node n) {
		if (commutes(n.op) && n.b < n.a)
			std::swap(n.a, n.b);
		const auto found = ids_.find(n);
		if (found != ids_.end())
			return found->second;
		const auto id = static_cast<Id>(nodes_.size());
		ids_.emplace(n, id);
		nodes_.push_back(n);
		return id;
	}

	Id Asm::num(double x, Type type) {
		return emit({Op::IMM, type, 0, 0, 0, 0, word(x)});
	}
	Id Asm::bits(Word x) {
		return emit({Op::IMM, Type::U64, 0, 0, 0, 0, x});
	}
	Id Asm::input(unsigned slot) {
		return emit({Op::LD, slot == 12 ? Type::F64 : Type::U64, 0, 0, 0, 0, slot});
	}
	Id Asm::unary(Op op, Type type, Id a) {
		return emit({op, type, 0, a});
	}
	Id Asm::binary(Op op, Type type, Id a, Id b) {
		return emit({op, type, 0, a, b});
	}
	Id Asm::sel(Id condition, Id yes, Id no) {
		const auto type = node(yes).type == Type::U64 ? Type::U64 : node(yes).type == Type::B1 && node(no).type == Type::B1 ? Type::B1 : Type::F64;
		return emit({Op::SEL, type, 0, condition, yes, no});
	}
	void Asm::root(Id id, float weight) {
		roots_.push_back({id, weight});
	}
	Graph Asm::finish() {
		Graph p;
		p.nodes_ = std::move(nodes_);
		p.roots_ = std::move(roots_);
		p.families_ = std::move(families_);
		p.sizes_ = sizes_;
		ids_.clear();
		return p;
	}

#endif

	namespace {

		// Cache slots name shared subgraphs, never new atomic instructions.
		enum class Shared { own, strong, clear, safe, wide, stop, control, area, push, count };
		enum class Pawn { attack, files, passed, open, count };

		// Builder visits every square; Native visits only the occupied subset.
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

#ifdef ELEGINUS_COMPILE
		class Builder : public Asm {
		public:
			using Value = Id;
			using Sum = std::vector<Id>;
			static constexpr bool direct = false;
			template <class F> Id memo(Shared, Id, F &&f) { return f(); }
			template <class F> Id pawn(Pawn, Id, F &&f) { return f(); }
			Squares squares(Id, int) const { return {~0ULL}; }
			Squares locations(Id) const { return {~0ULL}; }
			Id add(Id a, Id b) { return binary(Op::ADD, Type::F64, a, b); }
			Id sub(Id a, Id b) { return binary(Op::SUB, Type::F64, a, b); }
			Id mul(Id a, Id b) { return binary(Op::MUL, Type::F64, a, b); }
			Id land(Id a, Id b) { return sel(a, b, num(0, Type::B1)); }
			Id lor(Id a, Id b) { return sel(a, num(1, Type::B1), b); }
			Id lnot(Id a) { return equal(a, num(0)); }
			Id equal(Id a, Id b) { return binary(Op::EQ, Type::B1, a, b); }
			Id ge(Id a, Id b) { return binary(Op::GE, Type::B1, a, b); }
			Id maximum(Id a, Id b) { return binary(Op::MAX, Type::F64, a, b); }
			Id minimum(Id a, Id b) { return binary(Op::MIN, Type::F64, a, b); }
			Id bb(Word x) { return bits(x); }
			Id band(Id a, Id b) { return binary(Op::AND, Type::U64, a, b); }
			Id bor(Id a, Id b) { return binary(Op::OR, Type::U64, a, b); }
			Id bnot(Id a) { return unary(Op::NOT, Type::U64, a); }
			Id shl(Id a, unsigned n) { return binary(Op::SHL, Type::U64, a, bb(n)); }
			Id shr(Id a, unsigned n) { return binary(Op::SHR, Type::U64, a, bb(n)); }
			Id pop(Id a) { return unary(Op::POP, Type::F64, a); }
			Id anyset(Id a) { return lnot(equal(a, bb(0))); }
			Id pcs(Id role, int type) { return sel(equal(role, num(0)), input(type), input(6 + type)); }
			Id rel(Id role, Word mask) { return sel(equal(role, num(0)), bb(mask), bb(flip(mask))); }
			Id rsq(Id role, int square) { return sel(equal(role, num(0)), num(square), num(square ^ 56)); }
			Id right(Id role, int side) { return anyset(band(input(13), sel(equal(role, num(0)), bb(1ULL << side), bb(1ULL << (side + 2))))); }
			Id occ() {
				Id result = input(0);
				for (unsigned slot = 1; slot < 12; ++slot)
					result = bor(result, input(slot));
				return result;
			}

			// Direction order: forward, backward, east, west, then forward/backward diagonals.
			Id sh(Id x, Id role, int direction) {
				const auto east = band(x, bb(0x7F7F7F7F7F7F7F7FULL));
				const auto west = band(x, bb(0xFEFEFEFEFEFEFEFEULL));
				const auto white = equal(role, num(0));
				switch (direction) {
				case 0:
					return sel(white, shl(x, 8), shr(x, 8));
				case 1:
					return sel(white, shr(x, 8), shl(x, 8));
				case 2:
					return shl(east, 1);
				case 3:
					return shr(west, 1);
				case 4:
					return sel(white, shl(east, 9), shr(east, 7));
				case 5:
					return sel(white, shl(west, 7), shr(west, 9));
				case 6:
					return sel(white, shr(east, 7), shl(east, 9));
				case 7:
					return sel(white, shr(west, 9), shl(west, 7));
				default:
					throw std::invalid_argument("invalid shift direction");
				}
			}

			Id fill(Id x, Id role) {
				const auto white = equal(role, num(0));
				for (unsigned distance : {8U, 16U, 32U})
					x = bor(x, sel(white, shl(x, distance), shr(x, distance)));
				return x;
			}

			Id fromatk(Id role, int type, Id square) {
				const Node n = node(square);
				if (n.op == Op::SEL)
					return sel(n.a, fromatk(role, type, n.b), fromatk(role, type, n.c));
				if (n.op != Op::IMM)
					throw std::logic_error("attack macro requires a finite square expansion");
				const int s = static_cast<int>(real(n.imm));
				if (type == 0) {
					return sel(equal(role, num(0)), bb(chess::attacks::pawn(chess::Color::WHITE, chess::Square(s)).getBits()),
					    bb(chess::attacks::pawn(chess::Color::BLACK, chess::Square(s)).getBits()));
				}
				if (type == 1)
					return bb(chess::attacks::knight(chess::Square(s)).getBits());
				if (type == 5)
					return bb(chess::attacks::king(chess::Square(s)).getBits());
				Id result = bb(0);
				for (int dr = -1; dr <= 1; ++dr) {
					for (int df = -1; df <= 1; ++df) {
						if ((dr == 0 && df == 0) || (type == 2 && (dr == 0 || df == 0)) || (type == 3 && dr != 0 && df != 0))
							continue;
						Word ray = 0;
						for (int r = s / 8 + dr, f = s % 8 + df; r >= 0 && r < 8 && f >= 0 && f < 8; r += dr, f += df)
							ray |= 1ULL << (8 * r + f);
						if (!ray)
							continue;
						const auto blockers = band(occ(), bb(ray));
						Id span;
						if (8 * dr + df > 0) {
							// Include the first blocker; the zero-blocker case wraps to all bits set.
							const auto low = band(blockers, binary(Op::USUB, Type::U64, bb(0), blockers));
							span = binary(Op::UADD, Type::U64, low, binary(Op::USUB, Type::U64, low, bb(1)));
						} else {
							const auto shift = binary(Op::UADD, Type::U64, unary(Op::CLZ, Type::U64, blockers), bb(1));
							span = bnot(binary(Op::SHR, Type::U64, bb(~0ULL), shift));
						}
						result = bor(result, band(bb(ray), span));
					}
				}
				return result;
			}

			Id atk(Id role, int type = 6) { return attacks(role, type, false); }
			Id atk2(Id role) { return attacks(role, 6, true); }
			Id div(Id a, Id b) { return binary(Op::DIV, Type::F64, a, b); }
			Id abs(Id a) { return unary(Op::ABS, Type::F64, a); }
			Id gt(Id a, Id b) { return binary(Op::GT, Type::B1, a, b); }
			Id lt(Id a, Id b) { return binary(Op::LT, Type::B1, a, b); }
			Id le(Id a, Id b) { return binary(Op::LE, Type::B1, a, b); }

			Id sum(const std::vector<Id> &terms) {
				Id result = num(0);
				for (const auto term : terms)
					result = add(result, term);
				return result;
			}

		private:
			Id attacks(Id role, int type, bool twice) {
				Id seen = bb(0);
				Id repeated = bb(0);
				for (int piece = type == 6 ? 0 : type; piece <= (type == 6 ? 5 : type); ++piece) {
					for (int square = 0; square < 64; ++square) {
						const auto present = anyset(band(pcs(role, piece), bb(1ULL << square)));
						const auto current = sel(present, fromatk(role, piece, num(square)), bb(0));
						if (twice)
							repeated = bor(repeated, band(seen, current));
						seen = bor(seen, current);
					}
				}
				return twice ? repeated : seen;
			}
		};

#else
		// Fixed formulas execute directly on scalar and bitboard values.
		struct Signal {
			Word bits;
			Type type;
			Signal() = default;
			Signal(double x) : bits(word(x)), type(Type::F64) {}
			Signal(Word x, Type t) : bits(x), type(t) {}
		};

		struct Features {
			static constexpr bool compact = false;
			std::vector<Feature> &values;
			void begin(const std::array<double, 4> &) {}
			void put(std::uint32_t index, float value) { values.push_back({index, value}); }
		};

		template <class Output> class Native {
		public:
			using Value = Signal;
			struct Sum {
				double total = 0;
				void push_back(Value v) { total += real(v.bits); }
			};
			static constexpr bool direct = true;

			Native(const chess::Board &board, Output &out) : in(inputs(board)), occupied(board.occ().getBits()), output(out) {
				// Verify both full pawn bitboards on a hit; non-pawn pieces and turn are not dependencies.
				thread_local std::array<Pawns, 1024> table{};
				const Word hash = in[0] * 0x9e3779b97f4a7c15ULL ^ std::rotl(in[6] * 0xbf58476d1ce4e5b9ULL, 29);
				pawnCache = &table[(hash ^ (hash >> 32)) & (table.size() - 1)];
				if (!pawnCache->valid || pawnCache->white != in[0] || pawnCache->black != in[6]) {
					pawnCache->white = in[0];
					pawnCache->black = in[6];
					pawnCache->ready = 0;
					pawnCache->valid = true;
				}
				thread_local Attacks attacks;
				attackCache = &attacks;
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
							map = overlap = 0;
							auto pieces = in[i];
							while (pieces) {
								const int square = std::countr_zero(pieces);
								pieces &= pieces - 1;
								const auto a = fromatk(num(color), type, num(square)).bits;
								overlap |= map & a;
								map |= a;
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
			Value num(double x, Type t = Type::F64) const { return {word(x), t}; }
			template <class F> Value memo(Shared key, Value role, F &&f) {
				const auto i = 2 * static_cast<unsigned>(key) + color(role);
				if (!(ready & (1U << i))) {
					shared[i] = f();
					ready |= 1U << i;
				}
				return shared[i];
			}
			template <class F> Value pawn(Pawn key, Value role, F &&f) {
				const auto i = 2 * static_cast<unsigned>(key) + color(role);
				if (!(pawnCache->ready & (1U << i))) {
					pawnCache->values[i] = f();
					pawnCache->ready |= 1U << i;
				}
				return pawnCache->values[i];
			}
			Value bb(Word x) const { return {x, Type::U64}; }
			Value input(unsigned slot) const { return {in[slot], slot == 12 ? Type::F64 : Type::U64}; }
			Value add(Value a, Value b) const { return num(real(a.bits) + real(b.bits)); }
			Value sub(Value a, Value b) const { return num(real(a.bits) - real(b.bits)); }
			Value mul(Value a, Value b) const { return num(real(a.bits) * real(b.bits)); }
			Value land(Value a, Value b) const { return num(real(a.bits) != 0 && real(b.bits) != 0, Type::B1); }
			Value lor(Value a, Value b) const { return num(real(a.bits) != 0 || real(b.bits) != 0, Type::B1); }
			Value lnot(Value a) const { return num(real(a.bits) == 0, Type::B1); }
			Value equal(Value a, Value b) const { return num(a.type == Type::U64 ? a.bits == b.bits : real(a.bits) == real(b.bits), Type::B1); }
			Value div(Value a, Value b) const { return num(std::abs(real(b.bits)) > 1.0e-12 ? real(a.bits) / real(b.bits) : 0.0); }
			Value abs(Value a) const { return num(std::abs(real(a.bits))); }
			Value gt(Value a, Value b) const { return num(real(a.bits) > real(b.bits), Type::B1); }
			Value lt(Value a, Value b) const { return num(real(a.bits) < real(b.bits), Type::B1); }
			Value le(Value a, Value b) const { return num(real(a.bits) <= real(b.bits), Type::B1); }
			Value ge(Value a, Value b) const { return num(real(a.bits) >= real(b.bits), Type::B1); }
			Value maximum(Value a, Value b) const { return num(std::max(real(a.bits), real(b.bits))); }
			Value minimum(Value a, Value b) const { return num(std::min(real(a.bits), real(b.bits))); }
			Value band(Value a, Value b) const { return bb(a.bits & b.bits); }
			Value bor(Value a, Value b) const { return bb(a.bits | b.bits); }
			Value bnot(Value a) const { return bb(~a.bits); }
			Value pop(Value a) const { return num(std::popcount(a.bits)); }
			Value anyset(Value a) const { return num(a.bits != 0, Type::B1); }
			Value pcs(Value role, int type) const { return bb(in[6 * color(role) + type]); }
			Value rel(Value role, Word mask) const { return bb(color(role) == 0 ? mask : flip(mask)); }
			Value rsq(Value role, int square) const { return num(square ^ (color(role) == 0 ? 0 : 56)); }
			Value right(Value role, int side) const { return num((in[13] & (1ULL << (2 * color(role) + side))) != 0, Type::B1); }
			Value occ() const { return bb(occupied); }
			bool present(Value role, int type, int square) const { return (pcs(role, type).bits & rel(role, 1ULL << square).bits) != 0; }
			Squares squares(Value role, int type) const { return {rel(role, pcs(role, type).bits).bits}; }
			Squares locations(Value set) const { return {set.bits}; }
			Value sh(Value x, Value role, int direction) const {
				const Word east = x.bits & 0x7F7F7F7F7F7F7F7FULL, west = x.bits & 0xFEFEFEFEFEFEFEFEULL;
				const bool white = color(role) == 0;
				switch (direction) {
				case 0:
					return bb(white ? x.bits << 8 : x.bits >> 8);
				case 1:
					return bb(white ? x.bits >> 8 : x.bits << 8);
				case 2:
					return bb(east << 1);
				case 3:
					return bb(west >> 1);
				case 4:
					return bb(white ? east << 9 : east >> 7);
				case 5:
					return bb(white ? west << 7 : west >> 9);
				case 6:
					return bb(white ? east >> 7 : east << 9);
				case 7:
					return bb(white ? west >> 9 : west << 7);
				default:
					throw std::logic_error("invalid shift direction");
				}
			}
			Value fill(Value x, Value role) const {
				for (unsigned d : {8U, 16U, 32U})
					x.bits |= color(role) == 0 ? x.bits << d : x.bits >> d;
				return x;
			}
			Value fromatk(Value role, int type, Value square) {
				const int s = static_cast<int>(real(square.bits));
				if (type == 0)
					return bb(chess::attacks::pawn(static_cast<chess::Color>(color(role)), chess::Square(s)).getBits());
				if (type == 1)
					return bb(chess::attacks::knight(chess::Square(s)).getBits());
				if (type == 5)
					return bb(chess::attacks::king(chess::Square(s)).getBits());
				if (type == 4)
					return bor(fromatk(role, 2, square), fromatk(role, 3, square));
				auto &ray = attackCache->rays[type - 2][s];
				if (ray.map == 0 || ((ray.occupied ^ occupied) & ray.map))
					ray.map = (type == 2 ? chess::attacks::bishop(chess::Square(s), chess::Bitboard(occupied)) : chess::attacks::rook(chess::Square(s), chess::Bitboard(occupied)))
					              .getBits();
				ray.occupied = occupied;
				return bb(ray.map);
			}
			Value atk(Value role, int type = 6) const { return bb(attackCache->maps[color(role)][type]); }
			Value atk2(Value role) const { return bb(attackCache->maps[color(role)][7]); }
			Value sum(const Sum &terms) const { return num(terms.total); }
			void root(Value v, float) {
				const float x = static_cast<float>(real(v.bits));
				if (x != 0)
					output.put(index, x);
				++index;
			}
			void skip(unsigned n) { index += n; }
			template <class F> void group(Region region, bool active, F &&f) {
				if (active)
					f();
				else
					skip(catalog::sizes[static_cast<unsigned>(region)]);
			}
			bool compact() const { return Output::compact; }
			void coordinates(Value phase, Value open, Value freedom, Value exposure) { output.begin({real(phase.bits), real(open.bits), real(freedom.bits), real(exposure.bits)}); }
			void finish() const {}

		private:
			struct Pawns {
				Word white = 0, black = 0;
				std::array<Value, 2 * static_cast<unsigned>(Pawn::count)> values{};
				unsigned ready = 0;
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
				std::array<std::array<Ray, 64>, 2> rays{};
				bool valid = false;
			};
			static unsigned color(Value role) { return static_cast<unsigned>(real(role.bits)); }
			Pawns *pawnCache;
			Attacks *attackCache;
			std::array<Value, 2 * static_cast<unsigned>(Shared::count)> shared{};
			unsigned ready = 0;
			std::array<Word, 15> in;
			Word occupied;
			Output &output;
			unsigned index = 0;
		};

#endif
		enum class Family { general, mobility, pawn, king };

		template <class B> class Formulas {
			using Id = typename B::Value;
			using Sum = typename B::Sum;

		public:
			explicit Formulas(B builder = {}) : b(std::move(builder)) {
				z = b.num(0.0);
				o = b.num(1.0);
				us = b.input(12);
				them = b.sub(o, us);
				occ = b.occ();
				phase = makephase();
				open = makeopen();
				freedom = makefreedom();
				exposure = makeexposure();
				if constexpr (B::direct)
					b.coordinates(phase, open, freedom, exposure);
			}

			auto build() {
				reg(o, 0.15F, 0.17F, Family::general);
				material();
				pst();
				bishoppair();
				b.group(Region::pawns, has(0), [&] { pawns(); });
				mobility();
				pieces();
				threats();
				kings();
				endgames();
				// Context roots feed the graybox, not independently scored formulas.
				for (const auto id : {phase, open, freedom, exposure})
					b.root(id, 0.0F);
				for (const auto side : {us, them}) {
					for (int type = 0; type < 6; ++type) {
						b.root(b.mul(cnt(b.pcs(side, type)), b.num(1.0 / 16.0)), 0.0F);
					}
				}
				return b.finish();
			}

		private:
			bool has(int type) const {
				if constexpr (B::direct)
					return b.pcs(us, type).bits != 0 || b.pcs(them, type).bits != 0;
				return true;
			}
			static constexpr std::uint64_t filemask(int file) noexcept { return 0x0101010101010101ULL << file; }
			static constexpr std::uint64_t rankmask(int rank) noexcept { return 0xFFULL << (8 * rank); }

			static std::uint64_t ringmask(int square, int distance) noexcept {
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

			static std::uint64_t diagonalmask(int square) noexcept {
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

			Id cnt(Id set) { return b.pop(set); }
			Id diff(Id own, Id enemy) { return b.sub(own, enemy); }
			Id own(Id role) {
				return b.memo(Shared::own, role, [&] {
					Id result = b.pcs(role, 0);
					for (int type = 1; type < 6; ++type)
						result = b.bor(result, b.pcs(role, type));
					return result;
				});
			}

			Id pawnatt(Id role) {
				return b.pawn(Pawn::attack, role, [&] { return b.bor(b.sh(b.pcs(role, 0), role, 4), b.sh(b.pcs(role, 0), role, 5)); });
			}

			Id files(Id set, Id role, Id opponent) { return b.bor(b.fill(set, role), b.fill(set, opponent)); }

			Id passers(Id role, Id opponent) {
				return b.pawn(Pawn::passed, role, [&] {
					const auto span = b.fill(b.pcs(opponent, 0), opponent);
					const auto stops = b.bor(span, b.bor(b.sh(span, role, 2), b.sh(span, role, 3)));
					return b.band(b.pcs(role, 0), b.bnot(stops));
				});
			}

			Id strong(Id role, Id opponent) {
				return b.memo(Shared::strong, role, [&] { return b.bor(pawnatt(opponent), b.band(b.atk2(opponent), b.bnot(b.atk2(role)))); });
			}

			Id passerkingdistance(Id kingrole, Id passed, int distance) {
				Sum terms;
				for (int square : b.locations(passed)) {
					const auto pawn = b.anyset(b.band(passed, b.bb(1ULL << square)));
					const auto king = b.anyset(b.band(b.pcs(kingrole, 5), b.bb(ringmask(square, distance))));
					terms.push_back(b.land(pawn, king));
				}
				return b.sum(terms);
			}

			Id clamp01(Id value) { return b.minimum(o, b.maximum(z, value)); }

			Id makephase() {
				Id total = z;
				constexpr std::array<int, 6> units{{0, 1, 1, 2, 4, 0}};
				for (const auto role : {us, them}) {
					for (int type = 1; type <= 4; ++type) {
						total = b.add(total, b.mul(b.num(units[static_cast<std::size_t>(type)]), cnt(b.pcs(role, type))));
					}
				}
				return clamp01(b.div(total, b.num(24.0)));
			}

			Id makeopen() {
				return b.pawn(Pawn::open, us, [&] {
					Sum terms;
					const auto allpawns = b.bor(b.pcs(us, 0), b.pcs(them, 0));
					for (int file = 0; file < 8; ++file)
						terms.push_back(b.lnot(b.anyset(b.band(allpawns, b.bb(filemask(file))))));
					return b.div(b.sum(terms), b.num(8.0));
				});
			}

			Id makefreedom() {
				Id total = z;
				for (const auto role : {us, them}) {
					total = b.add(total, cnt(b.band(b.sh(b.pcs(role, 0), role, 0), b.bnot(occ))));
				}
				return clamp01(b.div(total, b.num(16.0)));
			}

			Id makeexposure() {
				Sum terms;
				const auto allpawns = b.bor(b.pcs(us, 0), b.pcs(them, 0));
				for (const auto role : {us, them}) {
					for (int square : b.squares(role, 5)) {
						const int file = square % 8;
						Sum exposed;
						for (int candidate = std::max(0, file - 1); candidate <= std::min(7, file + 1); ++candidate) {
							exposed.push_back(b.lnot(b.anyset(b.band(allpawns, b.bb(filemask(candidate))))));
						}
						const auto king = b.anyset(b.band(b.pcs(role, 5), b.rel(role, 1ULL << square)));
						terms.push_back(b.mul(king, b.sum(exposed)));
					}
				}
				return clamp01(b.div(b.sum(terms), b.num(6.0)));
			}

			Id structure(Family family) const noexcept {
				if (family == Family::pawn) {
					return freedom;
				}
				if (family == Family::king) {
					return exposure;
				}
				return open;
			}

			void reg(Id formula, float eg, float mg, Family family) {
				if constexpr (B::direct) {
					if (b.compact()) {
						b.root(formula, 0);
						b.skip(3);
						return;
					}
				} else {
					b.family(family == Family::pawn ? 1 : family == Family::king ? 2 : 0);
				}
				const auto ending = b.sub(o, phase);
				const auto opened = structure(family);
				const auto closed = b.sub(o, opened);
				b.root(b.mul(formula, b.mul(ending, closed)), eg);
				b.root(b.mul(formula, b.mul(ending, opened)), eg);
				b.root(b.mul(formula, b.mul(phase, closed)), mg);
				b.root(b.mul(formula, b.mul(phase, opened)), mg);
			}

			void material() {
				constexpr std::array<float, 5> eg{{0.235F, 0.703F, 0.743F, 1.280F, 2.340F}};
				constexpr std::array<float, 5> mg{{0.205F, 0.843F, 0.913F, 1.193F, 2.563F}};
				for (int type = 0; type < 5; ++type) {
					reg(diff(cnt(b.pcs(us, type)), cnt(b.pcs(them, type))), eg[static_cast<std::size_t>(type)], mg[static_cast<std::size_t>(type)], Family::general);
				}
			}

			void pst() {
				for (int type = 0; type < 6; ++type) {
					for (int square = 0; square < 64; ++square) {
						if constexpr (B::direct) {
							if (b.present(us, type, square) == b.present(them, type, square)) {
								b.skip(4);
								continue;
							}
						}
						const int file = square % 8;
						const int rank = square / 8;
						const float center = static_cast<float>(7 - std::abs(2 * file - 7) - std::abs(2 * rank - 7));
						float mg = 0.0F;
						float eg = 0.0F;
						if (type == 0) {
							mg = (5.0F * rank + center) / 400.0F;
							eg = (8.0F * rank + 0.5F * center) / 400.0F;
						} else if (type == 1) {
							mg = 5.0F * center / 400.0F;
							eg = 4.0F * center / 400.0F;
						} else if (type == 2) {
							mg = 3.0F * center / 400.0F;
							eg = mg;
						} else if (type == 3) {
							mg = (rank == 6 ? 18.0F : 1.5F * rank) / 400.0F;
							eg = (rank == 6 ? 10.0F : static_cast<float>(rank)) / 400.0F;
						} else if (type == 4) {
							mg = center / 400.0F;
							eg = 2.0F * center / 400.0F;
						} else {
							mg = ((rank == 0 ? 12.0F : -4.0F * rank) - 2.0F * center) / 400.0F;
							eg = 6.0F * center / 400.0F;
						}
						const auto friendly = cnt(b.band(b.pcs(us, type), b.rel(us, 1ULL << square)));
						const auto enemy = cnt(b.band(b.pcs(them, type), b.rel(them, 1ULL << square)));
						reg(diff(friendly, enemy), eg, mg, type == 5 ? Family::king : Family::general);
					}
				}
			}

			void bishoppair() {
				const auto friendly = b.ge(cnt(b.pcs(us, 2)), b.num(2.0));
				const auto enemy = b.ge(cnt(b.pcs(them, 2)), b.num(2.0));
				reg(diff(friendly, enemy), 0.18F, 0.12F, Family::general);
			}

			void pawns() {
				const auto fp = b.pcs(us, 0);
				const auto ep = b.pcs(them, 0);
				const auto fpass = passers(us, them);
				const auto epass = passers(them, us);
				const auto fatt = pawnatt(us);
				const auto eatt = pawnatt(them);
				const auto ffiles = b.pawn(Pawn::files, us, [&] { return files(fp, us, them); });
				const auto efiles = b.pawn(Pawn::files, them, [&] { return files(ep, them, us); });
				const auto fneighbours = b.bor(b.sh(ffiles, us, 2), b.sh(ffiles, us, 3));
				const auto eneighbours = b.bor(b.sh(efiles, them, 2), b.sh(efiles, them, 3));

				for (int rank = 1; rank < 7; ++rank) {
					const auto fm = b.rel(us, rankmask(rank));
					const auto em = b.rel(them, rankmask(rank));
					const auto fr = b.band(fpass, fm);
					const auto er = b.band(epass, em);
					const float progress = static_cast<float>(rank - 1);
					reg(diff(cnt(fr), cnt(er)), (20.0F + 14.0F * progress * progress) / 400.0F, (10.0F + 8.0F * progress * progress) / 400.0F, Family::pawn);
					const auto fclear = b.band(fr, b.memo(Shared::clear, us, [&] { return b.bnot(b.sh(b.fill(occ, them), them, 0)); }));
					const auto eclear = b.band(er, b.memo(Shared::clear, them, [&] { return b.bnot(b.sh(b.fill(occ, us), us, 0)); }));
					reg(diff(cnt(fclear), cnt(eclear)), (8.0F + 5.0F * progress) / 400.0F, (3.0F + 2.0F * progress) / 400.0F, Family::pawn);
					const auto fsafe = b.band(fr, b.memo(Shared::safe, us, [&] { return b.bnot(b.sh(b.fill(b.atk(them), them), them, 0)); }));
					const auto esafe = b.band(er, b.memo(Shared::safe, them, [&] { return b.bnot(b.sh(b.fill(b.atk(us), us), us, 0)); }));
					reg(diff(cnt(fsafe), cnt(esafe)), (8.0F + 6.0F * progress) / 400.0F, (3.0F + 2.0F * progress) / 400.0F, Family::pawn);
					const auto fwideatt = b.memo(Shared::wide, us, [&] { return b.bor(b.atk(them), b.bor(b.sh(b.atk(them), us, 2), b.sh(b.atk(them), us, 3))); });
					const auto ewideatt = b.memo(Shared::wide, them, [&] { return b.bor(b.atk(us), b.bor(b.sh(b.atk(us), them, 2), b.sh(b.atk(us), them, 3))); });
					const auto fstopclear = b.band(fr, b.memo(Shared::stop, us, [&] { return b.bnot(b.sh(b.fill(fwideatt, them), them, 0)); }));
					const auto estopclear = b.band(er, b.memo(Shared::stop, them, [&] { return b.bnot(b.sh(b.fill(ewideatt, us), us, 0)); }));
					reg(diff(cnt(fstopclear), cnt(estopclear)), (6.0F + 4.0F * progress) / 400.0F, (2.0F + 1.5F * progress) / 400.0F, Family::pawn);
					const auto fpushcontrol = b.memo(Shared::control, us, [&] { return b.bor(b.band(b.atk(us), b.bnot(b.atk(them))), b.band(b.atk2(us), b.bnot(b.atk2(them)))); });
					const auto epushcontrol =
					    b.memo(Shared::control, them, [&] { return b.bor(b.band(b.atk(them), b.bnot(b.atk(us))), b.band(b.atk2(them), b.bnot(b.atk2(us)))); });
					const auto fdefendedpush = b.band(fr, b.sh(fpushcontrol, us, 1));
					const auto edefendedpush = b.band(er, b.sh(epushcontrol, them, 1));
					reg(diff(cnt(fdefendedpush), cnt(edefendedpush)), (7.0F + 4.0F * progress) / 400.0F, (3.0F + 2.0F * progress) / 400.0F, Family::pawn);
					reg(diff(cnt(b.band(fr, fatt)), cnt(b.band(er, eatt))), (10.0F + 6.0F * progress) / 400.0F, (4.0F + 3.0F * progress) / 400.0F, Family::pawn);
					reg(diff(cnt(b.band(fr, b.sh(occ, them, 0))), cnt(b.band(er, b.sh(occ, us, 0)))), -(8.0F + 6.0F * progress) / 400.0F, -(3.0F + 3.0F * progress) / 400.0F,
					    Family::pawn);
					const auto fconnected = b.band(fr, b.bor(b.sh(fr, us, 2), b.sh(fr, us, 3)));
					const auto econnected = b.band(er, b.bor(b.sh(er, them, 2), b.sh(er, them, 3)));
					reg(diff(cnt(fconnected), cnt(econnected)), (5.0F + 3.5F * progress) / 400.0F, (2.0F + 1.5F * progress) / 400.0F, Family::pawn);
					const auto fphalanx = b.band(b.band(fp, b.sh(fp, us, 2)), fm);
					const auto ephalanx = b.band(b.band(ep, b.sh(ep, them, 2)), em);
					reg(diff(cnt(fphalanx), cnt(ephalanx)), (8.0F + 4.0F * progress) / 400.0F, (6.0F + 3.0F * progress) / 400.0F, Family::pawn);
					reg(diff(cnt(b.band(b.band(fp, fatt), fm)), cnt(b.band(b.band(ep, eatt), em))), (8.0F + 3.0F * progress) / 400.0F, (6.0F + 2.0F * progress) / 400.0F,
					    Family::pawn);
				}

				Id fdoubled = z;
				Id edoubled = z;
				Sum fislands;
				Sum eislands;
				for (int file = 0; file < 8; ++file) {
					const auto fcount = cnt(b.band(fp, b.bb(filemask(file))));
					const auto ecount = cnt(b.band(ep, b.bb(filemask(file))));
					fdoubled = b.add(fdoubled, b.maximum(z, b.sub(fcount, o)));
					edoubled = b.add(edoubled, b.maximum(z, b.sub(ecount, o)));
					const auto foccupied = b.anyset(b.band(fp, b.bb(filemask(file))));
					const auto eoccupied = b.anyset(b.band(ep, b.bb(filemask(file))));
					fislands.push_back(file == 0 ? foccupied : b.land(foccupied, b.lnot(b.anyset(b.band(fp, b.bb(filemask(file - 1)))))));
					eislands.push_back(file == 0 ? eoccupied : b.land(eoccupied, b.lnot(b.anyset(b.band(ep, b.bb(filemask(file - 1)))))));
				}
				reg(diff(fdoubled, edoubled), -0.04F, -0.03F, Family::pawn);
				reg(diff(cnt(b.band(fp, b.bnot(fneighbours))), cnt(b.band(ep, b.bnot(eneighbours)))), -0.03F, -0.025F, Family::pawn);
				const auto fback = b.band(b.band(fp, b.bnot(fatt)), b.sh(eatt, them, 0));
				const auto eback = b.band(b.band(ep, b.bnot(eatt)), b.sh(fatt, us, 0));
				reg(diff(cnt(fback), cnt(eback)), -0.035F, -0.03F, Family::pawn);
				reg(diff(b.sum(fislands), b.sum(eislands)), -0.025F, -0.015F, Family::pawn);
				// Total passer count supplies the omitted distance-three bucket for each king relation.
				for (int distance = 0; distance <= 7; ++distance) {
					if (distance == 3) {
						continue;
					}
					const float friendlyweight = static_cast<float>(3 - distance) / 400.0F;
					const float enemyweight = static_cast<float>(distance - 3) / 400.0F;
					reg(diff(passerkingdistance(us, fpass, distance), passerkingdistance(them, epass, distance)), friendlyweight, friendlyweight * 0.6F, Family::pawn);
					reg(diff(passerkingdistance(them, fpass, distance), passerkingdistance(us, epass, distance)), enemyweight, enemyweight * 0.6F, Family::pawn);
				}
			}

			Id sq(Id role, int square) { return b.rsq(role, square); }

			std::array<Id, 28> mobilitycounts(Id role, Id opponent, int type, int maximum) {
				std::array<Sum, 28> buckets{}; // A queen has at most 27 reachable squares.
				const auto area = b.memo(Shared::area, role, [&] {
					const auto pawns = b.pcs(role, 0);
					const auto blocked = b.band(pawns, b.sh(occ, role, 1));
					const auto early = b.band(pawns, b.rel(role, rankmask(1) | rankmask(2)));
					return b.band(b.bnot(own(role)), b.bnot(b.bor(blocked, b.bor(early, pawnatt(opponent)))));
				});
				for (int square : b.squares(role, type)) {
					const auto present = b.anyset(b.band(b.pcs(role, type), b.rel(role, 1ULL << square)));
					const auto count = cnt(b.band(b.fromatk(role, type, sq(role, square)), area));
					for (int bucket = 0; bucket <= maximum; ++bucket) {
						buckets[static_cast<std::size_t>(bucket)].push_back(b.land(present, b.equal(count, b.num(bucket))));
					}
				}
				std::array<Id, 28> result{};
				for (int bucket = 0; bucket <= maximum; ++bucket) {
					result[static_cast<std::size_t>(bucket)] = b.sum(buckets[static_cast<std::size_t>(bucket)]);
				}
				return result;
			}

			Id secondarymobility(Id role, Id opponent, int type) {
				Sum terms;
				Id unsafe = b.bor(pawnatt(opponent), b.bor(b.atk(opponent, 1), b.atk(opponent, 2)));
				if (type == 4) {
					unsafe = b.bor(unsafe, b.atk(opponent, 3));
				}
				const auto area = b.band(b.bnot(own(role)), b.bnot(unsafe));
				for (int square : b.squares(role, type)) {
					const auto present = b.anyset(b.band(b.pcs(role, type), b.rel(role, 1ULL << square)));
					terms.push_back(b.mul(present, cnt(b.band(b.fromatk(role, type, sq(role, square)), area))));
				}
				return b.sum(terms);
			}

			void mobility() {
				constexpr std::array<int, 4> maximum{{8, 13, 14, 27}};
				constexpr std::array<int, 4> reference{{4, 5, 7, 12}};
				constexpr std::array<float, 4> slope{{0.030F, 0.025F, 0.018F, 0.009F}};
				for (int type = 1; type <= 4; ++type) {
					b.group(static_cast<Region>(static_cast<unsigned>(Region::knightMobility) + type - 1), has(type), [&] {
						const auto friendly = mobilitycounts(us, them, type, maximum[static_cast<std::size_t>(type - 1)]);
						const auto enemy = mobilitycounts(them, us, type, maximum[static_cast<std::size_t>(type - 1)]);
						// Material already supplies the omitted reference bucket, so the remaining buckets span every mobility table without a null direction.
						for (int bucket = 0; bucket <= maximum[static_cast<std::size_t>(type - 1)]; ++bucket) {
							if (bucket == reference[static_cast<std::size_t>(type - 1)]) {
								continue;
							}
							const float weight = static_cast<float>(bucket - reference[static_cast<std::size_t>(type - 1)]) * slope[static_cast<std::size_t>(type - 1)];
							reg(diff(friendly[static_cast<std::size_t>(bucket)], enemy[static_cast<std::size_t>(bucket)]), weight * 0.8F, weight, Family::mobility);
						}
						if (type >= 3) {
							reg(diff(secondarymobility(us, them, type), secondarymobility(them, us, type)), type == 3 ? 0.008F : 0.004F, type == 3 ? 0.012F : 0.006F,
							    Family::mobility);
						}
					});
				}
			}

			Id rookline(Id role, Id opponent) {
				Sum terms;
				for (int square : b.squares(role, 3)) {
					const auto present = b.anyset(b.band(b.pcs(role, 3), b.rel(role, 1ULL << square)));
					const auto queens = b.bor(b.pcs(role, 4), b.pcs(opponent, 4));
					terms.push_back(b.mul(present, cnt(b.band(queens, b.bb(filemask(square % 8))))));
				}
				return b.sum(terms);
			}

			Id bishopxray(Id role, Id opponent) {
				Sum terms;
				for (int square : b.squares(role, 2)) {
					const auto present = b.anyset(b.band(b.pcs(role, 2), b.rel(role, 1ULL << square)));
					terms.push_back(b.mul(present, cnt(b.band(b.pcs(opponent, 0), b.rel(role, diagonalmask(square))))));
				}
				return b.sum(terms);
			}

			void pieces() {
				for (int type = 1; type <= 2; ++type) {
					const auto friendly = cnt(b.band(b.pcs(us, type), b.sh(b.pcs(us, 0), them, 0)));
					const auto enemy = cnt(b.band(b.pcs(them, type), b.sh(b.pcs(them, 0), us, 0)));
					reg(diff(friendly, enemy), 0.035F, 0.025F, Family::mobility);
				}
				constexpr std::uint64_t light = 0x55AA55AA55AA55AAULL;
				const auto bishoppawns = [&](Id role) {
					const auto l = b.rel(role, light);
					const auto d = b.bnot(l);
					return b.add(b.mul(cnt(b.band(b.pcs(role, 2), l)), cnt(b.band(b.pcs(role, 0), l))), b.mul(cnt(b.band(b.pcs(role, 2), d)), cnt(b.band(b.pcs(role, 0), d))));
				};
				reg(diff(bishoppawns(us), bishoppawns(them)), -0.006F, -0.008F, Family::general);
				reg(diff(cnt(b.band(b.pcs(us, 2), b.bnot(pawnatt(us)))), cnt(b.band(b.pcs(them, 2), b.bnot(pawnatt(them))))), -0.01F, -0.018F, Family::mobility);
				const auto central = filemask(2) | filemask(3) | filemask(4) | filemask(5);
				const auto fblocked = b.band(b.pcs(us, 0), b.band(b.sh(occ, us, 1), b.rel(us, central & (rankmask(1) | rankmask(2)))));
				const auto eblocked = b.band(b.pcs(them, 0), b.band(b.sh(occ, them, 1), b.rel(them, central & (rankmask(1) | rankmask(2)))));
				reg(diff(b.mul(cnt(b.pcs(us, 2)), cnt(fblocked)), b.mul(cnt(b.pcs(them, 2)), cnt(eblocked))), -0.004F, -0.010F, Family::mobility);
				reg(diff(bishopxray(us, them), bishopxray(them, us)), 0.012F, 0.018F, Family::mobility);
				reg(diff(cnt(b.band(b.pcs(us, 3), b.rel(us, rankmask(6)))), cnt(b.band(b.pcs(them, 3), b.rel(them, rankmask(6))))), 0.11F, 0.08F, Family::mobility);
				reg(diff(rookline(us, them), rookline(them, us)), 0.08F, 0.05F, Family::mobility);

				b.group(Region::rookfiles, has(3), [&] {
					Id fopen = z;
					Id eopen = z;
					Id fsemi = z;
					Id esemi = z;
					for (int file = 0; file < 8; ++file) {
						const auto mask = b.bb(filemask(file));
						const auto frooks = cnt(b.band(b.pcs(us, 3), mask));
						const auto erooks = cnt(b.band(b.pcs(them, 3), mask));
						const auto fp = b.anyset(b.band(b.pcs(us, 0), mask));
						const auto ep = b.anyset(b.band(b.pcs(them, 0), mask));
						fopen = b.add(fopen, b.mul(frooks, b.lnot(b.lor(fp, ep))));
						eopen = b.add(eopen, b.mul(erooks, b.lnot(b.lor(fp, ep))));
						fsemi = b.add(fsemi, b.mul(frooks, b.land(b.lnot(fp), ep)));
						esemi = b.add(esemi, b.mul(erooks, b.land(b.lnot(ep), fp)));
					}
					reg(diff(fopen, eopen), 0.0F, 0.20F, Family::mobility);
					reg(diff(fsemi, esemi), 0.03F, 0.10F, Family::mobility);
				});

				const auto outposts = [&](Id role, Id opponent, int type) {
					const auto ranks = b.rel(role, rankmask(3) | rankmask(4) | rankmask(5));
					const auto future = pawnatt(opponent);
					const auto viable = b.band(ranks, b.band(pawnatt(role), b.bnot(b.fill(future, opponent))));
					return cnt(b.band(b.pcs(role, type), viable));
				};
				for (int type = 1; type <= 2; ++type) {
					reg(diff(outposts(us, them, type), outposts(them, us, type)), 0.10F, 0.12F, Family::mobility);
				}

				const auto restricted = [&](Id role, Id opponent) {
					const auto guarded = strong(role, opponent);
					return cnt(b.band(b.band(b.atk(role), b.atk(opponent)), b.bnot(guarded)));
				};
				reg(diff(restricted(us, them), restricted(them, us)), 0.008F, 0.012F, Family::mobility);
				constexpr std::uint64_t center = 0x00003C3C3C000000ULL;
				const auto space = [&](Id role, Id opponent) { return cnt(b.band(b.band(b.atk(role), b.rel(role, center)), b.bnot(pawnatt(opponent)))); };
				reg(diff(space(us, them), space(them, us)), 0.004F, 0.012F, Family::mobility);
			}

			void threats() {
				const auto evalside = [&](Id role, Id opponent, int victim) {
					const auto targets = b.pcs(opponent, victim);
					const auto guarded = strong(role, opponent);
					const auto weak = b.band(targets, b.bnot(guarded));
					const auto hanging = b.band(targets, b.band(b.atk(role), b.bor(b.bnot(b.atk(opponent)), b.atk2(role))));
					const auto minor = b.band(weak, b.bor(b.atk(role, 1), b.atk(role, 2)));
					const auto rook = b.band(weak, b.atk(role, 3));
					const auto pushatt = b.memo(Shared::push, role, [&] {
						const auto pushed = b.band(b.sh(b.pcs(role, 0), role, 0), b.bnot(occ));
						return b.bor(b.sh(pushed, role, 4), b.sh(pushed, role, 5));
					});
					return std::array<Id, 5>{cnt(hanging), cnt(b.band(targets, pawnatt(role))), cnt(minor), cnt(rook), cnt(b.band(targets, pushatt))};
				};
				const auto kingthreat = [&](Id role, Id opponent) {
					const auto targets = own(opponent);
					const auto guarded = strong(role, opponent);
					return b.anyset(b.band(b.band(targets, b.bnot(guarded)), b.atk(role, 5)));
				};
				const auto queenpressure = [&](Id role, Id opponent, int type) {
					Sum terms;
					const auto guarded = strong(role, opponent);
					for (int square : b.squares(opponent, 4)) {
						const auto queen = b.anyset(b.band(b.pcs(opponent, 4), b.rel(opponent, 1ULL << square)));
						const auto sources = b.band(b.pcs(role, type), b.fromatk(role, type, sq(opponent, square)));
						terms.push_back(b.mul(queen, cnt(b.band(sources, b.bnot(guarded)))));
					}
					return b.sum(terms);
				};

				for (int victim = 0; victim < 5; ++victim) {
					b.group(static_cast<Region>(static_cast<unsigned>(Region::pawnThreat) + victim), has(victim), [&] {
						const auto friendly = evalside(us, them, victim);
						const auto enemy = evalside(them, us, victim);
						const float value = victim == 0 ? 0.07F : victim < 3 ? 0.16F : victim == 3 ? 0.22F : 0.30F;
						reg(diff(friendly[0], enemy[0]), value, value, Family::mobility);
						reg(diff(friendly[1], enemy[1]), value * 0.8F, value, Family::mobility);
						reg(diff(friendly[2], enemy[2]), value * 0.35F, value * 0.45F, Family::mobility);
						reg(diff(friendly[3], enemy[3]), value * 0.30F, value * 0.40F, Family::mobility);
						reg(diff(friendly[4], enemy[4]), value * 0.25F, value * 0.35F, Family::pawn);
					});
				}
				reg(diff(kingthreat(us, them), kingthreat(them, us)), 0.05F, 0.08F, Family::mobility);
				b.group(Region::queenPressure, has(4), [&] {
					for (int type = 1; type <= 3; ++type) {
						const float value = type == 1 ? 0.05F : type == 2 ? 0.06F : 0.04F;
						const auto friendly = queenpressure(us, them, type);
						const auto enemy = queenpressure(them, us, type);
						const auto fqueenless = b.equal(cnt(b.pcs(us, 4)), z);
						const auto equeenless = b.equal(cnt(b.pcs(them, 4)), z);
						reg(diff(b.mul(friendly, b.lnot(fqueenless)), b.mul(enemy, b.lnot(equeenless))), value * 0.5F, value, Family::mobility);
						reg(diff(b.mul(friendly, fqueenless), b.mul(enemy, equeenless)), value * 0.8F, value * 1.4F, Family::mobility);
					}
				});
			}

			Id ringatt(Id attacker, Id defender, int type, int distance) {
				Sum terms;
				for (int square : b.squares(defender, 5)) {
					const auto king = b.anyset(b.band(b.pcs(defender, 5), b.rel(defender, 1ULL << square)));
					const auto ring = b.rel(defender, ringmask(square, distance));
					terms.push_back(b.mul(king, cnt(b.band(b.atk(attacker, type), ring))));
				}
				return b.sum(terms);
			}

			Id potentialchecks(Id attacker, Id defender, int type) {
				Sum terms;
				for (int square : b.squares(defender, 5)) {
					const auto king = b.anyset(b.band(b.pcs(defender, 5), b.rel(defender, 1ULL << square)));
					// Reversing the defender's pawn direction gives the squares from which an attacking pawn checks the king.
					const auto geometry = b.fromatk(type == 0 ? defender : attacker, type, sq(defender, square));
					const auto destinations = b.band(geometry, b.band(b.atk(attacker, type), b.bnot(own(attacker))));
					terms.push_back(b.mul(king, cnt(destinations)));
				}
				return b.sum(terms);
			}

			Id escapes(Id defender, Id attacker) {
				Sum terms;
				for (int square : b.squares(defender, 5)) {
					const auto king = b.anyset(b.band(b.pcs(defender, 5), b.rel(defender, 1ULL << square)));
					const auto zone = b.rel(defender, ringmask(square, 1));
					terms.push_back(b.mul(king, cnt(b.band(zone, b.band(b.bnot(own(defender)), b.bnot(b.atk(attacker)))))));
				}
				return b.sum(terms);
			}

			Id kingpawns(Id defender, Id attacker, int distance, bool friendly) {
				Sum terms;
				for (int square : b.squares(defender, 5)) {
					const int rank = square / 8 + distance;
					if (rank >= 8) {
						continue;
					}
					const int file = square % 8;
					const auto king = b.anyset(b.band(b.pcs(defender, 5), b.rel(defender, 1ULL << square)));
					const auto pawns = friendly ? b.band(b.pcs(defender, 0), b.bnot(pawnatt(attacker))) : b.pcs(attacker, 0);
					for (int candidate = std::max(0, file - 1); candidate <= std::min(7, file + 1); ++candidate) {
						const auto exact = b.anyset(b.band(pawns, b.rel(defender, 1ULL << (rank * 8 + candidate))));
						std::uint64_t closer = 0;
						for (int near = square / 8 + 1; near < rank; ++near) {
							closer |= 1ULL << (near * 8 + candidate);
						}
						const auto nearest = closer == 0 ? exact : b.land(exact, b.lnot(b.anyset(b.band(pawns, b.rel(defender, closer)))));
						terms.push_back(b.land(king, nearest));
					}
				}
				return b.sum(terms);
			}

			Id kingopen(Id role) {
				Sum terms;
				const auto allpawns = b.bor(b.pcs(us, 0), b.pcs(them, 0));
				for (int square : b.squares(role, 5)) {
					Sum opened;
					for (int file = std::max(0, square % 8 - 1); file <= std::min(7, square % 8 + 1); ++file) {
						opened.push_back(b.lnot(b.anyset(b.band(allpawns, b.bb(filemask(file))))));
					}
					const auto king = b.anyset(b.band(b.pcs(role, 5), b.rel(role, 1ULL << square)));
					terms.push_back(b.mul(king, b.sum(opened)));
				}
				return b.sum(terms);
			}

			Id flank(Id map, Id defender) {
				Sum terms;
				for (int square : b.squares(defender, 5)) {
					std::uint64_t mask = 0;
					for (int file = std::max(0, square % 8 - 1); file <= std::min(7, square % 8 + 1); ++file) {
						mask |= filemask(file);
					}
					const auto king = b.anyset(b.band(b.pcs(defender, 5), b.rel(defender, 1ULL << square)));
					terms.push_back(b.mul(king, cnt(b.band(map, b.rel(defender, mask)))));
				}
				return b.sum(terms);
			}

			void kings() {
				Sum friendly;
				Sum enemy;
				for (int type = 0; type < 5; ++type) {
					const auto fi = ringatt(us, them, type, 1);
					const auto ei = ringatt(them, us, type, 1);
					friendly.push_back(fi);
					enemy.push_back(ei);
					const float pressure = type == 0 ? 0.02F : type < 3 ? 0.04F : type == 3 ? 0.05F : 0.07F;
					reg(diff(fi, ei), pressure * 0.25F, pressure, Family::king);
					reg(diff(ringatt(us, them, type, 2), ringatt(them, us, type, 2)), pressure * 0.10F, pressure * 0.35F, Family::king);
					reg(diff(potentialchecks(us, them, type), potentialchecks(them, us, type)), pressure * 0.15F, pressure * 0.45F, Family::king);
				}
				const auto fp = b.sum(friendly);
				const auto ep = b.sum(enemy);
				for (const int threshold : {2, 4, 6, 8}) {
					reg(diff(b.ge(fp, b.num(threshold)), b.ge(ep, b.num(threshold))), 0.0F, static_cast<float>(threshold) / 400.0F, Family::king);
				}
				reg(diff(escapes(us, them), escapes(them, us)), 0.01F, 0.02F, Family::king);
				b.group(Region::kingPawns, has(0), [&] {
					for (int distance = 1; distance <= 3; ++distance) {
						const float shelter = static_cast<float>(4 - distance) / 400.0F;
						const float storm = static_cast<float>(4 - distance) / 500.0F;
						reg(diff(kingpawns(us, them, distance, true), kingpawns(them, us, distance, true)), shelter * 0.5F, shelter, Family::king);
						reg(diff(kingpawns(them, us, distance, false), kingpawns(us, them, distance, false)), storm * 0.5F, storm, Family::king);
					}
				});
				reg(diff(kingopen(us), kingopen(them)), -0.015F, -0.04F, Family::king);
				reg(diff(flank(b.atk(us), us), flank(b.atk(them), them)), 0.001F, 0.002F, Family::king);
				reg(diff(flank(b.atk(us), them), flank(b.atk(them), us)), 0.001F, 0.003F, Family::king);
				reg(diff(flank(b.atk2(us), us), flank(b.atk2(them), them)), 0.002F, 0.004F, Family::king);
				reg(diff(flank(b.atk2(us), them), flank(b.atk2(them), us)), 0.002F, 0.006F, Family::king);
				const auto fblockedstorm = b.band(b.pcs(us, 0), b.sh(b.pcs(them, 0), them, 0));
				const auto eblockedstorm = b.band(b.pcs(them, 0), b.sh(b.pcs(us, 0), us, 0));
				reg(diff(flank(fblockedstorm, us), flank(eblockedstorm, them)), 0.006F, 0.014F, Family::king);
				const auto fnoqueen = b.equal(cnt(b.pcs(us, 4)), z);
				const auto enoqueen = b.equal(cnt(b.pcs(them, 4)), z);
				reg(diff(b.mul(fp, enoqueen), b.mul(ep, fnoqueen)), 0.0F, -0.01F, Family::king);
				for (int side = 0; side < 2; ++side) {
					const auto friendlyright = b.right(us, side);
					const auto enemyright = b.right(them, side);
					reg(diff(friendlyright, enemyright), 0.0F, side == 0 ? 0.03F : 0.02F, Family::king);
				}
			}

			Id nonpawn(Id role) {
				Id total = z;
				constexpr std::array<int, 5> value{{0, 3, 3, 5, 9}};
				for (int type = 1; type <= 4; ++type) {
					total = b.add(total, b.mul(b.num(value[static_cast<std::size_t>(type)]), cnt(b.pcs(role, type))));
				}
				return total;
			}

			void endgames() {
				constexpr std::uint64_t light = 0x55AA55AA55AA55AAULL;
				const auto onefb = b.equal(cnt(b.pcs(us, 2)), o);
				const auto oneeb = b.equal(cnt(b.pcs(them, 2)), o);
				const auto opposite = b.lor(b.land(b.anyset(b.band(b.pcs(us, 2), b.bb(light))), b.anyset(b.band(b.pcs(them, 2), b.bnot(b.bb(light))))),
				    b.land(b.anyset(b.band(b.pcs(us, 2), b.bnot(b.bb(light)))), b.anyset(b.band(b.pcs(them, 2), b.bb(light)))));
				const auto fpawns = cnt(b.pcs(us, 0));
				const auto epawns = cnt(b.pcs(them, 0));
				const auto material = b.add(diff(nonpawn(us), nonpawn(them)), diff(fpawns, epawns));
				const auto positive = b.gt(material, z);
				const auto negative = b.lt(material, z);
				const auto direction = diff(positive, negative);
				reg(b.mul(b.land(b.land(onefb, oneeb), opposite), direction), -0.10F, 0.0F, Family::general);
				const auto fpawnless = b.equal(fpawns, z);
				const auto epawnless = b.equal(epawns, z);
				const auto thin = b.le(b.abs(diff(nonpawn(us), nonpawn(them))), o);
				reg(diff(b.land(b.land(positive, fpawnless), thin), b.land(b.land(negative, epawnless), thin)), -0.30F, 0.0F, Family::general);

				Id symmetric = z;
				Id asymmetric = z;
				for (int file = 0; file < 8; ++file) {
					const auto fp = b.anyset(b.band(b.pcs(us, 0), b.bb(filemask(file))));
					const auto ep = b.anyset(b.band(b.pcs(them, 0), b.bb(filemask(file))));
					symmetric = b.add(symmetric, b.land(fp, ep));
					asymmetric = b.add(asymmetric, b.lor(b.land(fp, b.lnot(ep)), b.land(ep, b.lnot(fp))));
				}
				reg(b.mul(direction, b.add(fpawns, epawns)), 0.004F, 0.0F, Family::general);
				reg(b.mul(direction, symmetric), -0.010F, 0.0F, Family::general);
				reg(b.mul(direction, asymmetric), 0.014F, 0.0F, Family::general);
				reg(b.mul(direction, b.equal(phase, z)), 0.04F, 0.0F, Family::general);
				const auto strongpawns = diff(b.mul(positive, fpawns), b.mul(negative, epawns));
				reg(strongpawns, 0.012F, 0.0F, Family::general);
				const auto strongpassers = diff(b.mul(positive, cnt(passers(us, them))), b.mul(negative, cnt(passers(them, us))));
				reg(b.mul(b.land(b.land(onefb, oneeb), opposite), strongpassers), 0.025F, 0.0F, Family::general);
			}

			B b;
			Id z = 0;
			Id o = 0;
			Id us = 0;
			Id them = 0;
			Id occ = 0;
			Id phase = 0;
			Id open = 0;
			Id freedom = 0;
			Id exposure = 0;
		};

	} // namespace

#ifdef ELEGINUS_COMPILE
	void Graph::validate() const {
		if (nodes_.empty() || roots_.empty())
			throw std::runtime_error("empty formula program");
		if (families_.size() * 4 + kContext != roots_.size() || std::any_of(families_.begin(), families_.end(), [](auto f) { return f >= 3; }))
			throw std::runtime_error("invalid formula interpolation families");
		if (std::any_of(sizes_.begin(), sizes_.end(), [](auto size) { return size == 0 || size % 4 != 0; }))
			throw std::runtime_error("invalid formula activation region");
		for (Id id = 0; id < nodes_.size(); ++id) {
			const auto &n = nodes_[id];
			const int count = arity(n.op);
			if (count < 0 || n.type > Type::B1 || n.aux != 0)
				throw std::runtime_error("invalid instruction");
			if ((count >= 1 && n.a >= id) || (count >= 2 && n.b >= id) || (count >= 3 && n.c >= id)) {
				throw std::runtime_error("formula DAG is not topologically ordered");
			}
			const auto numeric = [](Type t) { return t == Type::F64 || t == Type::B1; };
			const auto a = count >= 1 ? nodes_[n.a].type : n.type;
			const auto b = count >= 2 ? nodes_[n.b].type : n.type;
			if (n.op == Op::IMM) {
				if (n.type != Type::U64 && (!std::isfinite(real(n.imm)) || (n.type == Type::B1 && real(n.imm) != 0 && real(n.imm) != 1))) {
					throw std::runtime_error("invalid numeric literal");
				}
			} else if (n.op == Op::LD) {
				if (n.imm >= 15 || n.type != (n.imm == 12 ? Type::F64 : Type::U64))
					throw std::runtime_error("invalid input register");
			} else if (n.op >= Op::NOT && n.op <= Op::CTZ) {
				if (a != Type::U64 || (count == 2 && b != Type::U64) || n.type != Type::U64)
					throw std::runtime_error("invalid bit instruction types");
			} else if (n.op == Op::POP) {
				if (a != Type::U64 || n.type != Type::F64)
					throw std::runtime_error("invalid POP types");
			} else if (n.op == Op::SEL) {
				const auto c = nodes_[n.c].type;
				if (a != Type::B1 ||
				    !((b == Type::U64 && c == Type::U64 && n.type == Type::U64) || (numeric(b) && numeric(c) && n.type == Type::F64) ||
				        (b == Type::B1 && c == Type::B1 && n.type == Type::B1))) {
					throw std::runtime_error("invalid SEL types");
				}
			} else if (n.op >= Op::EQ && n.op <= Op::GE) {
				if (n.type != Type::B1 || (!(numeric(a) && numeric(b)) && !(n.op == Op::EQ && a == Type::U64 && b == Type::U64))) {
					throw std::runtime_error("invalid comparison types");
				}
			} else if (!numeric(a) || (count == 2 && !numeric(b)) || n.type != Type::F64) {
				throw std::runtime_error("invalid arithmetic types");
			}
		}
		for (const auto &r : roots_) {
			if (r.node >= nodes_.size() || nodes_[r.node].type == Type::U64 || !std::isfinite(r.weight))
				throw std::runtime_error("invalid formula root");
		}
	}

	void Graph::write(const char *path) const {
		std::uint64_t hash = 14695981039346656037ULL;
		const auto add = [&](std::uint64_t x) {
			for (int i = 0; i < 8; ++i) {
				hash = (hash ^ (x & 255)) * 1099511628211ULL;
				x >>= 8;
			}
		};
		for (const auto &n : nodes_) {
			add(static_cast<unsigned>(n.op));
			add(static_cast<unsigned>(n.type));
			add(n.a);
			add(n.b);
			add(n.c);
			add(n.imm);
		}
		for (const auto &r : roots_)
			add(r.node);
		add(std::bit_cast<std::uint32_t>(kInputScale));
		std::ofstream out(path, std::ios::trunc);
		out.exceptions(std::ios::badbit | std::ios::failbit);
		out << "// Generated fixed formula metadata.\n#pragma once\n#include <cstdint>\nnamespace eleginus::catalog {\n";
		out << "inline constexpr float weights[]{\n" << std::scientific << std::setprecision(std::numeric_limits<float>::max_digits10);
		for (std::size_t i = 0; i < roots_.size(); ++i)
			out << roots_[i].weight << "F," << (i % 8 == 7 ? "\n" : " ");
		out << "\n};\ninline constexpr std::uint8_t families[]{\n";
		for (std::size_t i = 0; i < families_.size(); ++i)
			out << static_cast<unsigned>(families_[i]) << "," << (i % 24 == 23 ? "\n" : " ");
		out << "\n};\ninline constexpr unsigned sizes[]{\n";
		for (auto size : sizes_)
			out << size << ",";
		out << "\n};\ninline constexpr std::uint64_t signature = " << hash << "ULL;\n}\n";
	}
#else
	const Program &Program::fixed() {
		static const Program p = [] {
			Program result;
			result.weights_ = catalog::weights;
			result.families_ = catalog::families;
			result.signature_ = catalog::signature;
			return result;
		}();
		return p;
	}

	void Program::evaluate(const chess::Board &board, std::vector<Feature> &out) {
		out.clear();
		Features sink{out};
		Formulas<Native<Features>>(Native(board, sink)).build();
	}

	void Program::evaluate(const chess::Board &board, Evaluator &out) {
		Projection sink{out};
		Formulas<Native<Projection>>(Native(board, sink)).build();
	}

#endif
} // namespace eleginus

#ifdef ELEGINUS_COMPILE
int main(int argc, char **argv) {
	try {
		if (argc != 2)
			throw std::invalid_argument("expected metadata output path");
		const auto graph = eleginus::Formulas<eleginus::Builder>().build();
		graph.validate();
		graph.write(argv[1]);
		return 0;
	} catch (const std::exception &e) {
		std::cerr << "formula compilation failed: " << e.what() << '\n';
		return 1;
	}
}
#endif
