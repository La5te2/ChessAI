#pragma once

#include "chess.hpp"
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace eleginus {

	using Word = std::uint64_t;

	// A signal is the single value representation used throughout the formula algebra.
	struct Signal {
		Word bits = 0;
		Signal() = default;
		explicit Signal(Word value) : bits(value) {}
	};

	// These aliases identify the producing layer without changing representation.
	using AtomSignal = Signal;
	using InterSignal = Signal;

	// The fourteen irreducible board inputs of the formula language.
	enum class Atom : std::uint8_t { WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK, STM, CR, COUNT };

	inline constexpr std::size_t atomCount = static_cast<std::size_t>(Atom::COUNT);
	inline constexpr std::size_t atomIndex(Atom atom) noexcept { return static_cast<std::size_t>(atom); }
	inline constexpr std::size_t pieceAtomIndex(std::size_t color, std::size_t type) noexcept { return 6 * color + type; }

	inline std::int64_t number(Word value) noexcept { return std::bit_cast<std::int64_t>(value); }

	// FormulaAtoms defines the irreducible operations available to every formula.
	template <class Backend> class FormulaAtoms {
	public:
		// NUM constructs a signed integer signal; BB constructs a raw bitboard signal.
		AtomSignal NUM(std::int64_t value) const { return AtomSignal(static_cast<Word>(value)); }
		AtomSignal BB(Word value) const { return AtomSignal(value); }

		// INP reads one irreducible board input.
		AtomSignal INP(Atom atom) const { return AtomSignal(backend().input(atom)); }

		// Integer arithmetic primitives.
		AtomSignal ADD(InterSignal a, InterSignal b) const { return NUM(number(a.bits) + number(b.bits)); }
		AtomSignal SUB(InterSignal a, InterSignal b) const { return NUM(number(a.bits) - number(b.bits)); }
		AtomSignal MUL(InterSignal a, InterSignal b) const { return NUM(number(a.bits) * number(b.bits)); }
		AtomSignal ABS(InterSignal value) const { return NUM(std::abs(number(value.bits))); }

		// Logical primitives map their results to zero and one.
		AtomSignal LAND(InterSignal a, InterSignal b) const { return NUM(number(a.bits) != 0 && number(b.bits) != 0); }
		AtomSignal LOR(InterSignal a, InterSignal b) const { return NUM(number(a.bits) != 0 || number(b.bits) != 0); }
		AtomSignal LNOT(InterSignal value) const { return NUM(number(value.bits) == 0); }

		// Comparison primitives map their results to zero and one.
		AtomSignal EQ(InterSignal a, InterSignal b) const { return NUM(a.bits == b.bits); }
		AtomSignal GT(InterSignal a, InterSignal b) const { return NUM(number(a.bits) > number(b.bits)); }
		AtomSignal LT(InterSignal a, InterSignal b) const { return NUM(number(a.bits) < number(b.bits)); }
		AtomSignal LE(InterSignal a, InterSignal b) const { return NUM(number(a.bits) <= number(b.bits)); }
		AtomSignal GE(InterSignal a, InterSignal b) const { return NUM(number(a.bits) >= number(b.bits)); }

		// Bitboard primitives combine sets or reduce them to integers.
		AtomSignal AND(InterSignal a, InterSignal b) const { return BB(a.bits & b.bits); }
		AtomSignal OR(InterSignal a, InterSignal b) const { return BB(a.bits | b.bits); }
		AtomSignal NOT(InterSignal value) const { return BB(~value.bits); }
		AtomSignal POP(InterSignal value) const { return NUM(std::popcount(value.bits)); }
		AtomSignal ANY(InterSignal value) const { return NUM(value.bits != 0); }

		// Board primitives select piece sets, normalize coordinates and expose castling rights.
		AtomSignal PCS(InterSignal role, int type) const {
			return BB(backend().input(static_cast<Atom>(pieceAtomIndex(color(role), static_cast<std::size_t>(type)))));
		}
		AtomSignal REL(InterSignal role, Word mask) const { return BB(color(role) == 0 ? mask : flip(mask)); }
		AtomSignal SQ(InterSignal role, int square) const { return NUM(square ^ (color(role) == 0 ? 0 : 56)); }
		AtomSignal CR(InterSignal role, int wing) const {
			return NUM((backend().input(Atom::CR) & (1ULL << (2 * color(role) + wing))) != 0);
		}

		// SH performs one role-relative step in an orthogonal or diagonal direction.
		AtomSignal SH(InterSignal value, InterSignal role, int direction) const {
			const Word east = value.bits & 0x7F7F7F7F7F7F7F7FULL;
			const Word west = value.bits & 0xFEFEFEFEFEFEFEFEULL;
			const bool white = color(role) == 0;
			switch (direction) {
				case 0: return BB(white ? value.bits << 8 : value.bits >> 8);
				case 1: return BB(white ? value.bits >> 8 : value.bits << 8);
				case 2: return BB(east << 1);
				case 3: return BB(west >> 1);
				case 4: return BB(white ? east << 9 : east >> 7);
				case 5: return BB(white ? west << 7 : west >> 9);
				case 6: return BB(white ? east >> 7 : east << 9);
				case 7: return BB(white ? west >> 9 : west << 7);
				default: throw std::logic_error("invalid shift direction");
			}
		}

	private:
		const Backend &backend() const noexcept { return static_cast<const Backend &>(*this); }
		static unsigned color(InterSignal role) { return static_cast<unsigned>(number(role.bits)); }
		static Word flip(Word value) noexcept {
			value = ((value & 0x00FF00FF00FF00FFULL) << 8) | ((value >> 8) & 0x00FF00FF00FF00FFULL);
			value = ((value & 0x0000FFFF0000FFFFULL) << 16) | ((value >> 16) & 0x0000FFFF0000FFFFULL);
			return (value << 32) | (value >> 32);
		}
	};

	class FormulaSet {
	public:
		static float score(const chess::Board &board);
	};

} // namespace eleginus
