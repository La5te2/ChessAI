#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace eleginus {

	// Irreducible board inputs used by the fixed formula language.
	enum class Atom : std::uint8_t { WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK, STM, CR, COUNT };

	inline constexpr std::size_t atomCount = static_cast<std::size_t>(Atom::COUNT);
	inline constexpr std::size_t atomIndex(Atom atom) noexcept {
		return static_cast<std::size_t>(atom);
	}
	inline constexpr std::array<std::array<Atom, 6>, 2> pieceAtoms{{
		{Atom::WP, Atom::WN, Atom::WB, Atom::WR, Atom::WQ, Atom::WK},
		{Atom::BP, Atom::BN, Atom::BB, Atom::BR, Atom::BQ, Atom::BK},
	}};
	inline constexpr std::size_t pieceAtomIndex(std::size_t color, std::size_t type) noexcept {
		return 6 * color + type;
	}
	static_assert([] {
		for (std::size_t color = 0; color < pieceAtoms.size(); ++color) {
			for (std::size_t type = 0; type < pieceAtoms[color].size(); ++type) {
				if (atomIndex(pieceAtoms[color][type]) != pieceAtomIndex(color, type)) return false;
			}
		}
		return true;
	}());

} // namespace eleginus
