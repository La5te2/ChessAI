#pragma once

#include <cstddef>
#include <cstdint>

namespace eleginus {

	// Irreducible board inputs used by the fixed formula language.
	enum class Atom : std::uint8_t { WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK, STM, CR, COUNT };

	inline constexpr std::size_t atomCount = static_cast<std::size_t>(Atom::COUNT);
	inline constexpr std::size_t atomIndex(Atom atom) noexcept { return static_cast<std::size_t>(atom); }

} // namespace eleginus
