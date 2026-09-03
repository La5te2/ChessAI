#pragma once

#include "chess.hpp"
#include <cstdint>
#include <span>
#include <vector>

namespace eleginus {
	inline constexpr std::size_t kFormulaCount = 729;

	// A formula coordinate is an integer, directional HCE signal. Zero coordinates are omitted.
	struct Feature {
		std::uint16_t index = 0;
		std::int32_t score = 0;
		std::int32_t condition = 0;
	};

	class FormulaSet {
	public:
		static void evaluate(const chess::Board &board, std::vector<Feature> &out);
	};

} // namespace eleginus

namespace eleginus::detail {
	std::span<const float> initial() noexcept;
	void extract(const chess::Board &board, std::vector<Feature> &out);
} // namespace eleginus::detail
