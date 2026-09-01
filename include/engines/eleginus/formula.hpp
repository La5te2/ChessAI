#pragma once

#include "chess.hpp"
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace eleginus {
	inline constexpr std::size_t kFormulaCount = 622;

	// A formula coordinate is an integer, directional HCE signal. Zero coordinates are omitted.
	struct Feature {
		std::uint16_t index = 0;
		std::int32_t score = 0;
		std::int32_t condition = 0;
	};

	class FormulaSet {
	public:
		static const FormulaSet &fixed();
		static void evaluate(const chess::Board &board, std::vector<Feature> &out);
		static float evaluate(const chess::Board &board, std::span<const float> weights);
		static float evaluate(const chess::Board &board, std::span<const float> base, std::span<const std::uint16_t> rows,
		    std::span<const std::uint16_t> conditions, std::span<const float> relations);

		std::size_t size() const noexcept { return kFormulaCount; }
		std::span<const float> initial() const noexcept;
		std::span<const std::string_view> names() const noexcept;
	};

} // namespace eleginus

namespace eleginus::detail {
	void extract(const chess::Board &board, std::vector<Feature> &out);
	float score(const chess::Board &board, std::span<const float> weights);
	float score(const chess::Board &board, std::span<const float> base, std::span<const std::uint16_t> rows,
	    std::span<const std::uint16_t> conditions, std::span<const float> relations);
} // namespace eleginus::detail
