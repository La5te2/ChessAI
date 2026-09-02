#pragma once

#include "chess.hpp"
#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace eleginus {
	inline constexpr std::size_t kFormulaCount = 729;
	using FormulaMask = std::array<std::uint64_t, (kFormulaCount + 63) / 64>;

	// A formula coordinate is an integer, directional HCE signal. Zero coordinates are omitted.
	struct Feature {
		std::uint16_t index = 0;
		std::int32_t score = 0;
		std::int32_t condition = 0;
	};

	class FormulaSet {
	public:
		static void evaluate(const chess::Board &board, std::vector<Feature> &out);
		static float evaluate(const chess::Board &board, std::span<const float> weights);
		static float evaluate(const chess::Board &board, std::span<const float> weights, const FormulaMask &active);
		static float evaluate(const chess::Board &board, std::span<const float> base, std::span<const std::uint16_t> rows,
			std::span<const std::uint16_t> conditions, std::span<const float> relations);
		static float evaluate(const chess::Board &board, std::span<const float> base, std::span<const std::uint16_t> rows,
			std::span<const std::uint16_t> conditions, std::span<const float> relations, const FormulaMask &active);
	};

} // namespace eleginus

namespace eleginus::detail {
	std::span<const float> initial() noexcept;
	void extract(const chess::Board &board, std::vector<Feature> &out);
	float score(const chess::Board &board, std::span<const float> weights);
	float score(const chess::Board &board, std::span<const float> weights, const FormulaMask &active);
	float score(const chess::Board &board, std::span<const float> base, std::span<const std::uint16_t> rows,
		std::span<const std::uint16_t> conditions, std::span<const float> relations);
	float score(const chess::Board &board, std::span<const float> base, std::span<const std::uint16_t> rows,
		std::span<const std::uint16_t> conditions, std::span<const float> relations, const FormulaMask &active);
} // namespace eleginus::detail
