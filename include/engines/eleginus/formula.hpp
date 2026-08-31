#pragma once

#include "chess.hpp"
#include <cstdint>
#include <span>
#include <vector>

namespace eleginus {

	inline constexpr std::size_t kContext = 16;
	inline constexpr float kInputScale = 0.125F;

	struct Feature {
		std::uint32_t index = 0;
		float value = 0.0F;
	};

	class Evaluator;

	class Program {
	public:
		static const Program &fixed();
		static void evaluate(const chess::Board &board, std::vector<Feature> &out);
		static void evaluate(const chess::Board &board, Evaluator &out);

		std::span<const float> weights() const noexcept { return weights_; }
		std::span<const std::uint8_t> families() const noexcept { return families_; }
		std::uint64_t signature() const noexcept { return signature_; }

	private:
		std::span<const float> weights_;
		std::span<const std::uint8_t> families_;
		std::uint64_t signature_;
	};

} // namespace eleginus
