#pragma once

#include "chess.hpp"
#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace eleginus {
	struct FormulaContext {
		std::array<float, 5> material{};
		std::uint32_t pressureIndex = 0;
		std::array<float, 2> pressure{};
		std::array<std::array<float, 7>, 2> winnable{};
		std::array<std::array<float, 5>, 2> scale{};
	};

	struct FormulaParameter {
		float base = 0.0F;
		std::array<float, 5> material{};
	};

	struct FormulaGlobals {
		float pressureCenter = 0.0F;
		float pressureWidth = 0.0F;
		std::array<float, 7> winnable{};
		std::array<float, 5> scale{};
	};

	class FormulaSet {
	public:
		static float score(const chess::Board &board);
		static void features(const chess::Board &board, std::span<std::int32_t> signals, FormulaContext &context);
		static std::vector<FormulaParameter> parameters();
		static FormulaGlobals globals();
	};

} // namespace eleginus
