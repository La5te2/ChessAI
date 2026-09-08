#pragma once

#include "eleginus/formula.hpp"
#include <array>
#include <cstddef>
#include <filesystem>

namespace eleginus {

	// The parameter payload contains six coefficients per formula and fourteen score-adjustment values.
	inline constexpr std::size_t parameterCount = 6 * formulaCount + 14;
	using ParameterValues = std::array<float, parameterCount>;

	// The effective formula coefficient equals base plus the material response dotted with five
	// normalized counts: pawns, knights, bishops, rooks and queens.
	struct FormulaWeights {
		float base = 0.0F;
		std::array<float, 5> material{};
	};

	// These values control the king-pressure sigmoid, winnability adjustment and endgame scaling.
	struct AdjustmentWeights {
		float pressureCenter = 0.0F;
		float pressureWidth = 1.0F;
		std::array<float, 7> winnability{};
		std::array<float, 5> scaling{};
	};

	// All coefficients used by one evaluator.
	struct FormulaParameters {
		std::array<FormulaWeights, formulaCount> formulas{};
		AdjustmentWeights adjustments;
	};

	// Require finite coefficients and a positive king-pressure sigmoid width.
	void validateParameters(const FormulaParameters &parameters);
	// Return the source-defined starting parameters.
	FormulaParameters initialParameters();

	// Convert between structured parameters and the formula-major order used by files and tensors.
	ParameterValues flattenParameters(const FormulaParameters &parameters);
	FormulaParameters expandParameters(const ParameterValues &values);

	// Read or atomically replace one validated parameter file.
	FormulaParameters loadParameters(const std::filesystem::path &path);
	void saveParameters(const std::filesystem::path &path, const FormulaParameters &parameters);

} // namespace eleginus
