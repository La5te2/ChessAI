#pragma once

#include "eleginus/parameters.hpp"
#include "chess.hpp"
#include <array>
#include <cstdint>
#include <memory>

namespace eleginus {

	// Raw king-attack counts and the formula index at which their sigmoid difference is emitted.
	struct PressureValues {
		std::size_t formula = formulaCount;
		std::array<std::int32_t, 2> attacks{};
	};

	// Board facts used by the sign-dependent winnability and endgame-scaling adjustments.
	struct EndgameValues {
		std::int32_t pawns = 0;
		std::int32_t symmetricFiles = 0;
		std::int32_t asymmetricFiles = 0;
		std::int32_t pawnEnding = 0;
		std::array<std::int32_t, 2> sidePawns{};
		std::array<std::int32_t, 2> sidePassers{};
		std::int32_t oppositeBishops = 0;
		std::array<std::int32_t, 2> pawnless{};
		std::int32_t thinMaterial = 0;
		std::int32_t pureOppositeBishops = 0;
		std::int32_t mixedOppositeBishops = 0;
	};

	// Parameter-independent formula signals and adjustment inputs extracted from one board.
	struct FormulaValues {
		std::array<std::int32_t, formulaCount> signals{};
		std::array<float, 5> material{};
		PressureValues pressure;
		EndgameValues endgame;
	};

	// Store one parameter set and precompute its reusable evaluation data.
	class Evaluator {
	public:
		explicit Evaluator(FormulaParameters parameters);
		Evaluator(const Evaluator &) = delete;
		Evaluator &operator=(const Evaluator &) = delete;
		~Evaluator();

		// Evaluate a board from White's perspective.
		float score(const chess::Board &board) const;
		// Evaluate previously extracted board values with this parameter set.
		float score(const FormulaValues &values) const;

		// Extract the formula signals and adjustment inputs of one board.
		FormulaValues extract(const chess::Board &board) const;

	private:
		class State;
		std::unique_ptr<State> state;
	};

} // namespace eleginus
