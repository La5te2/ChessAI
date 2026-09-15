#pragma once

#include "chess.hpp"

namespace eleginus {

	// Evaluate the fixed source-defined formula set from White's perspective.
	float evaluate(const chess::Board &board);

} // namespace eleginus
