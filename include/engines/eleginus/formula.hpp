#pragma once

#include "chess.hpp"

namespace eleginus {
	class FormulaSet {
	public:
		static float score(const chess::Board &board);
	};

} // namespace eleginus
