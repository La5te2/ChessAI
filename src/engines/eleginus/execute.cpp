#include "eleginus/formula.hpp"

namespace eleginus {

	void FormulaSet::evaluate(const chess::Board &board, std::vector<Feature> &out) { detail::extract(board, out); }

	float FormulaSet::evaluate(const chess::Board &board, std::span<const float> weights) { return detail::score(board, weights); }

	float FormulaSet::evaluate(const chess::Board &board, std::span<const float> base, std::span<const std::uint16_t> rows,
	    std::span<const std::uint16_t> conditions, std::span<const float> relations) {
		return detail::score(board, base, rows, conditions, relations);
	}

} // namespace eleginus
