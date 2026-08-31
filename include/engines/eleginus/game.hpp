#pragma once

#include "chess.hpp"
#include <string>
#include <vector>

namespace eleginus {

	std::vector<chess::Move> legalmoves(const chess::Board &board);
	std::string moveToUci(const chess::Move &move);
	bool isGameOver(const chess::Board &board);

} // namespace eleginus
