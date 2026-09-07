#pragma once

#include "chess.hpp"
#include <string>
#include <vector>

namespace eleginus {
	// Return every legal move as a standard container for engine-facing callers.
	std::vector<chess::Move> legalmoves(const chess::Board &board);
	// Format one internal move with the chess library's UCI notation.
	std::string moveToUci(const chess::Move &move);
	// Report whether any chess-library terminal rule has ended the game.
	bool isGameOver(const chess::Board &board);

} // namespace eleginus
