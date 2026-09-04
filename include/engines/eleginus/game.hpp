#pragma once

#include "chess.hpp"
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace eleginus {
	using PackedBoard = chess::PackedBoard;
	inline constexpr std::size_t packedBoardSize = PackedBoard{}.size();

	std::vector<chess::Move> legalmoves(const chess::Board &board);
	std::string moveToUci(const chess::Move &move);
	bool isGameOver(const chess::Board &board);
	PackedBoard packBoard(const chess::Board &board);
	chess::Board unpackBoard(const PackedBoard &packed);

} // namespace eleginus
