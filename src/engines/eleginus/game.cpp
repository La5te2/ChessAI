#include "eleginus/game.hpp"

namespace eleginus {
	std::vector<chess::Move> legalmoves(const chess::Board &board) {
		chess::Movelist moves;
		chess::movegen::legalmoves(moves, board);
		return {moves.begin(), moves.end()};
	}

	std::string moveToUci(const chess::Move &move) {
		return chess::uci::moveToUci(move);
	}

	bool isGameOver(const chess::Board &board) {
		return board.isGameOver().first != chess::GameResultReason::NONE;
	}

	PackedBoard packBoard(const chess::Board &board) { return chess::Board::Compact::encode(board); }

	chess::Board unpackBoard(const PackedBoard &packed) { return chess::Board::Compact::decode(packed); }

} // namespace eleginus
