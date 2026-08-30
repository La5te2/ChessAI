#include "eleginus/game.hpp"

namespace eleginus {

std::vector<chess::Move> legal_moves(const chess::Board &board) {
	chess::Movelist moves;
	chess::movegen::legalmoves(moves, board);
	return {moves.begin(), moves.end()};
}

std::string move_uci(const chess::Move &move) {
	return chess::uci::moveToUci(move);
}

bool game_is_over(const chess::Board &board) {
	return board.isGameOver().first != chess::GameResultReason::NONE;
}

} // namespace eleginus
