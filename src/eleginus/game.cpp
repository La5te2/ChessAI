// Implements the Eleginus rule adapter without depending on another Gadidae architecture.

#include "eleginus/game.hpp"

namespace eleginus {

std::vector<chess::Move> legal_moves(const chess::Board &board) {
	chess::Movelist moves;
	chess::movegen::legalmoves(moves, board);
	return {moves.begin(), moves.end()};
}

namespace {

int encoded_destination(const chess::Move &move) {
	if (move.typeOf() != chess::Move::CASTLING) {
		return move.to().index();
	}
	const int from = move.from().index();
	const int rank = from / 8;
	return rank * 8 + (move.to().index() > from ? 6 : 2);
}

int oriented_square(int square, chess::Color side_to_move) {
	return side_to_move == chess::Color::WHITE ? square : square ^ 56;
}

bool action_frame_is_mirrored(const chess::Board &board) {
	const auto side = board.sideToMove();
	const int king = oriented_square(board.kingSq(side).index(), side);
	return king % 8 < 4;
}

} // namespace

int move_to_index(const chess::Move &move, chess::Color side_to_move,
				  bool horizontal_mirror) {
	int from = oriented_square(move.from().index(), side_to_move);
	int to = oriented_square(encoded_destination(move), side_to_move);
	if (horizontal_mirror) {
		from ^= 7;
		to ^= 7;
	}
	if (move.typeOf() == chess::Move::PROMOTION &&
		move.promotionType() != chess::PieceType::QUEEN) {
		const int direction = to % 8 - from % 8 + 1;
		const int piece = static_cast<int>(move.promotionType().internal()) -
						  static_cast<int>(chess::PieceType::KNIGHT);
		if (direction < 0 || direction > 2 || piece < 0 || piece > 2) {
			return -1;
		}
		return kBoardSquares * kBoardSquares + from * kUnderpromotionPlanes +
			   direction * 3 + piece;
	}
	return from * kBoardSquares + to;
}

int move_to_index(const chess::Move &move, const chess::Board &board) {
	return move_to_index(move, board.sideToMove(), action_frame_is_mirrored(board));
}

chess::Move index_to_move(int index, const chess::Board &board) {
	if (index < 0 || index >= kActionSize) {
		return chess::Move(chess::Move::NO_MOVE);
	}
	for (const auto &move : legal_moves(board)) {
		if (move_to_index(move, board) == index) {
			return move;
		}
	}
	return chess::Move(chess::Move::NO_MOVE);
}

std::string move_uci(const chess::Move &move) { return chess::uci::moveToUci(move); }

bool game_is_over(const chess::Board &board) {
	return board.isGameOver().first != chess::GameResultReason::NONE;
}

} // namespace eleginus
