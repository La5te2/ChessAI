#pragma once

// Eleginus chess-rule adapter and compact move codec. Rules stay outside the networks.

#include <string>
#include <vector>

#include "chess.hpp"

namespace eleginus {

inline constexpr int kBoardSquares = 64;
inline constexpr int kUnderpromotionPlanes = 9;
inline constexpr int kActionSize =
	kBoardSquares * kBoardSquares + kBoardSquares * kUnderpromotionPlanes;
inline constexpr const char *kArchType = "eleginus";
inline constexpr const char *kMoveEncoding = "sd_64x64_underpromo9";

/// Generates all legal moves; the neural evaluator never learns chess legality.
std::vector<chess::Move> legal_moves(const chess::Board &board);
/// Encodes a legal move in a stable source/destination action space.
int move_to_index(const chess::Move &move);
/// Resolves an action against the supplied position, rejecting illegal actions.
chess::Move index_to_move(int index, const chess::Board &board);
/// Formats a move in UCI coordinate notation.
std::string move_uci(const chess::Move &move);

/// Applies every terminal rule implemented by chess-library.
bool game_is_over(const chess::Board &board);
/// Returns the terminal score in {0, 0.5, 1} from the current side-to-move perspective.
float terminal_score_side_to_move(const chess::Board &board);

} // namespace eleginus
