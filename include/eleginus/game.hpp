#pragma once

// Eleginus chess-rule adapter and compact move codec. Rules stay outside the networks.

#include <string>
#include <vector>

#include "chess.hpp"

namespace eleginus {

inline constexpr int kBoardSquares = 64;
inline constexpr const char *kArchType = "eleginus";

/// Generates all legal moves; the neural evaluator never learns chess legality.
std::vector<chess::Move> legal_moves(const chess::Board &board);
/// Formats a move in UCI coordinate notation.
std::string move_uci(const chess::Move &move);

/// Applies every terminal rule implemented by chess-library.
bool game_is_over(const chess::Board &board);

} // namespace eleginus
