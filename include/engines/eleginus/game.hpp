#pragma once

#include "chess.hpp"
#include <string>
#include <vector>

namespace eleginus {

std::vector<chess::Move> legal_moves(const chess::Board &board);
std::string move_uci(const chess::Move &move);
bool game_is_over(const chess::Board &board);

} // namespace eleginus
