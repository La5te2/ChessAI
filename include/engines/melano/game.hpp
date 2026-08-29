#pragma once

// Melano chess rules, square-token state codec, and source-destination move codec.

#include "chess.hpp"
#include <array>
#include <cstdint>
#include <string>
#include <torch/types.h>
#include <vector>

namespace melano {

inline constexpr int kBoardSquares = 64;
inline constexpr int kStateFeatures = 66;
inline constexpr int kUnderpromotionPlanes = 9;
inline constexpr int kExpandedActionSize = kBoardSquares * kBoardSquares + kBoardSquares * kUnderpromotionPlanes;
inline constexpr int kActionSize = 1858;
inline constexpr int kUnderpromotionActionSize = 66;
inline constexpr int kOrdinaryActionSize = kActionSize - kUnderpromotionActionSize;
inline constexpr const char *kArchType = "melano";
inline constexpr const char *kStateEncoding = "melano_canonical_tokens66";
inline constexpr const char *kMoveEncoding = "melano_compact_sd1858";
inline constexpr const char *kTargetSchema = "melano_policy_value";

using PackedState = std::array<std::uint8_t, kStateFeatures>;

/// Generates every legal move in the supplied position.
std::vector<chess::Move> legal_moves(const chess::Board &board);
/// Maps a move to Melano's side-to-move canonical compact action space.
int move_to_index(const chess::Move &move, chess::Color side_to_move);
/// Maps a geometrically valid canonical expanded action to its compact index, or returns -1.
int compact_action_index(int index);
/// Restores a compact action index to its canonical expanded representation.
int expanded_action_index(int index);
/// Resolves an action index against a position and rejects illegal or ambiguous actions.
chess::Move index_to_move(int index, const chess::Board &board);
/// Formats a move as standard UCI coordinate notation.
std::string move_uci(const chess::Move &move);
/// Formats a legal move as SAN in the supplied pre-move position.
std::string move_san(const chess::Board &board, const chess::Move &move);

/// Packs a board in friendly/opposing side-to-move canonical coordinates.
PackedState encode_state(const chess::Board &board);
/// Encodes live boards into a compact uint8 tensor shaped [batch, 66].
torch::Tensor encode_boards(const std::vector<chess::Board> &boards, bool pinned_memory = false);

/// Applies all chess terminal rules represented by the chess library.
bool game_is_over(const chess::Board &board);
/// Returns -1, 0, or +1 from the current side-to-move perspective.
float terminal_value_side_to_move(const chess::Board &board);
/// Converts a terminal board to a PGN result token.
std::string game_result(const chess::Board &board);
/// Describes the rule that ended a game, including project max-ply truncation elsewhere.
std::string game_termination(const chess::Board &board);

/// Masks illegal actions and renormalizes legal mass, falling back to uniform legal play.
std::vector<float> normalize_legal_policy(const std::vector<float> &policy, const chess::Board &board);

/// Resolves auto/cpu/cuda while rejecting unavailable CUDA requests.
torch::Device resolve_device(const std::string &requested);

} // namespace melano
