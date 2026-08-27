#pragma once

// Gadus chess rules, persistent state codec, canonical network input, and move codec.

#include "chess.hpp"
#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <torch/types.h>
#include <vector>

namespace gadus {

inline constexpr int kBoardSquares = 64;
inline constexpr int kStatePlanes = 18;
inline constexpr int kInputPlanes = 17;
inline constexpr int kPolicyPlanes = 73;
inline constexpr int kExpandedActionSize = kBoardSquares * kPolicyPlanes;
inline constexpr int kActionSize = 1858;
inline constexpr const char *kArchType = "gadus";
inline constexpr const char *kStateEncoding = "gadus_18_planes";
inline constexpr const char *kMoveEncoding = "alphazero_64x73";
inline constexpr const char *kTargetSchema = "pv_supervised";

using PackedState = std::array<std::uint8_t, kStatePlanes * 8>;

/// Generates every legal move in the supplied position.
std::vector<chess::Move> legal_moves(const chess::Board &board);
/// Maps a move to the 1858-entry side-to-move canonical runtime action space.
int move_to_index(const chess::Move &move, chess::Color side_to_move);
/// Maps a move to the full physical-board 64x73 action space.
int full_action_index(const chess::Move &move);
/// Maps a full physical-board action index into the runtime canonical action space.
int canonical_action_index(int index, chess::Color side_to_move);
/// Maps a geometrically valid canonical 64x73 index to its runtime Policy index, or returns -1.
int compact_action_index(int index);
/// Restores a runtime Policy index to its canonical expanded 64x73 index.
int expanded_action_index(int index);
/// Returns the destination square encoded by a runtime Policy action.
int action_destination(int index);
/// Resolves an action index against a position and rejects illegal or ambiguous actions.
chess::Move index_to_move(int index, const chess::Board &board);
/// Formats a move as standard UCI coordinate notation.
std::string move_uci(const chess::Move &move);
/// Formats a legal move as SAN in the supplied pre-move position.
std::string move_san(const chess::Board &board, const chess::Move &move);

/// Bit-packs the Gadus 18 binary board planes for compact HDF5 storage.
PackedState encode_state(const chess::Board &board);
/// Expands packed rows into canonical float tensors shaped [count, 17, 8, 8].
torch::Tensor decode_states(const std::uint8_t *packed, std::int64_t count, bool pinned_memory = false);
/// Transfers packed rows first and expands their bits on the destination device.
torch::Tensor decode_states_device(const std::uint8_t *packed, std::int64_t count, const torch::Device &device);
/// Expands a contiguous uint8 tensor shaped [count, 18, 8] on its destination device.
torch::Tensor decode_states_device(const torch::Tensor &packed, const torch::Device &device);
/// Encodes live boards directly into a batched Gadus input tensor.
torch::Tensor encode_boards(const std::vector<chess::Board> &boards, bool pinned_memory = false);
/// Encodes live boards and expands packed planes on the destination device.
torch::Tensor encode_boards_device(const std::vector<chess::Board> &boards, const torch::Device &device);

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

} // namespace gadus
