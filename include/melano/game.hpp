#pragma once

// Melano chess rules, square-token state codec, source-destination move codec, and openings.

#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <vector>
#include <torch/torch.h>
#include "chess.hpp"

namespace melano {

inline constexpr int kBoardSquares = 64;
inline constexpr int kStateFeatures = 67;
inline constexpr int kUnderpromotionPlanes = 9;
inline constexpr int kActionSize = kBoardSquares * kBoardSquares +
								   kBoardSquares * kUnderpromotionPlanes;
inline constexpr const char *kArchType = "melano";
inline constexpr const char *kStateEncoding = "melano_square_tokens";
inline constexpr const char *kMoveEncoding = "sd_64x64_underpromo9";
inline constexpr const char *kTargetSchema = "pva_minimax_dueling";

using PackedState = std::array<std::uint8_t, kStateFeatures>;

/// Generates every legal move in the supplied position.
std::vector<chess::Move> legal_moves(const chess::Board &board);
/// Maps a move to Melano's source-destination and underpromotion action space.
int move_to_index(const chess::Move &move);
/// Resolves an action index against a position and rejects illegal or ambiguous actions.
chess::Move index_to_move(int index, const chess::Board &board);
/// Formats a move as standard UCI coordinate notation.
std::string move_uci(const chess::Move &move);
/// Formats a legal move as SAN in the supplied pre-move position.
std::string move_san(const chess::Board &board, const chess::Move &move);

/// Packs 64 piece tokens plus side, castling, and en-passant metadata.
PackedState encode_state(const chess::Board &board);
/// Expands packed rows into an int64 tensor shaped [count, 67].
torch::Tensor decode_states(const std::uint8_t *packed, std::int64_t count);
/// Encodes live boards directly into a batched Melano token tensor.
torch::Tensor encode_boards(const std::vector<chess::Board> &boards);

/// Applies all chess terminal rules represented by the chess library.
bool game_is_over(const chess::Board &board);
/// Returns -1, 0, or +1 from the current side-to-move perspective.
float terminal_value_side_to_move(const chess::Board &board);
/// Converts a terminal board to a PGN result token.
std::string game_result(const chess::Board &board);
/// Describes the rule that ended a game, including project max-ply truncation elsewhere.
std::string game_termination(const chess::Board &board);

/// Masks illegal actions and renormalizes legal mass, falling back to uniform legal play.
std::vector<float> normalize_legal_policy(const std::vector<float> &policy,
										  const chess::Board &board);

/// Resolves auto/cpu/cuda while rejecting unavailable CUDA requests.
torch::Device resolve_device(const std::string &requested);

struct OpeningSpec {
	std::string fen;
	bool candidate_white = true;
};

/// Expands a Polyglot book into unique FENs at the requested ply depth.
std::vector<std::string> load_opening_positions(const std::string &path, int book_plies,
												int max_positions, std::uint64_t seed);

/// Creates paired arena starts so candidate and baseline receive both colors per position.
std::vector<OpeningSpec> make_arena_specs(int games, const std::string &opening_book,
										  int book_plies, int max_positions, std::uint64_t seed);

} // namespace melano
