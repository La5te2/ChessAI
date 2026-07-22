#pragma once

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

std::vector<chess::Move> legal_moves(const chess::Board &board);
int move_to_index(const chess::Move &move);
chess::Move index_to_move(int index, const chess::Board &board);
std::string move_uci(const chess::Move &move);
std::string move_san(const chess::Board &board, const chess::Move &move);

PackedState encode_state(const chess::Board &board);
torch::Tensor decode_states(const std::uint8_t *packed, std::int64_t count);
torch::Tensor encode_boards(const std::vector<chess::Board> &boards);

bool game_is_over(const chess::Board &board);
float terminal_value_side_to_move(const chess::Board &board);
std::string game_result(const chess::Board &board);
std::string game_termination(const chess::Board &board);

std::vector<float> normalize_legal_policy(const std::vector<float> &policy,
										  const chess::Board &board);

torch::Device resolve_device(const std::string &requested);

struct OpeningSpec {
	std::string fen;
	bool candidate_white = true;
};

std::vector<std::string> load_opening_positions(const std::string &path, int book_plies,
												int max_positions, std::uint64_t seed);

std::vector<OpeningSpec> make_arena_specs(int games, const std::string &opening_book,
										  int book_plies, int max_positions, std::uint64_t seed);

} // namespace melano
