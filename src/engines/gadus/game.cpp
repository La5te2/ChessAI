// Implements Gadus's chess-facing codecs and rule queries.

#include "gadus/game.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <tuple>
#include <torch/cuda.h>

namespace gadus {

namespace {

constexpr std::array<std::pair<int, int>, 8> kRayDirections{{
    {-1, -1},
    {-1, 0},
    {-1, 1},
    {0, -1},
    {0, 1},
    {1, -1},
    {1, 0},
    {1, 1},
}};

constexpr std::array<std::pair<int, int>, 8> kKnightOffsets{{
    {-2, -1},
    {-2, 1},
    {-1, -2},
    {-1, 2},
    {1, -2},
    {1, 2},
    {2, -1},
    {2, 1},
}};

int encoded_destination(int source, int pattern) {
	const int rank = source / 8;
	const int file = source % 8;
	int rank_delta = 0;
	int file_delta = 0;
	if (pattern < 56) {
		const auto [rank_step, file_step] = kRayDirections[static_cast<std::size_t>(pattern / 7)];
		const int distance = pattern % 7 + 1;
		rank_delta = rank_step * distance;
		file_delta = file_step * distance;
	} else if (pattern < 64) {
		std::tie(rank_delta, file_delta) = kKnightOffsets[static_cast<std::size_t>(pattern - 56)];
	} else {
		if (rank != 6) {
			return -1;
		}
		rank_delta = 1;
		file_delta = (pattern - 64) / 3 - 1;
	}
	const int target_rank = rank + rank_delta;
	const int target_file = file + file_delta;
	return target_rank >= 0 && target_rank < 8 && target_file >= 0 && target_file < 8 ? target_rank * 8 + target_file : -1;
}

struct CompactActionMap {
	CompactActionMap() {
		compact.fill(-1);
		int next = 0;
		for (int expanded_index = 0; expanded_index < kExpandedActionSize; ++expanded_index) {
			const int source = expanded_index / kPolicyPlanes;
			const int pattern = expanded_index % kPolicyPlanes;
			if (encoded_destination(source, pattern) < 0) {
				continue;
			}
			if (next >= kActionSize) {
				throw std::logic_error("Gadus compact action map overflow");
			}
			compact[static_cast<std::size_t>(expanded_index)] = static_cast<std::int16_t>(next);
			expanded[static_cast<std::size_t>(next)] = static_cast<std::int16_t>(expanded_index);
			++next;
		}
		if (next != kActionSize) {
			throw std::logic_error("Gadus compact action map has the wrong size");
		}
	}

	std::array<std::int16_t, kExpandedActionSize> compact{};
	std::array<std::int16_t, kActionSize> expanded{};
};

const CompactActionMap &compact_action_map() {
	static const CompactActionMap mapping;
	return mapping;
}

// Map a colored piece to one of twelve binary piece planes.
int piece_plane(const chess::Piece &piece) {
	int type = static_cast<int>(piece.type().internal());
	if (type < 0 || type > 5) {
		throw std::runtime_error("invalid chess piece type");
	}
	if (piece.color() == chess::Color::BLACK) {
		type += 6;
	}
	return type;
}

// Convert expanded persistent planes to friendly/opposing canonical coordinates.
torch::Tensor canonicalize_expanded(torch::Tensor expanded) {
	auto white_to_move = expanded.slice(1, 12, 13).gt(0.5F);
	auto reflected = expanded.flip({2});
	auto friendly = torch::where(white_to_move, expanded.slice(1, 0, 6), reflected.slice(1, 6, 12));
	auto opposing = torch::where(white_to_move, expanded.slice(1, 6, 12), reflected.slice(1, 0, 6));
	auto friendly_king = torch::where(white_to_move, expanded.slice(1, 13, 14), reflected.slice(1, 15, 16));
	auto friendly_queen = torch::where(white_to_move, expanded.slice(1, 14, 15), reflected.slice(1, 16, 17));
	auto opposing_king = torch::where(white_to_move, expanded.slice(1, 15, 16), reflected.slice(1, 13, 14));
	auto opposing_queen = torch::where(white_to_move, expanded.slice(1, 16, 17), reflected.slice(1, 14, 15));
	auto en_passant = torch::where(white_to_move, expanded.slice(1, 17, 18), reflected.slice(1, 17, 18));
	return torch::cat({friendly, opposing, friendly_king, friendly_queen, opposing_king, opposing_queen, en_passant}, 1);
}

} // namespace

// Delegate legal move generation to chess-library and return an owning vector.
std::vector<chess::Move> legal_moves(const chess::Board &board) {
	chess::Movelist moves;
	chess::movegen::legalmoves(moves, board);
	return {moves.begin(), moves.end()};
}

// Normalize the library's king-to-rook castling representation to the king destination
// used by the persistent alphazero_64x73 training codec.
int policy_destination(const chess::Move &move) {
	if (move.typeOf() != chess::Move::CASTLING) {
		return move.to().index();
	}
	const int from = move.from().index();
	const int rank = from / 8;
	const bool king_side = move.to().index() > from;
	return rank * 8 + (king_side ? 6 : 2);
}

// Encode underpromotions separately, then sliding rays and knight jumps in the HDF5 73-plane codec.
int expanded_move_index(const chess::Move &move) {
	const int from = move.from().index();
	const int to = policy_destination(move);
	const int from_rank = from / 8;
	const int from_file = from % 8;
	const int to_rank = to / 8;
	const int to_file = to % 8;
	const int dr = to_rank - from_rank;
	const int dc = to_file - from_file;

	if (move.typeOf() == chess::Move::PROMOTION && move.promotionType() != chess::PieceType::QUEEN) {
		const int direction = dc + 1;
		const int piece = static_cast<int>(move.promotionType().internal()) - static_cast<int>(chess::PieceType::KNIGHT);
		if (direction < 0 || direction > 2 || piece < 0 || piece > 2) {
			throw std::invalid_argument("cannot encode underpromotion: " + move_uci(move));
		}
		return from * kPolicyPlanes + 64 + direction * 3 + piece;
	}

	for (int direction = 0; direction < static_cast<int>(kRayDirections.size()); ++direction) {
		for (int distance = 1; distance <= 7; ++distance) {
			if (dr == kRayDirections[direction].first * distance && dc == kRayDirections[direction].second * distance) {
				return from * kPolicyPlanes + direction * 7 + distance - 1;
			}
		}
	}

	for (int offset = 0; offset < static_cast<int>(kKnightOffsets.size()); ++offset) {
		if (dr == kKnightOffsets[offset].first && dc == kKnightOffsets[offset].second) {
			return from * kPolicyPlanes + 56 + offset;
		}
	}
	throw std::invalid_argument("cannot encode move: " + move_uci(move));
}

int move_to_index(const chess::Move &move, chess::Color side_to_move) {
	return canonical_action_index(expanded_move_index(move), side_to_move);
}

int hdf5_action_index(const chess::Move &move) {
	return expanded_move_index(move);
}

// Reflect the source rank and every rank-sensitive motion pattern for Black.
int canonical_action_index(int index, chess::Color side_to_move) {
	if (index < 0 || index >= kExpandedActionSize) {
		throw std::out_of_range("Gadus HDF5 action index");
	}

	const int source = index / kPolicyPlanes;
	const int pattern = index % kPolicyPlanes;
	const int canonical_source = side_to_move == chess::Color::WHITE ? source : (7 - source / 8) * 8 + source % 8;
	int canonical_pattern = pattern;
	if (side_to_move == chess::Color::BLACK && pattern < 56) {
		constexpr std::array<int, 8> reflected_directions{{5, 6, 7, 3, 4, 0, 1, 2}};
		const int direction = pattern / 7;
		canonical_pattern = reflected_directions[direction] * 7 + pattern % 7;
	} else if (side_to_move == chess::Color::BLACK && pattern < 64) {
		constexpr std::array<int, 8> reflected_knights{{6, 7, 4, 5, 2, 3, 0, 1}};
		canonical_pattern = 56 + reflected_knights[pattern - 56];
	}
	const int canonical = compact_action_index(canonical_source * kPolicyPlanes + canonical_pattern);
	if (canonical < 0) {
		throw std::logic_error("canonical Gadus action is geometrically invalid");
	}
	return canonical;
}

int compact_action_index(int index) {
	if (index < 0 || index >= kExpandedActionSize) {
		throw std::out_of_range("Gadus expanded action index");
	}
	return compact_action_map().compact[static_cast<std::size_t>(index)];
}

int expanded_action_index(int index) {
	if (index < 0 || index >= kActionSize) {
		throw std::out_of_range("Gadus compact action index");
	}
	return compact_action_map().expanded[static_cast<std::size_t>(index)];
}

int action_destination(int index) {
	if (index < 0 || index >= kActionSize) {
		throw std::out_of_range("Gadus action index");
	}
	const int expanded = expanded_action_index(index);
	const int destination = encoded_destination(expanded / kPolicyPlanes, expanded % kPolicyPlanes);
	if (destination < 0) {
		throw std::invalid_argument("Gadus action is geometrically invalid");
	}
	return destination;
}

// Decode by legal-move round-trip, which also validates position-dependent legality.
chess::Move index_to_move(int index, const chess::Board &board) {
	if (index < 0 || index >= kActionSize) {
		return chess::Move(chess::Move::NO_MOVE);
	}
	for (const auto &move : legal_moves(board)) {
		if (move_to_index(move, board.sideToMove()) == index) {
			return move;
		}
	}
	return chess::Move(chess::Move::NO_MOVE);
}

// Produce the protocol-level coordinate representation of a move.
std::string move_uci(const chess::Move &move) {
	return chess::uci::moveToUci(move);
}

// Produce SAN while turning library formatting failures into a stable UCI fallback.
std::string move_san(const chess::Board &board, const chess::Move &move) {
	try {
		return chess::uci::moveToSan(board, move);
	} catch (...) {
		return move_uci(move);
	}
}

// Pack each 8-square rank into one byte per plane to reduce HDF5 traffic by 8x.
PackedState encode_state(const chess::Board &board) {
	PackedState packed{};
	for (int square = 0; square < 64; ++square) {
		const auto piece = board.at(chess::Square(square));
		if (piece == chess::Piece::NONE) {
			continue;
		}
		const int plane = piece_plane(piece);
		const int rank = square / 8;
		const int file = square % 8;
		packed[plane * 8 + rank] |= static_cast<std::uint8_t>(1U << (7 - file));
	}

	if (board.sideToMove() == chess::Color::WHITE) {
		std::fill_n(packed.begin() + 12 * 8, 8, static_cast<std::uint8_t>(0xFF));
	}
	const auto rights = board.castlingRights();
	const std::array<bool, 4> castling{{
	    rights.has(chess::Color::WHITE, chess::Board::CastlingRights::Side::KING_SIDE),
	    rights.has(chess::Color::WHITE, chess::Board::CastlingRights::Side::QUEEN_SIDE),
	    rights.has(chess::Color::BLACK, chess::Board::CastlingRights::Side::KING_SIDE),
	    rights.has(chess::Color::BLACK, chess::Board::CastlingRights::Side::QUEEN_SIDE),
	}};
	for (int index = 0; index < 4; ++index) {
		if (castling[index]) {
			std::fill_n(packed.begin() + (13 + index) * 8, 8, static_cast<std::uint8_t>(0xFF));
		}
	}
	if (board.enpassantSq().is_valid()) {
		const int file = board.enpassantSq().file();
		for (int rank = 0; rank < 8; ++rank) {
			packed[17 * 8 + rank] |= static_cast<std::uint8_t>(1U << (7 - file));
		}
	}
	return packed;
}

// Expand persistent planes and convert each row to side-to-move canonical coordinates.
torch::Tensor decode_states(const std::uint8_t *packed, std::int64_t count, bool pinned_memory) {
	static const auto byte_planes = [] {
		std::array<std::array<float, 8>, 256> table{};
		for (std::size_t byte = 0; byte < table.size(); ++byte) {
			for (int file = 0; file < 8; ++file) {
				table[byte][file] = static_cast<float>((byte >> (7 - file)) & 1U);
			}
		}
		return table;
	}();
	auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
	if (pinned_memory) {
		options = options.pinned_memory(true);
	}
	auto output = torch::empty({count, kInputPlanes, 8, 8}, options);
	auto *destination = output.data_ptr<float>();
	for (std::int64_t item = 0; item < count; ++item) {
		const auto *state = packed + item * kStatePlanes * 8;
		const bool white_to_move = state[12 * 8] != 0;
		for (int plane = 0; plane < kInputPlanes; ++plane) {
			for (int rank = 0; rank < 8; ++rank) {
				int source_plane = 0;
				if (plane < 6) {
					source_plane = white_to_move ? plane : plane + 6;
				} else if (plane < 12) {
					source_plane = white_to_move ? plane : plane - 6;
				} else if (plane < 16) {
					constexpr std::array<int, 4> white_castling{{13, 14, 15, 16}};
					constexpr std::array<int, 4> black_castling{{15, 16, 13, 14}};
					source_plane = white_to_move ? white_castling[plane - 12] : black_castling[plane - 12];
				} else {
					source_plane = 17;
				}
				const int source_rank = white_to_move ? rank : 7 - rank;
				const auto byte = state[source_plane * 8 + source_rank];
				const auto offset = (static_cast<std::size_t>(item) * kInputPlanes * 8 * 8) + (static_cast<std::size_t>(plane) * 8 * 8) + (static_cast<std::size_t>(rank) * 8);
				std::memcpy(destination + offset, byte_planes[byte].data(), 8 * sizeof(float));
			}
		}
	}
	return output;
}

// Expand compact binary planes after they reach the inference or training device.
torch::Tensor decode_states_device(const torch::Tensor &packed, const torch::Device &device) {
	if (packed.scalar_type() != torch::kUInt8 || packed.dim() != 3 || packed.size(1) != kStatePlanes || packed.size(2) != 8) {
		throw std::invalid_argument("packed Gadus states must have shape [count, 18, 8]");
	}
	auto contiguous = packed.contiguous();
	if (!device.is_cuda()) {
		auto host = contiguous.to(torch::kCPU);
		return decode_states(host.data_ptr<std::uint8_t>(), host.size(0), false).to(device);
	}
	auto device_packed = contiguous.device() == device ? contiguous : contiguous.to(device, true);
	auto masks = torch::tensor({128, 64, 32, 16, 8, 4, 2, 1}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
	auto expanded = torch::bitwise_and(device_packed.unsqueeze(-1), masks).ne(0).to(torch::kFloat32);
	return canonicalize_expanded(expanded);
}

// Keep the compact 144-byte representation across PCIe, then expand its bits in parallel.
torch::Tensor decode_states_device(const std::uint8_t *packed, std::int64_t count, const torch::Device &device) {
	if (!device.is_cuda()) {
		return decode_states(packed, count, false).to(device);
	}
	auto host = torch::empty({count, kStatePlanes, 8}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true));
	std::memcpy(host.data_ptr<std::uint8_t>(), packed, static_cast<std::size_t>(count) * kStatePlanes * 8);
	return decode_states_device(host, device);
}

// Pack a live batch first, then use the same decoder as persisted training data.
torch::Tensor encode_boards(const std::vector<chess::Board> &boards, bool pinned_memory) {
	std::vector<std::uint8_t> packed(boards.size() * kStatePlanes * 8);
	for (std::size_t index = 0; index < boards.size(); ++index) {
		const auto state = encode_state(boards[index]);
		std::copy(state.begin(), state.end(), packed.begin() + index * state.size());
	}
	return decode_states(packed.data(), static_cast<std::int64_t>(boards.size()), pinned_memory);
}

// Encode to packed host rows and use device-side expansion for CUDA inference.
torch::Tensor encode_boards_device(const std::vector<chess::Board> &boards, const torch::Device &device) {
	std::vector<std::uint8_t> packed(boards.size() * kStatePlanes * 8);
	for (std::size_t index = 0; index < boards.size(); ++index) {
		const auto state = encode_state(boards[index]);
		std::copy(state.begin(), state.end(), packed.begin() + index * state.size());
	}
	return decode_states_device(packed.data(), static_cast<std::int64_t>(boards.size()), device);
}

// Report any library-recognized terminal reason, including mate and rule draws.
bool game_is_over(const chess::Board &board) {
	return board.isGameOver().first != chess::GameResultReason::NONE;
}

// Convert the terminal result to the current player's value convention.
float terminal_value_side_to_move(const chess::Board &board) {
	const auto outcome = board.isGameOver();
	if (outcome.first == chess::GameResultReason::NONE) {
		return std::numeric_limits<float>::quiet_NaN();
	}
	if (outcome.second == chess::GameResult::DRAW) {
		return 0.0F;
	}
	return outcome.second == chess::GameResult::WIN ? 1.0F : -1.0F;
}

// Translate the library outcome into the canonical PGN result token.
std::string game_result(const chess::Board &board) {
	const auto outcome = board.isGameOver();
	if (outcome.first == chess::GameResultReason::NONE) {
		return "*";
	}
	if (outcome.second == chess::GameResult::DRAW) {
		return "1/2-1/2";
	}
	const bool side_wins = outcome.second == chess::GameResult::WIN;
	const bool white_wins = side_wins == (board.sideToMove() == chess::Color::WHITE);
	return white_wins ? "1-0" : "0-1";
}

// Translate terminal reason enums to concise diagnostics and PGN metadata.
std::string game_termination(const chess::Board &board) {
	switch (board.isGameOver().first) {
	case chess::GameResultReason::CHECKMATE:
		return "checkmate";
	case chess::GameResultReason::STALEMATE:
		return "stalemate";
	case chess::GameResultReason::INSUFFICIENT_MATERIAL:
		return "insufficient material";
	case chess::GameResultReason::FIFTY_MOVE_RULE:
		return "fifty move rule";
	case chess::GameResultReason::THREEFOLD_REPETITION:
		return "threefold repetition";
	default:
		return "";
	}
}

// Restrict arbitrary network probabilities to legal actions and make their sum exactly one.
std::vector<float> normalize_legal_policy(const std::vector<float> &policy, const chess::Board &board) {
	std::vector<float> normalized(kActionSize, 0.0F);
	const auto moves = legal_moves(board);
	if (moves.empty()) {
		return normalized;
	}
	double total = 0.0;
	for (const auto &move : moves) {
		const int index = move_to_index(move, board.sideToMove());
		const float value = index < static_cast<int>(policy.size()) ? std::max(0.0F, policy[index]) : 0.0F;
		normalized[index] = value;
		total += value;
	}
	if (total <= 0.0) {
		const float uniform = 1.0F / static_cast<float>(moves.size());
		for (const auto &move : moves) {
			normalized[move_to_index(move, board.sideToMove())] = uniform;
		}
	} else {
		for (auto &value : normalized) {
			value = static_cast<float>(value / total);
		}
	}
	return normalized;
}

// Resolve device policy once at process startup and fail explicitly on unavailable CUDA.
torch::Device resolve_device(const std::string &requested) {
	if (requested == "auto") {
		return torch::Device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
	}
	if (requested.starts_with("cuda")) {
		if (!torch::cuda::is_available()) {
			throw std::runtime_error("CUDA was requested but this LibTorch build has no available CUDA device");
		}
		return torch::Device(requested);
	}
	if (requested == "cpu") {
		return torch::Device(torch::kCPU);
	}
	throw std::invalid_argument("unsupported device: " + requested);
}

} // namespace gadus
