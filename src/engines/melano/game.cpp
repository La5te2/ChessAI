// Implements Melano's chess-facing codecs and rule queries.

#include "melano/game.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <torch/cuda.h>

namespace melano {

namespace {

// Map an empty or colored piece to the categorical token vocabulary [0, 12].
int piece_token(const chess::Piece &piece) {
	int type = static_cast<int>(piece.type().internal());
	if (type < 0 || type > 5) {
		throw std::runtime_error("invalid chess piece type");
	}
	type += 1;
	if (piece.color() == chess::Color::BLACK) {
		type += 6;
	}
	return type;
}

} // namespace

// Delegate legal move generation to chess-library and return an owning vector.
std::vector<chess::Move> legal_moves(const chess::Board &board) {
	chess::Movelist moves;
	chess::movegen::legalmoves(moves, board);
	return {moves.begin(), moves.end()};
}

// Normalize the library's king-to-rook castling representation to the king destination
// used by the source-destination policy codec.
int policy_destination(const chess::Move &move) {
	if (move.typeOf() != chess::Move::CASTLING) {
		return move.to().index();
	}
	const int from = move.from().index();
	const int rank = from / 8;
	const bool king_side = move.to().index() > from;
	return rank * 8 + (king_side ? 6 : 2);
}

// Encode ordinary/queen moves as from*64+to and reserve a suffix for underpromotions.
int move_to_index(const chess::Move &move) {
	const int from = move.from().index();
	const int to = policy_destination(move);
	const int from_file = from % 8;
	const int to_file = to % 8;
	const int dc = to_file - from_file;

	if (move.typeOf() == chess::Move::PROMOTION && move.promotionType() != chess::PieceType::QUEEN) {
		const int direction = dc + 1;
		const int piece = static_cast<int>(move.promotionType().internal()) - static_cast<int>(chess::PieceType::KNIGHT);
		if (direction < 0 || direction > 2 || piece < 0 || piece > 2) {
			throw std::invalid_argument("cannot encode underpromotion: " + move_uci(move));
		}
		return kBoardSquares * kBoardSquares + from * kUnderpromotionPlanes + direction * 3 + piece;
	}
	return from * kBoardSquares + to;
}

// Decode by legal-move round-trip, which also validates position-dependent legality.
chess::Move index_to_move(int index, const chess::Board &board) {
	if (index < 0 || index >= kActionSize) {
		return chess::Move(chess::Move::NO_MOVE);
	}
	for (const auto &move : legal_moves(board)) {
		if (move_to_index(move) == index) {
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

// Store one categorical token per square followed by side, castling, and EP metadata.
PackedState encode_state(const chess::Board &board) {
	PackedState packed{};
	for (int square = 0; square < 64; ++square) {
		const auto piece = board.at(chess::Square(square));
		if (piece == chess::Piece::NONE) {
			continue;
		}
		packed[square] = static_cast<std::uint8_t>(piece_token(piece));
	}

	packed[64] = board.sideToMove() == chess::Color::WHITE ? 1 : 0;
	const auto rights = board.castlingRights();
	std::uint8_t castling = 0;
	if (rights.has(chess::Color::WHITE, chess::Board::CastlingRights::Side::KING_SIDE))
		castling |= 1;
	if (rights.has(chess::Color::WHITE, chess::Board::CastlingRights::Side::QUEEN_SIDE))
		castling |= 2;
	if (rights.has(chess::Color::BLACK, chess::Board::CastlingRights::Side::KING_SIDE))
		castling |= 4;
	if (rights.has(chess::Color::BLACK, chess::Board::CastlingRights::Side::QUEEN_SIDE))
		castling |= 8;
	packed[65] = castling;
	if (board.enpassantSq().is_valid()) {
		packed[66] = static_cast<std::uint8_t>(board.enpassantSq().file() + 1);
	}
	return packed;
}

// Expand compact byte tokens into optionally pinned int64 embedding indices.
torch::Tensor decode_states(const std::uint8_t *packed, std::int64_t count, bool pinned_memory) {
	auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
	if (pinned_memory) {
		options = options.pinned_memory(true);
	}
	auto output = torch::empty({count, kStateFeatures}, options);
	auto *destination = output.data_ptr<std::int64_t>();
	std::transform(packed, packed + count * kStateFeatures, destination, [](std::uint8_t value) { return static_cast<std::int64_t>(value); });
	return output;
}

// Keep Melano's one-byte categorical tokens compact across PCIe and widen on the GPU.
torch::Tensor decode_states_device(const std::uint8_t *packed, std::int64_t count, const torch::Device &device) {
	if (!device.is_cuda()) {
		return decode_states(packed, count, false).to(device);
	}
	auto host = torch::empty({count, kStateFeatures}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true));
	std::memcpy(host.data_ptr<std::uint8_t>(), packed, static_cast<std::size_t>(count) * kStateFeatures);
	return host.to(device, true).to(torch::kInt64);
}

// Encode live positions through exactly the same representation used by HDF5 rows.
torch::Tensor encode_boards(const std::vector<chess::Board> &boards, bool pinned_memory) {
	std::vector<std::uint8_t> packed(boards.size() * kStateFeatures);
	for (std::size_t index = 0; index < boards.size(); ++index) {
		const auto state = encode_state(boards[index]);
		std::copy(state.begin(), state.end(), packed.begin() + index * state.size());
	}
	return decode_states(packed.data(), static_cast<std::int64_t>(boards.size()), pinned_memory);
}

// Pack live positions once, then transfer the batch directly to the inference device.
torch::Tensor encode_boards_device(const std::vector<chess::Board> &boards, const torch::Device &device) {
	std::vector<std::uint8_t> packed(boards.size() * kStateFeatures);
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
		const int index = move_to_index(move);
		const float value = index < static_cast<int>(policy.size()) ? std::max(0.0F, policy[index]) : 0.0F;
		normalized[index] = value;
		total += value;
	}
	if (total <= 0.0) {
		const float uniform = 1.0F / static_cast<float>(moves.size());
		for (const auto &move : moves) {
			normalized[move_to_index(move)] = uniform;
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

} // namespace melano
