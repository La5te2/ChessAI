// Implements Melano's chess-facing codecs and rule queries.

#include "melano/game.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <torch/cuda.h>

namespace melano {

namespace {

struct CompactActionMap {
	CompactActionMap();

	std::array<std::int16_t, kExpandedActionSize> compact{};
	std::array<std::int16_t, kActionSize> expanded{};
};

// Map a piece to the friendly/opposing categorical token vocabulary [1, 12].
int piece_token(const chess::Piece &piece, chess::Color friendly) {
	int type = static_cast<int>(piece.type().internal());
	if (type < 0 || type > 5) {
		throw std::runtime_error("invalid chess piece type");
	}
	type += 1;
	if (piece.color() != friendly) {
		type += 6;
	}
	return type;
}

// Decode the destination of one canonical expanded action, or reject invalid geometry.
int expanded_destination(int index) {
	if (index < kBoardSquares * kBoardSquares) {
		const int source = index / kBoardSquares;
		const int destination = index % kBoardSquares;
		const int dr = destination / 8 - source / 8;
		const int df = destination % 8 - source % 8;
		const int adr = std::abs(dr);
		const int adf = std::abs(df);
		return source != destination && (dr == 0 || df == 0 || adr == adf || (adr == 1 && adf == 2) || (adr == 2 && adf == 1)) ? destination : -1;
	}

	const int suffix = index - kBoardSquares * kBoardSquares;
	const int source = suffix / kUnderpromotionPlanes;
	const int pattern = suffix % kUnderpromotionPlanes;
	if (source / 8 != 6) {
		return -1;
	}
	const int target_file = source % 8 + pattern / 3 - 1;
	return target_file >= 0 && target_file < 8 ? 7 * 8 + target_file : -1;
}

CompactActionMap::CompactActionMap() {
	compact.fill(-1);
	int next = 0;
	for (int expanded_index = 0; expanded_index < kExpandedActionSize; ++expanded_index) {
		if (expanded_destination(expanded_index) < 0) {
			continue;
		}
		if (next >= kActionSize) {
			throw std::logic_error("Melano compact action map overflow");
		}
		compact[static_cast<std::size_t>(expanded_index)] = static_cast<std::int16_t>(next);
		expanded[static_cast<std::size_t>(next)] = static_cast<std::int16_t>(expanded_index);
		++next;
	}
	if (next != kActionSize) {
		throw std::logic_error("Melano compact action map has the wrong size");
	}
}

const CompactActionMap &compact_action_map() {
	static const CompactActionMap mapping;
	return mapping;
}

int reflected_rank_square(int square) {
	return (7 - square / 8) * 8 + square % 8;
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
int expanded_move_index(const chess::Move &move) {
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

int move_to_index(const chess::Move &move, chess::Color side_to_move) {
	const int index = expanded_move_index(move);

	int canonical = index;
	if (side_to_move == chess::Color::BLACK) {
		if (index < kBoardSquares * kBoardSquares) {
			canonical = reflected_rank_square(index / kBoardSquares) * kBoardSquares + reflected_rank_square(index % kBoardSquares);
		} else {
			const int suffix = index - kBoardSquares * kBoardSquares;
			canonical = kBoardSquares * kBoardSquares + reflected_rank_square(suffix / kUnderpromotionPlanes) * kUnderpromotionPlanes + suffix % kUnderpromotionPlanes;
		}
	}
	const int compact = compact_action_index(canonical);
	if (compact < 0) {
		throw std::logic_error("canonical Melano action is geometrically invalid");
	}
	return compact;
}

int compact_action_index(int index) {
	if (index < 0 || index >= kExpandedActionSize) {
		throw std::out_of_range("Melano expanded action index");
	}
	return compact_action_map().compact[static_cast<std::size_t>(index)];
}

int expanded_action_index(int index) {
	if (index < 0 || index >= kActionSize) {
		throw std::out_of_range("Melano compact action index");
	}
	return compact_action_map().expanded[static_cast<std::size_t>(index)];
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

// Store friendly/opposing piece tokens in side-to-move canonical coordinates.
PackedState encode_state(const chess::Board &board) {
	PackedState packed{};
	const auto friendly = board.sideToMove();
	for (int square = 0; square < 64; ++square) {
		const auto piece = board.at(chess::Square(square));
		if (piece == chess::Piece::NONE) {
			continue;
		}
		const int canonical_square = friendly == chess::Color::WHITE ? square : reflected_rank_square(square);
		packed[canonical_square] = static_cast<std::uint8_t>(piece_token(piece, friendly));
	}

	const auto rights = board.castlingRights();
	const auto opposing = friendly == chess::Color::WHITE ? chess::Color::BLACK : chess::Color::WHITE;
	std::uint8_t castling = 0;
	if (rights.has(friendly, chess::Board::CastlingRights::Side::KING_SIDE))
		castling |= 1;
	if (rights.has(friendly, chess::Board::CastlingRights::Side::QUEEN_SIDE))
		castling |= 2;
	if (rights.has(opposing, chess::Board::CastlingRights::Side::KING_SIDE))
		castling |= 4;
	if (rights.has(opposing, chess::Board::CastlingRights::Side::QUEEN_SIDE))
		castling |= 8;
	packed[64] = castling;
	if (board.enpassantSq().is_valid()) {
		packed[65] = static_cast<std::uint8_t>(board.enpassantSq().file() + 1);
	}
	return packed;
}

// Encode live positions through exactly the same representation used by HDF5 rows.
torch::Tensor encode_boards(const std::vector<chess::Board> &boards, bool pinned_memory) {
	auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
	if (pinned_memory) {
		options = options.pinned_memory(true);
	}
	auto output = torch::empty({static_cast<std::int64_t>(boards.size()), kStateFeatures}, options);
	auto *destination = output.data_ptr<std::uint8_t>();
	for (std::size_t index = 0; index < boards.size(); ++index) {
		const auto state = encode_state(boards[index]);
		std::memcpy(destination + index * kStateFeatures, state.data(), state.size());
	}
	return output;
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

} // namespace melano
