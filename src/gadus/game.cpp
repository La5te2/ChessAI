// Implements Gadus's chess-facing codecs, rule queries, and Polyglot opening traversal.

#include "gadus/game.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <torch/cuda.h>

namespace gadus {

namespace {

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

// Read a Polyglot 16-bit field, whose on-disk byte order is big-endian.
std::uint16_t read_be16(std::istream &input) {
	std::uint8_t bytes[2]{};
	input.read(reinterpret_cast<char *>(bytes), 2);
	return static_cast<std::uint16_t>((bytes[0] << 8U) | bytes[1]);
}

// Read a Polyglot 32-bit field in big-endian order.
std::uint32_t read_be32(std::istream &input) {
	std::uint8_t bytes[4]{};
	input.read(reinterpret_cast<char *>(bytes), 4);
	return (static_cast<std::uint32_t>(bytes[0]) << 24U) |
		   (static_cast<std::uint32_t>(bytes[1]) << 16U) |
		   (static_cast<std::uint32_t>(bytes[2]) << 8U) | static_cast<std::uint32_t>(bytes[3]);
}

// Read a Polyglot 64-bit field in big-endian order.
std::uint64_t read_be64(std::istream &input) {
	std::uint8_t bytes[8]{};
	input.read(reinterpret_cast<char *>(bytes), 8);
	std::uint64_t value = 0;
	for (const auto byte : bytes) {
		value = (value << 8U) | byte;
	}
	return value;
}

// Match a raw Polyglot move against legal moves so special move flags remain correct.
chess::Move decode_polyglot_move(std::uint16_t raw, const chess::Board &board) {
	const int to_file = raw & 7;
	const int to_rank = (raw >> 3) & 7;
	const int from_file = (raw >> 6) & 7;
	const int from_rank = (raw >> 9) & 7;
	const int promotion = (raw >> 12) & 7;
	const int from = from_rank * 8 + from_file;
	const int to = to_rank * 8 + to_file;

	for (const auto &move : legal_moves(board)) {
		if (move.from().index() != from || move.to().index() != to) {
			continue;
		}
		if (promotion == 0 && move.typeOf() != chess::Move::PROMOTION) {
			return move;
		}
		if (promotion > 0 && move.typeOf() == chess::Move::PROMOTION &&
			static_cast<int>(move.promotionType().internal()) == promotion) {
			return move;
		}
	}
	return chess::Move(chess::Move::NO_MOVE);
}

// Use repetition-relevant FEN fields as the opening traversal identity.
std::string state_key(const chess::Board &board) { return board.getFen(false); }

} // namespace

// Delegate legal move generation to chess-library and return an owning vector.
std::vector<chess::Move> legal_moves(const chess::Board &board) {
	chess::Movelist moves;
	chess::movegen::legalmoves(moves, board);
	return {moves.begin(), moves.end()};
}

// Normalize the library's king-to-rook castling representation to the king destination
// used by the historical alphazero_64x73 training codec.
int policy_destination(const chess::Move &move) {
	if (move.typeOf() != chess::Move::CASTLING) {
		return move.to().index();
	}
	const int from = move.from().index();
	const int rank = from / 8;
	const bool king_side = move.to().index() > from;
	return rank * 8 + (king_side ? 6 : 2);
}

// Encode underpromotions separately, then sliding rays and knight jumps in 73 planes.
int move_to_index(const chess::Move &move) {
	const int from = move.from().index();
	const int to = policy_destination(move);
	const int from_rank = from / 8;
	const int from_file = from % 8;
	const int to_rank = to / 8;
	const int to_file = to % 8;
	const int dr = to_rank - from_rank;
	const int dc = to_file - from_file;

	if (move.typeOf() == chess::Move::PROMOTION &&
		move.promotionType() != chess::PieceType::QUEEN) {
		const int direction = dc + 1;
		const int piece = static_cast<int>(move.promotionType().internal()) -
						  static_cast<int>(chess::PieceType::KNIGHT);
		if (direction < 0 || direction > 2 || piece < 0 || piece > 2) {
			throw std::invalid_argument("cannot encode underpromotion: " + move_uci(move));
		}
		return from * kPolicyPlanes + 64 + direction * 3 + piece;
	}

	constexpr std::array<std::pair<int, int>, 8> directions{{
		{-1, -1},
		{-1, 0},
		{-1, 1},
		{0, -1},
		{0, 1},
		{1, -1},
		{1, 0},
		{1, 1},
	}};
	for (int direction = 0; direction < static_cast<int>(directions.size()); ++direction) {
		for (int distance = 1; distance <= 7; ++distance) {
			if (dr == directions[direction].first * distance &&
				dc == directions[direction].second * distance) {
				return from * kPolicyPlanes + direction * 7 + distance - 1;
			}
		}
	}

	constexpr std::array<std::pair<int, int>, 8> knights{{
		{-2, -1},
		{-2, 1},
		{-1, -2},
		{-1, 2},
		{1, -2},
		{1, 2},
		{2, -1},
		{2, 1},
	}};
	for (int offset = 0; offset < static_cast<int>(knights.size()); ++offset) {
		if (dr == knights[offset].first && dc == knights[offset].second) {
			return from * kPolicyPlanes + 56 + offset;
		}
	}
	throw std::invalid_argument("cannot encode move: " + move_uci(move));
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
std::string move_uci(const chess::Move &move) { return chess::uci::moveToUci(move); }

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

// Expand packed rank bits to the floating-point NCHW tensor consumed by convolutions.
torch::Tensor decode_states(const std::uint8_t *packed, std::int64_t count,
							bool pinned_memory) {
	auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
	if (pinned_memory) {
		options = options.pinned_memory(true);
	}
	auto output = torch::zeros({count, kStatePlanes, 8, 8}, options);
	auto accessor = output.accessor<float, 4>();
	for (std::int64_t item = 0; item < count; ++item) {
		const auto *state = packed + item * kStatePlanes * 8;
		for (int plane = 0; plane < kStatePlanes; ++plane) {
			for (int rank = 0; rank < 8; ++rank) {
				const auto byte = state[plane * 8 + rank];
				for (int file = 0; file < 8; ++file) {
					accessor[item][plane][rank][file] =
						static_cast<float>((byte >> (7 - file)) & 1U);
				}
			}
		}
	}
	return output;
}

// Pack a live batch first, then use the same decoder as persisted training data.
torch::Tensor encode_boards(const std::vector<chess::Board> &boards, bool pinned_memory) {
	std::vector<std::uint8_t> packed(boards.size() * kStatePlanes * 8);
	for (std::size_t index = 0; index < boards.size(); ++index) {
		const auto state = encode_state(boards[index]);
		std::copy(state.begin(), state.end(), packed.begin() + index * state.size());
	}
	return decode_states(packed.data(), static_cast<std::int64_t>(boards.size()),
						 pinned_memory);
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
std::vector<float> normalize_legal_policy(const std::vector<float> &policy,
										  const chess::Board &board) {
	std::vector<float> normalized(kActionSize, 0.0F);
	const auto moves = legal_moves(board);
	if (moves.empty()) {
		return normalized;
	}
	double total = 0.0;
	for (const auto &move : moves) {
		const int index = move_to_index(move);
		const float value =
			index < static_cast<int>(policy.size()) ? std::max(0.0F, policy[index]) : 0.0F;
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
			throw std::runtime_error(
				"CUDA was requested but this LibTorch build has no available CUDA device");
		}
		return torch::Device(requested);
	}
	if (requested == "cpu") {
		return torch::Device(torch::kCPU);
	}
	throw std::invalid_argument("unsupported device: " + requested);
}

// Breadth-first traverse Polyglot entries and emit unique non-terminal frontier positions.
std::vector<std::string> load_opening_positions(const std::string &path, int book_plies,
												int max_positions, std::uint64_t seed) {
	if (path.empty()) {
		return {std::string(chess::constants::STARTPOS)};
	}
	std::ifstream input(path, std::ios::binary);
	if (!input) {
		throw std::runtime_error("opening book not found: " + path);
	}
	std::unordered_multimap<std::uint64_t, std::uint16_t> entries;
	while (input.peek() != std::char_traits<char>::eof()) {
		const auto key = read_be64(input);
		const auto move = read_be16(input);
		static_cast<void>(read_be16(input));
		static_cast<void>(read_be32(input));
		if (!input) {
			throw std::runtime_error("truncated Polyglot opening book: " + path);
		}
		entries.emplace(key, move);
	}

	struct Pending {
		chess::Board board;
		int ply;
	};
	std::queue<Pending> queue;
	queue.push({chess::Board(), 0});
	std::unordered_set<std::string> visited;
	std::unordered_set<std::string> emitted;
	std::vector<std::string> positions;
	std::mt19937_64 rng(seed);

	while (!queue.empty() && static_cast<int>(positions.size()) < std::max(1, max_positions)) {
		auto current = std::move(queue.front());
		queue.pop();
		const auto key = state_key(current.board) + "|" + std::to_string(current.ply);
		if (!visited.insert(key).second) {
			continue;
		}

		std::vector<chess::Move> moves;
		if (current.ply < std::max(1, book_plies)) {
			const auto range = entries.equal_range(current.board.hash());
			for (auto it = range.first; it != range.second; ++it) {
				const auto move = decode_polyglot_move(it->second, current.board);
				if (move.move() != chess::Move::NO_MOVE) {
					moves.push_back(move);
				}
			}
			std::shuffle(moves.begin(), moves.end(), rng);
		}

		if (current.ply > 0 && (current.ply >= std::max(1, book_plies) || moves.empty())) {
			const auto fen = current.board.getFen();
			if (!game_is_over(current.board) && emitted.insert(state_key(current.board)).second) {
				positions.push_back(fen);
			}
			continue;
		}
		for (const auto &move : moves) {
			auto child = current.board;
			child.makeMove(move);
			if (!game_is_over(child)) {
				queue.push({std::move(child), current.ply + 1});
			}
		}
	}
	if (positions.empty()) {
		positions.push_back(std::string(chess::constants::STARTPOS));
	}
	return positions;
}

// Pair each selected FEN with swapped candidate colors to remove first-move/color bias.
std::vector<OpeningSpec> make_arena_specs(int games, const std::string &opening_book,
										  int book_plies, int max_positions, std::uint64_t seed) {
	if (games <= 0) {
		return {};
	}
	if (opening_book.empty()) {
		std::vector<OpeningSpec> specs;
		specs.reserve(games);
		for (int index = 0; index < games; ++index) {
			specs.push_back({std::string(chess::constants::STARTPOS), index % 2 == 0});
		}
		return specs;
	}

	auto positions = load_opening_positions(opening_book, book_plies, max_positions, seed);
	const int required = (games + 1) / 2;
	if (static_cast<int>(positions.size()) < required) {
		throw std::runtime_error(
			"arena requires enough unique opening states: required=" + std::to_string(required) +
			" available=" + std::to_string(positions.size()));
	}
	std::mt19937_64 rng(seed);
	std::shuffle(positions.begin(), positions.end(), rng);
	std::vector<OpeningSpec> specs;
	specs.reserve(games);
	for (int pair = 0; static_cast<int>(specs.size()) < games; ++pair) {
		specs.push_back({positions[pair], true});
		if (static_cast<int>(specs.size()) < games) {
			specs.push_back({positions[pair], false});
		}
	}
	return specs;
}

} // namespace gadus
