#include "eleginus/features.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace eleginus {

namespace {

constexpr int kBiasOffset = 0;
constexpr int kMaterialOffset = 1;
constexpr int kPieceSquareOffset = kMaterialOffset + 5;
constexpr int kMobilityOffset = kPieceSquareOffset + 6 * 64;
constexpr int kAttackOffset = kMobilityOffset + 6;
constexpr int kPairOffset = kAttackOffset + 2 * 6 * 6;
constexpr int kPawnOffset = kPairOffset + 2 * 6 * 6 * 225;
constexpr int kKingOffset = kPawnOffset + 4 * 8;
constexpr int kCastlingOffset = kKingOffset + 6;
static_assert(kCastlingOffset + 2 == FeatureMap::kPrimitiveFeatures);

enum ControlFeature {
	friendly_control,
	enemy_control,
	contested_control,
	friendly_exclusive_control,
	enemy_exclusive_control,
	friendly_hanging_count,
	enemy_hanging_count,
	friendly_hanging_value,
	enemy_hanging_value,
	friendly_overloaded,
	enemy_overloaded,
	friendly_king_escapes,
	enemy_king_escapes,
	friendly_king_pressure,
	enemy_king_pressure
};
static_assert(enemy_king_pressure + 1 == FeatureMap::kControlFeatures);

enum TopologyFeature {
	locked_pawn_pairs,
	pawn_files,
	open_files,
	pawn_contacts,
	all_slider_reach,
	all_safe_mobility,
	friendly_pawn_advances,
	enemy_pawn_advances,
	friendly_pawn_breaks,
	enemy_pawn_breaks,
	friendly_slider_reach,
	enemy_slider_reach,
	friendly_safe_mobility,
	enemy_safe_mobility,
	friendly_regions,
	enemy_regions,
	friendly_largest_region,
	enemy_largest_region
};
static_assert(enemy_largest_region + 1 == FeatureMap::kTopologyFeatures);

enum TransitionFeature {
	legal_moves,
	quiet_moves,
	captures,
	pawn_moves,
	king_moves,
	non_pawn_quiet_moves,
	best_pawn_advance_delta,
	best_pawn_break_delta,
	best_slider_reach_delta,
	best_safe_mobility_delta,
	best_king_mobility_delta,
	best_control_delta,
	mean_pawn_advance_delta,
	mean_pawn_break_delta,
	mean_slider_reach_delta,
	mean_safe_mobility_delta,
	mean_king_mobility_delta,
	mean_control_delta,
	preserve_pawn_advances,
	preserve_pawn_breaks,
	preserve_slider_reach,
	preserve_safe_mobility,
	preserve_king_mobility,
	preserve_control,
	delta_moments
};

constexpr int kStructuralDimensions = 6;
constexpr int kDeltaMoments = kStructuralDimensions * (kStructuralDimensions + 1) / 2;
static_assert(delta_moments + kDeltaMoments == FeatureMap::kTransitionFeatures);

struct StructuralSummary {
	int pawn_advances = 0;
	int pawn_breaks = 0;
	int slider_reach = 0;
	int safe_mobility = 0;
	int king_mobility = 0;
	int control = 0;
};

struct Token {
	chess::Square square;
	chess::Color color;
	int side = 0;
	int type = 0;
	int relative_square = 0;
	int canonical_square = 0;
	chess::Bitboard attacks;
};

struct ControlField {
	std::array<std::array<std::uint8_t, 64>, 2> count{};
	std::array<std::array<std::uint16_t, 64>, 2> minimum{};
	std::array<chess::Bitboard, 2> map{};
};

struct Coordinates {
	float material = 0.0F;
	float openness = 0.0F;
	float pawn_freedom = 0.0F;
	float king_exposure = 0.0F;
	float constraint = 0.0F;
};

enum class Family { general, mobility, pawn, king, transition };

struct Scratch {
	std::array<float, FeatureMap::kBaseFeatures> values{};
	std::array<std::uint32_t, FeatureMap::kBaseFeatures> stamps{};
	std::vector<std::uint32_t> active;
	std::uint32_t generation = 0;

	void reset() {
		if (++generation == 0) {
			stamps.fill(0);
			generation = 1;
		}
		active.clear();
	}

	void add(int index, float value) {
		if (value == 0.0F) {
			return;
		}
		const auto position = static_cast<std::size_t>(index);
		if (stamps[position] != generation) {
			stamps[position] = generation;
			values[position] = value;
			active.push_back(static_cast<std::uint32_t>(index));
		} else {
			values[position] += value;
		}
	}

	float get(std::uint32_t index) const noexcept {
		return stamps[index] == generation ? values[index] : 0.0F;
	}
};

chess::Bitboard attacks_for(const chess::Board &board, const Token &piece) {
	switch (piece.type) {
	case 0:
		return chess::attacks::pawn(piece.color, piece.square);
	case 1:
		return chess::attacks::knight(piece.square);
	case 2:
		return chess::attacks::bishop(piece.square, board.occ());
	case 3:
		return chess::attacks::rook(piece.square, board.occ());
	case 4:
		return chess::attacks::queen(piece.square, board.occ());
	case 5:
		return chess::attacks::king(piece.square);
	default:
		throw std::logic_error("invalid Eleginus piece type");
	}
}

chess::Bitboard attacks_for(const chess::Board &board, chess::Square square, chess::Piece piece) {
	const Token token{square, piece.color(), 0, static_cast<int>(piece.type().internal()), 0, 0, {}};
	return attacks_for(board, token);
}

int piece_value(int type) noexcept {
	constexpr std::array<int, 6> values{{100, 320, 330, 500, 900, 20000}};
	return values[static_cast<std::size_t>(type)];
}

ControlField control_field(const std::vector<Token> &pieces) {
	ControlField field;
	for (auto &side : field.minimum) {
		side.fill(std::numeric_limits<std::uint16_t>::max());
	}
	for (const auto &piece : pieces) {
		const auto color = static_cast<std::size_t>(piece.color);
		auto attacks = piece.attacks;
		field.map[color] |= attacks;
		while (attacks) {
			const auto square = static_cast<std::size_t>(attacks.pop());
			field.count[color][square] = static_cast<std::uint8_t>(std::min(255, field.count[color][square] + 1));
			field.minimum[color][square] = static_cast<std::uint16_t>(
				std::min<int>(field.minimum[color][square], piece_value(piece.type)));
		}
	}
	return field;
}

ControlField control_field(const chess::Board &board) {
	ControlField field;
	for (auto &side : field.minimum) {
		side.fill(std::numeric_limits<std::uint16_t>::max());
	}
	for (int square = 0; square < 64; ++square) {
		const auto source = chess::Square(square);
		const auto piece = board.at(source);
		if (piece == chess::Piece::NONE) {
			continue;
		}
		const auto color = static_cast<std::size_t>(piece.color());
		const int type = static_cast<int>(piece.type().internal());
		auto attacks = attacks_for(board, source, piece);
		field.map[color] |= attacks;
		while (attacks) {
			const auto target = static_cast<std::size_t>(attacks.pop());
			field.count[color][target] = static_cast<std::uint8_t>(std::min(255, field.count[color][target] + 1));
			field.minimum[color][target] = static_cast<std::uint16_t>(std::min<int>(field.minimum[color][target], piece_value(type)));
		}
	}
	return field;
}

std::array<StructuralSummary, 2> structural_summaries(const chess::Board &board, const ControlField &field) {
	std::array<StructuralSummary, 2> summaries;
	for (int square = 0; square < 64; ++square) {
		const auto source = chess::Square(square);
		const auto piece = board.at(source);
		if (piece == chess::Piece::NONE) {
			continue;
		}
		const auto color = piece.color();
		auto &summary = summaries[static_cast<std::size_t>(color)];
		const auto own = board.us(color);
		const auto enemy = board.us(~color);
		const auto enemy_attacks = field.map[static_cast<std::size_t>(~color)];
		const int type = static_cast<int>(piece.type().internal());
		const auto attacks = attacks_for(board, source, piece);
		summary.safe_mobility += static_cast<int>((attacks & ~own & ~enemy_attacks).count());
		if (type >= 2 && type <= 4) {
			summary.slider_reach += static_cast<int>((attacks & ~own).count());
		}
		if (type == 5) {
			summary.king_mobility = static_cast<int>((attacks & ~own & ~enemy_attacks).count());
		}
		if (type != 0) {
			continue;
		}
		const int direction = color == chess::Color::WHITE ? 8 : -8;
		const int destination = square + direction;
		if (destination >= 0 && destination < 64 && board.at(chess::Square(destination)) == chess::Piece::NONE) {
			++summary.pawn_advances;
			const int relative_rank = static_cast<int>(source.relative_square(color).rank().internal());
			const int double_destination = destination + direction;
			if (relative_rank == 1 && double_destination >= 0 && double_destination < 64 &&
				board.at(chess::Square(double_destination)) == chess::Piece::NONE) {
				++summary.pawn_advances;
			}
		}
		summary.pawn_breaks += static_cast<int>((attacks & enemy & board.pieces(chess::PieceType::PAWN, ~color)).count());
	}
	for (int side = 0; side < 2; ++side) {
		summaries[static_cast<std::size_t>(side)].control = static_cast<int>(field.map[static_cast<std::size_t>(side)].count());
	}
	return summaries;
}

std::array<StructuralSummary, 2> structural_summaries(const chess::Board &board) {
	return structural_summaries(board, control_field(board));
}

std::array<int, 6> summary_values(const StructuralSummary &summary) {
	return {{summary.pawn_advances, summary.pawn_breaks, summary.slider_reach, summary.safe_mobility,
		summary.king_mobility, summary.control}};
}

std::pair<int, int> accessibility(const chess::Board &board, chess::Color color) {
	chess::Bitboard pawn_attacks;
	auto enemy_pawns = board.pieces(chess::PieceType::PAWN, ~color);
	while (enemy_pawns) {
		const auto square = chess::Square(enemy_pawns.pop());
		pawn_attacks |= chess::attacks::pawn(~color, square);
	}
	const auto barriers = board.pieces(chess::PieceType::PAWN) | pawn_attacks;
	auto remaining = ~barriers;
	int regions = 0;
	int largest = 0;
	while (remaining) {
		++regions;
		int size = 0;
		chess::Bitboard frontier = chess::Bitboard::fromSquare(chess::Square(remaining.pop()));
		while (frontier) {
			auto next = frontier;
			frontier = {};
			while (next) {
				const auto square = chess::Square(next.pop());
				++size;
				frontier |= chess::attacks::king(square) & remaining;
			}
			remaining &= ~frontier;
		}
		largest = std::max(largest, size);
	}
	return {regions, largest};
}

int pair_index(int family, int first_type, int second_type, int first_square, int second_square) {
	const int dr = second_square / 8 - first_square / 8;
	const int df = second_square % 8 - first_square % 8;
	const int displacement = (dr + 7) * 15 + df + 7;
	return kPairOffset + (((family * 6 + first_type) * 6 + second_type) * 225 + displacement);
}

float phase_of(const std::vector<Token> &pieces) {
	constexpr std::array<int, 6> phase_value{{0, 1, 1, 2, 4, 0}};
	int phase = 0;
	for (const auto &piece : pieces) {
		phase += phase_value[static_cast<std::size_t>(piece.type)];
	}
	return std::min(24, phase) / 24.0F;
}

Family family_of(std::uint32_t index) noexcept {
	if (index >= FeatureMap::kTransitionOffset) {
		return Family::transition;
	}
	if (index >= FeatureMap::kTopologyOffset) {
		const int topology = static_cast<int>(index) - FeatureMap::kTopologyOffset;
		if (topology == locked_pawn_pairs || topology == pawn_files || topology == pawn_contacts ||
			topology == friendly_pawn_advances || topology == enemy_pawn_advances || topology == friendly_pawn_breaks ||
			topology == enemy_pawn_breaks) {
			return Family::pawn;
		}
		return Family::mobility;
	}
	if (index >= FeatureMap::kControlOffset) {
		const int control = static_cast<int>(index) - FeatureMap::kControlOffset;
		if (control == friendly_king_escapes || control == enemy_king_escapes || control == friendly_king_pressure ||
			control == enemy_king_pressure) {
			return Family::king;
		}
		return Family::mobility;
	}
	if (index >= kKingOffset) {
		return Family::king;
	}
	if (index >= kPawnOffset) {
		return Family::pawn;
	}
	if ((index >= kMobilityOffset && index < kPairOffset)) {
		return Family::mobility;
	}
	return Family::general;
}

std::array<float, FeatureMap::kRegimes> mixture(Family family, const Coordinates &coordinates) noexcept {
	float structure = coordinates.openness;
	switch (family) {
	case Family::pawn:
		structure = coordinates.pawn_freedom;
		break;
	case Family::king:
		structure = coordinates.king_exposure;
		break;
	case Family::transition:
		structure = coordinates.constraint;
		break;
	case Family::general:
	case Family::mobility:
		break;
	}
	const float endgame = 1.0F - coordinates.material;
	const float closed = 1.0F - structure;
	return {{endgame * closed, endgame * structure, coordinates.material * closed, coordinates.material * structure}};
}

} // namespace

void FeatureMap::extract(const chess::Board &board, std::vector<Feature> &output) const {
	thread_local std::vector<Token> pieces;
	pieces.clear();
	pieces.reserve(32);
	const auto friendly = board.sideToMove();
	for (int square = 0; square < 64; ++square) {
		const auto piece = board.at(chess::Square(square));
		if (piece == chess::Piece::NONE) {
			continue;
		}
		const int side = piece.color() == friendly ? 0 : 1;
		const int relative = square ^ (piece.color() == chess::Color::BLACK ? 56 : 0);
		const int canonical = square ^ (friendly == chess::Color::BLACK ? 56 : 0);
		pieces.push_back({chess::Square(square), piece.color(), side, static_cast<int>(piece.type().internal()), relative, canonical, {}});
	}
	for (auto &piece : pieces) {
		piece.attacks = attacks_for(board, piece);
	}
	const auto controls = control_field(pieces);

	thread_local Scratch scratch;
	scratch.reset();
	scratch.active.reserve(1400);
	scratch.add(kBiasOffset, 1.0F);

	std::array<std::array<int, 6>, 2> counts{};
	std::array<std::array<int, 8>, 2> pawns_by_file{};
	for (const auto &piece : pieces) {
		const float sign = piece.side == 0 ? 1.0F : -1.0F;
		++counts[static_cast<std::size_t>(piece.side)][static_cast<std::size_t>(piece.type)];
		if (piece.type == 0) {
			++pawns_by_file[static_cast<std::size_t>(piece.side)][static_cast<std::size_t>(piece.square.file().internal())];
		}
		scratch.add(kPieceSquareOffset + piece.type * 64 + piece.relative_square, sign);
	}
	for (int type = 0; type < 5; ++type) {
		scratch.add(kMaterialOffset + type, static_cast<float>(counts[0][type] - counts[1][type]));
	}

	const auto friendly_index = static_cast<std::size_t>(friendly);
	const auto enemy_index = static_cast<std::size_t>(~friendly);
	const auto contested = controls.map[friendly_index] & controls.map[enemy_index];
	const int control = FeatureMap::kControlOffset;
	scratch.add(control + friendly_control, static_cast<float>(controls.map[friendly_index].count()) / 64.0F);
	scratch.add(control + enemy_control, static_cast<float>(controls.map[enemy_index].count()) / 64.0F);
	scratch.add(control + contested_control, static_cast<float>(contested.count()) / 64.0F);
	scratch.add(control + friendly_exclusive_control,
		static_cast<float>((controls.map[friendly_index] & ~controls.map[enemy_index]).count()) / 64.0F);
	scratch.add(control + enemy_exclusive_control,
		static_cast<float>((controls.map[enemy_index] & ~controls.map[friendly_index]).count()) / 64.0F);

	std::array<int, 2> hanging_count{};
	std::array<int, 2> hanging_value{};
	std::array<int, 2> overloaded_count{};
	for (const auto &piece : pieces) {
		if (piece.type == 5) {
			continue;
		}
		const auto own = static_cast<std::size_t>(piece.color);
		const auto enemy = static_cast<std::size_t>(~piece.color);
		const auto square = static_cast<std::size_t>(piece.square.index());
		const bool attacked = controls.count[enemy][square] != 0;
		const bool loose = controls.count[own][square] == 0;
		const bool cheaper_attacker = controls.minimum[enemy][square] < piece_value(piece.type);
		const bool outnumbered = controls.count[enemy][square] > controls.count[own][square];
		if (attacked && (loose || cheaper_attacker || outnumbered)) {
			++hanging_count[own];
			hanging_value[own] += piece_value(piece.type);
		}
	}
	for (const auto &defender : pieces) {
		const auto own = static_cast<std::size_t>(defender.color);
		const auto enemy = static_cast<std::size_t>(~defender.color);
		int obligations = 0;
		for (const auto &target : pieces) {
			if (target.color != defender.color || target.type == 5 || target.square == defender.square) {
				continue;
			}
			const auto square = static_cast<std::size_t>(target.square.index());
			if (static_cast<bool>(defender.attacks & chess::Bitboard::fromSquare(target.square)) && controls.count[enemy][square] != 0 &&
				controls.count[own][square] == 1) {
				++obligations;
			}
		}
		overloaded_count[own] += obligations >= 2 ? 1 : 0;
	}
	std::array<int, 2> king_escapes{};
	std::array<int, 2> king_pressure{};
	for (int side = 0; side < 2; ++side) {
		const auto color = static_cast<chess::Color>(side);
		const auto king = board.kingSq(color);
		const auto ring = chess::attacks::king(king);
		king_escapes[static_cast<std::size_t>(side)] =
			static_cast<int>((ring & ~board.us(color) & ~controls.map[static_cast<std::size_t>(~color)]).count());
		auto pressure_squares = ring | chess::Bitboard::fromSquare(king);
		while (pressure_squares) {
			king_pressure[static_cast<std::size_t>(side)] +=
				controls.count[static_cast<std::size_t>(~color)][static_cast<std::size_t>(pressure_squares.pop())];
		}
	}
	scratch.add(control + friendly_hanging_count, static_cast<float>(hanging_count[friendly_index]) / 8.0F);
	scratch.add(control + enemy_hanging_count, static_cast<float>(hanging_count[enemy_index]) / 8.0F);
	scratch.add(control + friendly_hanging_value, static_cast<float>(hanging_value[friendly_index]) / 900.0F);
	scratch.add(control + enemy_hanging_value, static_cast<float>(hanging_value[enemy_index]) / 900.0F);
	scratch.add(control + friendly_overloaded, static_cast<float>(overloaded_count[friendly_index]) / 4.0F);
	scratch.add(control + enemy_overloaded, static_cast<float>(overloaded_count[enemy_index]) / 4.0F);
	scratch.add(control + friendly_king_escapes, static_cast<float>(king_escapes[friendly_index]) / 8.0F);
	scratch.add(control + enemy_king_escapes, static_cast<float>(king_escapes[enemy_index]) / 8.0F);
	scratch.add(control + friendly_king_pressure, static_cast<float>(king_pressure[friendly_index]) / 16.0F);
	scratch.add(control + enemy_king_pressure, static_cast<float>(king_pressure[enemy_index]) / 16.0F);

	for (const auto &piece : pieces) {
		const float sign = piece.side == 0 ? 1.0F : -1.0F;
		const auto own = board.us(piece.color);
		scratch.add(kMobilityOffset + piece.type, sign * static_cast<float>((piece.attacks & ~own).count()) / 8.0F);
		auto occupied_targets = piece.attacks & board.occ();
		while (occupied_targets) {
			const auto target = board.at(chess::Square(occupied_targets.pop()));
			const int family = target.color() == piece.color ? 1 : 0;
			scratch.add(kAttackOffset + (family * 6 + piece.type) * 6 + static_cast<int>(target.type().internal()), sign);
		}
	}

	for (std::size_t first = 0; first < pieces.size(); ++first) {
		for (std::size_t second = 0; second < pieces.size(); ++second) {
			if (first == second) {
				continue;
			}
			const auto &a = pieces[first];
			const auto &b = pieces[second];
			if (a.side == b.side) {
				scratch.add(pair_index(0, a.type, b.type, a.relative_square, b.relative_square), a.side == 0 ? 1.0F : -1.0F);
			} else if (a.side == 0) {
				scratch.add(pair_index(1, a.type, b.type, a.canonical_square, b.canonical_square), 1.0F);
			} else {
				scratch.add(pair_index(1, a.type, b.type, a.canonical_square, b.canonical_square), -1.0F);
			}
		}
	}

	for (const auto &piece : pieces) {
		if (piece.type != 0) {
			continue;
		}
		const int side = piece.side;
		const float sign = side == 0 ? 1.0F : -1.0F;
		const int file = static_cast<int>(piece.square.file().internal());
		const int rank = static_cast<int>(piece.square.relative_square(piece.color).rank().internal());
		const auto own_pawns = board.pieces(chess::PieceType::PAWN, piece.color);
		const auto enemy_pawns = board.pieces(chess::PieceType::PAWN, ~piece.color);
		if (pawns_by_file[side][file] > 1) {
			scratch.add(kPawnOffset + rank, sign);
		}
		const bool left_pawn = file > 0 && pawns_by_file[side][file - 1] > 0;
		const bool right_pawn = file < 7 && pawns_by_file[side][file + 1] > 0;
		if (!left_pawn && !right_pawn) {
			scratch.add(kPawnOffset + 8 + rank, sign);
		}
		bool passed = true;
		auto enemies = enemy_pawns;
		while (enemies) {
			const chess::Square enemy_square(enemies.pop());
			const int file_distance = std::abs(enemy_square.file() - piece.square.file());
			int forward = enemy_square.rank() - piece.square.rank();
			if (piece.color == chess::Color::BLACK) {
				forward = -forward;
			}
			if (file_distance <= 1 && forward > 0) {
				passed = false;
				break;
			}
		}
		if (passed) {
			scratch.add(kPawnOffset + 16 + rank, sign);
		}
		if (chess::attacks::pawn(~piece.color, piece.square) & own_pawns) {
			scratch.add(kPawnOffset + 24 + rank, sign);
		}
	}

	for (int side = 0; side < 2; ++side) {
		const auto attacker_color = side == 0 ? friendly : ~friendly;
		const auto king_color = ~attacker_color;
		const auto ring = chess::attacks::king(board.kingSq(king_color)) | chess::Bitboard::fromSquare(board.kingSq(king_color));
		const float sign = side == 0 ? 1.0F : -1.0F;
		for (const auto &piece : pieces) {
			if (piece.color == attacker_color) {
				scratch.add(kKingOffset + piece.type, sign * static_cast<float>((piece.attacks & ring).count()));
			}
		}
	}

	const auto rights = board.castlingRights();
	using Side = chess::Board::CastlingRights::Side;
	scratch.add(kCastlingOffset, static_cast<float>(rights.has(friendly, Side::KING_SIDE) - rights.has(~friendly, Side::KING_SIDE)));
	scratch.add(kCastlingOffset + 1, static_cast<float>(rights.has(friendly, Side::QUEEN_SIDE) - rights.has(~friendly, Side::QUEEN_SIDE)));

	const auto summaries = structural_summaries(board, controls);
	const auto &friendly_summary = summaries[static_cast<std::size_t>(friendly)];
	const auto &enemy_summary = summaries[static_cast<std::size_t>(~friendly)];
	int locked_pairs = 0;
	int occupied_pawn_files = 0;
	int open_file_count = 0;
	int contact_count = 0;
	const auto white_pawns = board.pieces(chess::PieceType::PAWN, chess::Color::WHITE);
	const auto black_pawns = board.pieces(chess::PieceType::PAWN, chess::Color::BLACK);
	for (int file = 0; file < 8; ++file) {
		const chess::Bitboard file_mask(0x0101010101010101ULL << file);
		const bool white = static_cast<bool>(white_pawns & file_mask);
		const bool black = static_cast<bool>(black_pawns & file_mask);
		occupied_pawn_files += white && black ? 1 : 0;
		open_file_count += !white && !black ? 1 : 0;
	}
	auto pawns = white_pawns;
	while (pawns) {
		const auto square = chess::Square(pawns.pop());
		const int destination = square.index() + 8;
		locked_pairs += destination < 64 && static_cast<bool>(black_pawns & chess::Bitboard::fromSquare(chess::Square(destination))) ? 1 : 0;
		contact_count += static_cast<int>((chess::attacks::pawn(chess::Color::WHITE, square) & black_pawns).count());
	}
	pawns = black_pawns;
	while (pawns) {
		const auto square = chess::Square(pawns.pop());
		contact_count += static_cast<int>((chess::attacks::pawn(chess::Color::BLACK, square) & white_pawns).count());
	}
	const int topology = FeatureMap::kTopologyOffset;
	scratch.add(topology + locked_pawn_pairs, static_cast<float>(locked_pairs) / 8.0F);
	scratch.add(topology + pawn_files, static_cast<float>(occupied_pawn_files) / 8.0F);
	scratch.add(topology + open_files, static_cast<float>(open_file_count) / 8.0F);
	scratch.add(topology + pawn_contacts, static_cast<float>(contact_count) / 16.0F);
	scratch.add(topology + all_slider_reach, static_cast<float>(friendly_summary.slider_reach + enemy_summary.slider_reach) / 64.0F);
	scratch.add(topology + all_safe_mobility, static_cast<float>(friendly_summary.safe_mobility + enemy_summary.safe_mobility) / 128.0F);
	scratch.add(topology + friendly_pawn_advances, static_cast<float>(friendly_summary.pawn_advances) / 16.0F);
	scratch.add(topology + enemy_pawn_advances, static_cast<float>(enemy_summary.pawn_advances) / 16.0F);
	scratch.add(topology + friendly_pawn_breaks, static_cast<float>(friendly_summary.pawn_breaks) / 8.0F);
	scratch.add(topology + enemy_pawn_breaks, static_cast<float>(enemy_summary.pawn_breaks) / 8.0F);
	scratch.add(topology + friendly_slider_reach, static_cast<float>(friendly_summary.slider_reach) / 64.0F);
	scratch.add(topology + enemy_slider_reach, static_cast<float>(enemy_summary.slider_reach) / 64.0F);
	scratch.add(topology + friendly_safe_mobility, static_cast<float>(friendly_summary.safe_mobility) / 64.0F);
	scratch.add(topology + enemy_safe_mobility, static_cast<float>(enemy_summary.safe_mobility) / 64.0F);
	const auto friendly_access = accessibility(board, friendly);
	const auto enemy_access = accessibility(board, ~friendly);
	scratch.add(topology + friendly_regions, static_cast<float>(friendly_access.first) / 8.0F);
	scratch.add(topology + enemy_regions, static_cast<float>(enemy_access.first) / 8.0F);
	scratch.add(topology + friendly_largest_region, static_cast<float>(friendly_access.second) / 64.0F);
	scratch.add(topology + enemy_largest_region, static_cast<float>(enemy_access.second) / 64.0F);

	Coordinates coordinates;
	coordinates.material = phase_of(pieces);
	coordinates.openness = std::clamp((static_cast<float>(open_file_count) / 8.0F +
		1.0F - static_cast<float>(locked_pairs) / 8.0F +
		static_cast<float>(friendly_summary.slider_reach + enemy_summary.slider_reach) / 128.0F) / 3.0F, 0.0F, 1.0F);
	coordinates.pawn_freedom = std::clamp((static_cast<float>(friendly_summary.pawn_advances + enemy_summary.pawn_advances) / 32.0F +
		static_cast<float>(friendly_summary.pawn_breaks + enemy_summary.pawn_breaks) / 16.0F) * 0.5F, 0.0F, 1.0F);
	coordinates.king_exposure = std::clamp((static_cast<float>(king_pressure[0] + king_pressure[1]) / 32.0F +
		1.0F - static_cast<float>(king_escapes[0] + king_escapes[1]) / 16.0F) * 0.5F, 0.0F, 1.0F);
	coordinates.constraint = std::clamp(1.0F -
		static_cast<float>(friendly_summary.safe_mobility + enemy_summary.safe_mobility) / 128.0F, 0.0F, 1.0F);

	if (board.occ().count() <= FeatureMap::kTransitionPieceLimit) {
		chess::Movelist moves;
		chess::movegen::legalmoves(moves, board);
		if (!moves.empty()) {
			std::array<int, kStructuralDimensions> best_delta;
			best_delta.fill(std::numeric_limits<int>::min());
			std::array<float, kStructuralDimensions> delta_sum{};
			std::array<int, kStructuralDimensions> preserving{};
			std::array<float, kDeltaMoments> moment_sum{};
			int quiet_count = 0;
			int capture_count = 0;
			int pawn_move_count = 0;
			int king_move_count = 0;
			int non_pawn_quiet_count = 0;
			const auto before = summary_values(friendly_summary);
			constexpr std::array<float, kStructuralDimensions> delta_scale{{4.0F, 4.0F, 16.0F, 32.0F, 8.0F, 32.0F}};
			chess::Board successor = board;
			for (const auto move : moves) {
				const auto moving_piece = board.at(move.from()).type();
				const bool capture = board.isCapture(move);
				const bool pawn_move = moving_piece == chess::PieceType::PAWN;
				const bool king_move = moving_piece == chess::PieceType::KING;
				quiet_count += !capture && move.typeOf() != chess::Move::PROMOTION ? 1 : 0;
				capture_count += capture ? 1 : 0;
				pawn_move_count += pawn_move ? 1 : 0;
				king_move_count += king_move ? 1 : 0;
				non_pawn_quiet_count += !capture && !pawn_move && move.typeOf() != chess::Move::PROMOTION ? 1 : 0;
				successor.makeMove(move);
				const auto successor_summaries = structural_summaries(successor);
				const auto after = summary_values(successor_summaries[static_cast<std::size_t>(friendly)]);
				successor.unmakeMove(move);
				std::array<float, kStructuralDimensions> normalized_delta{};
				for (std::size_t index = 0; index < before.size(); ++index) {
					const int delta = after[index] - before[index];
					best_delta[index] = std::max(best_delta[index], delta);
					normalized_delta[index] = static_cast<float>(delta) / delta_scale[index];
					delta_sum[index] += normalized_delta[index];
					preserving[index] += delta >= 0 ? 1 : 0;
				}
				int moment = 0;
				for (int left = 0; left < kStructuralDimensions; ++left) {
					for (int right = left; right < kStructuralDimensions; ++right) {
						moment_sum[static_cast<std::size_t>(moment++)] +=
							normalized_delta[static_cast<std::size_t>(left)] * normalized_delta[static_cast<std::size_t>(right)];
					}
				}
			}
			const int transition = FeatureMap::kTransitionOffset;
			const float move_scale = 1.0F / static_cast<float>(moves.size());
			scratch.add(transition + legal_moves, static_cast<float>(moves.size()) / 32.0F);
			scratch.add(transition + quiet_moves, static_cast<float>(quiet_count) / 32.0F);
			scratch.add(transition + captures, static_cast<float>(capture_count) / 16.0F);
			scratch.add(transition + pawn_moves, static_cast<float>(pawn_move_count) / 16.0F);
			scratch.add(transition + king_moves, static_cast<float>(king_move_count) / 8.0F);
			scratch.add(transition + non_pawn_quiet_moves, static_cast<float>(non_pawn_quiet_count) / 32.0F);
			for (std::size_t index = 0; index < best_delta.size(); ++index) {
				scratch.add(transition + best_pawn_advance_delta + static_cast<int>(index),
					static_cast<float>(best_delta[index]) / delta_scale[index]);
				scratch.add(transition + mean_pawn_advance_delta + static_cast<int>(index), delta_sum[index] * move_scale);
				scratch.add(transition + preserve_pawn_advances + static_cast<int>(index),
					static_cast<float>(preserving[index]) * move_scale);
			}
			for (int moment = 0; moment < kDeltaMoments; ++moment) {
				scratch.add(transition + delta_moments + moment, moment_sum[static_cast<std::size_t>(moment)] * move_scale);
			}
		}
	}

	output.clear();
	output.reserve(kRegimes * scratch.active.size());
	for (const auto index : scratch.active) {
		const float value = scratch.get(index);
		if (value == 0.0F) {
			continue;
		}
		const auto gate = mixture(family_of(index), coordinates);
		for (std::uint32_t regime = 0; regime < kRegimes; ++regime) {
			if (gate[regime] != 0.0F) {
				output.push_back({kRegimes * index + regime, value * gate[regime]});
			}
		}
	}
}

void FeatureMap::initialize(std::vector<float> &weights) const {
	weights.assign(kFixedFeatures, 0.0F);
	auto set = [&](int base, float middlegame_cp, float endgame_cp) {
		const auto offset = static_cast<std::size_t>(kRegimes * base);
		weights[offset] = endgame_cp;
		weights[offset + 1] = endgame_cp;
		weights[offset + 2] = middlegame_cp;
		weights[offset + 3] = middlegame_cp;
	};

	constexpr std::array<float, 5> middlegame_material{{82.0F, 337.0F, 365.0F, 477.0F, 1025.0F}};
	constexpr std::array<float, 5> endgame_material{{94.0F, 281.0F, 297.0F, 512.0F, 936.0F}};
	for (int type = 0; type < 5; ++type) {
		set(kMaterialOffset + type, middlegame_material[static_cast<std::size_t>(type)], endgame_material[static_cast<std::size_t>(type)]);
	}

	for (int type = 0; type < 6; ++type) {
		for (int square = 0; square < 64; ++square) {
			const float file = static_cast<float>(square % 8);
			const float rank = static_cast<float>(square / 8);
			const float center = 7.0F - std::abs(2.0F * file - 7.0F) - std::abs(2.0F * rank - 7.0F);
			float middlegame = 0.0F;
			float endgame = 0.0F;
			if (type == 0) {
				middlegame = 5.0F * rank + center;
				endgame = 8.0F * rank + 0.5F * center;
			} else if (type == 1) {
				middlegame = 5.0F * center;
				endgame = 4.0F * center;
			} else if (type == 2) {
				middlegame = 3.0F * center;
				endgame = 3.0F * center;
			} else if (type == 3) {
				middlegame = rank == 6.0F ? 18.0F : rank * 1.5F;
				endgame = rank == 6.0F ? 10.0F : rank;
			} else if (type == 4) {
				middlegame = center;
				endgame = 2.0F * center;
			} else {
				const float back_rank = rank == 0.0F ? 12.0F : -4.0F * rank;
				middlegame = back_rank - 2.0F * center;
				endgame = 6.0F * center;
			}
			set(kPieceSquareOffset + type * 64 + square, middlegame, endgame);
		}
	}

	constexpr std::array<float, 6> mobility{{1.0F, 4.0F, 4.0F, 2.0F, 1.0F, 0.0F}};
	for (int type = 0; type < 6; ++type) {
		set(kMobilityOffset + type, 8.0F * mobility[static_cast<std::size_t>(type)], 8.0F * mobility[static_cast<std::size_t>(type)]);
	}
	constexpr std::array<float, 6> victim{{3.0F, 10.0F, 10.0F, 16.0F, 28.0F, 0.0F}};
	for (int attacker = 0; attacker < 6; ++attacker) {
		for (int target = 0; target < 6; ++target) {
			set(kAttackOffset + attacker * 6 + target, victim[static_cast<std::size_t>(target)], victim[static_cast<std::size_t>(target)]);
			set(kAttackOffset + 36 + attacker * 6 + target, 2.0F, 2.0F);
		}
	}

	for (int displacement = 0; displacement < 225; ++displacement) {
		set(kPairOffset + ((2 * 6 + 2) * 225 + displacement), 12.5F, 12.5F);
	}
	for (int rank = 0; rank < 8; ++rank) {
		set(kPawnOffset + rank, -12.0F, -16.0F);
		set(kPawnOffset + 8 + rank, -10.0F, -12.0F);
		set(kPawnOffset + 16 + rank, 4.0F * rank * rank, 7.0F * rank * rank);
		set(kPawnOffset + 24 + rank, 6.0F, 8.0F);
	}
	constexpr std::array<float, 6> king_pressure{{2.0F, 6.0F, 6.0F, 8.0F, 12.0F, 0.0F}};
	for (int type = 0; type < 6; ++type) {
		set(kKingOffset + type, king_pressure[static_cast<std::size_t>(type)], king_pressure[static_cast<std::size_t>(type)] * 0.25F);
	}
	set(kCastlingOffset, 12.0F, 0.0F);
	set(kCastlingOffset + 1, 8.0F, 0.0F);
}

} // namespace eleginus
