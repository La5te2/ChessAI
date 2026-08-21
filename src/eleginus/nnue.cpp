// Implements sparse feature deltas and incremental Value inference.

#include "eleginus/nnue.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace eleginus {

namespace {

constexpr int kAccumulatorRefreshInterval = 256;

int oriented_square(int square, chess::Color perspective) {
	return perspective == chess::Color::WHITE ? square : square ^ 56;
}

int horizontally_mirrored_square(int square, int king_square) {
	return king_square % 8 < 4 ? square ^ 7 : square;
}

int relative_piece_index(const chess::Piece &piece, chess::Color perspective) {
	const int type = static_cast<int>(piece.type().internal());
	if (type < 0 || type > 5) {
		throw std::runtime_error("invalid piece in Eleginus feature encoder");
	}
	return type + (piece.color() == perspective ? 0 : 6);
}

int castling_feature(const chess::Board &board, chess::Color perspective) {
	const auto rights = board.castlingRights();
	const auto opponent = ~perspective;
	int mask = 0;
	if (rights.has(perspective, chess::Board::CastlingRights::Side::KING_SIDE))
		mask |= 1;
	if (rights.has(perspective, chess::Board::CastlingRights::Side::QUEEN_SIDE))
		mask |= 2;
	if (rights.has(opponent, chess::Board::CastlingRights::Side::KING_SIDE))
		mask |= 4;
	if (rights.has(opponent, chess::Board::CastlingRights::Side::QUEEN_SIDE))
		mask |= 8;
	return kEncodedCastlingFeatureBase + mask;
}

int en_passant_feature(const chess::Board &board) {
	int file = 0;
	if (board.enpassantSq().is_valid())
		file = board.enpassantSq().index() % 8 + 1;
	return kEncodedEnPassantFeatureBase + file;
}

void require_size(const std::vector<float> &values, std::size_t expected, const char *name) {
	if (values.size() != expected) {
		throw std::invalid_argument(std::string("invalid Eleginus ") + name + " size");
	}
}

void add_feature(FeatureAccumulator &accumulator, const std::vector<float> &table,
				 int feature, float sign) {
	const auto offset = static_cast<std::size_t>(feature) * kValueFeatureWidth;
	for (int channel = 0; channel < kValueFeatureWidth; ++channel) {
		accumulator[static_cast<std::size_t>(channel)] +=
			sign * table[offset + static_cast<std::size_t>(channel)];
	}
}

FloatAccumulator refresh_accumulator(const EncodedFeatures &encoded,
									 const std::vector<float> &table,
									 const std::vector<float> &bias) {
	FloatAccumulator result;
	result.white_to_move = encoded.white_to_move;
	for (const int feature : encoded.perspective[0])
		result.piece_count += feature != kEncodedPaddingFeature;
	result.piece_count -= 2;
	for (int perspective = 0; perspective < kPerspectiveCount; ++perspective) {
		auto &values = result.perspective[static_cast<std::size_t>(perspective)];
		std::copy(bias.begin(), bias.end(), values.begin());
		const auto canonical = canonicalize_features(
			encoded.perspective[static_cast<std::size_t>(perspective)]);
		for (const int feature : canonical) {
			if (feature != kPaddingFeature) {
				add_feature(values, table, feature, 1.0F);
			}
		}
	}
	return result;
}

void update_accumulator_in_place(FloatAccumulator &current,
								 const EncodedFeatures &old_features,
								 const EncodedFeatures &new_features,
								 const std::vector<float> &table) {
	current.white_to_move = new_features.white_to_move;
	current.piece_count = 0;
	for (const int feature : new_features.perspective[0])
		current.piece_count += feature != kEncodedPaddingFeature;
	current.piece_count -= 2;
	for (int perspective = 0; perspective < kPerspectiveCount; ++perspective) {
		const auto old_list = canonicalize_features(
			old_features.perspective[static_cast<std::size_t>(perspective)]);
		const auto new_list = canonicalize_features(
			new_features.perspective[static_cast<std::size_t>(perspective)]);
		auto &values = current.perspective[static_cast<std::size_t>(perspective)];
		for (const int feature : old_list) {
			if (feature != kPaddingFeature &&
				std::find(new_list.begin(), new_list.end(), feature) == new_list.end())
				add_feature(values, table, feature, -1.0F);
		}
		for (const int feature : new_list) {
			if (feature != kPaddingFeature &&
				std::find(old_list.begin(), old_list.end(), feature) == old_list.end())
				add_feature(values, table, feature, 1.0F);
		}
	}
}

FloatAccumulator update_accumulator(const FloatAccumulator &current,
									const EncodedFeatures &old_features,
									const EncodedFeatures &new_features,
									const std::vector<float> &table) {
	FloatAccumulator result = current;
	update_accumulator_in_place(result, old_features, new_features, table);
	return result;
}

struct PieceToken {
	int type = 0;
	int square = 0;
};

struct ControlLayout {
	std::array<std::array<std::uint8_t, 64>, kPerspectiveCount> occupancy{};
	std::array<std::array<std::uint8_t, 64>, kPerspectiveCount> piece_type{};
	std::array<std::array<std::uint64_t, 64>, kPerspectiveCount> attacks{};
	std::array<std::array<float, kMaterialFeatureWidth>, kPerspectiveCount> material{};
};

bool inside(int rank, int file) {
	return rank >= 0 && rank < 8 && file >= 0 && file < 8;
}

template <typename Add>
void attack_targets(const PieceToken &piece, const std::array<std::uint8_t, 64> &occupancy,
					Add add) {
	const int rank = piece.square / 8;
	const int file = piece.square % 8;
	const int kind = piece.type % 6;
	if (kind == 0) {
		const int direction = piece.type < 6 ? 1 : -1;
		for (const int df : {-1, 1}) {
			if (inside(rank + direction, file + df))
				add((rank + direction) * 8 + file + df);
		}
		return;
	}
	if (kind == 1) {
		constexpr std::array offsets{
			std::pair{-2, -1}, std::pair{-2, 1}, std::pair{-1, -2}, std::pair{-1, 2},
			std::pair{1, -2}, std::pair{1, 2}, std::pair{2, -1}, std::pair{2, 1},
		};
		for (const auto &[dr, df] : offsets) {
			if (inside(rank + dr, file + df))
				add((rank + dr) * 8 + file + df);
		}
		return;
	}
	if (kind == 5) {
		for (int dr = -1; dr <= 1; ++dr) {
			for (int df = -1; df <= 1; ++df) {
				if ((dr != 0 || df != 0) && inside(rank + dr, file + df))
					add((rank + dr) * 8 + file + df);
			}
		}
		return;
	}
	const bool diagonal = kind == 2 || kind == 4;
	const bool straight = kind == 3 || kind == 4;
	for (int dr = -1; dr <= 1; ++dr) {
		for (int df = -1; df <= 1; ++df) {
			if (dr == 0 && df == 0)
				continue;
			if ((dr != 0 && df != 0 && !diagonal) || ((dr == 0 || df == 0) && !straight))
				continue;
			for (int distance = 1; distance < 8; ++distance) {
				const int target_rank = rank + dr * distance;
				const int target_file = file + df * distance;
				if (!inside(target_rank, target_file))
					break;
				const int target = target_rank * 8 + target_file;
				add(target);
				if (occupancy[static_cast<std::size_t>(target)] != 0)
					break;
			}
		}
	}
}

std::uint64_t attack_mask(const PieceToken &piece,
						  const std::array<std::uint8_t, 64> &occupancy) {
	std::uint64_t result = 0;
	attack_targets(piece, occupancy, [&](int target) {
		result |= std::uint64_t{1} << target;
	});
	return result;
}

bool is_slider(std::uint8_t piece_type) {
	if (piece_type == 0)
		return false;
	const int kind = (piece_type - 1) % 6;
	return kind == 2 || kind == 3 || kind == 4;
}

bool changed_square_affects_slider(int source, std::uint8_t piece_type, int changed_square,
								   const std::array<std::uint8_t, 64> &occupancy) {
	if (!is_slider(piece_type) || source == changed_square)
		return false;
	const int source_rank = source / 8;
	const int source_file = source % 8;
	const int target_rank = changed_square / 8;
	const int target_file = changed_square % 8;
	const int rank_delta = target_rank - source_rank;
	const int file_delta = target_file - source_file;
	const bool diagonal = std::abs(rank_delta) == std::abs(file_delta);
	const bool straight = rank_delta == 0 || file_delta == 0;
	const int kind = (piece_type - 1) % 6;
	if ((diagonal && kind != 2 && kind != 4) ||
		(straight && kind != 3 && kind != 4) || (!diagonal && !straight))
		return false;
	const int rank_step = (rank_delta > 0) - (rank_delta < 0);
	const int file_step = (file_delta > 0) - (file_delta < 0);
	int rank = source_rank + rank_step;
	int file = source_file + file_step;
	while (rank != target_rank || file != target_file) {
		if (occupancy[static_cast<std::size_t>(rank * 8 + file)] != 0)
			return false;
		rank += rank_step;
		file += file_step;
	}
	return true;
}

ControlLayout control_layout(const EncodedFeatures &encoded) {
	ControlLayout result;
	for (int perspective = 0; perspective < kPerspectiveCount; ++perspective) {
		const auto canonical = canonicalize_features(
			encoded.perspective[static_cast<std::size_t>(perspective)]);
		auto &occupancy = result.occupancy[static_cast<std::size_t>(perspective)];
		auto &piece_type = result.piece_type[static_cast<std::size_t>(perspective)];
		std::array<int, 6> own{};
		std::array<int, 6> opponent{};
		for (const int feature : canonical) {
			if (feature < 0 || feature >= kPieceFeatureCount)
				continue;
			const int code = feature % (12 * 64);
			const PieceToken piece{code / 64, code % 64};
			occupancy[static_cast<std::size_t>(piece.square)] =
				static_cast<std::uint8_t>(piece.type + 1);
			piece_type[static_cast<std::size_t>(piece.square)] =
				static_cast<std::uint8_t>(piece.type + 1);
			auto &counts = piece.type < 6 ? own : opponent;
			++counts[static_cast<std::size_t>(piece.type % 6)];
		}
		auto &material = result.material[static_cast<std::size_t>(perspective)];
		for (int type = 0; type < 5; ++type)
			material[static_cast<std::size_t>(type)] =
				static_cast<float>(own[static_cast<std::size_t>(type)] -
					opponent[static_cast<std::size_t>(type)]);
		material[5] = static_cast<float>((own[2] >= 2) - (opponent[2] >= 2));
	}
	return result;
}

ControlLayout control_runtime_features(const EncodedFeatures &encoded) {
	ControlLayout result = control_layout(encoded);
	for (int perspective = 0; perspective < kPerspectiveCount; ++perspective) {
		for (int square = 0; square < 64; ++square) {
			const auto type = result.piece_type[static_cast<std::size_t>(perspective)]
				[static_cast<std::size_t>(square)];
			if (type != 0) {
				result.attacks[static_cast<std::size_t>(perspective)]
					[static_cast<std::size_t>(square)] =
					attack_mask(PieceToken{type - 1, square},
						result.occupancy[static_cast<std::size_t>(perspective)]);
			}
		}
	}
	return result;
}

ControlEdge control_edge(int perspective, int source_square, int target_square,
						 std::uint8_t encoded_piece_type) {
	const int piece_type = encoded_piece_type - 1;
	const int from_rank = source_square / 8;
	const int from_file = source_square % 8;
	const int target_rank = target_square / 8;
	const int target_file = target_square % 8;
	return ControlEdge{
		static_cast<std::uint16_t>(piece_type * 64 + source_square),
		static_cast<std::uint16_t>(piece_type * 64 + target_square),
		static_cast<std::uint16_t>(piece_type * 225 +
			(target_rank - from_rank + 7) * 15 + target_file - from_file + 7),
		static_cast<std::uint8_t>(perspective),
		static_cast<std::uint8_t>(target_square),
		static_cast<std::uint8_t>(piece_type < 6 ? 0 : 1),
	};
}

void add_control_message(std::array<float, kControlWidth> &field,
						 const ValueWeights &weights, const ControlEdge &edge, float sign) {
	const auto source = static_cast<std::size_t>(edge.source) * kControlWidth;
	const auto target = static_cast<std::size_t>(edge.target) * kControlWidth;
	const auto geometry = static_cast<std::size_t>(edge.geometry) * kControlWidth;
	for (int channel = 0; channel < kControlWidth; ++channel) {
		const auto index = static_cast<std::size_t>(channel);
		const float latent = weights.control_source[source + index] +
			weights.control_target[target + index] + weights.control_geometry[geometry + index];
		field[index] += sign * latent;
	}
}

std::array<float, kControlLocalWidth> local_square(
	const ControlAccumulator &control, const ValueWeights &weights, int perspective, int square) {
	constexpr int input_width = kControlWidth * 2 + kControlCountWidth * 2 +
		kControlOccupancyWidth + kControlSquareWidth;
	std::array<float, input_width> input{};
	int cursor = 0;
	for (int ownership = 0; ownership < 2; ++ownership) {
		const auto &field = control.field[static_cast<std::size_t>(perspective * 2 + ownership)]
			[static_cast<std::size_t>(square)];
		for (const float value : field)
			input[static_cast<std::size_t>(cursor++)] = value;
	}
	for (int ownership = 0; ownership < 2; ++ownership) {
		const int count = std::min<int>(control.count[static_cast<std::size_t>(perspective)]
			[static_cast<std::size_t>(ownership)][static_cast<std::size_t>(square)], 7);
		const auto offset = static_cast<std::size_t>(count * kControlCountWidth);
		for (int channel = 0; channel < kControlCountWidth; ++channel)
			input[static_cast<std::size_t>(cursor++)] =
				weights.control_count[offset + static_cast<std::size_t>(channel)];
	}
	const auto occupancy_offset = static_cast<std::size_t>(
		control.occupancy[static_cast<std::size_t>(perspective)][static_cast<std::size_t>(square)] *
		kControlOccupancyWidth);
	for (int channel = 0; channel < kControlOccupancyWidth; ++channel)
		input[static_cast<std::size_t>(cursor++)] =
			weights.control_occupancy[occupancy_offset + static_cast<std::size_t>(channel)];
	const auto square_offset = static_cast<std::size_t>(square * kControlSquareWidth);
	for (int channel = 0; channel < kControlSquareWidth; ++channel)
		input[static_cast<std::size_t>(cursor++)] =
			weights.control_square[square_offset + static_cast<std::size_t>(channel)];

	std::array<float, kControlLocalWidth> result{};
	for (int row = 0; row < kControlLocalWidth; ++row) {
		float value = weights.control_local_bias[static_cast<std::size_t>(row)];
		const auto offset = static_cast<std::size_t>(row * input_width);
		for (int column = 0; column < input_width; ++column)
			value += weights.control_local_weight[offset + static_cast<std::size_t>(column)] *
				input[static_cast<std::size_t>(column)];
		value = std::clamp(value, 0.0F, 1.0F);
		result[static_cast<std::size_t>(row)] = value * value;
	}
	return result;
}

void add_pooling_contribution(ControlAccumulator &control, const ValueWeights &weights,
							  int perspective,
							  const std::array<float, kControlLocalWidth> &local, float sign) {
	std::array<float, kControlAttentionHeads * kControlAttentionKeyWidth> key{};
	std::array<float, kControlAttentionWidth> value{};
	for (int row = 0; row < kControlAttentionHeads * kControlAttentionKeyWidth; ++row) {
		float total = weights.attention_key_bias[static_cast<std::size_t>(row)];
		for (int column = 0; column < kControlLocalWidth; ++column)
			total += weights.attention_key_weight[static_cast<std::size_t>(
				row * kControlLocalWidth + column)] * local[static_cast<std::size_t>(column)];
		key[static_cast<std::size_t>(row)] = total;
	}
	for (int row = 0; row < kControlAttentionWidth; ++row) {
		float total = weights.attention_value_bias[static_cast<std::size_t>(row)];
		for (int column = 0; column < kControlLocalWidth; ++column)
			total += weights.attention_value_weight[static_cast<std::size_t>(
				row * kControlLocalWidth + column)] * local[static_cast<std::size_t>(column)];
		value[static_cast<std::size_t>(row)] = total;
	}
	auto &mean = control.mean[static_cast<std::size_t>(perspective)];
	auto &numerator = control.attention_numerator[static_cast<std::size_t>(perspective)];
	for (int channel = 0; channel < kControlLocalWidth; ++channel)
		mean[static_cast<std::size_t>(channel)] +=
			sign * local[static_cast<std::size_t>(channel)] / 64.0F;
	for (int head = 0; head < kControlAttentionHeads; ++head) {
		const auto key_offset = static_cast<std::size_t>(head * kControlAttentionKeyWidth);
		const auto query_offset = static_cast<std::size_t>(
			(control.bucket * kControlAttentionHeads + head) * kControlAttentionKeyWidth);
		float logit = 0.0F;
		for (int channel = 0; channel < kControlAttentionKeyWidth; ++channel)
			logit += weights.attention_query[query_offset + static_cast<std::size_t>(channel)] *
				key[key_offset + static_cast<std::size_t>(channel)];
		logit = std::clamp(logit / std::sqrt(static_cast<float>(kControlAttentionKeyWidth)),
			-8.0F, 8.0F);
		const float attention_weight = std::exp(logit);
		for (int channel = 0; channel < kControlAttentionHeadWidth; ++channel) {
			const auto value_index = static_cast<std::size_t>(
				head * kControlAttentionHeadWidth + channel);
			numerator[static_cast<std::size_t>(head)][static_cast<std::size_t>(channel)] +=
				sign * attention_weight * value[value_index];
		}
		control.attention_denominator[static_cast<std::size_t>(perspective)]
			[static_cast<std::size_t>(head)] += sign * attention_weight;
	}
}

void finish_pooling(ControlAccumulator &control, int perspective) {
	for (int head = 0; head < kControlAttentionHeads; ++head) {
		const float denominator = std::max(
			control.attention_denominator[static_cast<std::size_t>(perspective)]
				[static_cast<std::size_t>(head)], 1.0e-12F);
		for (int channel = 0; channel < kControlAttentionHeadWidth; ++channel) {
			const auto output = static_cast<std::size_t>(
				head * kControlAttentionHeadWidth + channel);
			control.attention[static_cast<std::size_t>(perspective)][output] =
				control.attention_numerator[static_cast<std::size_t>(perspective)]
					[static_cast<std::size_t>(head)][static_cast<std::size_t>(channel)] /
				denominator;
		}
	}
}

void refresh_pooling(ControlAccumulator &control, const ValueWeights &weights) {
	for (int perspective = 0; perspective < kPerspectiveCount; ++perspective) {
		control.mean[static_cast<std::size_t>(perspective)].fill(0.0F);
		for (auto &head : control.attention_numerator[static_cast<std::size_t>(perspective)])
			head.fill(0.0F);
		control.attention_denominator[static_cast<std::size_t>(perspective)].fill(0.0F);
		for (int square = 0; square < 64; ++square)
			add_pooling_contribution(control, weights, perspective,
				control.local[static_cast<std::size_t>(perspective)]
					[static_cast<std::size_t>(square)], 1.0F);
		finish_pooling(control, perspective);
	}
}

ControlAccumulator refresh_control(const ControlLayout &features, const ValueWeights &weights,
								   int bucket) {
	ControlAccumulator result;
	result.occupancy = features.occupancy;
	result.piece_type = features.piece_type;
	result.attacks = features.attacks;
	result.material = features.material;
	result.bucket = bucket;
	for (int perspective = 0; perspective < kPerspectiveCount; ++perspective) {
		for (int source = 0; source < 64; ++source) {
			const auto type = result.piece_type[static_cast<std::size_t>(perspective)]
				[static_cast<std::size_t>(source)];
			auto attacks = result.attacks[static_cast<std::size_t>(perspective)]
				[static_cast<std::size_t>(source)];
			while (attacks != 0) {
				const int target = std::countr_zero(attacks);
				attacks &= attacks - 1;
				const auto edge = control_edge(perspective, source, target, type);
				add_control_message(result.field[static_cast<std::size_t>(edge.perspective * 2 +
					edge.ownership)][static_cast<std::size_t>(edge.square)], weights, edge, 1.0F);
				auto &count = result.count[static_cast<std::size_t>(edge.perspective)]
					[static_cast<std::size_t>(edge.ownership)][static_cast<std::size_t>(edge.square)];
				if (count < std::numeric_limits<std::uint8_t>::max())
					++count;
			}
		}
	}
	for (int perspective = 0; perspective < kPerspectiveCount; ++perspective)
		for (int square = 0; square < 64; ++square)
			result.local[static_cast<std::size_t>(perspective)][static_cast<std::size_t>(square)] =
				local_square(result, weights, perspective, square);
	refresh_pooling(result, weights);
	return result;
}

void update_control(ControlAccumulator &result, const ControlLayout &features,
					const ValueWeights &weights, int bucket) {
	std::array<std::array<bool, 64>, kPerspectiveCount> dirty{};
	for (int perspective = 0; perspective < kPerspectiveCount; ++perspective) {
		std::uint64_t changed_squares = 0;
		for (int square = 0; square < 64; ++square) {
			if (result.occupancy[static_cast<std::size_t>(perspective)]
					[static_cast<std::size_t>(square)] !=
				features.occupancy[static_cast<std::size_t>(perspective)]
					[static_cast<std::size_t>(square)])
				changed_squares |= std::uint64_t{1} << square;
		}
		for (int source = 0; source < 64; ++source) {
			const auto old_type = result.piece_type[static_cast<std::size_t>(perspective)]
				[static_cast<std::size_t>(source)];
			const auto new_type = features.piece_type[static_cast<std::size_t>(perspective)]
				[static_cast<std::size_t>(source)];
			bool affected = old_type != new_type;
			if (!affected && is_slider(old_type)) {
				auto changed = changed_squares;
				while (changed != 0 && !affected) {
					const int square = std::countr_zero(changed);
					changed &= changed - 1;
					affected = changed_square_affects_slider(source, old_type, square,
						result.occupancy[static_cast<std::size_t>(perspective)]) ||
						changed_square_affects_slider(source, new_type, square,
							features.occupancy[static_cast<std::size_t>(perspective)]);
				}
			}
			if (!affected)
				continue;
			auto old_attacks = result.attacks[static_cast<std::size_t>(perspective)]
				[static_cast<std::size_t>(source)];
			std::uint64_t new_attacks = 0;
			if (new_type != 0) {
				new_attacks = attack_mask(PieceToken{new_type - 1, source},
					features.occupancy[static_cast<std::size_t>(perspective)]);
			}
			const auto removed = old_type == new_type ? old_attacks & ~new_attacks : old_attacks;
			const auto added = old_type == new_type ? new_attacks & ~old_attacks : new_attacks;
			auto apply_edges = [&](std::uint64_t mask, std::uint8_t type, float sign) {
				while (mask != 0) {
					const int target = std::countr_zero(mask);
					mask &= mask - 1;
					const auto edge = control_edge(perspective, source, target, type);
					add_control_message(result.field[static_cast<std::size_t>(edge.perspective * 2 +
						edge.ownership)][static_cast<std::size_t>(edge.square)], weights, edge, sign);
					auto &count = result.count[static_cast<std::size_t>(edge.perspective)]
						[static_cast<std::size_t>(edge.ownership)][static_cast<std::size_t>(edge.square)];
					if (sign < 0.0F) {
						if (count == 0)
							throw std::runtime_error("Eleginus control count underflow");
						--count;
					} else if (count < std::numeric_limits<std::uint8_t>::max()) {
						++count;
					}
					dirty[static_cast<std::size_t>(perspective)]
						[static_cast<std::size_t>(target)] = true;
				}
			};
			if (removed != 0)
				apply_edges(removed, old_type, -1.0F);
			if (added != 0)
				apply_edges(added, new_type, 1.0F);
			result.attacks[static_cast<std::size_t>(perspective)]
				[static_cast<std::size_t>(source)] = new_attacks;
			result.piece_type[static_cast<std::size_t>(perspective)]
				[static_cast<std::size_t>(source)] = new_type;
		}
		for (int square = 0; square < 64; ++square) {
			if (result.occupancy[static_cast<std::size_t>(perspective)]
				[static_cast<std::size_t>(square)] !=
				features.occupancy[static_cast<std::size_t>(perspective)]
				[static_cast<std::size_t>(square)])
				dirty[static_cast<std::size_t>(perspective)][static_cast<std::size_t>(square)] = true;
		}
	}
	result.occupancy = features.occupancy;
	result.material = features.material;
	const bool bucket_changed = result.bucket != bucket;
	result.bucket = bucket;
	for (int perspective = 0; perspective < kPerspectiveCount; ++perspective) {
		for (int square = 0; square < 64; ++square) {
			if (dirty[static_cast<std::size_t>(perspective)][static_cast<std::size_t>(square)]) {
				if (!bucket_changed)
					add_pooling_contribution(result, weights, perspective,
						result.local[static_cast<std::size_t>(perspective)]
							[static_cast<std::size_t>(square)], -1.0F);
				result.local[static_cast<std::size_t>(perspective)][static_cast<std::size_t>(square)] =
					local_square(result, weights, perspective, square);
				if (!bucket_changed)
					add_pooling_contribution(result, weights, perspective,
						result.local[static_cast<std::size_t>(perspective)]
							[static_cast<std::size_t>(square)], 1.0F);
			}
		}
		if (!bucket_changed)
			finish_pooling(result, perspective);
	}
	if (bucket_changed)
		refresh_pooling(result, weights);
}

using DenseInput = std::array<float, kValueDenseWidth * 2>;
using HiddenValues = std::array<float, kValueHiddenWidth>;
using BottleneckValues = std::array<float, kValueBottleneckWidth>;

DenseInput oriented_value_input(const ValueAccumulator &accumulator) {
	DenseInput input{};
	const int first = accumulator.features.white_to_move ? 0 : 1;
	const int second = 1 - first;
	for (int side = 0; side < kPerspectiveCount; ++side) {
		const int perspective = side == 0 ? first : second;
		const auto output_offset = static_cast<std::size_t>(side * kValueDenseWidth);
		for (int channel = 0; channel < kValueAccumulatorWidth; ++channel) {
			const float value = std::clamp(
				accumulator.features.perspective[static_cast<std::size_t>(perspective)]
					[static_cast<std::size_t>(channel)], 0.0F, 1.0F);
			input[output_offset + static_cast<std::size_t>(channel)] = value * value;
		}
		for (int channel = 0; channel < kControlLocalWidth; ++channel)
			input[output_offset + kValueAccumulatorWidth + static_cast<std::size_t>(channel)] =
				accumulator.control.mean[static_cast<std::size_t>(perspective)]
					[static_cast<std::size_t>(channel)];
		for (int channel = 0; channel < kControlAttentionWidth; ++channel)
			input[output_offset + kValueAccumulatorWidth + kControlLocalWidth +
				static_cast<std::size_t>(channel)] =
				accumulator.control.attention[static_cast<std::size_t>(perspective)]
					[static_cast<std::size_t>(channel)];
	}
	return input;
}

template <std::size_t OutputWidth, std::size_t InputWidth>
std::array<float, OutputWidth> linear_relu_bucket(const std::array<float, InputWidth> &input,
												  const std::vector<float> &weight,
												  const std::vector<float> &bias,
												  int bucket) {
	std::array<float, OutputWidth> output{};
	const auto first_row = static_cast<std::size_t>(bucket) * OutputWidth;
	for (std::size_t row = 0; row < OutputWidth; ++row) {
		const auto source_row = first_row + row;
		float sum = bias[source_row];
		const auto offset = source_row * InputWidth;
		for (std::size_t column = 0; column < input.size(); ++column)
			sum += weight[offset + column] * input[column];
		output[row] = std::max(0.0F, sum);
	}
	return output;
}

} // namespace

EncodedFeatures encode_features(const chess::Board &board) {
	EncodedFeatures result;
	for (auto &perspective : result.perspective) {
		perspective.fill(kEncodedPaddingFeature);
	}
	result.white_to_move = board.sideToMove() == chess::Color::WHITE;

	for (int perspective_index = 0; perspective_index < kPerspectiveCount; ++perspective_index) {
		const auto perspective =
			perspective_index == 0 ? chess::Color::WHITE : chess::Color::BLACK;
		const int oriented_king = oriented_square(board.kingSq(perspective).index(), perspective);
		auto &features = result.perspective[static_cast<std::size_t>(perspective_index)];
		int cursor = 0;
		for (int square = 0; square < kBoardSquares; ++square) {
			const auto piece = board.at(chess::Square(square));
			if (piece == chess::Piece::NONE) {
				continue;
			}
			const int piece_index = relative_piece_index(piece, perspective);
			const int oriented = oriented_square(square, perspective);
			features[static_cast<std::size_t>(cursor++)] =
				oriented_king * 12 * 64 + piece_index * 64 + oriented;
		}
		features[static_cast<std::size_t>(cursor++)] =
			castling_feature(board, perspective);
		features[static_cast<std::size_t>(cursor)] =
			en_passant_feature(board);
	}
	return result;
}

PerspectiveFeatures canonicalize_features(const PerspectiveFeatures &features) {
	int king_square = -1;
	for (const int feature : features) {
		if (feature >= 0 && feature < kEncodedPieceFeatureCount) {
			king_square = feature / (12 * 64);
			break;
		}
	}
	if (king_square < 0 || king_square >= 64)
		throw std::runtime_error("Eleginus encoded perspective contains no king bucket");
	const bool mirror = king_square % 8 < 4;
	const int canonical_king = horizontally_mirrored_square(king_square, king_square);
	const int king_bucket = (canonical_king / 8) * 4 + canonical_king % 8 - 4;

	PerspectiveFeatures result{};
	for (std::size_t index = 0; index < features.size(); ++index) {
		const int feature = features[index];
		if (feature == kEncodedPaddingFeature) {
			result[index] = kPaddingFeature;
		} else if (feature >= 0 && feature < kEncodedPieceFeatureCount) {
			const int remainder = feature % (12 * 64);
			const int piece = remainder / 64;
			const int square = remainder % 64;
			result[index] = king_bucket * 12 * 64 + piece * 64 +
				horizontally_mirrored_square(square, king_square);
		} else if (feature >= kEncodedCastlingFeatureBase &&
			feature < kEncodedEnPassantFeatureBase) {
			int mask = feature - kEncodedCastlingFeatureBase;
			if (mirror) {
				mask = ((mask & 1) << 1) | ((mask & 2) >> 1) |
					((mask & 4) << 1) | ((mask & 8) >> 1);
			}
			result[index] = kCastlingFeatureBase + mask;
		} else if (feature >= kEncodedEnPassantFeatureBase &&
			feature < kEncodedFeatureCount) {
			int code = feature - kEncodedEnPassantFeatureBase;
			if (mirror && code > 0)
				code = 9 - code;
			result[index] = kEnPassantFeatureBase + code;
		} else {
			throw std::runtime_error("Eleginus encoded feature is out of range");
		}
	}
	return result;
}

ControlFeatures control_features(const EncodedFeatures &encoded) {
	const ControlLayout layout = control_runtime_features(encoded);
	ControlFeatures result;
	result.occupancy = layout.occupancy;
	result.material = layout.material;
	result.edges.reserve(256);
	for (int perspective = 0; perspective < kPerspectiveCount; ++perspective) {
		for (int square = 0; square < 64; ++square) {
			const auto type = layout.piece_type[static_cast<std::size_t>(perspective)]
				[static_cast<std::size_t>(square)];
			if (type != 0) {
				auto targets = layout.attacks[static_cast<std::size_t>(perspective)]
					[static_cast<std::size_t>(square)];
				while (targets != 0) {
					const int target = std::countr_zero(targets);
					targets &= targets - 1;
					result.edges.push_back(control_edge(perspective, square, target, type));
				}
			}
		}
	}
	std::sort(result.edges.begin(), result.edges.end());
	return result;
}

std::array<std::array<float, kMaterialFeatureWidth>, kPerspectiveCount>
material_features(const EncodedFeatures &encoded) {
	return control_layout(encoded).material;
}

int value_bucket(int piece_count) noexcept {
	return std::clamp((piece_count - 1) / 4, 0, kValueBucketCount - 1);
}

CpuValue::CpuValue(ValueWeights weights) : weights_(std::move(weights)) {
	require_size(weights_.feature_table,
		static_cast<std::size_t>(kFeatureVocabulary) * kValueFeatureWidth,
		"Value feature table");
	require_size(weights_.accumulator_bias, kValueFeatureWidth, "Value accumulator bias");
	require_size(weights_.control_source,
		static_cast<std::size_t>(kControlSourceVocabulary) * kControlWidth,
		"control source table");
	require_size(weights_.control_target,
		static_cast<std::size_t>(kControlSourceVocabulary) * kControlWidth,
		"control target table");
	require_size(weights_.control_geometry,
		static_cast<std::size_t>(kControlGeometryVocabulary) * kControlWidth,
		"control geometry table");
	require_size(weights_.control_occupancy,
		static_cast<std::size_t>(kControlOccupancyVocabulary) * kControlOccupancyWidth,
		"control occupancy table");
	require_size(weights_.control_count,
		static_cast<std::size_t>(kControlCountVocabulary) * kControlCountWidth,
		"control count table");
	require_size(weights_.control_square, 64 * kControlSquareWidth, "control square table");
	require_size(weights_.control_local_weight,
		static_cast<std::size_t>(kControlLocalWidth) *
			(kControlWidth * 2 + kControlCountWidth * 2 + kControlOccupancyWidth +
			 kControlSquareWidth),
		"control local weight");
	require_size(weights_.control_local_bias, kControlLocalWidth, "control local bias");
	require_size(weights_.attention_key_weight,
		static_cast<std::size_t>(kControlAttentionHeads * kControlAttentionKeyWidth) *
			kControlLocalWidth,
		"attention key weight");
	require_size(weights_.attention_key_bias,
		kControlAttentionHeads * kControlAttentionKeyWidth, "attention key bias");
	require_size(weights_.attention_value_weight,
		static_cast<std::size_t>(kControlAttentionWidth) * kControlLocalWidth,
		"attention value weight");
	require_size(weights_.attention_value_bias, kControlAttentionWidth, "attention value bias");
	require_size(weights_.attention_query,
		static_cast<std::size_t>(kValueBucketCount * kControlAttentionHeads) *
			kControlAttentionKeyWidth,
		"attention query");
	require_size(weights_.material_weight, kMaterialFeatureWidth, "material weight");
	require_size(weights_.hidden_weight,
		static_cast<std::size_t>(kValueBucketCount) * kValueHiddenWidth *
			kValueDenseWidth * 2,
		"Value hidden weight");
	require_size(weights_.hidden_bias, kValueBucketCount * kValueHiddenWidth,
		"Value hidden bias");
	require_size(weights_.bottleneck_weight,
		static_cast<std::size_t>(kValueBucketCount) * kValueBottleneckWidth *
			kValueHiddenWidth,
		"Value bottleneck weight");
	require_size(weights_.bottleneck_bias, kValueBucketCount * kValueBottleneckWidth,
		"Value bottleneck bias");
	require_size(weights_.output_weight, kValueBucketCount * kValueBottleneckWidth,
		"Value output weight");
	require_size(weights_.output_bias, kValueBucketCount, "Value output bias");
}

ValueAccumulator CpuValue::refresh(const chess::Board &board) const {
	const auto encoded = encode_features(board);
	const auto pieces = control_runtime_features(encoded);
	auto feature_accumulator = refresh_accumulator(
		encoded, weights_.feature_table, weights_.accumulator_bias);
	const int bucket = value_bucket(feature_accumulator.piece_count);
	return ValueAccumulator{
		encoded,
		std::move(feature_accumulator),
		refresh_control(pieces, weights_, bucket),
		0,
	};
}

ValueAccumulator CpuValue::update(const ValueAccumulator &current, const chess::Board &before,
								  const chess::Board &after) const {
	const auto old_encoded = encode_features(before);
	const auto new_encoded = encode_features(after);
	auto feature_accumulator = update_accumulator(current.features, old_encoded, new_encoded,
		weights_.feature_table);
	const int bucket = value_bucket(feature_accumulator.piece_count);
	auto control = current.control;
	update_control(control, control_layout(new_encoded), weights_, bucket);
	int incremental_updates = current.incremental_updates + 1;
	if (incremental_updates >= kAccumulatorRefreshInterval) {
		feature_accumulator = refresh_accumulator(
			new_encoded, weights_.feature_table, weights_.accumulator_bias);
		control = refresh_control(control_runtime_features(new_encoded), weights_,
			value_bucket(feature_accumulator.piece_count));
		incremental_updates = 0;
	}
	return ValueAccumulator{
		new_encoded,
		std::move(feature_accumulator),
		std::move(control),
		incremental_updates,
	};
}

ValueUndoState CpuValue::apply(ValueAccumulator &current, const chess::Board &after) const {
	ValueUndoState undo{current.encoded};
	const auto new_encoded = encode_features(after);
	update_accumulator_in_place(current.features, current.encoded, new_encoded,
		weights_.feature_table);
	const int bucket = value_bucket(current.features.piece_count);
	update_control(current.control, control_layout(new_encoded), weights_, bucket);
	current.encoded = new_encoded;
	if (++current.incremental_updates >= kAccumulatorRefreshInterval) {
		current.features = refresh_accumulator(
			new_encoded, weights_.feature_table, weights_.accumulator_bias);
		current.control = refresh_control(control_runtime_features(new_encoded), weights_,
			value_bucket(current.features.piece_count));
		current.incremental_updates = 0;
	}
	return undo;
}

void CpuValue::undo(ValueAccumulator &current, const ValueUndoState &undo) const {
	update_accumulator_in_place(current.features, current.encoded, undo.encoded,
		weights_.feature_table);
	const int bucket = value_bucket(current.features.piece_count);
	update_control(current.control, control_layout(undo.encoded), weights_, bucket);
	current.encoded = undo.encoded;
	if (++current.incremental_updates >= kAccumulatorRefreshInterval) {
		current.features = refresh_accumulator(
			undo.encoded, weights_.feature_table, weights_.accumulator_bias);
		current.control = refresh_control(control_runtime_features(undo.encoded), weights_,
			value_bucket(current.features.piece_count));
		current.incremental_updates = 0;
	}
}

float CpuValue::evaluate(const ValueAccumulator &accumulator) const {
	const int bucket = value_bucket(accumulator.features.piece_count);
	const auto input = oriented_value_input(accumulator);
	const HiddenValues hidden = linear_relu_bucket<kValueHiddenWidth>(
		input, weights_.hidden_weight, weights_.hidden_bias, bucket);
	const BottleneckValues bottleneck = linear_relu_bucket<kValueBottleneckWidth>(
		hidden, weights_.bottleneck_weight, weights_.bottleneck_bias, bucket);
	float output = weights_.output_bias[static_cast<std::size_t>(bucket)];
	const auto output_offset = static_cast<std::size_t>(bucket * kValueBottleneckWidth);
	for (int channel = 0; channel < kValueBottleneckWidth; ++channel) {
		output += weights_.output_weight[output_offset + static_cast<std::size_t>(channel)] *
				  bottleneck[static_cast<std::size_t>(channel)];
	}
	const int first = accumulator.features.white_to_move ? 0 : 1;
	const int second = 1 - first;
	for (int feature = 0; feature < kMaterialFeatureWidth; ++feature)
		output += weights_.material_weight[static_cast<std::size_t>(feature)] *
			accumulator.control.material[static_cast<std::size_t>(first)]
				[static_cast<std::size_t>(feature)];
	const auto psqt = static_cast<std::size_t>(kValueAccumulatorWidth + bucket);
	output += 0.5F *
		(accumulator.features.perspective[static_cast<std::size_t>(first)][psqt] -
		 accumulator.features.perspective[static_cast<std::size_t>(second)][psqt]);
	return output;
}

} // namespace eleginus
