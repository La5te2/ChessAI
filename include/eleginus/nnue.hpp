#pragma once

// Eleginus sparse chess features and incremental Value evaluator.

#include <array>
#include <compare>
#include <cstdint>
#include <vector>

#include "eleginus/game.hpp"

namespace eleginus {

inline constexpr int kPerspectiveCount = 2;
inline constexpr int kEncodedKingSquareCount = 64;
inline constexpr int kEncodedPieceFeatureCount = kEncodedKingSquareCount * 12 * 64;
inline constexpr int kEncodedCastlingFeatureBase = kEncodedPieceFeatureCount;
inline constexpr int kEncodedEnPassantFeatureBase = kEncodedCastlingFeatureBase + 16;
inline constexpr int kEncodedFeatureCount = kEncodedEnPassantFeatureBase + 9;
inline constexpr int kEncodedPaddingFeature = kEncodedFeatureCount;
inline constexpr int kEncodedFeatureVocabulary = kEncodedFeatureCount + 1;
inline constexpr int kKingBucketCount = 32;
inline constexpr int kPieceFeatureCount = kKingBucketCount * 12 * 64;
inline constexpr int kCastlingFeatureBase = kPieceFeatureCount;
inline constexpr int kEnPassantFeatureBase = kCastlingFeatureBase + 16;
inline constexpr int kFeatureCount = kEnPassantFeatureBase + 9;
inline constexpr int kFeatureSlots = 34; // 32 pieces plus castling and en-passant context.
inline constexpr int kPaddingFeature = kFeatureCount;
inline constexpr int kFeatureVocabulary = kFeatureCount + 1;
inline constexpr int kControlPieceTypes = 12;
inline constexpr int kControlSourceVocabulary = kControlPieceTypes * 64;
inline constexpr int kControlGeometryVocabulary = kControlPieceTypes * 15 * 15;
inline constexpr int kControlWidth = 16;
inline constexpr int kControlCountVocabulary = 8;
inline constexpr int kControlCountWidth = 4;
inline constexpr int kControlOccupancyVocabulary = 13;
inline constexpr int kControlOccupancyWidth = 4;
inline constexpr int kControlSquareWidth = 4;
inline constexpr int kControlLocalWidth = 16;
inline constexpr int kControlAttentionHeads = 4;
inline constexpr int kControlAttentionKeyWidth = 8;
inline constexpr int kControlAttentionWidth = 16;
inline constexpr int kControlAttentionHeadWidth =
	kControlAttentionWidth / kControlAttentionHeads;
static_assert(kControlAttentionWidth % kControlAttentionHeads == 0);
inline constexpr int kMaterialFeatureWidth = 6;
inline constexpr int kValueAccumulatorWidth = 144;
inline constexpr int kValueDenseWidth =
	kValueAccumulatorWidth + kControlLocalWidth + kControlAttentionWidth;
inline constexpr int kValueHiddenWidth = 32;
inline constexpr int kValueBottleneckWidth = 32;
inline constexpr int kValueBucketCount = 8;
inline constexpr int kValueFeatureWidth = kValueAccumulatorWidth + kValueBucketCount;

using PerspectiveFeatures = std::array<std::int32_t, kFeatureSlots>;
using FeatureAccumulator = std::array<float, kValueFeatureWidth>;

struct EncodedFeatures {
	std::array<PerspectiveFeatures, kPerspectiveCount> perspective{};
	bool white_to_move = true;
};

/// Builds king-conditioned sparse features from both color perspectives.
EncodedFeatures encode_features(const chess::Board &board);
/// Maps one stable 64-king-square feature sequence into the mirrored 32-bucket network vocabulary.
PerspectiveFeatures canonicalize_features(const PerspectiveFeatures &features);

/// Floating-point accumulator used by the first custom CPU inference path.
struct FloatAccumulator {
	std::array<FeatureAccumulator, kPerspectiveCount> perspective{};
	bool white_to_move = true;
	int piece_count = 0;
};

struct ControlEdge {
	std::uint16_t source = 0;
	std::uint16_t target = 0;
	std::uint16_t geometry = 0;
	std::uint8_t perspective = 0;
	std::uint8_t square = 0;
	std::uint8_t ownership = 0;

	auto operator<=>(const ControlEdge &) const = default;
};

struct ControlFeatures {
	std::vector<ControlEdge> edges;
	std::array<std::array<std::uint8_t, 64>, kPerspectiveCount> occupancy{};
	std::array<std::array<float, kMaterialFeatureWidth>, kPerspectiveCount> material{};
};

/// Derives canonical pseudo-attack edges, occupancy and material from sparse features.
ControlFeatures control_features(const EncodedFeatures &encoded);
/// Returns the six signed material features for both canonical perspectives.
std::array<std::array<float, kMaterialFeatureWidth>, kPerspectiveCount>
material_features(const EncodedFeatures &encoded);

struct ControlAccumulator {
	std::array<std::array<std::array<float, kControlWidth>, 64>,
		kPerspectiveCount * 2> field{};
	std::array<std::array<std::array<std::uint8_t, 64>, 2>, kPerspectiveCount> count{};
	std::array<std::array<std::uint8_t, 64>, kPerspectiveCount> occupancy{};
	std::array<std::array<std::array<float, kControlLocalWidth>, 64>,
		kPerspectiveCount> local{};
	std::array<std::array<float, kControlLocalWidth>, kPerspectiveCount> mean{};
	std::array<std::array<float, kControlAttentionWidth>, kPerspectiveCount> attention{};
	std::array<std::array<std::array<float, kControlAttentionHeadWidth>,
		kControlAttentionHeads>, kPerspectiveCount> attention_numerator{};
	std::array<std::array<float, kControlAttentionHeads>, kPerspectiveCount>
		attention_denominator{};
	std::array<std::array<float, kMaterialFeatureWidth>, kPerspectiveCount> material{};
	std::array<std::array<std::uint8_t, 64>, kPerspectiveCount> piece_type{};
	std::array<std::array<std::uint64_t, 64>, kPerspectiveCount> attacks{};
	int bucket = 0;
};

/// Complete incremental state consumed by the Value network.
struct ValueAccumulator {
	EncodedFeatures encoded;
	FloatAccumulator features;
	ControlAccumulator control;
	int incremental_updates = 0;
};

struct ValueUndoState {
	EncodedFeatures encoded;
};

struct ValueWeights {
	std::vector<float> feature_table;
	std::vector<float> accumulator_bias;
	std::vector<float> control_source;
	std::vector<float> control_target;
	std::vector<float> control_geometry;
	std::vector<float> control_occupancy;
	std::vector<float> control_count;
	std::vector<float> control_square;
	std::vector<float> control_local_weight;
	std::vector<float> control_local_bias;
	std::vector<float> attention_key_weight;
	std::vector<float> attention_key_bias;
	std::vector<float> attention_value_weight;
	std::vector<float> attention_value_bias;
	std::vector<float> attention_query;
	std::vector<float> material_weight;
	std::vector<float> hidden_weight;
	std::vector<float> hidden_bias;
	std::vector<float> bottleneck_weight;
	std::vector<float> bottleneck_bias;
	std::vector<float> output_weight;
	std::vector<float> output_bias;
};

/// Selects one of eight material-dependent Value subnetworks.
int value_bucket(int piece_count) noexcept;

/// Immutable, Torch-free evaluator for the Value network.
class CpuValue {
	public:
	explicit CpuValue(ValueWeights weights);
	ValueAccumulator refresh(const chess::Board &board) const;
	ValueAccumulator update(const ValueAccumulator &current, const chess::Board &before,
							const chess::Board &after) const;
	ValueUndoState apply(ValueAccumulator &current, const chess::Board &after) const;
	void undo(ValueAccumulator &current, const ValueUndoState &undo) const;
	float evaluate(const ValueAccumulator &accumulator) const;
	const ValueWeights &weights() const noexcept { return weights_; }

	private:
	ValueWeights weights_;
};

} // namespace eleginus
