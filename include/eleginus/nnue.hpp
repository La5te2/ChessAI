#pragma once

// Eleginus sparse chess features and independent incremental Policy/Value evaluators.

#include <array>
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
inline constexpr int kPolicyAccumulatorWidth = 128;
inline constexpr int kPolicyHiddenWidth = 128;
inline constexpr int kValueAccumulatorWidth = 512;
inline constexpr int kValueHiddenWidth = 32;
inline constexpr int kValueBottleneckWidth = 32;
inline constexpr int kValueBucketCount = 8;
inline constexpr int kValueFeatureWidth = kValueAccumulatorWidth + kValueBucketCount;

using PerspectiveFeatures = std::array<std::int32_t, kFeatureSlots>;

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
	std::array<std::vector<float>, kPerspectiveCount> perspective;
	bool white_to_move = true;
	int piece_count = 0;
};

struct PolicyWeights {
	std::vector<float> feature_table;
	std::vector<float> accumulator_bias;
	std::vector<float> hidden_weight;
	std::vector<float> hidden_bias;
	std::vector<float> output_weight;
	std::vector<float> output_bias;
};

struct ValueWeights {
	std::vector<float> feature_table;
	std::vector<float> accumulator_bias;
	std::vector<float> hidden_weight;
	std::vector<float> hidden_bias;
	std::vector<float> bottleneck_weight;
	std::vector<float> bottleneck_bias;
	std::vector<float> output_weight;
	std::vector<float> output_bias;
};

/// Selects one of eight material-dependent Value subnetworks.
int value_bucket(int piece_count) noexcept;

/// Immutable, Torch-free evaluator for the independent Policy network.
class CpuPolicy {
	public:
	explicit CpuPolicy(PolicyWeights weights);
	FloatAccumulator refresh(const chess::Board &board) const;
	FloatAccumulator update(const FloatAccumulator &current, const chess::Board &before,
							const chess::Board &after) const;
	std::vector<float> evaluate(const FloatAccumulator &accumulator,
							const std::vector<chess::Move> &moves) const;
	const PolicyWeights &weights() const noexcept { return weights_; }

	private:
	PolicyWeights weights_;
};

/// Immutable, Torch-free evaluator for the independent Value network.
class CpuValue {
	public:
	explicit CpuValue(ValueWeights weights);
	FloatAccumulator refresh(const chess::Board &board) const;
	FloatAccumulator update(const FloatAccumulator &current, const chess::Board &before,
							const chess::Board &after) const;
	float evaluate(const FloatAccumulator &accumulator) const;
	const ValueWeights &weights() const noexcept { return weights_; }

	private:
	ValueWeights weights_;
};

} // namespace eleginus
