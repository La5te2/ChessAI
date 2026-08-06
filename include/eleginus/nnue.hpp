#pragma once

// Eleginus sparse chess features and incrementally maintained Value accumulator state.

#include <array>
#include <cstdint>
#include <vector>

#include "eleginus/game.hpp"

namespace eleginus {

inline constexpr int kPerspectiveCount = 2;
inline constexpr int kPieceFeatureCount = 64 * 12 * 64;
inline constexpr int kCastlingFeatureBase = kPieceFeatureCount;
inline constexpr int kEnPassantFeatureBase = kCastlingFeatureBase + 16;
inline constexpr int kFeatureCount = kEnPassantFeatureBase + 9;
inline constexpr int kFeatureSlots = 34; // 32 pieces plus castling and en-passant context.
inline constexpr int kPaddingFeature = kFeatureCount;
inline constexpr int kFeatureVocabulary = kFeatureCount + 1;
inline constexpr int kValueAccumulatorWidth = 256;
inline constexpr int kValueHiddenWidth = 64;
inline constexpr int kValueBottleneckWidth = 32;

using PerspectiveFeatures = std::array<std::int32_t, kFeatureSlots>;

struct EncodedFeatures {
	std::array<PerspectiveFeatures, kPerspectiveCount> perspective{};
	bool white_to_move = true;
};

/// Builds king-conditioned sparse features from both color perspectives.
EncodedFeatures encode_features(const chess::Board &board);

/// Floating-point accumulator used by the first custom CPU inference path.
struct FloatAccumulator {
	std::array<std::vector<float>, kPerspectiveCount> perspective;
	bool white_to_move = true;
};

struct ValueWeights {
	std::vector<float> feature_table;
	std::vector<float> hidden_weight;
	std::vector<float> hidden_bias;
	std::vector<float> bottleneck_weight;
	std::vector<float> bottleneck_bias;
	std::vector<float> output_weight;
	float output_bias = 0.0F;
};

/// Immutable, Torch-free evaluator for the shared Value network.
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
