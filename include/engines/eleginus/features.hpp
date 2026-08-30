#pragma once

#include "chess.hpp"
#include <cstdint>
#include <vector>

namespace eleginus {

struct Feature {
	std::uint32_t index = 0;
	float value = 0.0F;
};

class FeatureMap {
public:
	static constexpr int kPrimitiveFeatures = 16708;
	static constexpr int kControlOffset = kPrimitiveFeatures;
	static constexpr int kControlFeatures = 15;
	static constexpr int kCoreFeatures = kControlOffset + kControlFeatures;
	static constexpr int kTopologyOffset = kCoreFeatures;
	static constexpr int kTopologyFeatures = 18;
	static constexpr int kTransitionOffset = kTopologyOffset + kTopologyFeatures;
	static constexpr int kTransitionFeatures = 45;
	static constexpr int kBaseFeatures = kTransitionOffset + kTransitionFeatures;
	static constexpr int kRegimes = 4;
	static constexpr int kFixedFeatures = kRegimes * kBaseFeatures;
	static constexpr int kTransitionPieceLimit = 10;

	void extract(const chess::Board &board, std::vector<Feature> &output) const;
	void initialize(std::vector<float> &weights) const;
};

} // namespace eleginus
