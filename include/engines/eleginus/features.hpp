#pragma once

#include "chess.hpp"
#include <cstdint>
#include <span>
#include <vector>

namespace eleginus {

struct Feature {
	std::uint32_t index = 0;
	float value = 0.0F;
};

struct FeatureTerm {
	std::uint32_t left = 0;
	std::uint32_t right = 0;

	friend bool operator==(const FeatureTerm &, const FeatureTerm &) = default;
};

class FeatureMap {
public:
	static constexpr int kCoreFeatures = 16708;
	static constexpr int kTopologyOffset = kCoreFeatures;
	static constexpr int kTopologyFeatures = 18;
	static constexpr int kTransitionOffset = kTopologyOffset + kTopologyFeatures;
	static constexpr int kTransitionFeatures = 45;
	static constexpr int kBaseFeatures = kTransitionOffset + kTransitionFeatures;
	static constexpr int kFixedFeatures = 2 * kBaseFeatures;
	static constexpr int kTransitionPieceLimit = 10;

	void extract(const chess::Board &board, std::vector<Feature> &output) const;
	void extract(const chess::Board &board, std::span<const FeatureTerm> terms, std::vector<Feature> &output,
		std::vector<Feature> *candidate_atoms = nullptr) const;
	void initialize(std::vector<float> &weights) const;
	static std::size_t candidate_terms() noexcept;
};

} // namespace eleginus
