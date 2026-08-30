#pragma once

#include "eleginus/features.hpp"
#include <cstdint>
#include <filesystem>
#include <vector>

namespace eleginus {

inline constexpr std::uint32_t kArchitectureType = 3;

class Model {
public:
	Model();

	static Model load(const std::filesystem::path &path);
	void save(const std::filesystem::path &path) const;

	float score(const chess::Board &board) const;
	float score(const std::vector<Feature> &features) const noexcept;
	int centipawns(const chess::Board &board) const;
	void extract(const chess::Board &board, std::vector<Feature> &features, std::vector<Feature> *candidate_atoms = nullptr) const;
	std::size_t add_terms(const std::vector<FeatureTerm> &terms);

	const std::vector<float> &weights() const noexcept { return weights_; }
	std::vector<float> &weights() noexcept { return weights_; }
	const std::vector<FeatureTerm> &terms() const noexcept { return terms_; }

private:
	enum class Initialization { defaults, empty };
	explicit Model(Initialization initialization);

	FeatureMap features_;
	std::vector<FeatureTerm> terms_;
	std::vector<float> weights_;
};

} // namespace eleginus
