#pragma once

#include "eleginus/features.hpp"
#include <cstdint>
#include <filesystem>
#include <vector>

namespace eleginus {

inline constexpr std::uint32_t kArchitectureType = 3;

struct Evaluation {
	float centipawns = 0.0F;
	float uncertainty = 0.0F;
};

class Model {
public:
	Model();

	static Model load(const std::filesystem::path &path);
	void save(const std::filesystem::path &path) const;

	float score(const chess::Board &board) const;
	float score(const std::vector<Feature> &features) const noexcept;
	float uncertainty(const chess::Board &board) const;
	float uncertainty(const std::vector<Feature> &features) const noexcept;
	Evaluation evaluate(const chess::Board &board) const;
	int centipawns(const chess::Board &board) const;
	void extract(const chess::Board &board, std::vector<Feature> &features, std::vector<Feature> *candidate_atoms = nullptr) const;
	std::size_t add_terms(const std::vector<FeatureTerm> &terms);

	const std::vector<float> &weights() const noexcept { return weights_; }
	std::vector<float> &weights() noexcept { return weights_; }
	const std::vector<float> &uncertainty_weights() const noexcept { return uncertainty_weights_; }
	std::vector<float> &uncertainty_weights() noexcept { return uncertainty_weights_; }
	const std::vector<FeatureTerm> &terms() const noexcept { return terms_; }

private:
	enum class Initialization { defaults, empty };
	explicit Model(Initialization initialization);

	FeatureMap features_;
	std::vector<FeatureTerm> terms_;
	std::vector<float> weights_;
	std::vector<float> uncertainty_weights_;
};

} // namespace eleginus
