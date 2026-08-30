#include "eleginus/model.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace eleginus {

namespace {

constexpr std::array<char, 8> kMagic{{'E', 'L', 'E', 'G', 'I', 'N', 'U', 'S'}};
struct Header {
	std::array<char, 8> magic{};
	std::uint32_t type_id = 0;
	std::uint32_t fixed_features = 0;
};
static_assert(sizeof(Header) == 16);

} // namespace

Model::Model() : Model(Initialization::defaults) {}

Model::Model(Initialization initialization) {
	if (initialization == Initialization::defaults) {
		features_.initialize(weights_);
		uncertainty_weights_.assign(weights_.size(), 0.0F);
		constexpr float initial_uncertainty_logit = -2.1972246F;
		std::fill_n(uncertainty_weights_.begin(), FeatureMap::kRegimes, initial_uncertainty_logit);
	}
}

Model Model::load(const std::filesystem::path &path) {
	std::ifstream stream(path, std::ios::binary);
	if (!stream) {
		throw std::runtime_error("cannot open Eleginus model: " + path.string());
	}
	Header header;
	stream.read(reinterpret_cast<char *>(&header), sizeof(header));
	if (!stream || header.magic != kMagic || header.type_id != kArchitectureType || header.fixed_features != FeatureMap::kFixedFeatures) {
		throw std::runtime_error("unsupported Eleginus model: " + path.string());
	}
	const auto expected_size = sizeof(Header) + 2ULL * FeatureMap::kFixedFeatures * sizeof(float);
	if (std::filesystem::file_size(path) != expected_size) {
		throw std::runtime_error("Eleginus model size does not match its feature map: " + path.string());
	}
	Model model(Initialization::empty);
	model.weights_.resize(FeatureMap::kFixedFeatures);
	const auto weight_bytes = static_cast<std::streamsize>(model.weights_.size() * sizeof(float));
	stream.read(reinterpret_cast<char *>(model.weights_.data()), weight_bytes);
	model.uncertainty_weights_.resize(model.weights_.size());
	stream.read(reinterpret_cast<char *>(model.uncertainty_weights_.data()), weight_bytes);
	if (!stream) {
		throw std::runtime_error("truncated Eleginus model: " + path.string());
	}
	if (!std::all_of(model.weights_.begin(), model.weights_.end(), [](float value) { return std::isfinite(value); })) {
		throw std::runtime_error("nonfinite Eleginus model weights: " + path.string());
	}
	if (!std::all_of(model.uncertainty_weights_.begin(), model.uncertainty_weights_.end(), [](float value) { return std::isfinite(value); })) {
		throw std::runtime_error("nonfinite Eleginus uncertainty weights: " + path.string());
	}
	return model;
}

void Model::save(const std::filesystem::path &path) const {
	if (!path.parent_path().empty()) {
		std::filesystem::create_directories(path.parent_path());
	}
	std::ofstream stream(path, std::ios::binary | std::ios::trunc);
	if (!stream) {
		throw std::runtime_error("cannot create Eleginus model: " + path.string());
	}
	const Header header{kMagic, kArchitectureType, FeatureMap::kFixedFeatures};
	stream.write(reinterpret_cast<const char *>(&header), sizeof(header));
	stream.write(reinterpret_cast<const char *>(weights_.data()), static_cast<std::streamsize>(weights_.size() * sizeof(float)));
	stream.write(reinterpret_cast<const char *>(uncertainty_weights_.data()),
		static_cast<std::streamsize>(uncertainty_weights_.size() * sizeof(float)));
	if (!stream) {
		throw std::runtime_error("cannot write Eleginus model: " + path.string());
	}
}

float Model::score(const chess::Board &board) const {
	thread_local std::vector<Feature> active;
	extract(board, active);
	return score(active);
}

float Model::score(const std::vector<Feature> &features) const noexcept {
	float score = 0.0F;
	for (const auto &feature : features) {
		score += weights_[feature.index] * feature.value;
	}
	return score;
}

float Model::uncertainty(const chess::Board &board) const {
	thread_local std::vector<Feature> active;
	extract(board, active);
	return uncertainty(active);
}

float Model::uncertainty(const std::vector<Feature> &features) const noexcept {
	float logit = 0.0F;
	for (const auto &feature : features) {
		logit += uncertainty_weights_[feature.index] * feature.value;
	}
	return 1.0F / (1.0F + std::exp(-std::clamp(logit, -20.0F, 20.0F)));
}

Evaluation Model::evaluate(const chess::Board &board) const {
	thread_local std::vector<Feature> active;
	extract(board, active);
	return {score(active), uncertainty(active)};
}

int Model::centipawns(const chess::Board &board) const {
	return static_cast<int>(std::lround(std::clamp(score(board), -25000.0F, 25000.0F)));
}

void Model::extract(const chess::Board &board, std::vector<Feature> &features) const {
	features_.extract(board, features);
}

} // namespace eleginus
