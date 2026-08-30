#include "eleginus/model.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace eleginus {

namespace {

constexpr std::array<char, 8> kMagic{{'E', 'L', 'E', 'G', 'I', 'N', 'U', 'S'}};
struct Header {
	std::array<char, 8> magic{};
	std::uint32_t type_id = 0;
	std::uint32_t fixed_features = 0;
	std::uint32_t terms = 0;
};
static_assert(sizeof(Header) == 20);
static_assert(sizeof(FeatureTerm) == 8);

} // namespace

Model::Model() : Model(Initialization::defaults) {}

Model::Model(Initialization initialization) {
	if (initialization == Initialization::defaults) {
		features_.initialize(weights_);
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
	const auto expected_size = sizeof(Header) + static_cast<std::uintmax_t>(header.terms) * sizeof(FeatureTerm) +
		(static_cast<std::uintmax_t>(FeatureMap::kFixedFeatures) + 2ULL * header.terms) * sizeof(float);
	if (std::filesystem::file_size(path) != expected_size) {
		throw std::runtime_error("Eleginus model size does not match its feature map: " + path.string());
	}
	Model model(Initialization::empty);
	model.terms_.resize(header.terms);
	stream.read(reinterpret_cast<char *>(model.terms_.data()), static_cast<std::streamsize>(model.terms_.size() * sizeof(FeatureTerm)));
	for (std::size_t index = 0; index < model.terms_.size(); ++index) {
		const auto &term = model.terms_[index];
		const auto current = model.terms_.begin() + static_cast<std::ptrdiff_t>(index);
		const bool duplicate = std::find(model.terms_.begin(), current, term) != current;
		if (term.left >= FeatureMap::kBaseFeatures || term.right >= FeatureMap::kBaseFeatures || term.left > term.right || duplicate) {
			throw std::runtime_error("invalid Eleginus feature term: " + path.string());
		}
	}
	model.weights_.resize(FeatureMap::kFixedFeatures + 2 * model.terms_.size());
	const auto weight_bytes = static_cast<std::streamsize>(model.weights_.size() * sizeof(float));
	stream.read(reinterpret_cast<char *>(model.weights_.data()), weight_bytes);
	if (!stream) {
		throw std::runtime_error("truncated Eleginus model: " + path.string());
	}
	if (!std::all_of(model.weights_.begin(), model.weights_.end(), [](float value) { return std::isfinite(value); })) {
		throw std::runtime_error("nonfinite Eleginus model weights: " + path.string());
	}
	return model;
}

void Model::save(const std::filesystem::path &path) const {
	if (terms_.size() > std::numeric_limits<std::uint32_t>::max()) {
		throw std::runtime_error("Eleginus model has too many feature terms");
	}
	if (!path.parent_path().empty()) {
		std::filesystem::create_directories(path.parent_path());
	}
	std::ofstream stream(path, std::ios::binary | std::ios::trunc);
	if (!stream) {
		throw std::runtime_error("cannot create Eleginus model: " + path.string());
	}
	const Header header{kMagic, kArchitectureType, FeatureMap::kFixedFeatures, static_cast<std::uint32_t>(terms_.size())};
	stream.write(reinterpret_cast<const char *>(&header), sizeof(header));
	stream.write(reinterpret_cast<const char *>(terms_.data()), static_cast<std::streamsize>(terms_.size() * sizeof(FeatureTerm)));
	stream.write(reinterpret_cast<const char *>(weights_.data()), static_cast<std::streamsize>(weights_.size() * sizeof(float)));
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

int Model::centipawns(const chess::Board &board) const {
	return static_cast<int>(std::lround(std::clamp(score(board), -25000.0F, 25000.0F)));
}

void Model::extract(const chess::Board &board, std::vector<Feature> &features, std::vector<Feature> *candidate_atoms) const {
	features_.extract(board, terms_, features, candidate_atoms);
}

std::size_t Model::add_terms(const std::vector<FeatureTerm> &terms) {
	const std::size_t previous = terms_.size();
	for (auto term : terms) {
		if (term.left > term.right) {
			std::swap(term.left, term.right);
		}
		if (term.right >= FeatureMap::kBaseFeatures) {
			throw std::invalid_argument("Eleginus feature term is outside the base feature map");
		}
		if (std::find(terms_.begin(), terms_.end(), term) == terms_.end()) {
			terms_.push_back(term);
			weights_.push_back(0.0F);
			weights_.push_back(0.0F);
		}
	}
	return terms_.size() - previous;
}

} // namespace eleginus
