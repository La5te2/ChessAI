#include "eleginus/features.hpp"
#include "eleginus/model.hpp"
#include "eleginus/search.hpp"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string &message) {
	if (!condition) {
		throw std::runtime_error(message);
	}
}

float base_value(const std::vector<eleginus::Feature> &features, int base) {
	float value = 0.0F;
	for (const auto &feature : features) {
		if (feature.index == static_cast<std::uint32_t>(2 * base) || feature.index == static_cast<std::uint32_t>(2 * base + 1)) {
			value += feature.value;
		}
	}
	return value;
}

} // namespace

int main() {
	try {
		eleginus::FeatureMap map;
		require(eleginus::FeatureMap::candidate_terms() == 162735, "candidate feature universe has the wrong size");
		std::vector<eleginus::Feature> features;
		map.extract(chess::Board(), features);
		require(!features.empty(), "start position has no active features");
		std::vector<bool> seen(eleginus::FeatureMap::kFixedFeatures);
		for (const auto &feature : features) {
			require(feature.index < eleginus::FeatureMap::kFixedFeatures, "feature index is outside the model");
			require(!seen[feature.index], "active features are not unique");
			seen[feature.index] = true;
		}
		require(base_value(features, eleginus::FeatureMap::kTransitionOffset) == 0.0F,
			"dense positions unexpectedly compute successor features");

		const chess::Board closed("4k3/8/8/4p3/4P3/8/8/4K3 w - - 0 1");
		const chess::Board open("4k3/8/4p3/8/4P3/8/8/4K3 w - - 0 1");
		std::vector<eleginus::Feature> closed_features;
		std::vector<eleginus::Feature> open_features;
		map.extract(closed, closed_features);
		map.extract(open, open_features);
		require(base_value(closed_features, eleginus::FeatureMap::kTopologyOffset) >
			base_value(open_features, eleginus::FeatureMap::kTopologyOffset), "pawn locks are not represented");

		const chess::Board white_to_move("8/8/8/8/8/2k5/2p5/K7 w - - 0 1");
		const chess::Board black_to_move("8/8/8/8/8/2k5/2p5/K7 b - - 0 1");
		std::vector<eleginus::Feature> white_transition;
		std::vector<eleginus::Feature> black_transition;
		map.extract(white_to_move, white_transition);
		map.extract(black_to_move, black_transition);
		const float white_moves = base_value(white_transition, eleginus::FeatureMap::kTransitionOffset);
		const float black_moves = base_value(black_transition, eleginus::FeatureMap::kTransitionOffset);
		require(white_moves > 0.0F && black_moves > 0.0F, "sparse positions have no successor features");
		require(white_moves != black_moves, "successor features ignore the side to move");

		const eleginus::Model model;
		const chess::Board white_queen("4k3/8/8/8/8/8/4Q3/4K3 w - - 0 1");
		const chess::Board black_view("4k3/8/8/8/8/8/4Q3/4K3 b - - 0 1");
		require(model.centipawns(white_queen) >= 890, "material initialization does not value a queen");
		require(model.centipawns(black_view) <= -890, "material initialization is not side-to-move relative");
		require(std::abs(model.score(white_queen) - model.centipawns(white_queen)) < 0.51F, "model score is not expressed in centipawns");

		auto extended = model;
		std::vector<eleginus::Feature> extended_features;
		std::vector<eleginus::Feature> atoms;
		extended.extract(white_queen, extended_features, &atoms);
		require(atoms.size() >= 2, "test position has too few candidate atoms");
		const float original_score = extended.score(extended_features);
		require(extended.add_terms({{atoms[0].index, atoms[1].index}}) == 1, "feature term was not added");
		extended.extract(white_queen, extended_features);
		require(extended.score(extended_features) == original_score, "zero-initialized feature term changed the score");

		const auto path = std::filesystem::temp_directory_path() / "eleginus-test.pth";
		extended.save(path);
		const auto loaded = eleginus::Model::load(path);
		std::filesystem::remove(path);
		require(loaded.weights() == extended.weights(), "model save/load changed weights");
		require(loaded.terms() == extended.terms(), "model save/load changed feature terms");

		eleginus::SearchOptions options;
		options.depth = 2;
		options.hash_mb = 1;
		const chess::Board mate("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1");
		const auto result = eleginus::Searcher(model, options).search(mate);
		require(result.move.move() != chess::Move::NO_MOVE, "search returned no move");
		require(result.score_cp >= 29000, "search did not find the forced mate");

		options.depth = 1;
		const auto sparse_result = eleginus::Searcher(model, options).search(white_to_move);
		require(sparse_result.evaluated_nodes > sparse_result.root.size(), "sparse search did not perform a mandatory-move probe");

		std::cout << "Eleginus tests passed\n";
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "Eleginus tests failed: " << error.what() << '\n';
		return 1;
	}
}
