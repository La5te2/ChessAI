#include "eleginus/features.hpp"
#include "eleginus/game.hpp"
#include "eleginus/model.hpp"
#include "eleginus/search.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <future>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

constexpr float kDecisiveMarginCp = 100.0F;
constexpr int kTermParameters = 2;

struct Options {
	std::filesystem::path output;
	std::filesystem::path resume;
	int games = 100;
	int depth = 2;
	int max_plies = 200;
	int hash_mb = 16;
	int workers = static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
	float learning_rate = 1.0F;
	float weight_decay = 1.0e-6F;
	float exploration = 0.08F;
	float temperature_cp = 80.0F;
	int log_every = 10;
	int discover_every = 64;
	std::uint64_t seed = 2026;
};

struct Sample {
	std::vector<eleginus::Feature> features;
	std::vector<eleginus::Feature> atoms;
	std::array<float, 2> phase{};
	chess::Color side = chess::Color::WHITE;
};

struct GameRecord {
	std::vector<Sample> samples;
	std::optional<chess::Color> winner;
};

struct CandidateStat {
	std::array<double, 2> gradient{};
	std::array<std::array<double, 2>, 2> curvature{};
	double loss = 0.0;
	std::uint32_t games = 0;
};

using CandidateStats = std::unordered_map<std::uint64_t, CandidateStat>;

struct DiscoveryState {
	CandidateStats candidates;
	std::vector<float> weights;
};

std::string value_after(int argc, char **argv, int &index) {
	if (index + 1 >= argc) {
		throw std::invalid_argument(std::string("missing value after ") + argv[index]);
	}
	return argv[++index];
}

Options parse_options(int argc, char **argv) {
	Options options;
	for (int index = 1; index < argc; ++index) {
		const std::string argument = argv[index];
		if (argument == "--out") {
			options.output = value_after(argc, argv, index);
		} else if (argument == "--resume") {
			options.resume = value_after(argc, argv, index);
		} else if (argument == "--games") {
			options.games = std::stoi(value_after(argc, argv, index));
		} else if (argument == "--depth") {
			options.depth = std::stoi(value_after(argc, argv, index));
		} else if (argument == "--max-plies") {
			options.max_plies = std::stoi(value_after(argc, argv, index));
		} else if (argument == "--hash") {
			options.hash_mb = std::stoi(value_after(argc, argv, index));
		} else if (argument == "--workers") {
			options.workers = std::stoi(value_after(argc, argv, index));
		} else if (argument == "--lr") {
			options.learning_rate = std::stof(value_after(argc, argv, index));
		} else if (argument == "--weight-decay") {
			options.weight_decay = std::stof(value_after(argc, argv, index));
		} else if (argument == "--exploration") {
			options.exploration = std::stof(value_after(argc, argv, index));
		} else if (argument == "--temperature") {
			options.temperature_cp = std::stof(value_after(argc, argv, index));
		} else if (argument == "--log-every") {
			options.log_every = std::stoi(value_after(argc, argv, index));
		} else if (argument == "--discover-every") {
			options.discover_every = std::stoi(value_after(argc, argv, index));
		} else if (argument == "--seed") {
			options.seed = std::stoull(value_after(argc, argv, index));
		} else if (argument == "--help") {
			std::cout << "usage: train --out eleginus.pth [--resume model.pth] [--games 100] [--depth 2] "
					  << "[--max-plies 200] [--hash 16] [--workers N] [--lr 1] [--exploration 0.08] "
					  << "[--temperature 80] [--discover-every 64] [--seed 2026]\n";
			std::exit(0);
		} else {
			throw std::invalid_argument("unknown option: " + argument);
		}
	}
	if (options.output.empty() || options.games < 0 || options.depth <= 0 || options.depth > 16 || options.max_plies <= 0 ||
		options.hash_mb <= 0 || options.workers <= 0 || options.workers > 256 || options.learning_rate <= 0.0F || options.weight_decay < 0.0F ||
		options.exploration < 0.0F || options.exploration > 1.0F || options.temperature_cp <= 0.0F || options.log_every <= 0 ||
		options.discover_every <= 0) {
		throw std::invalid_argument("invalid or incomplete Eleginus training options");
	}
	return options;
}

chess::Move choose_move(const eleginus::SearchResult &result, const Options &options, std::mt19937_64 &random) {
	if (result.root.empty()) {
		return chess::Move(chess::Move::NO_MOVE);
	}
	std::uniform_real_distribution<float> unit(0.0F, 1.0F);
	if (unit(random) < options.exploration) {
		std::uniform_int_distribution<std::size_t> move(0, result.root.size() - 1);
		return result.root[move(random)].move;
	}
	const int best = std::max_element(result.root.begin(), result.root.end(), [](const auto &left, const auto &right) {
		return left.score_cp < right.score_cp;
	})->score_cp;
	std::vector<double> mass;
	mass.reserve(result.root.size());
	double total = 0.0;
	for (const auto &candidate : result.root) {
		const double value = std::exp(std::clamp((candidate.score_cp - best) / static_cast<double>(options.temperature_cp), -40.0, 0.0));
		mass.push_back(value);
		total += value;
	}
	std::uniform_real_distribution<double> sample(0.0, total);
	double selected = sample(random);
	for (std::size_t index = 0; index < mass.size(); ++index) {
		selected -= mass[index];
		if (selected <= 0.0) {
			return result.root[index].move;
		}
	}
	return result.root.back().move;
}

std::optional<chess::Color> terminal_winner(const chess::Board &board) {
	const auto [reason, result] = board.isGameOver();
	if (reason == chess::GameResultReason::NONE || result == chess::GameResult::DRAW) {
		return std::nullopt;
	}
	return result == chess::GameResult::WIN ? board.sideToMove() : ~board.sideToMove();
}

std::uint64_t game_seed(std::uint64_t seed, std::uint64_t game) noexcept {
	std::uint64_t value = seed + 0x9e3779b97f4a7c15ULL * (game + 1U);
	value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
	value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
	return value ^ (value >> 31U);
}

GameRecord play_game(const eleginus::Model &model, const Options &options, std::uint64_t seed) {
	GameRecord record;
	record.samples.reserve(static_cast<std::size_t>(options.max_plies));
	chess::Board board;
	eleginus::SearchOptions search_options;
	search_options.depth = options.depth;
	search_options.hash_mb = static_cast<std::size_t>(options.hash_mb);
	search_options.multipv = 256;
	eleginus::Searcher searcher(model, search_options);
	std::mt19937_64 random(seed);
	for (int ply = 0; ply < options.max_plies && !eleginus::game_is_over(board); ++ply) {
		Sample sample;
		sample.side = board.sideToMove();
		model.extract(board, sample.features, &sample.atoms);
		for (const auto &feature : sample.features) {
			if (feature.index < sample.phase.size()) {
				sample.phase[feature.index] = feature.value;
			}
		}
		record.samples.push_back(std::move(sample));
		const auto result = searcher.search(board);
		const auto move = choose_move(result, options, random);
		if (move.move() == chess::Move::NO_MOVE) {
			break;
		}
		board.makeMove(move);
	}
	record.winner = terminal_winner(board);
	return record;
}

std::uint64_t candidate_key(std::uint32_t left, std::uint32_t right) noexcept {
	if (left > right) {
		std::swap(left, right);
	}
	return (static_cast<std::uint64_t>(left) << 32U) | right;
}

eleginus::FeatureTerm candidate_term(std::uint64_t key) noexcept {
	return {static_cast<std::uint32_t>(key >> 32U), static_cast<std::uint32_t>(key)};
}

struct Objective {
	double loss = 0.0;
	float gradient = 0.0F;
	float curvature = 0.0F;
};

float score_with(const std::vector<eleginus::Feature> &features, const std::vector<float> &weights) noexcept {
	float score = 0.0F;
	for (const auto &feature : features) {
		score += weights[feature.index] * feature.value;
	}
	return score;
}

Objective objective(float score, chess::Color side, std::optional<chess::Color> winner) noexcept {
	if (!winner) {
		return {0.5 * static_cast<double>(score) * score, score, 1.0F};
	}
	const float label = side == *winner ? 1.0F : -1.0F;
	const float shortfall = std::max(0.0F, kDecisiveMarginCp - label * score);
	return {0.5 * static_cast<double>(shortfall) * shortfall, -label * shortfall, shortfall > 0.0F ? 1.0F : 0.0F};
}

void accumulate_candidates(const Sample &sample, const Objective &objective, CandidateStats &statistics) {
	if (objective.curvature == 0.0F) {
		return;
	}
	for (std::size_t left = 0; left < sample.atoms.size(); ++left) {
		for (std::size_t right = left; right < sample.atoms.size(); ++right) {
			const float value = sample.atoms[left].value * sample.atoms[right].value;
			if (value == 0.0F) {
				continue;
			}
			const auto key = candidate_key(sample.atoms[left].index, sample.atoms[right].index);
			auto &statistic = statistics[key];
			statistic.loss += objective.loss;
			std::array<double, 2> phased{};
			for (std::size_t phase = 0; phase < phased.size(); ++phase) {
				phased[phase] = static_cast<double>(value) * sample.phase[phase];
				statistic.gradient[phase] += objective.gradient * phased[phase];
			}
			for (std::size_t row = 0; row < phased.size(); ++row) {
				for (std::size_t column = 0; column < phased.size(); ++column) {
					statistic.curvature[row][column] += objective.curvature * phased[row] * phased[column];
				}
			}
		}
	}
}

struct Selection {
	std::optional<eleginus::FeatureTerm> term;
	double evidence = -std::numeric_limits<double>::infinity();
	double gain = 0.0;
	std::uint32_t support = 0;
};

std::pair<double, int> candidate_gain(const CandidateStat &statistic) noexcept {
	const double a = statistic.curvature[0][0];
	const double b = statistic.curvature[0][1];
	const double c = statistic.curvature[1][1];
	const double trace = a + c;
	if (!(trace > 0.0) || !std::isfinite(trace)) {
		return {0.0, 0};
	}
	const double g0 = statistic.gradient[0];
	const double g1 = statistic.gradient[1];
	const double angle = 0.5 * std::atan2(2.0 * b, a - c);
	const double cosine = std::cos(angle);
	const double sine = std::sin(angle);
	const double difference = std::hypot(a - c, 2.0 * b);
	const std::array<double, 2> eigenvalues{{0.5 * (trace + difference), std::max(0.0, 0.5 * (trace - difference))}};
	const std::array<double, 2> projections{{cosine * g0 + sine * g1, -sine * g0 + cosine * g1}};
	const double tolerance = 2.0 * std::numeric_limits<float>::epsilon() * eigenvalues[0];
	double gain = 0.0;
	int rank = 0;
	for (std::size_t index = 0; index < eigenvalues.size(); ++index) {
		if (eigenvalues[index] > tolerance) {
			gain += 0.5 * projections[index] * projections[index] / eigenvalues[index];
			++rank;
		}
	}
	return {std::min(statistic.loss, std::max(0.0, gain)), rank};
}

Selection select_term(const eleginus::Model &model, const DiscoveryState &discovery) {
	Selection best;
	if (discovery.candidates.empty()) {
		return best;
	}
	std::unordered_set<std::uint64_t> existing;
	existing.reserve(model.terms().size());
	for (const auto &term : model.terms()) {
		existing.insert(candidate_key(term.left, term.right));
	}
	const auto universe = eleginus::FeatureMap::candidate_terms();
	const auto remaining = universe > existing.size() ? universe - existing.size() : 0U;
	if (remaining == 0) {
		return best;
	}
	for (const auto &[key, statistic] : discovery.candidates) {
		if (existing.contains(key) || statistic.games < 2 || !(statistic.loss > 0.0)) {
			continue;
		}
		const auto [gain, identifiable_directions] = candidate_gain(statistic);
		if (identifiable_directions == 0 || !(gain > 0.0) || !std::isfinite(gain)) {
			continue;
		}
		const double observations = static_cast<double>(statistic.games);
		if (observations <= kTermParameters + 1.0) {
			continue;
		}
		const double squared_error = 2.0 * statistic.loss;
		const double residual = std::max(squared_error - 2.0 * gain,
			squared_error * std::numeric_limits<float>::epsilon());
		const double correction = 2.0 * kTermParameters * (kTermParameters + 1.0) /
			(observations - kTermParameters - 1.0);
		const double complexity = kTermParameters * std::log(observations) + correction + 2.0 * std::log(static_cast<double>(remaining));
		const double evidence = observations * std::log(squared_error / residual) - complexity;
		if (evidence > best.evidence || (evidence == best.evidence && best.term && key < candidate_key(best.term->left, best.term->right))) {
			best = {candidate_term(key), evidence, gain, statistic.games};
		}
	}
	return best;
}

double update(eleginus::Model &model, const std::vector<Sample> &samples, std::optional<chess::Color> winner,
	std::vector<float> &squared_gradient, DiscoveryState &discovery, const Options &options, std::mt19937_64 &random) {
	std::vector<std::size_t> order(samples.size());
	for (std::size_t index = 0; index < order.size(); ++index) {
		order[index] = index;
	}
	std::shuffle(order.begin(), order.end(), random);
	auto &weights = model.weights();
	double loss = 0.0;
	CandidateStats game_candidates;
	for (const auto sample_index : order) {
		const auto &sample = samples[sample_index];
		const float score = model.score(sample.features);
		const auto training_objective = objective(score, sample.side, winner);
		loss += training_objective.loss;
		const float discovery_score = score_with(sample.features, discovery.weights);
		const auto discovery_objective = objective(discovery_score, sample.side, winner);
		accumulate_candidates(sample, discovery_objective, game_candidates);
		for (const auto &feature : sample.features) {
			const auto index = feature.index;
			const float gradient = training_objective.gradient * feature.value + options.weight_decay * weights[index];
			squared_gradient[index] += gradient * gradient;
			weights[index] -= options.learning_rate * gradient / std::sqrt(squared_gradient[index]);
		}
	}
	if (!samples.empty()) {
		const double inverse_samples = 1.0 / static_cast<double>(samples.size());
		for (const auto &[key, game] : game_candidates) {
			auto &total = discovery.candidates[key];
			total.loss += game.loss * inverse_samples;
			for (std::size_t row = 0; row < total.gradient.size(); ++row) {
				total.gradient[row] += game.gradient[row] * inverse_samples;
				for (std::size_t column = 0; column < total.gradient.size(); ++column) {
					total.curvature[row][column] += game.curvature[row][column] * inverse_samples;
				}
			}
			++total.games;
		}
	}
	return samples.empty() ? 0.0 : loss / static_cast<double>(samples.size());
}

} // namespace

int main(int argc, char **argv) {
	try {
		const auto options = parse_options(argc, argv);
		auto model = options.resume.empty() ? eleginus::Model() : eleginus::Model::load(options.resume);
		std::vector<float> squared_gradient(model.weights().size(), 1.0e-6F);
		std::mt19937_64 update_random(options.seed);
		DiscoveryState discovery;
		discovery.weights = model.weights();
		int white_wins = 0;
		int black_wins = 0;
		int draws = 0;
		double loss = 0.0;

		std::cout << "self-play start: out=" << options.output.string() << " features=" << model.weights().size()
				  << " games=" << options.games << " depth=" << options.depth << " workers=" << options.workers
				  << " seed=" << options.seed << '\n';
		int completed = 0;
		while (completed < options.games) {
			const int batch = std::min(options.workers, options.games - completed);
			std::vector<std::future<GameRecord>> futures;
			futures.reserve(static_cast<std::size_t>(batch));
			for (int offset = 0; offset < batch; ++offset) {
				const auto seed = game_seed(options.seed, static_cast<std::uint64_t>(completed + offset));
				futures.push_back(std::async(std::launch::async, [&model, &options, seed] { return play_game(model, options, seed); }));
			}
			std::vector<GameRecord> records;
			records.reserve(static_cast<std::size_t>(batch));
			for (auto &future : futures) {
				records.push_back(future.get());
			}
			for (auto &record : records) {
				++completed;
				if (!record.winner) {
					++draws;
				} else if (*record.winner == chess::Color::WHITE) {
					++white_wins;
				} else {
					++black_wins;
				}
				loss += update(model, record.samples, record.winner, squared_gradient, discovery, options, update_random);
				if (completed % options.discover_every == 0) {
					const auto selection = select_term(model, discovery);
					std::size_t added = 0;
					if (selection.term && selection.evidence > 0.0) {
						added = model.add_terms({*selection.term});
					}
					squared_gradient.resize(model.weights().size(), 1.0e-6F);
					std::cout << "feature discovery: games=" << completed << " candidates=" << discovery.candidates.size()
							  << " added=" << added << " evidence=" << selection.evidence << " gain=" << selection.gain
							  << " support=" << selection.support << " terms=" << model.terms().size() << '\n';
					discovery = {};
					discovery.weights = model.weights();
				}
				if (completed % options.log_every == 0 || completed == options.games) {
					const int interval = completed % options.log_every == 0 ? options.log_every : completed % options.log_every;
					std::cout << "self-play: games=" << completed << " white=" << white_wins << " draws=" << draws
							  << " black=" << black_wins << " loss=" << loss / static_cast<double>(interval) << '\n';
					loss = 0.0;
					model.save(options.output);
				}
			}
		}
		model.save(options.output);
		std::cout << "self-play finished: model=" << options.output.string() << '\n';
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "train error: " << error.what() << '\n';
		return 1;
	}
}
