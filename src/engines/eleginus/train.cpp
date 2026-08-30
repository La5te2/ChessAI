#include "eleginus/features.hpp"
#include "eleginus/game.hpp"
#include "eleginus/model.hpp"
#include "eleginus/search.hpp"
#include <torch/torch.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <future>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr float kTdLambda = 0.7F;
constexpr float kValueScaleCp = 400.0F;

struct Options {
	std::filesystem::path output;
	std::filesystem::path resume;
	int games = 100;
	int depth = 2;
	int max_plies = 320;
	int hash_mb = 16;
	int workers = static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
	float learning_rate = 1.0F;
	float weight_decay = 1.0e-6F;
	float exploration = 0.08F;
	float temperature_cp = 80.0F;
	int log_every = 10;
	std::uint64_t seed = 2026;
	std::string device = "auto";
};

enum class Outcome { white, draw, black, truncated };

struct Sample {
	std::vector<eleginus::Feature> features;
	chess::Color side = chess::Color::WHITE;
};

struct GameRecord {
	std::vector<Sample> samples;
	Outcome outcome = Outcome::truncated;
};

struct Objective {
	double loss = 0.0;
	float gradient = 0.0F;
};

struct SparseBatch {
	std::vector<std::int64_t> indices;
	std::vector<float> value_contributions;
	std::vector<float> uncertainty_contributions;
	std::size_t samples = 0;
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
		} else if (argument == "--seed") {
			options.seed = std::stoull(value_after(argc, argv, index));
		} else if (argument == "--device") {
			options.device = value_after(argc, argv, index);
		} else if (argument == "--help") {
			std::cout << "usage: train --out eleginus.pth [--resume model.pth] [--games 100] [--depth 2] "
					  << "[--max-plies 320] [--hash 16] [--workers N] [--lr 1] [--exploration 0.08] "
					  << "[--temperature 80] [--device auto] [--seed 2026]\n";
			std::exit(0);
		} else {
			throw std::invalid_argument("unknown option: " + argument);
		}
	}
	if (options.output.empty() || options.games < 0 || options.depth <= 0 || options.depth > 16 || options.max_plies <= 0 ||
		options.hash_mb <= 0 || options.workers <= 0 || options.workers > 256 || options.learning_rate <= 0.0F ||
		options.weight_decay < 0.0F || options.exploration < 0.0F || options.exploration > 1.0F || options.temperature_cp <= 0.0F ||
		options.log_every <= 0) {
		throw std::invalid_argument("invalid or incomplete Eleginus training options");
	}
	return options;
}

torch::Device resolve_device(const std::string &requested) {
	if (requested == "auto") {
		return torch::Device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
	}
	if (requested == "cuda" && !torch::cuda::is_available()) {
		throw std::runtime_error("CUDA was requested but is not available");
	}
	if (requested == "cuda" || requested == "cpu") {
		return torch::Device(requested);
	}
	throw std::invalid_argument("Eleginus training device must be auto, cpu or cuda");
}

class SparseOptimizer {
public:
	SparseOptimizer(const eleginus::Model &model, torch::Device device) : device_(std::move(device)) {
		weights_ = tensor_from(model.weights());
		uncertainty_ = tensor_from(model.uncertainty_weights());
		value_accumulator_ = torch::full_like(weights_, 1.0e-6F);
		uncertainty_accumulator_ = torch::full_like(uncertainty_, 1.0e-6F);
	}

	void step(eleginus::Model &model, const std::vector<std::int64_t> &indices, const std::vector<float> &value_contributions,
		const std::vector<float> &uncertainty_contributions, std::size_t samples, const Options &options) {
		if (indices.empty() || samples == 0) {
			return;
		}
		torch::NoGradGuard guard;
		const auto index = torch::from_blob(const_cast<std::int64_t *>(indices.data()), {static_cast<std::int64_t>(indices.size())},
			torch::TensorOptions().dtype(torch::kInt64)).clone().to(device_);
		const auto value_source = torch::from_blob(const_cast<float *>(value_contributions.data()),
			{static_cast<std::int64_t>(value_contributions.size())}, torch::TensorOptions().dtype(torch::kFloat32)).clone().to(device_);
		const auto uncertainty_source = torch::from_blob(const_cast<float *>(uncertainty_contributions.data()),
			{static_cast<std::int64_t>(uncertainty_contributions.size())}, torch::TensorOptions().dtype(torch::kFloat32)).clone().to(device_);
		auto value_gradient = torch::zeros_like(weights_);
		auto uncertainty_gradient = torch::zeros_like(uncertainty_);
		value_gradient.index_add_(0, index, value_source).div_(static_cast<double>(samples));
		uncertainty_gradient.index_add_(0, index, uncertainty_source).div_(static_cast<double>(samples));
		value_gradient.add_(weights_, options.weight_decay);
		uncertainty_gradient.add_(uncertainty_, options.weight_decay);
		value_accumulator_.addcmul_(value_gradient, value_gradient);
		uncertainty_accumulator_.addcmul_(uncertainty_gradient, uncertainty_gradient);
		weights_.addcdiv_(value_gradient, value_accumulator_.sqrt(), -options.learning_rate);
		uncertainty_.addcdiv_(uncertainty_gradient, uncertainty_accumulator_.sqrt(), -options.learning_rate / kValueScaleCp);
		sync(model);
	}

	const torch::Device &device() const noexcept { return device_; }

private:
	torch::Tensor tensor_from(const std::vector<float> &values) const {
		return torch::from_blob(const_cast<float *>(values.data()), {static_cast<std::int64_t>(values.size())},
			torch::TensorOptions().dtype(torch::kFloat32)).clone().to(device_);
	}

	void sync(eleginus::Model &model) const {
		const auto value_cpu = weights_.to(torch::kCPU).contiguous();
		const auto uncertainty_cpu = uncertainty_.to(torch::kCPU).contiguous();
		std::copy_n(value_cpu.data_ptr<float>(), model.weights().size(), model.weights().begin());
		std::copy_n(uncertainty_cpu.data_ptr<float>(), model.uncertainty_weights().size(), model.uncertainty_weights().begin());
	}

	torch::Device device_;
	torch::Tensor weights_;
	torch::Tensor uncertainty_;
	torch::Tensor value_accumulator_;
	torch::Tensor uncertainty_accumulator_;
};

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

Outcome terminal_outcome(const chess::Board &board) {
	const auto [reason, result] = board.isGameOver();
	if (reason == chess::GameResultReason::NONE) {
		return Outcome::truncated;
	}
	if (result == chess::GameResult::DRAW) {
		return Outcome::draw;
	}
	const auto winner = result == chess::GameResult::WIN ? board.sideToMove() : ~board.sideToMove();
	return winner == chess::Color::WHITE ? Outcome::white : Outcome::black;
}

float outcome_value(Outcome outcome) noexcept {
	return outcome == Outcome::white ? 1.0F : outcome == Outcome::black ? -1.0F : 0.0F;
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
	search_options.multipv = 1;
	search_options.capture_principal_variation = true;
	eleginus::Searcher searcher(model, search_options);
	std::mt19937_64 random(seed);
	for (int ply = 0; ply < options.max_plies && !eleginus::game_is_over(board); ++ply) {
		const auto result = searcher.search(board);
		if (result.principal_variation.empty()) {
			break;
		}
		chess::Board leaf = board;
		for (const auto move : result.principal_variation) {
			leaf.makeMove(move);
		}
		Sample sample;
		sample.side = leaf.sideToMove();
		model.extract(leaf, sample.features);
		record.samples.push_back(std::move(sample));
		const auto move = choose_move(result, options, random);
		if (move.move() == chess::Move::NO_MOVE) {
			break;
		}
		board.makeMove(move);
	}
	record.outcome = terminal_outcome(board);
	if (record.outcome == Outcome::truncated) {
		record.samples.clear();
	}
	return record;
}

float score_with(const std::vector<eleginus::Feature> &features, const std::vector<float> &weights) noexcept {
	float score = 0.0F;
	for (const auto &feature : features) {
		score += weights[feature.index] * feature.value;
	}
	return score;
}

float white_prediction(float score, chess::Color side) noexcept {
	return std::tanh((side == chess::Color::WHITE ? score : -score) / kValueScaleCp);
}

std::vector<float> td_returns(const eleginus::Model &model, const GameRecord &record) {
	std::vector<float> predictions(record.samples.size());
	for (std::size_t index = 0; index < record.samples.size(); ++index) {
		predictions[index] = white_prediction(model.score(record.samples[index].features), record.samples[index].side);
	}
	std::vector<float> returns(predictions.size());
	float accumulated = 0.0F;
	for (std::size_t reverse = predictions.size(); reverse-- > 0;) {
		const float successor = reverse + 1 < predictions.size() ? predictions[reverse + 1] : outcome_value(record.outcome);
		accumulated = successor - predictions[reverse] + kTdLambda * accumulated;
		returns[reverse] = accumulated;
	}
	return returns;
}

Objective objective(float score, chess::Color side, float target) noexcept {
	const float perspective = side == chess::Color::WHITE ? 1.0F : -1.0F;
	const float prediction = std::tanh(perspective * score / kValueScaleCp);
	const float error = prediction - target;
	const float derivative = perspective * (1.0F - prediction * prediction);
	return {0.5 * static_cast<double>(error) * error, error * derivative};
}

double update(const eleginus::Model &model, const GameRecord &record, SparseBatch &batch) {
	const auto returns = td_returns(model, record);
	const auto &uncertainty_weights = model.uncertainty_weights();
	double loss = 0.0;
	for (std::size_t sample_index = 0; sample_index < record.samples.size(); ++sample_index) {
		const auto &sample = record.samples[sample_index];
		const float prediction = white_prediction(model.score(sample.features), sample.side);
		const float target = std::clamp(prediction + returns[sample_index], -1.0F, 1.0F);
		const auto value_objective = objective(model.score(sample.features), sample.side, target);
		const float uncertainty_target = std::clamp(std::abs(returns[sample_index]), 0.0F, 1.0F);
		const float uncertainty_logit = score_with(sample.features, uncertainty_weights);
		const float uncertainty = 1.0F / (1.0F + std::exp(-std::clamp(uncertainty_logit, -20.0F, 20.0F)));
		loss += value_objective.loss - uncertainty_target * std::log(std::max(uncertainty, 1.0e-7F)) -
			(1.0F - uncertainty_target) * std::log(std::max(1.0F - uncertainty, 1.0e-7F));
		for (const auto &feature : sample.features) {
			batch.indices.push_back(static_cast<std::int64_t>(feature.index));
			batch.value_contributions.push_back(value_objective.gradient * feature.value);
			batch.uncertainty_contributions.push_back((uncertainty - uncertainty_target) * feature.value);
		}
	}
	batch.samples += record.samples.size();
	return record.samples.empty() ? 0.0 : loss / static_cast<double>(record.samples.size());
}

} // namespace

int main(int argc, char **argv) {
	try {
		const auto options = parse_options(argc, argv);
		auto model = options.resume.empty() ? eleginus::Model() : eleginus::Model::load(options.resume);
		SparseOptimizer optimizer(model, resolve_device(options.device));
		int white_wins = 0;
		int black_wins = 0;
		int draws = 0;
		int discarded = 0;
		std::uint64_t attempted = 0;
		double loss = 0.0;

		std::cout << "self-play start: out=" << options.output.string() << " features=" << model.weights().size()
				  << " games=" << options.games << " max_plies=" << options.max_plies << " depth=" << options.depth
				  << " workers=" << options.workers << " device=" << optimizer.device() << " seed=" << options.seed << '\n';
		int completed = 0;
		int logged = 0;
		while (completed < options.games) {
			const int batch = std::min(options.workers, options.games - completed);
			std::vector<std::future<GameRecord>> futures;
			futures.reserve(static_cast<std::size_t>(batch));
			for (int offset = 0; offset < batch; ++offset) {
				const auto seed = game_seed(options.seed, attempted++);
				futures.push_back(std::async(std::launch::async, [&model, &options, seed] { return play_game(model, options, seed); }));
			}
			std::vector<GameRecord> records;
			records.reserve(futures.size());
			for (auto &future : futures) {
				records.push_back(future.get());
			}
			SparseBatch gradients;
			bool log_due = false;
			for (auto &record : records) {
				if (record.outcome == Outcome::truncated) {
					++discarded;
					continue;
				}
				++completed;
				++logged;
				if (record.outcome == Outcome::draw) {
					++draws;
				} else if (record.outcome == Outcome::white) {
					++white_wins;
				} else {
					++black_wins;
				}
				loss += update(model, record, gradients);
				if (completed % options.log_every == 0 || completed == options.games) {
					log_due = true;
				}
				if (completed == options.games) {
					break;
				}
			}
			optimizer.step(model, gradients.indices, gradients.value_contributions, gradients.uncertainty_contributions,
				gradients.samples, options);
			if (log_due) {
				std::cout << "self-play: games=" << completed << " discarded=" << discarded << " white=" << white_wins
						  << " draws=" << draws << " black=" << black_wins << " loss=" << loss / static_cast<double>(logged) << '\n';
				loss = 0.0;
				logged = 0;
				model.save(options.output);
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
