#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "gadus/game.hpp"

namespace gadus {

struct SupervisedBatch {
	torch::Tensor states;
	torch::Tensor moves;
	torch::Tensor values;
};

struct DatasetInfo {
	std::int64_t length = 0;
	int has_comments = 1;
	std::string arch_type;
	std::string state_encoding;
	std::string move_encoding;
	std::string target_schema;
};

class SupervisedH5 {
	public:
	explicit SupervisedH5(const std::filesystem::path &path);
	~SupervisedH5();
	SupervisedH5(const SupervisedH5 &) = delete;
	SupervisedH5 &operator=(const SupervisedH5 &) = delete;
	SupervisedH5(SupervisedH5 &&) noexcept;
	SupervisedH5 &operator=(SupervisedH5 &&) noexcept;

	const DatasetInfo &info() const noexcept;
	SupervisedBatch read(const std::vector<std::int64_t> &indices) const;

	private:
	struct Impl;
	Impl *impl_;
};

struct PreprocessOptions {
	std::filesystem::path input = "data/games.pgn";
	std::filesystem::path output = "data/games.gadus.h5";
	std::int64_t max_games = -1;
	int chunk_size = 16384;
	int has_comments = 1;
	int compression_level = 1;
	int log_every = 10000;
};

void preprocess_pgn(const PreprocessOptions &options);

struct TrainOptions {
	std::filesystem::path data = "data/games.gadus.h5";
	std::filesystem::path output = "models/gadus.pth";
	int channels = 128;
	int blocks = 10;
	int epochs = 10;
	int batch_size = 512;
	std::int64_t max_steps = -1;
	double learning_rate = 1e-3;
	double weight_decay = 1e-4;
	double value_weight = 0.25;
	int save_every = 5000;
	int log_every = 100;
	std::uint64_t seed = 2026;
	std::string device = "auto";
};

void train_supervised(const TrainOptions &options);

} // namespace gadus
