#pragma once

// Melano PGN-to-HDF5 preprocessing and one-shot supervised P/V/A training.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
#include <torch/types.h>
#include "melano/game.hpp"

namespace melano {

struct SupervisedBatch {
	torch::Tensor states;
	torch::Tensor next_states;
	torch::Tensor moves;
	torch::Tensor values;
	torch::Tensor next_values;
	torch::Tensor advantage_moves;
	torch::Tensor advantage_values;
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
	/// Opens and validates a Melano HDF5 dataset and its architecture schema.
	explicit SupervisedH5(const std::filesystem::path &path);
	/// Closes all HDF5 handles owned by this reader.
	~SupervisedH5();
	/// Dataset handles have unique ownership and cannot be copied.
	SupervisedH5(const SupervisedH5 &) = delete;
	/// Dataset handles have unique ownership and cannot be copy-assigned.
	SupervisedH5 &operator=(const SupervisedH5 &) = delete;
	/// Transfers ownership of an open dataset reader.
	SupervisedH5(SupervisedH5 &&) noexcept;
	/// Replaces this reader with another reader's open HDF5 handles.
	SupervisedH5 &operator=(SupervisedH5 &&) noexcept;

	/// Returns immutable schema and row-count metadata.
	const DatasetInfo &info() const noexcept;
	/// Reads arbitrary rows and decodes state, move, value, and advantage tensors.
	SupervisedBatch read(const std::vector<std::int64_t> &indices) const;

	private:
	struct Impl;
	Impl *impl_;
};

struct PreprocessOptions {
	std::filesystem::path input = "data/games.pgn";
	std::filesystem::path output = "data/games.melano.h5";
	std::int64_t max_games = -1;
	int chunk_size = 16384;
	int has_comments = 1;
	int compression_level = 1;
	int log_every = 10000;
};

/// Parses PGN games and writes Melano-specific policy, value, and advantage targets.
void preprocess_pgn(const PreprocessOptions &options);

struct TrainOptions {
	std::filesystem::path data = "data/games.melano.h5";
	std::filesystem::path output = "models/melano.pth";
	int channels = 128;
	int blocks = 10;
	int epochs = 10;
	int batch_size = 512;
	std::int64_t max_steps = -1;
	double learning_rate = 1e-3;
	double weight_decay = 1e-4;
	double value_weight = 0.25;
	double dueling_q_weight = 0.5;
	double dynamics_weight = 0.25;
	double imagined_value_weight = 0.25;
	int save_every = 5000;
	int log_every = 100;
	std::uint64_t seed = 2026;
	std::string device = "auto";
};

/// Trains a new Melano model from scratch and atomically writes the final checkpoint.
void train_supervised(const TrainOptions &options);

} // namespace melano
