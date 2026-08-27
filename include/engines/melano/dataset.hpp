#pragma once

// Melano PGN-to-HDF5 preprocessing and one-shot supervised policy/value training.

#include "melano/game.hpp"
#include "melano/precision.hpp"
#include <cstdint>
#include <filesystem>
#include <string>
#include <torch/types.h>
#include <vector>

namespace melano {

struct SupervisedBatch {
	torch::Tensor states;
	torch::Tensor moves;
	torch::Tensor values;
};

struct DatasetInfo {
	std::int64_t length = 0;
	std::int64_t chunk_rows = 1;
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
	/// Reads arbitrary rows and decodes state, move, and value tensors.
	SupervisedBatch read(const std::vector<std::int64_t> &indices, bool pinned_memory = false) const;
	/// Reads one contiguous range while preserving state, move, and value alignment.
	SupervisedBatch read_contiguous(std::int64_t begin, std::int64_t count, bool pinned_memory = false) const;

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

/// Parses PGN games and writes Melano-specific state, policy, and value targets.
void preprocess_pgn(const PreprocessOptions &options);

struct TrainOptions {
	std::filesystem::path data = "data/games.melano.h5";
	std::filesystem::path output = "models/melano/melano.pth";
	int channels = 128;
	int blocks = 10;
	int epochs = 10;
	int batch_size = 512;
	std::int64_t max_steps = -1;
	double learning_rate = 1e-3;
	double weight_decay = 1e-4;
	double value_weight = 0.25;
	double grad_clip = 1.0;
	int save_every = 5000;
	int log_every = 100;
	std::uint64_t seed = 2026;
	std::string device = "auto";
	ComputePrecision precision = ComputePrecision::Fp32;
};

/// Trains a new Melano model from scratch and atomically writes the final checkpoint.
void train_supervised(const TrainOptions &options);

} // namespace melano
