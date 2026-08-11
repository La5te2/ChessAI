#pragma once

// Eleginus sparse supervised datasets, PGN preprocessing, and joint Policy/Value fitting.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <torch/types.h>

#include "eleginus/model.hpp"

namespace eleginus {

inline constexpr const char *kStateEncoding = "eleginus_sparse_features_v1";
inline constexpr const char *kTargetSchema = "policy_value_perspective_resolved";

struct DatasetInfo {
	std::int64_t length = 0;
	int has_comments = 0;
	std::string source;
};

struct Batch {
	std::vector<EncodedFeatures> features;
	torch::Tensor moves;
	torch::Tensor values;
};

class H5Dataset {
	public:
	explicit H5Dataset(const std::filesystem::path &path);
	~H5Dataset();

	H5Dataset(const H5Dataset &) = delete;
	H5Dataset &operator=(const H5Dataset &) = delete;
	H5Dataset(H5Dataset &&) noexcept;
	H5Dataset &operator=(H5Dataset &&) noexcept;

	const DatasetInfo &info() const noexcept;
	Batch read_contiguous(std::int64_t begin, std::int64_t count) const;

	private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

struct WriterOptions {
	std::filesystem::path output;
	int has_comments = 0;
	int chunk_size = 4096;
	int compression_level = 4;
	std::string source = "supervised";
};

class H5Writer {
	public:
	explicit H5Writer(const WriterOptions &options);
	~H5Writer();

	H5Writer(const H5Writer &) = delete;
	H5Writer &operator=(const H5Writer &) = delete;
	H5Writer(H5Writer &&) noexcept;
	H5Writer &operator=(H5Writer &&) noexcept;

	void append(const std::vector<EncodedFeatures> &features,
				const std::vector<std::uint16_t> &moves, const std::vector<float> &values);
	std::int64_t size() const noexcept;
	void flush();

	private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

struct PreprocessOptions {
	std::filesystem::path input = "data/games.pgn";
	std::filesystem::path output = "data/games.eleginus.h5";
	std::int64_t max_games = -1;
	int chunk_size = 4096;
	int has_comments = 1;
	int compression_level = 4;
	int log_every = 1000;
};

void preprocess_pgn(const PreprocessOptions &options);

struct TrainOptions {
	std::filesystem::path data;
	int epochs = 1;
	int batch_size = 512;
	std::int64_t max_steps = 0;
	double learning_rate = 1.0e-3;
	double weight_decay = 1.0e-5;
	int log_every = 100;
	std::uint64_t seed = 2026;
};

struct TrainStats {
	std::int64_t steps = 0;
	std::int64_t samples = 0;
};

TrainStats train_from_h5(Model &model, const TrainOptions &options, const torch::Device &device);

} // namespace eleginus
