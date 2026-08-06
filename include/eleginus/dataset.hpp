#pragma once

// Eleginus sparse Value datasets, PGN preprocessing, and supervised Value fitting.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <torch/types.h>

#include "eleginus/model.hpp"

namespace eleginus {

inline constexpr const char *kValueStateEncoding = "eleginus_sparse_features_v1";
inline constexpr const char *kValueTargetSchema = "side_to_move_expectation_01_v1";

struct ValueDatasetInfo {
	std::int64_t length = 0;
	int has_comments = 0;
	std::string source;
};

struct ValueBatch {
	std::vector<EncodedFeatures> features;
	torch::Tensor moves;
	torch::Tensor values;
};

class ValueH5 {
	public:
	explicit ValueH5(const std::filesystem::path &path);
	~ValueH5();

	ValueH5(const ValueH5 &) = delete;
	ValueH5 &operator=(const ValueH5 &) = delete;
	ValueH5(ValueH5 &&) noexcept;
	ValueH5 &operator=(ValueH5 &&) noexcept;

	const ValueDatasetInfo &info() const noexcept;
	ValueBatch read_contiguous(std::int64_t begin, std::int64_t count) const;

	private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

struct ValueWriterOptions {
	std::filesystem::path output;
	int has_comments = 0;
	int chunk_size = 4096;
	int compression_level = 4;
	std::string source = "supervised";
};

class ValueH5Writer {
	public:
	explicit ValueH5Writer(const ValueWriterOptions &options);
	~ValueH5Writer();

	ValueH5Writer(const ValueH5Writer &) = delete;
	ValueH5Writer &operator=(const ValueH5Writer &) = delete;
	ValueH5Writer(ValueH5Writer &&) noexcept;
	ValueH5Writer &operator=(ValueH5Writer &&) noexcept;

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

struct ValueTrainOptions {
	std::filesystem::path data;
	int epochs = 1;
	int batch_size = 512;
	std::int64_t max_steps = 0;
	double learning_rate = 1.0e-3;
	double weight_decay = 1.0e-5;
	int log_every = 100;
	std::uint64_t seed = 2026;
};

struct ValueTrainStats {
	std::int64_t steps = 0;
	std::int64_t samples = 0;
	double mean_loss = 0.0;
};

ValueTrainStats train_value_from_h5(ValueNetwork &value, const ValueTrainOptions &options,
									const torch::Device &device);

} // namespace eleginus
