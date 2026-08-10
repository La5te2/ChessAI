// Implements the architecture-locked HDF5 format and joint Policy/Value fitting.

#include "eleginus/dataset.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <regex>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <hdf5.h>
#include <torch/nn/functional/loss.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/optim.h>

#include "eleginus/game.hpp"

namespace eleginus {

namespace {

void require_h5(herr_t status, const std::string &operation) {
	if (status < 0)
		throw std::runtime_error("HDF5 failed to " + operation);
}

hid_t require_id(hid_t identifier, const std::string &operation) {
	if (identifier < 0)
		throw std::runtime_error("HDF5 failed to " + operation);
	return identifier;
}

void write_string_attribute(hid_t object, const char *name, const std::string &value) {
	const hid_t space = require_id(H5Screate(H5S_SCALAR), "create attribute space");
	const hid_t type = require_id(H5Tcopy(H5T_C_S1), "copy attribute string type");
	require_h5(H5Tset_size(type, value.size() + 1), "set attribute string size");
	const hid_t attribute = require_id(
		H5Acreate2(object, name, type, space, H5P_DEFAULT, H5P_DEFAULT), name);
	require_h5(H5Awrite(attribute, type, value.c_str()), name);
	H5Aclose(attribute);
	H5Tclose(type);
	H5Sclose(space);
}

void write_int_attribute(hid_t object, const char *name, std::int64_t value) {
	const hid_t space = require_id(H5Screate(H5S_SCALAR), "create attribute space");
	const hid_t attribute = require_id(
		H5Acreate2(object, name, H5T_STD_I64LE, space, H5P_DEFAULT, H5P_DEFAULT), name);
	require_h5(H5Awrite(attribute, H5T_NATIVE_INT64, &value), name);
	H5Aclose(attribute);
	H5Sclose(space);
}

std::string read_string_attribute(hid_t object, const char *name) {
	if (H5Aexists(object, name) <= 0)
		throw std::runtime_error(std::string("missing HDF5 attribute: ") + name);
	const hid_t attribute = require_id(H5Aopen(object, name, H5P_DEFAULT), name);
	const hid_t type = require_id(H5Aget_type(attribute), name);
	const auto size = H5Tget_size(type);
	std::vector<char> buffer(size + 1, '\0');
	require_h5(H5Aread(attribute, type, buffer.data()), name);
	H5Tclose(type);
	H5Aclose(attribute);
	return buffer.data();
}

std::string read_optional_string_attribute(hid_t object, const char *name) {
	return H5Aexists(object, name) > 0 ? read_string_attribute(object, name) : std::string{};
}

std::int64_t read_int_attribute(hid_t object, const char *name) {
	if (H5Aexists(object, name) <= 0)
		throw std::runtime_error(std::string("missing HDF5 attribute: ") + name);
	std::int64_t value = 0;
	const hid_t attribute = require_id(H5Aopen(object, name, H5P_DEFAULT), name);
	require_h5(H5Aread(attribute, H5T_NATIVE_INT64, &value), name);
	H5Aclose(attribute);
	return value;
}

hid_t create_dataset(hid_t file, const char *name, const std::vector<hsize_t> &initial,
					 const std::vector<hsize_t> &maximum, const std::vector<hsize_t> &chunk,
					 hid_t type, int compression_level) {
	const hid_t space = require_id(
		H5Screate_simple(static_cast<int>(initial.size()), initial.data(), maximum.data()), name);
	const hid_t properties = require_id(H5Pcreate(H5P_DATASET_CREATE), name);
	require_h5(H5Pset_chunk(properties, static_cast<int>(chunk.size()), chunk.data()), name);
	if (compression_level > 0) {
		require_h5(H5Pset_shuffle(properties), "enable shuffle filter");
		require_h5(H5Pset_deflate(properties, static_cast<unsigned>(compression_level)),
			"enable deflate filter");
	}
	const hid_t dataset = require_id(
		H5Dcreate2(file, name, type, space, H5P_DEFAULT, properties, H5P_DEFAULT), name);
	H5Pclose(properties);
	H5Sclose(space);
	return dataset;
}

void write_slice(hid_t dataset, hid_t type, const void *data,
				 const std::vector<hsize_t> &start, const std::vector<hsize_t> &count) {
	const hid_t file_space = require_id(H5Dget_space(dataset), "get write space");
	require_h5(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start.data(), nullptr,
		count.data(), nullptr), "select write slice");
	const hid_t memory_space = require_id(
		H5Screate_simple(static_cast<int>(count.size()), count.data(), nullptr),
		"create write memory space");
	require_h5(H5Dwrite(dataset, type, memory_space, file_space, H5P_DEFAULT, data),
		"write dataset slice");
	H5Sclose(memory_space);
	H5Sclose(file_space);
}

void read_slice(hid_t dataset, hid_t type, void *data, const std::vector<hsize_t> &start,
				const std::vector<hsize_t> &count) {
	const hid_t file_space = require_id(H5Dget_space(dataset), "get read space");
	require_h5(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start.data(), nullptr,
		count.data(), nullptr), "select read slice");
	const hid_t memory_space = require_id(
		H5Screate_simple(static_cast<int>(count.size()), count.data(), nullptr),
		"create read memory space");
	require_h5(H5Dread(dataset, type, memory_space, file_space, H5P_DEFAULT, data),
		"read dataset slice");
	H5Sclose(memory_space);
	H5Sclose(file_space);
}

std::optional<double> comment_score_white(const std::string &comment) {
	static const std::regex score_pattern(
		R"((^|[^[:alnum:]_.])([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))(?=$|[^[:alnum:]_.]))");
	std::smatch match;
	if (!std::regex_search(comment, match, score_pattern))
		return std::nullopt;
	return std::stod(match[2].str());
}

float comment_value(const std::string &comment, chess::Color turn) {
	const double white = comment_score_white(comment).value_or(0.0);
	const double side = turn == chess::Color::WHITE ? white : -white;
	return static_cast<float>((std::tanh(side / 3.0) + 1.0) * 0.5);
}

float result_value(const std::string &result, chess::Color turn) {
	float white = 0.5F;
	if (result == "1-0")
		white = 1.0F;
	else if (result == "0-1")
		white = 0.0F;
	return turn == chess::Color::WHITE ? white : 1.0F - white;
}

struct StopPgnParsing {};

class PreprocessVisitor : public chess::pgn::Visitor {
	public:
	PreprocessVisitor(const PreprocessOptions &options, H5Writer &writer)
		: options_(options), writer_(writer) {}

	void startPgn() override {
		if (options_.max_games >= 0 && games_ >= options_.max_games)
			throw StopPgnParsing{};
		board_ = chess::Board();
		result_ = "*";
		previous_comment_.clear();
		game_has_eval_ = false;
		game_features_.clear();
		game_moves_.clear();
		game_values_.clear();
	}

	void header(std::string_view key, std::string_view value) override {
		if (key == "Result")
			result_ = std::string(value);
		if (key == "FEN" && !value.empty())
			board_ = chess::Board(value);
	}

	void startMoves() override {}

	void move(std::string_view san, std::string_view comment) override {
		try {
			const auto parsed = chess::uci::parseSan(board_, san);
			game_features_.push_back(encode_features(board_));
			game_moves_.push_back(
				static_cast<std::uint16_t>(move_to_index(parsed, board_.sideToMove())));
			game_values_.push_back(options_.has_comments
				? comment_value(previous_comment_, board_.sideToMove())
				: result_value(result_, board_.sideToMove()));
			if (comment_score_white(std::string(comment)).has_value())
				game_has_eval_ = true;
			board_.makeMove(parsed);
			previous_comment_ = std::string(comment);
		} catch (const std::exception &) {
			++skipped_moves_;
		}
	}

	void endPgn() override {
		if ((options_.has_comments && !game_has_eval_) ||
			(!options_.has_comments && result_ != "1-0" && result_ != "0-1" &&
			 result_ != "1/2-1/2")) {
			++skipped_games_;
			return;
		}
		writer_.append(game_features_, game_moves_, game_values_);
		++games_;
		if (options_.log_every > 0 && games_ % options_.log_every == 0) {
			std::cout << "preprocess progress: games=" << games_
					  << " positions=" << writer_.size()
					  << " skipped_moves=" << skipped_moves_
					  << " skipped_games=" << skipped_games_ << std::endl;
		}
	}

	std::int64_t games() const noexcept { return games_; }
	std::int64_t skipped_moves() const noexcept { return skipped_moves_; }
	std::int64_t skipped_games() const noexcept { return skipped_games_; }

	private:
	PreprocessOptions options_;
	H5Writer &writer_;
	chess::Board board_;
	std::string result_;
	std::string previous_comment_;
	bool game_has_eval_ = false;
	std::vector<EncodedFeatures> game_features_;
	std::vector<std::uint16_t> game_moves_;
	std::vector<float> game_values_;
	std::int64_t games_ = 0;
	std::int64_t skipped_moves_ = 0;
	std::int64_t skipped_games_ = 0;
};

} // namespace

struct H5Writer::Impl {
	explicit Impl(const WriterOptions &writer_options) : options(writer_options) {
		if (options.output.empty())
			throw std::invalid_argument("Eleginus HDF5 output path is empty");
		if (options.has_comments != 0 && options.has_comments != 1)
			throw std::invalid_argument("has_comments must be 0 or 1");
		if (options.chunk_size <= 0 || options.compression_level < 0 ||
			options.compression_level > 9)
			throw std::invalid_argument("invalid Eleginus HDF5 chunk or compression setting");
		if (!options.output.parent_path().empty())
			std::filesystem::create_directories(options.output.parent_path());
		file = require_id(H5Fcreate(options.output.string().c_str(), H5F_ACC_TRUNC,
			H5P_DEFAULT, H5P_DEFAULT), "create Eleginus HDF5 file");
		write_string_attribute(file, "arch_type", kArchType);
		write_string_attribute(file, "state_encoding", kStateEncoding);
		write_string_attribute(file, "move_encoding", kMoveEncoding);
		write_string_attribute(file, "target_schema", kTargetSchema);
		write_string_attribute(file, "value_perspective", "side_to_move");
		write_string_attribute(file, "source", options.source);
		write_int_attribute(file, "has_cmt", options.has_comments);
		if (options.has_comments) {
			write_string_attribute(file, "comment_eval_perspective", "white");
			write_string_attribute(file, "comment_value_transform",
				"(tanh(side_to_move_pawn_score/3)+1)/2");
		}
		const auto chunk = static_cast<hsize_t>(options.chunk_size);
		states = create_dataset(file, "states", {0, kPerspectiveCount, kFeatureSlots},
			{H5S_UNLIMITED, kPerspectiveCount, kFeatureSlots},
			{chunk, kPerspectiveCount, kFeatureSlots}, H5T_STD_U16LE,
			options.compression_level);
		moves = create_dataset(file, "moves", {0}, {H5S_UNLIMITED}, {chunk},
			H5T_STD_U16LE, options.compression_level);
		values = create_dataset(file, "values", {0}, {H5S_UNLIMITED}, {chunk},
			H5T_IEEE_F32LE, options.compression_level);
	}

	~Impl() {
		if (states >= 0)
			H5Dclose(states);
		if (moves >= 0)
			H5Dclose(moves);
		if (values >= 0)
			H5Dclose(values);
		if (file >= 0)
			H5Fclose(file);
	}

	WriterOptions options;
	hid_t file = -1;
	hid_t states = -1;
	hid_t moves = -1;
	hid_t values = -1;
	std::int64_t length = 0;
};

H5Writer::H5Writer(const WriterOptions &options)
	: impl_(std::make_unique<Impl>(options)) {}

H5Writer::~H5Writer() = default;
H5Writer::H5Writer(H5Writer &&) noexcept = default;
H5Writer &H5Writer::operator=(H5Writer &&) noexcept = default;

void H5Writer::append(const std::vector<EncodedFeatures> &features,
						   const std::vector<std::uint16_t> &moves,
						   const std::vector<float> &target_values) {
	if (features.size() != moves.size() || features.size() != target_values.size())
		throw std::invalid_argument("Eleginus HDF5 state/move/value rows are misaligned");
	if (features.empty())
		return;
	std::vector<std::uint16_t> packed(
		features.size() * static_cast<std::size_t>(kPerspectiveCount * kFeatureSlots));
	std::size_t cursor = 0;
	for (std::size_t row = 0; row < features.size(); ++row) {
		const int first = features[row].white_to_move ? 0 : 1;
		const int second = 1 - first;
		for (const int perspective_index : {first, second}) {
			const auto &perspective =
				features[row].perspective[static_cast<std::size_t>(perspective_index)];
			for (const int feature : perspective) {
				if (feature < 0 || feature >= kFeatureVocabulary)
					throw std::invalid_argument("Eleginus HDF5 feature index is out of range");
				packed[cursor++] = static_cast<std::uint16_t>(feature);
			}
		}
		if (!std::isfinite(target_values[row]) || target_values[row] < 0.0F ||
			target_values[row] > 1.0F)
			throw std::invalid_argument("Eleginus Value target must be finite and in [0,1]");
		if (moves[row] >= kActionSize)
			throw std::invalid_argument("Eleginus HDF5 move index is out of range");
	}
	const hsize_t old_length = static_cast<hsize_t>(impl_->length);
	const hsize_t count = static_cast<hsize_t>(features.size());
	const hsize_t next = old_length + count;
	const hsize_t state_dimensions[] = {next, kPerspectiveCount, kFeatureSlots};
	const hsize_t scalar_dimensions[] = {next};
	require_h5(H5Dset_extent(impl_->states, state_dimensions), "extend Eleginus states");
	require_h5(H5Dset_extent(impl_->moves, scalar_dimensions), "extend Eleginus moves");
	require_h5(H5Dset_extent(impl_->values, scalar_dimensions), "extend Eleginus values");
	write_slice(impl_->states, H5T_NATIVE_UINT16, packed.data(), {old_length, 0, 0},
		{count, kPerspectiveCount, kFeatureSlots});
	write_slice(impl_->moves, H5T_NATIVE_UINT16, moves.data(), {old_length}, {count});
	write_slice(impl_->values, H5T_NATIVE_FLOAT, target_values.data(), {old_length}, {count});
	impl_->length += static_cast<std::int64_t>(features.size());
}

std::int64_t H5Writer::size() const noexcept { return impl_->length; }

void H5Writer::flush() {
	require_h5(H5Fflush(impl_->file, H5F_SCOPE_GLOBAL), "flush Eleginus HDF5 file");
}

struct H5Dataset::Impl {
	explicit Impl(const std::filesystem::path &path) {
		file = require_id(H5Fopen(path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT),
			"open Eleginus HDF5 file");
		if (read_string_attribute(file, "arch_type") != kArchType ||
			read_string_attribute(file, "state_encoding") != kStateEncoding ||
			read_string_attribute(file, "move_encoding") != kMoveEncoding ||
			read_string_attribute(file, "target_schema") != kTargetSchema ||
			read_string_attribute(file, "value_perspective") != "side_to_move")
			throw std::runtime_error("HDF5 file is not an Eleginus supervised dataset");
		info.has_comments = static_cast<int>(read_int_attribute(file, "has_cmt"));
		info.source = read_optional_string_attribute(file, "source");
		states = require_id(H5Dopen2(file, "states", H5P_DEFAULT), "open Eleginus states");
		moves = require_id(H5Dopen2(file, "moves", H5P_DEFAULT), "open Eleginus moves");
		values = require_id(H5Dopen2(file, "values", H5P_DEFAULT), "open Eleginus values");
		const hid_t state_space = require_id(H5Dget_space(states), "get Eleginus state shape");
		if (H5Sget_simple_extent_ndims(state_space) != 3)
			throw std::runtime_error("Eleginus states must have rank three");
		hsize_t state_dimensions[3]{};
		H5Sget_simple_extent_dims(state_space, state_dimensions, nullptr);
		H5Sclose(state_space);
		if (state_dimensions[1] != kPerspectiveCount || state_dimensions[2] != kFeatureSlots)
			throw std::runtime_error("Eleginus state dimensions do not match [N,2,34]");
		info.length = static_cast<std::int64_t>(state_dimensions[0]);
		for (const auto [dataset, name] : {
				 std::pair<hid_t, const char *>{moves, "moves"},
				 std::pair<hid_t, const char *>{values, "values"}}) {
			const hid_t space = require_id(H5Dget_space(dataset), std::string("get ") + name);
			if (H5Sget_simple_extent_ndims(space) != 1)
				throw std::runtime_error(std::string("Eleginus ") + name + " must have rank one");
			hsize_t dimensions[1]{};
			H5Sget_simple_extent_dims(space, dimensions, nullptr);
			H5Sclose(space);
			if (dimensions[0] != state_dimensions[0])
				throw std::runtime_error("Eleginus HDF5 datasets are not row aligned");
		}
	}

	~Impl() {
		if (states >= 0)
			H5Dclose(states);
		if (moves >= 0)
			H5Dclose(moves);
		if (values >= 0)
			H5Dclose(values);
		if (file >= 0)
			H5Fclose(file);
	}

	hid_t file = -1;
	hid_t states = -1;
	hid_t moves = -1;
	hid_t values = -1;
	DatasetInfo info;
};

H5Dataset::H5Dataset(const std::filesystem::path &path) : impl_(std::make_unique<Impl>(path)) {}
H5Dataset::~H5Dataset() = default;
H5Dataset::H5Dataset(H5Dataset &&) noexcept = default;
H5Dataset &H5Dataset::operator=(H5Dataset &&) noexcept = default;

const DatasetInfo &H5Dataset::info() const noexcept { return impl_->info; }

Batch H5Dataset::read_contiguous(std::int64_t begin, std::int64_t count) const {
	if (begin < 0 || count < 0 || begin + count > impl_->info.length)
		throw std::out_of_range("Eleginus HDF5 read range is out of bounds");
	Batch batch;
	batch.features.resize(static_cast<std::size_t>(count));
	batch.moves = torch::empty({count}, torch::TensorOptions().dtype(torch::kInt64));
	batch.values = torch::empty({count}, torch::TensorOptions().dtype(torch::kFloat32));
	if (count == 0)
		return batch;
	std::vector<std::uint16_t> packed(static_cast<std::size_t>(count) *
		static_cast<std::size_t>(kPerspectiveCount * kFeatureSlots));
	std::vector<std::uint16_t> packed_moves(static_cast<std::size_t>(count));
	read_slice(impl_->states, H5T_NATIVE_UINT16, packed.data(),
		{static_cast<hsize_t>(begin), 0, 0},
		{static_cast<hsize_t>(count), kPerspectiveCount, kFeatureSlots});
	read_slice(impl_->moves, H5T_NATIVE_UINT16, packed_moves.data(), {static_cast<hsize_t>(begin)},
		{static_cast<hsize_t>(count)});
	read_slice(impl_->values, H5T_NATIVE_FLOAT, batch.values.data_ptr<float>(),
		{static_cast<hsize_t>(begin)}, {static_cast<hsize_t>(count)});
	std::size_t cursor = 0;
	auto *move_data = batch.moves.data_ptr<std::int64_t>();
	for (std::int64_t row = 0; row < count; ++row) {
		for (auto &perspective : batch.features[static_cast<std::size_t>(row)].perspective)
			for (auto &feature : perspective)
				feature = packed[cursor++];
		batch.features[static_cast<std::size_t>(row)].white_to_move = true;
		move_data[row] = packed_moves[static_cast<std::size_t>(row)];
	}
	return batch;
}

void preprocess_pgn(const PreprocessOptions &options) {
	if (options.has_comments != 0 && options.has_comments != 1)
		throw std::invalid_argument("has_comments must be 0 or 1");
	std::ifstream input(options.input);
	if (!input)
		throw std::runtime_error("PGN not found: " + options.input.string());
	WriterOptions writer_options;
	writer_options.output = options.output;
	writer_options.has_comments = options.has_comments;
	writer_options.chunk_size = options.chunk_size;
	writer_options.compression_level = options.compression_level;
	writer_options.source = options.has_comments ? "pgn_comments" : "pgn_result";
	H5Writer writer(writer_options);
	PreprocessVisitor visitor(options, writer);
	std::cout << "Eleginus preprocess start: input=" << options.input.string()
			  << " output=" << options.output.string()
			  << " has_cmt=" << options.has_comments << std::endl;
	try {
		chess::pgn::StreamParser parser(input);
		const auto error = parser.readGames(visitor);
		if (error.hasError() && error.code() != chess::pgn::StreamParserError::NotEnoughData)
			throw std::runtime_error("PGN parse failed: " + error.message());
	} catch (const StopPgnParsing &) {
	}
	writer.flush();
	std::cout << "Eleginus preprocess summary: games=" << visitor.games()
			  << " positions=" << writer.size()
			  << " skipped_moves=" << visitor.skipped_moves()
			  << " skipped_games=" << visitor.skipped_games()
			  << " output=" << options.output.string() << std::endl;
}

TrainStats train_from_h5(Model &model, const TrainOptions &options,
						 const torch::Device &device) {
	if (!model)
		throw std::invalid_argument("cannot train an empty Eleginus model");
	if (options.data.empty() || options.epochs <= 0 || options.batch_size <= 0 ||
		options.max_steps < 0 || options.learning_rate <= 0.0 || options.weight_decay < 0.0)
		throw std::invalid_argument("invalid Eleginus training options");
	H5Dataset data(options.data);
	if (data.info().length <= 0)
		throw std::runtime_error("Eleginus supervised dataset is empty: " + options.data.string());
	torch::manual_seed(static_cast<std::int64_t>(options.seed));
	std::mt19937_64 rng(options.seed);
	torch::optim::AdamW policy_optimizer(model->policy->parameters(),
		torch::optim::AdamWOptions(options.learning_rate).weight_decay(options.weight_decay));
	torch::optim::AdamW value_optimizer(model->value->parameters(),
		torch::optim::AdamWOptions(options.learning_rate).weight_decay(options.weight_decay));
	model->train();
	TrainStats stats;
	const std::int64_t chunk_rows =
		std::max<std::int64_t>(4096, static_cast<std::int64_t>(options.batch_size) * 16);
	std::vector<std::int64_t> chunks;
	for (std::int64_t begin = 0; begin < data.info().length; begin += chunk_rows)
		chunks.push_back(begin);
	bool stop = false;
	for (int epoch = 0; epoch < options.epochs && !stop; ++epoch) {
		std::shuffle(chunks.begin(), chunks.end(), rng);
		for (const auto chunk_begin : chunks) {
			const auto chunk_count = std::min(chunk_rows, data.info().length - chunk_begin);
			auto chunk = data.read_contiguous(chunk_begin, chunk_count);
			std::vector<std::int64_t> order(static_cast<std::size_t>(chunk_count));
			std::iota(order.begin(), order.end(), 0);
			std::shuffle(order.begin(), order.end(), rng);
			const auto *chunk_targets = chunk.values.data_ptr<float>();
			const auto *chunk_moves = chunk.moves.data_ptr<std::int64_t>();
			for (std::int64_t offset = 0; offset < chunk_count; offset += options.batch_size) {
				if (options.max_steps > 0 && stats.steps >= options.max_steps) {
					stop = true;
					break;
				}
				const auto count = std::min<std::int64_t>(options.batch_size, chunk_count - offset);
				std::vector<EncodedFeatures> encoded;
				encoded.reserve(static_cast<std::size_t>(count));
				auto value_targets =
					torch::empty({count, 1}, torch::TensorOptions().dtype(torch::kFloat32));
				auto move_targets =
					torch::empty({count}, torch::TensorOptions().dtype(torch::kInt64));
				auto *value_data = value_targets.data_ptr<float>();
				auto *move_data = move_targets.data_ptr<std::int64_t>();
				for (std::int64_t row = 0; row < count; ++row) {
					const auto source = order[static_cast<std::size_t>(offset + row)];
					encoded.push_back(chunk.features[static_cast<std::size_t>(source)]);
					value_data[row] = chunk_targets[source];
					move_data[row] = chunk_moves[source];
				}
				auto [features, side] = encode_feature_batch(encoded, device);
				value_targets = value_targets.to(device);
				move_targets = move_targets.to(device);
				policy_optimizer.zero_grad();
				value_optimizer.zero_grad();
				auto logits = model->policy->forward(features, side);
				auto value_prediction = model->value->forward(features, side);
				auto policy_loss = torch::nn::functional::cross_entropy(logits, move_targets);
				auto value_loss = torch::mse_loss(value_prediction, value_targets);
				auto loss = policy_loss + value_loss;
				loss.backward();
				torch::nn::utils::clip_grad_norm_(model->policy->parameters(), 1.0);
				torch::nn::utils::clip_grad_norm_(model->value->parameters(), 1.0);
				policy_optimizer.step();
				value_optimizer.step();
				const double policy_loss_value = policy_loss.item<double>();
				const double value_loss_value = value_loss.item<double>();
				const double loss_value = loss.item<double>();
				++stats.steps;
				stats.samples += count;
				stats.mean_policy_loss += policy_loss_value;
				stats.mean_value_loss += value_loss_value;
				stats.mean_loss += loss_value;
				if (options.log_every > 0 && stats.steps % options.log_every == 0)
					std::cout << "Eleginus training: epoch=" << (epoch + 1)
							  << " step=" << stats.steps
							  << " policy=" << policy_loss_value
							  << " value=" << value_loss_value
							  << " loss=" << loss_value << std::endl;
			}
			if (stop)
				break;
		}
	}
	model->eval();
	if (stats.steps > 0) {
		stats.mean_policy_loss /= static_cast<double>(stats.steps);
		stats.mean_value_loss /= static_cast<double>(stats.steps);
		stats.mean_loss /= static_cast<double>(stats.steps);
	}
	return stats;
}

} // namespace eleginus
