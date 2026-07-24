// Implements Melano PGN parsing, schema-checked HDF5 I/O, and fresh P/V/A training.

#include "melano/dataset.hpp"
#include <hdf5.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <regex>
#include <stdexcept>
#include <utility>
#include <torch/optim.h>
#include "melano/checkpoint.hpp"

namespace melano {

namespace {

// Turn a negative HDF5 status into an operation-specific C++ exception.
void require_h5(herr_t status, const std::string &operation) {
	if (status < 0) {
		throw std::runtime_error("HDF5 operation failed: " + operation);
	}
}

// Validate an HDF5 handle before it can leak into later API calls.
hid_t require_id(hid_t id, const std::string &operation) {
	if (id < 0) {
		throw std::runtime_error("HDF5 operation failed: " + operation);
	}
	return id;
}

// Persist a null-terminated string schema attribute.
void write_string_attribute(hid_t object, const char *name, const std::string &value) {
	const hid_t space = require_id(H5Screate(H5S_SCALAR), "create attribute space");
	const hid_t type = require_id(H5Tcopy(H5T_C_S1), "copy string type");
	require_h5(H5Tset_size(type, value.size() + 1), "set string attribute size");
	const hid_t attribute =
		require_id(H5Acreate2(object, name, type, space, H5P_DEFAULT, H5P_DEFAULT), name);
	require_h5(H5Awrite(attribute, type, value.c_str()), name);
	H5Aclose(attribute);
	H5Tclose(type);
	H5Sclose(space);
}

// Persist a portable little-endian int64 metadata attribute.
void write_int_attribute(hid_t object, const char *name, std::int64_t value) {
	const hid_t space = require_id(H5Screate(H5S_SCALAR), "create attribute space");
	const hid_t attribute =
		require_id(H5Acreate2(object, name, H5T_STD_I64LE, space, H5P_DEFAULT, H5P_DEFAULT), name);
	require_h5(H5Awrite(attribute, H5T_NATIVE_INT64, &value), name);
	H5Aclose(attribute);
	H5Sclose(space);
}

// Read a required string attribute used for architecture/schema validation.
std::string read_string_attribute(hid_t object, const char *name) {
	if (H5Aexists(object, name) <= 0) {
		throw std::runtime_error(std::string("HDF5 missing required attribute: ") + name);
	}
	const hid_t attribute = require_id(H5Aopen(object, name, H5P_DEFAULT), name);
	const hid_t type = require_id(H5Aget_type(attribute), name);
	const auto size = H5Tget_size(type);
	std::vector<char> buffer(size + 1, '\0');
	require_h5(H5Aread(attribute, type, buffer.data()), name);
	H5Tclose(type);
	H5Aclose(attribute);
	return std::string(buffer.data());
}

// Read a required integer attribute used for row counts and comment mode.
std::int64_t read_int_attribute(hid_t object, const char *name) {
	if (H5Aexists(object, name) <= 0) {
		throw std::runtime_error(std::string("HDF5 missing required attribute: ") + name);
	}
	std::int64_t value = 0;
	const hid_t attribute = require_id(H5Aopen(object, name, H5P_DEFAULT), name);
	require_h5(H5Aread(attribute, H5T_NATIVE_INT64, &value), name);
	H5Aclose(attribute);
	return value;
}

class H5Writer {
	public:
	// Create a fresh Melano file with aligned state, policy, value, and advantage rows.
	explicit H5Writer(const PreprocessOptions &options) : options_(options) {
		if (!options.output.parent_path().empty()) {
			std::filesystem::create_directories(options.output.parent_path());
		}
		file_ = require_id(
			H5Fcreate(options.output.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT),
			"create output file");
		write_string_attribute(file_, "arch_type", kArchType);
		write_string_attribute(file_, "state_encoding", kStateEncoding);
		write_string_attribute(file_, "move_encoding", kMoveEncoding);
		write_string_attribute(file_, "target_schema", kTargetSchema);
		write_string_attribute(file_, "value_perspective", "side_to_move");
		write_int_attribute(file_, "has_cmt", options.has_comments);
		if (options.has_comments) {
			write_string_attribute(file_, "comment_eval_perspective", "white");
			write_string_attribute(file_, "comment_value_transform",
								   "tanh(side_to_move_pawn_score/3)");
		}
		states_ = create_dataset(
			"states", {0, kStateFeatures}, {H5S_UNLIMITED, kStateFeatures},
			{static_cast<hsize_t>(std::max(1, options.chunk_size)), kStateFeatures}, H5T_STD_U8LE);
		next_states_ = create_dataset(
			"next_states", {0, kStateFeatures}, {H5S_UNLIMITED, kStateFeatures},
			{static_cast<hsize_t>(std::max(1, options.chunk_size)), kStateFeatures}, H5T_STD_U8LE);
		moves_ =
			create_dataset("moves", {0}, {H5S_UNLIMITED},
						   {static_cast<hsize_t>(std::max(1, options.chunk_size))}, H5T_STD_U16LE);
		values_ =
			create_dataset("values", {0}, {H5S_UNLIMITED},
						   {static_cast<hsize_t>(std::max(1, options.chunk_size))}, H5T_IEEE_F32LE);
		next_values_ =
			create_dataset("next_values", {0}, {H5S_UNLIMITED},
						   {static_cast<hsize_t>(std::max(1, options.chunk_size))}, H5T_IEEE_F32LE);
		advantage_moves_ =
			create_dataset("adv_moves", {0}, {H5S_UNLIMITED},
						   {static_cast<hsize_t>(std::max(1, options.chunk_size))}, H5T_STD_U16LE);
		advantage_values_ =
			create_dataset("adv_values", {0}, {H5S_UNLIMITED},
						   {static_cast<hsize_t>(std::max(1, options.chunk_size))}, H5T_IEEE_F32LE);
	}

	// Close datasets before their owning HDF5 file.
	~H5Writer() {
		if (states_ >= 0)
			H5Dclose(states_);
		if (next_states_ >= 0)
			H5Dclose(next_states_);
		if (moves_ >= 0)
			H5Dclose(moves_);
		if (values_ >= 0)
			H5Dclose(values_);
		if (next_values_ >= 0)
			H5Dclose(next_values_);
		if (advantage_moves_ >= 0)
			H5Dclose(advantage_moves_);
		if (advantage_values_ >= 0)
			H5Dclose(advantage_values_);
		if (file_ >= 0)
			H5Fclose(file_);
	}

	// Append one aligned block of P/V/A supervision and encoded token states.
	void append(const std::vector<PackedState> &states,
				const std::vector<PackedState> &next_states,
				const std::vector<std::uint16_t> &moves,
				const std::vector<float> &values,
				const std::vector<float> &next_values,
				const std::vector<std::uint16_t> &advantage_moves,
				const std::vector<float> &advantage_values) {
		if (states.empty())
			return;
		if (states.size() != next_states.size() || states.size() != moves.size() ||
			states.size() != values.size() || states.size() != next_values.size() ||
			states.size() != advantage_moves.size() || states.size() != advantage_values.size()) {
			throw std::runtime_error("preprocess buffers have mismatched lengths");
		}
		const hsize_t count = states.size();
		const hsize_t old = size_;
		const hsize_t next = old + count;
		extend(states_, {next, kStateFeatures});
		extend(next_states_, {next, kStateFeatures});
		extend(moves_, {next});
		extend(values_, {next});
		extend(next_values_, {next});
		extend(advantage_moves_, {next});
		extend(advantage_values_, {next});

		write_slice(states_, H5T_NATIVE_UINT8, states.data(), {old, 0}, {count, kStateFeatures});
		write_slice(next_states_, H5T_NATIVE_UINT8, next_states.data(), {old, 0},
					{count, kStateFeatures});
		write_slice(moves_, H5T_NATIVE_UINT16, moves.data(), {old}, {count});
		write_slice(values_, H5T_NATIVE_FLOAT, values.data(), {old}, {count});
		write_slice(next_values_, H5T_NATIVE_FLOAT, next_values.data(), {old}, {count});
		write_slice(advantage_moves_, H5T_NATIVE_UINT16, advantage_moves.data(), {old}, {count});
		write_slice(advantage_values_, H5T_NATIVE_FLOAT, advantage_values.data(), {old}, {count});
		size_ = next;
	}

	// Record final counters and flush all HDF5 buffers to disk.
	void finish(std::int64_t games, std::int64_t skipped_moves, std::int64_t skipped_games) {
		write_int_attribute(file_, "games", games);
		write_int_attribute(file_, "positions", static_cast<std::int64_t>(size_));
		write_int_attribute(file_, "skipped_moves", skipped_moves);
		write_int_attribute(file_, "skipped_games_no_cmt", skipped_games);
		require_h5(H5Fflush(file_, H5F_SCOPE_GLOBAL), "flush output file");
	}

	// Return the number of aligned position rows written so far.
	std::int64_t size() const { return static_cast<std::int64_t>(size_); }

	private:
	// Create an unlimited chunked dataset with optional shuffle+deflate compression.
	hid_t create_dataset(const char *name, const std::vector<hsize_t> &initial,
						 const std::vector<hsize_t> &maximum, const std::vector<hsize_t> &chunk,
						 hid_t type) {
		const hid_t space = require_id(
			H5Screate_simple(static_cast<int>(initial.size()), initial.data(), maximum.data()),
			name);
		const hid_t properties = require_id(H5Pcreate(H5P_DATASET_CREATE), name);
		require_h5(H5Pset_chunk(properties, static_cast<int>(chunk.size()), chunk.data()), name);
		if (options_.compression_level > 0) {
			require_h5(H5Pset_shuffle(properties), "enable shuffle filter");
			require_h5(H5Pset_deflate(properties, options_.compression_level),
					   "enable gzip filter");
		}
		const hid_t dataset = require_id(
			H5Dcreate2(file_, name, type, space, H5P_DEFAULT, properties, H5P_DEFAULT), name);
		H5Pclose(properties);
		H5Sclose(space);
		return dataset;
	}

	// Grow an extensible dataset to the supplied absolute shape.
	static void extend(hid_t dataset, const std::vector<hsize_t> &dimensions) {
		require_h5(H5Dset_extent(dataset, dimensions.data()), "extend dataset");
	}

	// Write a contiguous memory block into one selected file hyperslab.
	static void write_slice(hid_t dataset, hid_t type, const void *data,
							const std::vector<hsize_t> &start, const std::vector<hsize_t> &count) {
		const hid_t file_space = require_id(H5Dget_space(dataset), "get dataset space");
		require_h5(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start.data(), nullptr,
									   count.data(), nullptr),
				   "select append slice");
		const hid_t memory_space =
			require_id(H5Screate_simple(static_cast<int>(count.size()), count.data(), nullptr),
					   "create append memory space");
		require_h5(H5Dwrite(dataset, type, memory_space, file_space, H5P_DEFAULT, data),
				   "append dataset");
		H5Sclose(memory_space);
		H5Sclose(file_space);
	}

	PreprocessOptions options_;
	hid_t file_ = -1;
	hid_t states_ = -1;
	hid_t next_states_ = -1;
	hid_t moves_ = -1;
	hid_t values_ = -1;
	hid_t next_values_ = -1;
	hid_t advantage_moves_ = -1;
	hid_t advantage_values_ = -1;
	hsize_t size_ = 0;
};

// Parse a CCRL-style white-perspective signed pawn evaluation from a PGN comment.
std::optional<double> comment_score_white(const std::string &comment) {
	static const std::regex score_pattern(
		R"((^|[^A-Za-z0-9_.])([+-](?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))(?:/[0-9]+)?)");
	std::smatch match;
	if (!std::regex_search(comment, match, score_pattern))
		return std::nullopt;
	return std::stod(match[2].str());
}

// Convert white-perspective pawn score to bounded side-to-move V=tanh(score/3).
float comment_value(const std::string &comment, chess::Color turn) {
	const double white = comment_score_white(comment).value_or(0.0);
	const double side = turn == chess::Color::WHITE ? white : -white;
	return static_cast<float>(std::tanh(side / 3.0));
}

// Convert a PGN game result to an exact side-to-move terminal target.
float result_value(const std::string &result, chess::Color turn) {
	float white = 0.0F;
	if (result == "1-0")
		white = 1.0F;
	if (result == "0-1")
		white = -1.0F;
	return turn == chess::Color::WHITE ? white : -white;
}

struct StopPgnParsing {};

class PreprocessVisitor : public chess::pgn::Visitor {
	public:
	// Bind parser callbacks to one Melano writer and option set.
	PreprocessVisitor(const PreprocessOptions &options, H5Writer &writer)
		: options_(options), writer_(writer) {}

	// Reset per-game state and stop cleanly after max_games.
	void startPgn() override {
		if (options_.max_games >= 0 && games_ >= options_.max_games) {
			throw StopPgnParsing{};
		}
		board_ = chess::Board();
		result_ = "*";
		previous_comment_.clear();
		game_has_eval_ = false;
		game_states_.clear();
		game_next_states_.clear();
		game_moves_.clear();
		game_values_.clear();
		game_next_values_.clear();
		game_advantage_moves_.clear();
		game_advantage_values_.clear();
	}

	// Capture Result and optional non-starting FEN headers before move parsing.
	void header(std::string_view key, std::string_view value) override {
		if (key == "Result")
			result_ = std::string(value);
		if (key == "FEN" && !value.empty())
			board_ = chess::Board(value);
	}

	// Satisfy the visitor interface; no setup is needed after headers.
	void startMoves() override {}

	// Encode s and derive A(s,a)=clamp(V_after_same_perspective-V(s), -2, 0).
	void move(std::string_view san, std::string_view comment) override {
		try {
			const auto move = chess::uci::parseSan(board_, san);
			const auto move_index = static_cast<std::uint16_t>(move_to_index(move));
			chess::Board next_board = board_;
			next_board.makeMove(move);
			game_states_.push_back(encode_state(board_));
			game_next_states_.push_back(encode_state(next_board));
			game_moves_.push_back(move_index);
			game_advantage_moves_.push_back(move_index);
			const float before = options_.has_comments
				? comment_value(previous_comment_, board_.sideToMove())
				: result_value(result_, board_.sideToMove());
			game_values_.push_back(before);
			const float next_value = options_.has_comments
				? comment_value(std::string(comment), next_board.sideToMove())
				: result_value(result_, next_board.sideToMove());
			game_next_values_.push_back(next_value);
			float advantage = 0.0F;
			if (options_.has_comments && comment_score_white(previous_comment_).has_value() &&
				comment_score_white(std::string(comment)).has_value()) {
				const float after = comment_value(std::string(comment), board_.sideToMove());
				advantage = std::clamp(after - before, -2.0F, 0.0F);
			}
			game_advantage_values_.push_back(advantage);
			if (comment_score_white(std::string(comment)).has_value())
				game_has_eval_ = true;
			board_ = std::move(next_board);
			previous_comment_ = std::string(comment);
		} catch (const std::exception &) {
			++skipped_moves_;
		}
	}

	// Commit complete games, rejecting comment-required games with no evaluation.
	void endPgn() override {
		if (options_.has_comments && !game_has_eval_) {
			++skipped_games_;
			return;
		}
		writer_.append(game_states_, game_next_states_, game_moves_, game_values_, game_next_values_,
					   game_advantage_moves_, game_advantage_values_);
		++games_;
		if (options_.log_every > 0 && games_ % options_.log_every == 0) {
			std::cout << "preprocess progress: games=" << games_ << " positions=" << writer_.size()
					  << " skipped_moves=" << skipped_moves_
					  << " skipped_games_no_cmt=" << skipped_games_ << std::endl;
		}
	}

	// Return the number of games committed to HDF5.
	std::int64_t games() const { return games_; }
	// Return the number of SAN moves that failed to parse.
	std::int64_t skipped_moves() const { return skipped_moves_; }
	// Return the number of comment-required games rejected without evaluations.
	std::int64_t skipped_games() const { return skipped_games_; }

	private:
	PreprocessOptions options_;
	H5Writer &writer_;
	chess::Board board_;
	std::string result_;
	std::string previous_comment_;
	bool game_has_eval_ = false;
	std::vector<PackedState> game_states_;
	std::vector<PackedState> game_next_states_;
	std::vector<std::uint16_t> game_moves_;
	std::vector<float> game_values_;
	std::vector<float> game_next_values_;
	std::vector<std::uint16_t> game_advantage_moves_;
	std::vector<float> game_advantage_values_;
	std::int64_t games_ = 0;
	std::int64_t skipped_moves_ = 0;
	std::int64_t skipped_games_ = 0;
};

// Build an ordered union of one-row hyperslabs for arbitrary batch indices.
void select_rows(hid_t space, const std::vector<std::int64_t> &indices, int rank) {
	require_h5(H5Sselect_none(space), "clear dataset selection");
	for (const auto index : indices) {
		if (rank == 2) {
			const hsize_t start[] = {static_cast<hsize_t>(index), 0};
			const hsize_t count[] = {1, kStateFeatures};
			require_h5(H5Sselect_hyperslab(space, H5S_SELECT_OR, start, nullptr, count, nullptr),
					   "select state rows");
		} else {
			const hsize_t start[] = {static_cast<hsize_t>(index)};
			const hsize_t count[] = {1};
			require_h5(H5Sselect_hyperslab(space, H5S_SELECT_OR, start, nullptr, count, nullptr),
					   "select scalar rows");
		}
	}
}

// Validate one rank-two state dataset against the architecture's fixed token encoding.
void require_state_shape(hid_t dataset, const char *name, std::int64_t expected_rows) {
	const hid_t space = require_id(H5Dget_space(dataset), std::string("get ") + name + " shape");
	hsize_t dimensions[2]{};
	if (H5Sget_simple_extent_ndims(space) != 2) {
		H5Sclose(space);
		throw std::runtime_error(std::string(name) + " must have rank 2");
	}
	H5Sget_simple_extent_dims(space, dimensions, nullptr);
	H5Sclose(space);
	if (dimensions[1] != kStateFeatures ||
		(expected_rows >= 0 && dimensions[0] != static_cast<hsize_t>(expected_rows))) {
		throw std::runtime_error(std::string(name) + " must have shape [N,67]");
	}
}

// Validate one scalar target dataset and keep every training field row-aligned.
void require_scalar_shape(hid_t dataset, const char *name, std::int64_t expected_rows) {
	const hid_t space = require_id(H5Dget_space(dataset), std::string("get ") + name + " shape");
	hsize_t dimensions[1]{};
	if (H5Sget_simple_extent_ndims(space) != 1) {
		H5Sclose(space);
		throw std::runtime_error(std::string(name) + " must have rank 1");
	}
	H5Sget_simple_extent_dims(space, dimensions, nullptr);
	H5Sclose(space);
	if (dimensions[0] != static_cast<hsize_t>(expected_rows)) {
		throw std::runtime_error(std::string(name) + " must have shape [N]");
	}
}

} // namespace

struct SupervisedH5::Impl {
	// Open required datasets and reject any non-Melano schema before reading data.
	explicit Impl(const std::filesystem::path &path) {
		file = require_id(H5Fopen(path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT),
						  "open supervised data");
		info.arch_type = read_string_attribute(file, "arch_type");
		info.state_encoding = read_string_attribute(file, "state_encoding");
		info.move_encoding = read_string_attribute(file, "move_encoding");
		info.target_schema = read_string_attribute(file, "target_schema");
		info.has_comments = static_cast<int>(read_int_attribute(file, "has_cmt"));
		if (info.arch_type != kArchType || info.state_encoding != kStateEncoding ||
			info.move_encoding != kMoveEncoding || info.target_schema != kTargetSchema) {
			throw std::runtime_error("HDF5 schema does not match the Melano architecture");
		}
		states = require_id(H5Dopen2(file, "states", H5P_DEFAULT), "open states");
		next_states =
			require_id(H5Dopen2(file, "next_states", H5P_DEFAULT), "open next_states");
		moves = require_id(H5Dopen2(file, "moves", H5P_DEFAULT), "open moves");
		values = require_id(H5Dopen2(file, "values", H5P_DEFAULT), "open values");
		next_values =
			require_id(H5Dopen2(file, "next_values", H5P_DEFAULT), "open next_values");
		advantage_moves = require_id(H5Dopen2(file, "adv_moves", H5P_DEFAULT), "open adv_moves");
		advantage_values = require_id(H5Dopen2(file, "adv_values", H5P_DEFAULT), "open adv_values");
		const hid_t space = require_id(H5Dget_space(states), "get states shape");
		hsize_t dimensions[2]{};
		if (H5Sget_simple_extent_ndims(space) != 2) {
			H5Sclose(space);
			throw std::runtime_error("states must have rank 2");
		}
		H5Sget_simple_extent_dims(space, dimensions, nullptr);
		H5Sclose(space);
		if (dimensions[1] != kStateFeatures)
			throw std::runtime_error("states must have shape [N,67]");
		info.length = static_cast<std::int64_t>(dimensions[0]);
		if (info.length <= 0)
			throw std::runtime_error("supervised HDF5 is empty");
		require_state_shape(next_states, "next_states", info.length);
		require_scalar_shape(moves, "moves", info.length);
		require_scalar_shape(values, "values", info.length);
		require_scalar_shape(next_values, "next_values", info.length);
		require_scalar_shape(advantage_moves, "adv_moves", info.length);
		require_scalar_shape(advantage_values, "adv_values", info.length);
	}

	// Close every opened dataset before closing the HDF5 file.
	~Impl() {
		if (states >= 0)
			H5Dclose(states);
		if (next_states >= 0)
			H5Dclose(next_states);
		if (moves >= 0)
			H5Dclose(moves);
		if (values >= 0)
			H5Dclose(values);
		if (next_values >= 0)
			H5Dclose(next_values);
		if (advantage_moves >= 0)
			H5Dclose(advantage_moves);
		if (advantage_values >= 0)
			H5Dclose(advantage_values);
		if (file >= 0)
			H5Fclose(file);
	}

	hid_t file = -1;
	hid_t states = -1;
	hid_t next_states = -1;
	hid_t moves = -1;
	hid_t values = -1;
	hid_t next_values = -1;
	hid_t advantage_moves = -1;
	hid_t advantage_values = -1;
	DatasetInfo info;
};

// Allocate the private HDF5 implementation after successful validation.
SupervisedH5::SupervisedH5(const std::filesystem::path &path) : impl_(new Impl(path)) {}
// Release the owned implementation and all HDF5 handles.
SupervisedH5::~SupervisedH5() { delete impl_; }
// Transfer the pimpl pointer and null the source to preserve single ownership.
SupervisedH5::SupervisedH5(SupervisedH5 &&other) noexcept
	: impl_(std::exchange(other.impl_, nullptr)) {}
// Release current handles before taking ownership from another reader.
SupervisedH5 &SupervisedH5::operator=(SupervisedH5 &&other) noexcept {
	if (this != &other) {
		delete impl_;
		impl_ = std::exchange(other.impl_, nullptr);
	}
	return *this;
}

// Expose validated immutable metadata without another HDF5 call.
const DatasetInfo &SupervisedH5::info() const noexcept { return impl_->info; }

// Read sorted HDF5 rows into owned state, move, value, and advantage tensors.
SupervisedBatch SupervisedH5::read(const std::vector<std::int64_t> &requested) const {
	if (requested.empty())
		throw std::invalid_argument("cannot read an empty HDF5 batch");
	auto indices = requested;
	std::sort(indices.begin(), indices.end());
	for (const auto index : indices) {
		if (index < 0 || index >= impl_->info.length)
			throw std::out_of_range("HDF5 row index");
	}
	const hsize_t batch = indices.size();
	std::vector<std::uint8_t> packed(batch * kStateFeatures);
	std::vector<std::uint8_t> packed_next(batch * kStateFeatures);
	std::vector<std::uint16_t> moves(batch);
	std::vector<float> values(batch);
	std::vector<float> next_values(batch);
	std::vector<std::uint16_t> advantage_moves(batch);
	std::vector<float> advantage_values(batch);

	const hid_t state_space = require_id(H5Dget_space(impl_->states), "get states selection");
	select_rows(state_space, indices, 2);
	const hsize_t state_dims[] = {batch, kStateFeatures};
	const hid_t state_memory = require_id(H5Screate_simple(2, state_dims, nullptr), "state memory");
	require_h5(H5Dread(impl_->states, H5T_NATIVE_UINT8, state_memory, state_space, H5P_DEFAULT,
					   packed.data()),
			   "read state rows");
	H5Sclose(state_memory);
	H5Sclose(state_space);
	const hid_t next_state_space =
		require_id(H5Dget_space(impl_->next_states), "get next_states selection");
	select_rows(next_state_space, indices, 2);
	const hid_t next_state_memory =
		require_id(H5Screate_simple(2, state_dims, nullptr), "next_state memory");
	require_h5(H5Dread(impl_->next_states, H5T_NATIVE_UINT8, next_state_memory,
					  next_state_space, H5P_DEFAULT, packed_next.data()),
			   "read next_state rows");
	H5Sclose(next_state_memory);
	H5Sclose(next_state_space);

	for (const auto &[dataset, type, destination] : {
			 std::tuple<hid_t, hid_t, void *>{impl_->moves, H5T_NATIVE_UINT16, moves.data()},
			 std::tuple<hid_t, hid_t, void *>{impl_->values, H5T_NATIVE_FLOAT, values.data()},
			 std::tuple<hid_t, hid_t, void *>{impl_->next_values, H5T_NATIVE_FLOAT,
												next_values.data()},
			 std::tuple<hid_t, hid_t, void *>{impl_->advantage_moves, H5T_NATIVE_UINT16,
												advantage_moves.data()},
			 std::tuple<hid_t, hid_t, void *>{impl_->advantage_values, H5T_NATIVE_FLOAT,
												advantage_values.data()},
		 }) {
		const hid_t space = require_id(H5Dget_space(dataset), "get scalar selection");
		select_rows(space, indices, 1);
		const hsize_t dimensions[] = {batch};
		const hid_t memory = require_id(H5Screate_simple(1, dimensions, nullptr), "scalar memory");
		require_h5(H5Dread(dataset, type, memory, space, H5P_DEFAULT, destination),
				   "read scalar rows");
		H5Sclose(memory);
		H5Sclose(space);
	}

	return {
		decode_states(packed.data(), static_cast<std::int64_t>(batch)),
		decode_states(packed_next.data(), static_cast<std::int64_t>(batch)),
		torch::from_blob(moves.data(), {static_cast<std::int64_t>(batch)}, torch::kUInt16)
			.clone()
			.to(torch::kInt64),
		torch::from_blob(values.data(), {static_cast<std::int64_t>(batch)}, torch::kFloat32)
			.clone(),
		torch::from_blob(next_values.data(), {static_cast<std::int64_t>(batch)}, torch::kFloat32)
			.clone(),
		torch::from_blob(advantage_moves.data(), {static_cast<std::int64_t>(batch)}, torch::kUInt16)
			.clone()
			.to(torch::kInt64),
		torch::from_blob(advantage_values.data(), {static_cast<std::int64_t>(batch)},
							 torch::kFloat32)
			.clone(),
	};
}

// Stream PGN input through the visitor and finalize a fresh Melano dataset.
void preprocess_pgn(const PreprocessOptions &options) {
	if (options.has_comments != 0 && options.has_comments != 1) {
		throw std::invalid_argument("has_comments must be 0 or 1");
	}
	std::ifstream input(options.input);
	if (!input)
		throw std::runtime_error("PGN not found: " + options.input.string());
	std::cout << "preprocess start: input=" << options.input.string()
			  << " output=" << options.output.string() << " arch_type=" << kArchType
			  << " has_cmt=" << options.has_comments << std::endl;
	H5Writer writer(options);
	PreprocessVisitor visitor(options, writer);
	try {
		chess::pgn::StreamParser parser(input);
		const auto error = parser.readGames(visitor);
		if (error.hasError() && error.code() != chess::pgn::StreamParserError::NotEnoughData) {
			throw std::runtime_error("PGN parse failed: " + error.message());
		}
	} catch (const StopPgnParsing &) {
	}
	writer.finish(visitor.games(), visitor.skipped_moves(), visitor.skipped_games());
	std::cout << "preprocess summary: games=" << visitor.games() << " positions=" << writer.size()
			  << " skipped_moves=" << visitor.skipped_moves()
			  << " skipped_games_no_cmt=" << visitor.skipped_games()
			  << " output=" << options.output.string() << std::endl;
}

// Optimize policy, value, dueling action value, latent transition, and successor value.
void train_supervised(const TrainOptions &options) {
	torch::manual_seed(static_cast<std::int64_t>(options.seed));
	const auto device = resolve_device(options.device);
	SupervisedH5 data(options.data);
	auto model = Model(options.channels, options.blocks);
	model->to(device);
	model->train();
	torch::optim::AdamW optimizer(
		model->parameters(),
		torch::optim::AdamWOptions(options.learning_rate).weight_decay(options.weight_decay));

	std::vector<std::int64_t> order(static_cast<std::size_t>(data.info().length));
	std::iota(order.begin(), order.end(), 0);
	std::mt19937_64 rng(options.seed);
	std::int64_t global_step = 0;
	bool stop = false;
	std::cout << "training start: data=" << options.data.string()
			  << " out=" << options.output.string() << " arch_type=" << kArchType
			  << " device=" << device.str() << " epochs=" << options.epochs
			  << " batch_size=" << options.batch_size << " max_steps=" << options.max_steps
			  << std::endl;
	std::cout << "created model: channels=" << options.channels << " blocks=" << options.blocks
			  << " parameters=" << parameter_count(model) << std::endl;

	for (int epoch = 0; epoch < options.epochs && !stop; ++epoch) {
		std::shuffle(order.begin(), order.end(), rng);
		double policy_total = 0.0;
		double value_total = 0.0;
		double dueling_q_total = 0.0;
		double dynamics_total = 0.0;
		double imagined_value_total = 0.0;
		std::int64_t batches = 0;
		for (std::int64_t begin = 0; begin < data.info().length; begin += options.batch_size) {
			const auto end = std::min<std::int64_t>(begin + options.batch_size, data.info().length);
			std::vector<std::int64_t> indices(order.begin() + begin, order.begin() + end);
			auto batch = data.read(indices);
			auto states = batch.states.to(device, true);
			auto next_states = batch.next_states.to(device, true);
			auto moves = batch.moves.to(device, true);
			auto values = batch.values.to(device, true);
			auto next_values = batch.next_values.to(device, true);
			auto advantage_moves = batch.advantage_moves.to(device, true);
			auto advantage_values = batch.advantage_values.to(device, true);
			optimizer.zero_grad();

			auto tokens = model->encode(states);
			auto output = model->predict(tokens);
			auto policy_loss = torch::nn::functional::cross_entropy(output.policy, moves);
			auto predicted_value = output.value.squeeze(1);
			auto value_loss = torch::mse_loss(predicted_value, values);
			auto selected_advantage = output.advantages.gather(1, advantage_moves.unsqueeze(1)).squeeze(1);
			auto dueling_q_loss = torch::zeros({}, states.options().dtype(torch::kFloat32));
			if (data.info().has_comments) {
				auto predicted_q = torch::clamp(predicted_value + selected_advantage, -1.0, 1.0);
				auto target_q = torch::clamp(values + advantage_values, -1.0, 1.0);
				dueling_q_loss = torch::mse_loss(predicted_q, target_q);
			}
			auto predicted_next = model->transition(tokens, moves);
			torch::Tensor target_next;
			{
				torch::NoGradGuard no_grad;
				target_next = model->encode(next_states).detach();
			}
			// Cosine consistency avoids making latent scale itself a learning target.
			auto predicted_unit = predicted_next /
				predicted_next.square().sum(-1, true).sqrt().clamp_min(1e-8);
			auto target_unit =
				target_next / target_next.square().sum(-1, true).sqrt().clamp_min(1e-8);
			auto dynamics_loss = 1.0 - (predicted_unit * target_unit).sum(-1).mean();
			auto imagined = model->predict(predicted_next);
			auto imagined_value_loss =
				torch::mse_loss(imagined.value.squeeze(1), next_values);
			auto loss = policy_loss + options.value_weight * value_loss +
					options.dueling_q_weight * dueling_q_loss +
					options.dynamics_weight * dynamics_loss +
					options.imagined_value_weight * imagined_value_loss;
			loss.backward();
			optimizer.step();

			++global_step;
			++batches;
			policy_total += policy_loss.item<double>();
			value_total += value_loss.item<double>();
			dueling_q_total += dueling_q_loss.item<double>();
			dynamics_total += dynamics_loss.item<double>();
			imagined_value_total += imagined_value_loss.item<double>();
			if (options.log_every > 0 &&
				(global_step == 1 || global_step % options.log_every == 0)) {
				std::cout << "train step: epoch=" << epoch << " global_step=" << global_step
						  << " policy=" << policy_loss.item<double>()
						  << " value=" << value_loss.item<double>()
						  << " dueling_q=" << dueling_q_loss.item<double>()
						  << " dynamics=" << dynamics_loss.item<double>()
						  << " imagined_value=" << imagined_value_loss.item<double>()
						  << " loss=" << loss.item<double>() << std::endl;
			}
			if (options.save_every > 0 && global_step % options.save_every == 0) {
				save_checkpoint_atomic(options.output, model,
								   {options.channels, options.blocks});
				std::cout << "checkpoint saved: path=" << options.output.string()
						  << " global_step=" << global_step << std::endl;
			}
			if (options.max_steps >= 0 && global_step >= options.max_steps) {
				stop = true;
				break;
			}
		}
		save_checkpoint_atomic(options.output, model, {options.channels, options.blocks});
		std::cout << "epoch=" << epoch << ", steps=" << global_step
				  << ", policy=" << policy_total / std::max<std::int64_t>(1, batches)
				  << ", value=" << value_total / std::max<std::int64_t>(1, batches)
				  << ", dueling_q=" << dueling_q_total / std::max<std::int64_t>(1, batches)
				  << ", dynamics=" << dynamics_total / std::max<std::int64_t>(1, batches)
				  << ", imagined_value="
				  << imagined_value_total / std::max<std::int64_t>(1, batches)
				  << std::endl;
	}
	std::cout << "training finished: " << options.output.string() << std::endl;
}

} // namespace melano
