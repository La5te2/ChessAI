// Implements Melano PGN parsing, schema-checked HDF5 I/O, and Policy/Value training.

#include "melano/dataset.hpp"
#include "melano/checkpoint.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <future>
#include <hdf5.h>
#include <iostream>
#include <numeric>
#include <random>
#include <regex>
#include <stdexcept>
#include <torch/optim.h>
#include <utility>

namespace melano {

namespace {

// Use the standard Transformer warmup/inverse-square-root schedule without another CLI knob.
double scheduled_learning_rate(double peak_learning_rate, std::int64_t step, std::int64_t warmup_steps) {
	const auto positive_step = std::max<std::int64_t>(1, step);
	const auto positive_warmup = std::max<std::int64_t>(1, warmup_steps);
	if (positive_step <= positive_warmup) {
		return peak_learning_rate * static_cast<double>(positive_step) / static_cast<double>(positive_warmup);
	}
	return peak_learning_rate * std::sqrt(static_cast<double>(positive_warmup) / static_cast<double>(positive_step));
}

// Apply the current scheduled rate to every AdamW parameter group.
void set_learning_rate(torch::optim::AdamW &optimizer, double learning_rate) {
	for (auto &group : optimizer.param_groups()) {
		static_cast<torch::optim::AdamWOptions &>(group.options()).lr(learning_rate);
	}
}

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
	const hid_t attribute = require_id(H5Acreate2(object, name, type, space, H5P_DEFAULT, H5P_DEFAULT), name);
	require_h5(H5Awrite(attribute, type, value.c_str()), name);
	H5Aclose(attribute);
	H5Tclose(type);
	H5Sclose(space);
}

// Persist a portable little-endian int64 metadata attribute.
void write_int_attribute(hid_t object, const char *name, std::int64_t value) {
	const hid_t space = require_id(H5Screate(H5S_SCALAR), "create attribute space");
	const hid_t attribute = require_id(H5Acreate2(object, name, H5T_STD_I64LE, space, H5P_DEFAULT, H5P_DEFAULT), name);
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
	// Create a fresh Melano file with aligned state, policy, and value rows.
	explicit H5Writer(const PreprocessOptions &options) : options_(options) {
		if (!options.output.parent_path().empty()) {
			std::filesystem::create_directories(options.output.parent_path());
		}
		file_ = require_id(H5Fcreate(options.output.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), "create output file");
		write_string_attribute(file_, "arch_type", kArchType);
		write_string_attribute(file_, "state_encoding", kStateEncoding);
		write_string_attribute(file_, "move_encoding", kMoveEncoding);
		write_string_attribute(file_, "target_schema", kTargetSchema);
		write_string_attribute(file_, "value_perspective", "side_to_move");
		write_int_attribute(file_, "has_cmt", options.has_comments);
		if (options.has_comments) {
			write_string_attribute(file_, "comment_eval_perspective", "white");
			write_string_attribute(file_, "comment_value_transform", "tanh(side_to_move_pawn_score/3)");
		}
		states_ =
		    create_dataset("states", {0, kStateFeatures}, {H5S_UNLIMITED, kStateFeatures}, {static_cast<hsize_t>(std::max(1, options.chunk_size)), kStateFeatures}, H5T_STD_U8LE);
		moves_ = create_dataset("moves", {0}, {H5S_UNLIMITED}, {static_cast<hsize_t>(std::max(1, options.chunk_size))}, H5T_STD_U16LE);
		values_ = create_dataset("values", {0}, {H5S_UNLIMITED}, {static_cast<hsize_t>(std::max(1, options.chunk_size))}, H5T_IEEE_F32LE);
	}

	// Close datasets before their owning HDF5 file.
	~H5Writer() {
		if (states_ >= 0)
			H5Dclose(states_);
		if (moves_ >= 0)
			H5Dclose(moves_);
		if (values_ >= 0)
			H5Dclose(values_);
		if (file_ >= 0)
			H5Fclose(file_);
	}

	// Append one aligned block of encoded states, policy actions, and values.
	void append(const std::vector<PackedState> &states, const std::vector<std::uint16_t> &moves, const std::vector<float> &values) {
		if (states.empty())
			return;
		if (states.size() != moves.size() || states.size() != values.size()) {
			throw std::runtime_error("preprocess buffers have mismatched lengths");
		}
		const hsize_t count = states.size();
		const hsize_t old = size_;
		const hsize_t next = old + count;
		extend(states_, {next, kStateFeatures});
		extend(moves_, {next});
		extend(values_, {next});

		write_slice(states_, H5T_NATIVE_UINT8, states.data(), {old, 0}, {count, kStateFeatures});
		write_slice(moves_, H5T_NATIVE_UINT16, moves.data(), {old}, {count});
		write_slice(values_, H5T_NATIVE_FLOAT, values.data(), {old}, {count});
		size_ = next;
	}

	// Record final counters and flush all HDF5 buffers to disk.
	void finish(std::int64_t games, std::int64_t skipped_games) {
		write_int_attribute(file_, "games", games);
		write_int_attribute(file_, "positions", static_cast<std::int64_t>(size_));
		write_int_attribute(file_, "skipped_games", skipped_games);
		require_h5(H5Fflush(file_, H5F_SCOPE_GLOBAL), "flush output file");
	}

	// Return the number of aligned position rows written so far.
	std::int64_t size() const { return static_cast<std::int64_t>(size_); }

private:
	// Create an unlimited chunked dataset with optional shuffle+deflate compression.
	hid_t create_dataset(const char *name, const std::vector<hsize_t> &initial, const std::vector<hsize_t> &maximum, const std::vector<hsize_t> &chunk, hid_t type) {
		const hid_t space = require_id(H5Screate_simple(static_cast<int>(initial.size()), initial.data(), maximum.data()), name);
		const hid_t properties = require_id(H5Pcreate(H5P_DATASET_CREATE), name);
		require_h5(H5Pset_chunk(properties, static_cast<int>(chunk.size()), chunk.data()), name);
		if (options_.compression_level > 0) {
			require_h5(H5Pset_shuffle(properties), "enable shuffle filter");
			require_h5(H5Pset_deflate(properties, options_.compression_level), "enable gzip filter");
		}
		const hid_t dataset = require_id(H5Dcreate2(file_, name, type, space, H5P_DEFAULT, properties, H5P_DEFAULT), name);
		H5Pclose(properties);
		H5Sclose(space);
		return dataset;
	}

	// Grow an extensible dataset to the supplied absolute shape.
	static void extend(hid_t dataset, const std::vector<hsize_t> &dimensions) { require_h5(H5Dset_extent(dataset, dimensions.data()), "extend dataset"); }

	// Write a contiguous memory block into one selected file hyperslab.
	static void write_slice(hid_t dataset, hid_t type, const void *data, const std::vector<hsize_t> &start, const std::vector<hsize_t> &count) {
		const hid_t file_space = require_id(H5Dget_space(dataset), "get dataset space");
		require_h5(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start.data(), nullptr, count.data(), nullptr), "select append slice");
		const hid_t memory_space = require_id(H5Screate_simple(static_cast<int>(count.size()), count.data(), nullptr), "create append memory space");
		require_h5(H5Dwrite(dataset, type, memory_space, file_space, H5P_DEFAULT, data), "append dataset");
		H5Sclose(memory_space);
		H5Sclose(file_space);
	}

	PreprocessOptions options_;
	hid_t file_ = -1;
	hid_t states_ = -1;
	hid_t moves_ = -1;
	hid_t values_ = -1;
	hsize_t size_ = 0;
};

// Parse a CCRL-style white-perspective signed pawn evaluation from a PGN comment.
std::optional<double> comment_score_white(const std::string &comment) {
	static const std::regex score_pattern(R"((^|[^A-Za-z0-9_.])([+-](?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))(?:/[0-9]+)?)");
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

// Recognize numeric movetext metadata after the PGN parser has removed leading digits.
bool is_detached_numeric_metadata(std::string_view token) {
	if (token.empty())
		return false;
	bool has_separator = false;
	for (const char character : token) {
		if (character == ':' || character == ',') {
			has_separator = true;
			continue;
		}
		if (character < '0' || character > '9')
			return false;
	}
	return has_separator;
}

class PreprocessVisitor : public chess::pgn::Visitor {
public:
	// Bind parser callbacks to one Melano writer and option set.
	PreprocessVisitor(const PreprocessOptions &options, H5Writer &writer) : options_(options), writer_(writer) {}

	// Reset per-game state and stop cleanly after max_games.
	void startPgn() override {
		if (options_.max_games >= 0 && games_ >= options_.max_games) {
			throw StopPgnParsing{};
		}
		board_ = chess::Board();
		result_ = "*";
		previous_comment_.clear();
		game_has_eval_ = false;
		game_invalid_ = false;
		game_states_.clear();
		game_moves_.clear();
		game_values_.clear();
		detached_comment_.clear();
		collecting_detached_comment_ = false;
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

	// Encode the pre-move state and attach its policy action and side-to-move value target.
	void move(std::string_view san, std::string_view comment) override {
		if (san.empty()) {
			if (options_.has_comments)
				attach_detached_comment(comment);
			return;
		}
		if (consume_detached_comment(san))
			return;
		if (is_detached_numeric_metadata(san)) {
			if (options_.has_comments && !comment.empty())
				attach_detached_comment(comment);
			return;
		}
		if (game_invalid_)
			return;
		try {
			const auto move = chess::uci::parseSan(board_, san);
			const auto move_index = static_cast<std::uint16_t>(move_to_index(move));
			game_states_.push_back(encode_state(board_));
			game_moves_.push_back(move_index);
			const float value = options_.has_comments ? comment_value(previous_comment_, board_.sideToMove()) : result_value(result_, board_.sideToMove());
			game_values_.push_back(value);
			if (comment_score_white(std::string(comment)).has_value())
				game_has_eval_ = true;
			board_.makeMove(move);
			previous_comment_ = std::string(comment);
		} catch (const std::exception &error) {
			game_invalid_ = true;
			if (parse_warning_count_ < 8) {
				std::cerr << "Melano preprocess rejected game after SAN '" << san << "': " << error.what() << std::endl;
				++parse_warning_count_;
			}
		}
	}

	// Commit complete games whose SAN sequence and requested target source are valid.
	void endPgn() override {
		if (game_invalid_ || (options_.has_comments && !game_has_eval_)) {
			++skipped_games_;
			return;
		}
		writer_.append(game_states_, game_moves_, game_values_);
		++games_;
		if (options_.log_every > 0 && games_ % options_.log_every == 0) {
			std::cout << "preprocess progress: games=" << games_ << " positions=" << writer_.size() << " skipped=" << skipped_games_ << std::endl;
		}
	}

	// Return the number of games committed to HDF5.
	std::int64_t games() const { return games_; }
	// Return the number of games rejected for invalid SAN or a missing requested target.
	std::int64_t skipped_games() const { return skipped_games_; }

private:
	// Attach a parser-separated comment to the state reached by the preceding move.
	void attach_detached_comment(std::string_view comment) {
		previous_comment_ = std::string(comment);
		if (comment_score_white(previous_comment_).has_value())
			game_has_eval_ = true;
	}

	// Consume a brace comment that the PGN parser exposed through its SAN callback.
	bool consume_detached_comment(std::string_view token) {
		if (!collecting_detached_comment_ && (token.empty() || token.front() != '{'))
			return false;
		if (!detached_comment_.empty())
			detached_comment_.push_back(' ');
		detached_comment_.append(token);
		collecting_detached_comment_ = detached_comment_.find('}') == std::string::npos;
		if (collecting_detached_comment_)
			return true;

		const auto open = detached_comment_.find('{');
		const auto close = detached_comment_.rfind('}');
		attach_detached_comment(std::string_view(detached_comment_).substr(open + 1, close - open - 1));
		detached_comment_.clear();
		return true;
	}

	PreprocessOptions options_;
	H5Writer &writer_;
	chess::Board board_;
	std::string result_;
	std::string previous_comment_;
	bool game_has_eval_ = false;
	bool game_invalid_ = false;
	std::vector<PackedState> game_states_;
	std::vector<std::uint16_t> game_moves_;
	std::vector<float> game_values_;
	std::string detached_comment_;
	bool collecting_detached_comment_ = false;
	std::int64_t games_ = 0;
	std::int64_t skipped_games_ = 0;
	int parse_warning_count_ = 0;
};

// Build an ordered union of one-row hyperslabs for arbitrary batch indices.
void select_rows(hid_t space, const std::vector<std::int64_t> &indices, int rank) {
	require_h5(H5Sselect_none(space), "clear dataset selection");
	for (const auto index : indices) {
		if (rank == 2) {
			const hsize_t start[] = {static_cast<hsize_t>(index), 0};
			const hsize_t count[] = {1, kStateFeatures};
			require_h5(H5Sselect_hyperslab(space, H5S_SELECT_OR, start, nullptr, count, nullptr), "select state rows");
		} else {
			const hsize_t start[] = {static_cast<hsize_t>(index)};
			const hsize_t count[] = {1};
			require_h5(H5Sselect_hyperslab(space, H5S_SELECT_OR, start, nullptr, count, nullptr), "select scalar rows");
		}
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
		file = require_id(H5Fopen(path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), "open supervised data");
		info.arch_type = read_string_attribute(file, "arch_type");
		info.state_encoding = read_string_attribute(file, "state_encoding");
		info.move_encoding = read_string_attribute(file, "move_encoding");
		info.target_schema = read_string_attribute(file, "target_schema");
		info.has_comments = static_cast<int>(read_int_attribute(file, "has_cmt"));
		if (info.arch_type != kArchType || info.state_encoding != kStateEncoding || info.move_encoding != kMoveEncoding || info.target_schema != kTargetSchema) {
			throw std::runtime_error("HDF5 schema does not match the Melano architecture");
		}
		states = require_id(H5Dopen2(file, "states", H5P_DEFAULT), "open states");
		moves = require_id(H5Dopen2(file, "moves", H5P_DEFAULT), "open moves");
		values = require_id(H5Dopen2(file, "values", H5P_DEFAULT), "open values");
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
		const hid_t properties = require_id(H5Dget_create_plist(states), "get states creation properties");
		if (H5Pget_layout(properties) == H5D_CHUNKED) {
			hsize_t chunk_dimensions[2]{};
			if (H5Pget_chunk(properties, 2, chunk_dimensions) != 2 || chunk_dimensions[0] == 0) {
				H5Pclose(properties);
				throw std::runtime_error("states has an invalid HDF5 chunk shape");
			}
			info.chunk_rows = static_cast<std::int64_t>(chunk_dimensions[0]);
		} else {
			info.chunk_rows = info.length;
		}
		H5Pclose(properties);
		require_scalar_shape(moves, "moves", info.length);
		require_scalar_shape(values, "values", info.length);
	}

	// Close every opened dataset before closing the HDF5 file.
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

// Allocate the private HDF5 implementation after successful validation.
SupervisedH5::SupervisedH5(const std::filesystem::path &path) : impl_(new Impl(path)) {
}
// Release the owned implementation and all HDF5 handles.
SupervisedH5::~SupervisedH5() {
	delete impl_;
}
// Transfer the pimpl pointer and null the source to preserve single ownership.
SupervisedH5::SupervisedH5(SupervisedH5 &&other) noexcept : impl_(std::exchange(other.impl_, nullptr)) {
}
// Release current handles before taking ownership from another reader.
SupervisedH5 &SupervisedH5::operator=(SupervisedH5 &&other) noexcept {
	if (this != &other) {
		delete impl_;
		impl_ = std::exchange(other.impl_, nullptr);
	}
	return *this;
}

// Expose validated immutable metadata without another HDF5 call.
const DatasetInfo &SupervisedH5::info() const noexcept {
	return impl_->info;
}

// Read sorted HDF5 rows into owned, optionally pinned state, move, and value tensors.
SupervisedBatch SupervisedH5::read(const std::vector<std::int64_t> &requested, bool pinned_memory) const {
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
	std::vector<std::uint16_t> moves(batch);
	auto index_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
	auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
	if (pinned_memory) {
		index_options = index_options.pinned_memory(true);
		float_options = float_options.pinned_memory(true);
	}
	auto move_tensor = torch::empty({static_cast<std::int64_t>(batch)}, index_options);
	auto value_tensor = torch::empty({static_cast<std::int64_t>(batch)}, float_options);

	const hid_t state_space = require_id(H5Dget_space(impl_->states), "get states selection");
	select_rows(state_space, indices, 2);
	const hsize_t state_dims[] = {batch, kStateFeatures};
	const hid_t state_memory = require_id(H5Screate_simple(2, state_dims, nullptr), "state memory");
	require_h5(H5Dread(impl_->states, H5T_NATIVE_UINT8, state_memory, state_space, H5P_DEFAULT, packed.data()), "read state rows");
	H5Sclose(state_memory);
	H5Sclose(state_space);

	for (const auto &[dataset, type, destination] : {
	         std::tuple<hid_t, hid_t, void *>{impl_->moves, H5T_NATIVE_UINT16, moves.data()},
	         std::tuple<hid_t, hid_t, void *>{impl_->values, H5T_NATIVE_FLOAT, value_tensor.data_ptr<float>()},
	     }) {
		const hid_t space = require_id(H5Dget_space(dataset), "get scalar selection");
		select_rows(space, indices, 1);
		const hsize_t dimensions[] = {batch};
		const hid_t memory = require_id(H5Screate_simple(1, dimensions, nullptr), "scalar memory");
		require_h5(H5Dread(dataset, type, memory, space, H5P_DEFAULT, destination), "read scalar rows");
		H5Sclose(memory);
		H5Sclose(space);
	}
	auto *move_destination = move_tensor.data_ptr<std::int64_t>();
	for (std::size_t index = 0; index < moves.size(); ++index) {
		move_destination[index] = moves[index];
	}

	return {
	    decode_states(packed.data(), static_cast<std::int64_t>(batch), pinned_memory),
	    std::move(move_tensor),
	    std::move(value_tensor),
	};
}

// Read one physical HDF5 range while preserving state, move, and value alignment.
SupervisedBatch SupervisedH5::read_contiguous(std::int64_t begin, std::int64_t count, bool pinned_memory) const {
	if (begin < 0 || count <= 0 || begin > impl_->info.length - count)
		throw std::out_of_range("contiguous HDF5 row range");
	const hsize_t batch = static_cast<hsize_t>(count);
	std::vector<std::uint8_t> packed(batch * kStateFeatures);
	std::vector<std::uint16_t> moves(batch);
	auto index_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
	auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
	if (pinned_memory) {
		index_options = index_options.pinned_memory(true);
		float_options = float_options.pinned_memory(true);
	}
	auto move_tensor = torch::empty({count}, index_options);
	auto value_tensor = torch::empty({count}, float_options);

	const hid_t state_space = require_id(H5Dget_space(impl_->states), "get state range");
	const hsize_t state_start[] = {static_cast<hsize_t>(begin), 0};
	const hsize_t state_count[] = {batch, kStateFeatures};
	require_h5(H5Sselect_hyperslab(state_space, H5S_SELECT_SET, state_start, nullptr, state_count, nullptr), "select state range");
	const hid_t state_memory = require_id(H5Screate_simple(2, state_count, nullptr), "create state range memory");
	require_h5(H5Dread(impl_->states, H5T_NATIVE_UINT8, state_memory, state_space, H5P_DEFAULT, packed.data()), "read state range");
	H5Sclose(state_memory);
	H5Sclose(state_space);

	for (const auto &[dataset, type, destination] : {
	         std::tuple<hid_t, hid_t, void *>{impl_->moves, H5T_NATIVE_UINT16, moves.data()},
	         std::tuple<hid_t, hid_t, void *>{impl_->values, H5T_NATIVE_FLOAT, value_tensor.data_ptr<float>()},
	     }) {
		const hid_t space = require_id(H5Dget_space(dataset), "get scalar range");
		const hsize_t start[] = {static_cast<hsize_t>(begin)};
		const hsize_t dimensions[] = {batch};
		require_h5(H5Sselect_hyperslab(space, H5S_SELECT_SET, start, nullptr, dimensions, nullptr), "select scalar range");
		const hid_t memory = require_id(H5Screate_simple(1, dimensions, nullptr), "create scalar range memory");
		require_h5(H5Dread(dataset, type, memory, space, H5P_DEFAULT, destination), "read scalar range");
		H5Sclose(memory);
		H5Sclose(space);
	}
	auto *move_destination = move_tensor.data_ptr<std::int64_t>();
	for (std::size_t index = 0; index < moves.size(); ++index) {
		move_destination[index] = moves[index];
	}

	return {
	    decode_states(packed.data(), count, pinned_memory),
	    std::move(move_tensor),
	    std::move(value_tensor),
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
	std::cout << "preprocess start: input=" << options.input.string() << " output=" << options.output.string() << " arch_type=" << kArchType << " has_cmt=" << options.has_comments
	          << std::endl;
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
	writer.finish(visitor.games(), visitor.skipped_games());
	std::cout << "preprocess summary: games=" << visitor.games() << " positions=" << writer.size() << " skipped=" << visitor.skipped_games()
	          << " output=" << options.output.string() << std::endl;
}

// Optimize the exact-state Policy and Value outputs from supervised targets.
void train_supervised(const TrainOptions &options) {
	if (options.channels <= 0 || options.blocks <= 0) {
		throw std::invalid_argument("channels and blocks must be positive");
	}
	if (options.batch_size <= 0 || options.epochs < 0) {
		throw std::invalid_argument("batch-size must be positive and epochs must be nonnegative");
	}
	torch::manual_seed(static_cast<std::int64_t>(options.seed));
	const auto device = resolve_device(options.device);
	validate_compute_precision(options.precision, device);
	SupervisedH5 data(options.data);
	auto model = Model(options.channels, options.blocks);
	model->to(device);
	model->train();
	torch::optim::AdamW optimizer(model->parameters(), torch::optim::AdamWOptions(options.learning_rate).weight_decay(options.weight_decay));
	const auto steps_per_epoch = std::max<std::int64_t>(1, (data.info().length + options.batch_size - 1) / options.batch_size);
	const auto epoch_step_limit = std::max<std::int64_t>(1, static_cast<std::int64_t>(options.epochs) * steps_per_epoch);
	const auto planned_steps = options.max_steps > 0 ? std::min(options.max_steps, epoch_step_limit) : epoch_step_limit;
	const auto warmup_steps = std::min<std::int64_t>(planned_steps, std::min<std::int64_t>(2000, std::max<std::int64_t>(100, planned_steps / 100)));

	std::mt19937_64 rng(options.seed);
	std::int64_t global_step = 0;
	bool stop = false;
	std::cout << "training start: data=" << options.data.string() << " out=" << options.output.string() << " arch_type=" << kArchType << " device=" << device.str()
	          << " epochs=" << options.epochs << " batch_size=" << options.batch_size << " max_steps=" << options.max_steps
	          << " precision=" << compute_precision_name(options.precision) << " grad_clip=" << options.grad_clip << " lr_peak=" << options.learning_rate
	          << " lr_warmup_steps=" << warmup_steps << std::endl;
	std::cout << "created model: channels=" << options.channels << " blocks=" << options.blocks << " parameters=" << parameter_count(model) << std::endl;
	std::cout << "training input: rows=" << data.info().length << " hdf5_chunk_rows=" << data.info().chunk_rows << " loader=chunk_shuffle_prefetch" << std::endl;

	for (int epoch = 0; epoch < options.epochs && !stop; ++epoch) {
		std::vector<std::int64_t> chunk_starts;
		for (std::int64_t begin = 0; begin < data.info().length; begin += data.info().chunk_rows) {
			chunk_starts.push_back(begin);
		}
		std::shuffle(chunk_starts.begin(), chunk_starts.end(), rng);
		auto metric_totals = torch::zeros({2}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
		std::int64_t batches = 0;
		auto launch_read = [&](std::size_t chunk_index) {
			const auto begin = chunk_starts[chunk_index];
			const auto count = std::min(data.info().chunk_rows, data.info().length - begin);
			return std::async(std::launch::async, [&data, begin, count, &device] { return data.read_contiguous(begin, count, device.is_cuda()); });
		};
		auto prefetched = launch_read(0);
		for (std::size_t chunk_index = 0; chunk_index < chunk_starts.size() && !stop; ++chunk_index) {
			auto chunk = prefetched.get();
			if (chunk_index + 1 < chunk_starts.size())
				prefetched = launch_read(chunk_index + 1);

			const auto chunk_rows = chunk.states.size(0);
			std::vector<std::int64_t> local_order(static_cast<std::size_t>(chunk_rows));
			std::iota(local_order.begin(), local_order.end(), 0);
			std::shuffle(local_order.begin(), local_order.end(), rng);
			auto permutation = torch::from_blob(local_order.data(), {chunk_rows}, torch::TensorOptions().dtype(torch::kInt64)).clone();
			chunk.states = chunk.states.index_select(0, permutation);
			chunk.moves = chunk.moves.index_select(0, permutation);
			chunk.values = chunk.values.index_select(0, permutation);

			for (std::int64_t begin = 0; begin < chunk_rows; begin += options.batch_size) {
				const auto count = std::min<std::int64_t>(options.batch_size, chunk_rows - begin);
				auto states = chunk.states.narrow(0, begin, count).to(device, true);
				auto moves = chunk.moves.narrow(0, begin, count).to(device, true);
				auto values = chunk.values.narrow(0, begin, count).to(device, true);
				const double learning_rate = scheduled_learning_rate(options.learning_rate, global_step + 1, warmup_steps);
				set_learning_rate(optimizer, learning_rate);
				optimizer.zero_grad();

				torch::Tensor policy;
				torch::Tensor predicted_value;
				{
					AutocastGuard autocast(options.precision, device);
					std::tie(policy, predicted_value) = model->forward(states);
				}
				auto policy_loss = torch::nn::functional::cross_entropy(policy.to(torch::kFloat32), moves);
				auto value_loss = torch::mse_loss(predicted_value.squeeze(1).to(torch::kFloat32), values);
				auto loss = policy_loss + options.value_weight * value_loss;
				loss.backward();
				double gradient_norm = 0.0;
				if (options.grad_clip > 0.0) {
					gradient_norm = torch::nn::utils::clip_grad_norm_(model->parameters(), options.grad_clip, 2.0, true);
				}
				optimizer.step();

				++global_step;
				++batches;
				metric_totals.add_(torch::stack({policy_loss.detach(), value_loss.detach()}));
				if (options.log_every > 0 && (global_step == 1 || global_step % options.log_every == 0)) {
					auto metrics = torch::stack({policy_loss.detach(), value_loss.detach(), loss.detach()}).to(torch::kCPU).contiguous();
					auto metric_values = metrics.accessor<float, 1>();
					std::cout << "train step: epoch=" << epoch << " global_step=" << global_step << " policy=" << metric_values[0] << " value=" << metric_values[1]
					          << " loss=" << metric_values[2] << " grad_norm_before_clip=" << gradient_norm << " lr=" << learning_rate << std::endl;
				}
				if (options.save_every > 0 && global_step % options.save_every == 0) {
					save_checkpoint_atomic(options.output, model, {options.channels, options.blocks});
					std::cout << "checkpoint saved: path=" << options.output.string() << " global_step=" << global_step << std::endl;
				}
				if (options.max_steps > 0 && global_step >= options.max_steps) {
					stop = true;
					break;
				}
			}
		}
		save_checkpoint_atomic(options.output, model, {options.channels, options.blocks});
		auto epoch_metrics = metric_totals.to(torch::kCPU).contiguous();
		auto epoch_values = epoch_metrics.accessor<float, 1>();
		std::cout << "epoch=" << epoch << ", steps=" << global_step << ", policy=" << epoch_values[0] / std::max<std::int64_t>(1, batches)
		          << ", value=" << epoch_values[1] / std::max<std::int64_t>(1, batches) << std::endl;
	}
	std::cout << "training finished: " << options.output.string() << std::endl;
}

} // namespace melano
