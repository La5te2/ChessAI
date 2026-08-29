// Implements Melano JSONL preprocessing, schema-checked HDF5 I/O, and Policy/Value training.

#include "melano/dataset.hpp"
#include "melano/checkpoint.hpp"
#include "melano/cuda.hpp"
#include <ATen/ops/_foreach_mul.h>
#include <ATen/ops/_foreach_norm.h>
#include <ATen/ops/index_select.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <future>
#include <hdf5.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
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

// Preserve pinned allocation while applying one shared row permutation to a loaded tensor.
torch::Tensor shuffled_rows(const torch::Tensor &input, const torch::Tensor &permutation, bool pinned_memory) {
	auto options = torch::TensorOptions().dtype(input.scalar_type()).device(torch::kCPU);
	if (pinned_memory) {
		options = options.pinned_memory(true);
	}
	auto output = torch::empty(input.sizes(), options);
	at::index_select_out(output, input, 0, permutation);
	return output;
}

// Clip gradients entirely on their device without materializing the norm on the host.
void clip_gradient_norm_async(const std::vector<torch::Tensor> &parameters, double maximum_norm) {
	std::vector<torch::Tensor> gradients;
	gradients.reserve(parameters.size());
	for (const auto &parameter : parameters) {
		if (parameter.grad().defined()) {
			gradients.push_back(parameter.grad().detach());
		}
	}
	if (gradients.empty()) {
		return;
	}
	const auto norms = at::_foreach_norm(gradients, 2.0);
	const auto total_norm = torch::stack(norms).norm(2.0);
	at::_assert_async(torch::isfinite(total_norm), "The total norm of order 2.0 for gradients is non-finite");
	const auto coefficient = (maximum_norm / (total_norm + 1.0e-6)).clamp_max(1.0);
	torch::NoGradGuard guard;
	at::_foreach_mul_(gradients, coefficient);
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
		write_string_attribute(file_, "value_transform", "tanh(side_to_move_pawn_score/3)");
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

	// Record input-line counters and flush all HDF5 buffers to disk.
	void finish(std::int64_t records, std::int64_t skipped_records) {
		write_int_attribute(file_, "positions", static_cast<std::int64_t>(size_));
		write_int_attribute(file_, "records", records);
		write_int_attribute(file_, "skipped_records", skipped_records);
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

// Convert a White-perspective centipawn score to bounded side-to-move value.
float cp_value(int centipawns, chess::Color turn) {
	const double white = static_cast<double>(centipawns) / 100.0;
	const double side = turn == chess::Color::WHITE ? white : -white;
	return static_cast<float>(std::tanh(side / 3.0));
}

// Convert a White-perspective mate sign to an exact side-to-move endpoint.
float mate_value(int mate, chess::Color turn) {
	const float white = mate > 0 ? 1.0F : -1.0F;
	return turn == chess::Color::WHITE ? white : -white;
}

std::string complete_fen(const std::string &fen) {
	const auto fields = static_cast<int>(std::count(fen.begin(), fen.end(), ' ')) + 1;
	if (fields == 4) {
		return fen + " 0 1";
	}
	if (fields == 6) {
		return fen;
	}
	throw std::invalid_argument("FEN must contain four or six fields");
}

chess::Move parse_pv_move(const chess::Board &board, std::string_view token) {
	for (const auto &move : legal_moves(board)) {
		if (chess::uci::moveToUci(move) == token || chess::uci::moveToUci(move, true) == token) {
			return move;
		}
	}
	throw std::invalid_argument("PV starts with an illegal move: " + std::string(token));
}

struct EvaluationTarget {
	chess::Move move;
	float value = 0.0F;
};

// Select the deepest evaluation, break equal depths by nodes, and use its first PV.
EvaluationTarget evaluation_target(const nlohmann::json &record, const chess::Board &board) {
	if (!record.contains("evals") || !record.at("evals").is_array()) {
		throw std::invalid_argument("record has no evaluation array");
	}
	const nlohmann::json *selected = nullptr;
	int selected_depth = -1;
	std::int64_t selected_knodes = -1;
	for (const auto &evaluation : record.at("evals")) {
		if (!evaluation.is_object() || !evaluation.contains("pvs") || !evaluation.at("pvs").is_array() || evaluation.at("pvs").empty()) {
			continue;
		}
		const int depth = evaluation.value("depth", -1);
		const std::int64_t knodes = evaluation.value("knodes", std::int64_t{-1});
		if (selected == nullptr || depth > selected_depth || (depth == selected_depth && knodes > selected_knodes)) {
			selected = &evaluation;
			selected_depth = depth;
			selected_knodes = knodes;
		}
	}
	if (selected == nullptr) {
		throw std::invalid_argument("record has no principal variation");
	}

	const auto &pv = selected->at("pvs").front();
	if (!pv.is_object() || !pv.contains("line") || !pv.at("line").is_string()) {
		throw std::invalid_argument("selected principal variation has no line");
	}
	const std::string line = pv.at("line").get<std::string>();
	const auto separator = line.find(' ');
	const auto first_move_length = separator == std::string::npos ? line.size() : separator;
	const std::string_view first_move(line.data(), first_move_length);
	if (first_move.empty()) {
		throw std::invalid_argument("selected principal variation is empty");
	}

	float value = 0.0F;
	if (pv.contains("cp") && pv.at("cp").is_number_integer()) {
		value = cp_value(pv.at("cp").get<int>(), board.sideToMove());
	} else if (pv.contains("mate") && pv.at("mate").is_number_integer() && pv.at("mate").get<int>() != 0) {
		value = mate_value(pv.at("mate").get<int>(), board.sideToMove());
	} else {
		throw std::invalid_argument("selected principal variation has no cp or mate score");
	}
	return {parse_pv_move(board, first_move), value};
}

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
			throw std::runtime_error("states must have shape [N,66]");
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
	auto state_options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
	auto move_options = torch::TensorOptions().dtype(torch::kInt16).device(torch::kCPU);
	auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
	if (pinned_memory) {
		state_options = state_options.pinned_memory(true);
		move_options = move_options.pinned_memory(true);
		float_options = float_options.pinned_memory(true);
	}
	auto state_tensor = torch::empty({static_cast<std::int64_t>(batch), kStateFeatures}, state_options);
	auto move_tensor = torch::empty({static_cast<std::int64_t>(batch)}, move_options);
	auto value_tensor = torch::empty({static_cast<std::int64_t>(batch)}, float_options);

	const hid_t state_space = require_id(H5Dget_space(impl_->states), "get states selection");
	select_rows(state_space, indices, 2);
	const hsize_t state_dims[] = {batch, kStateFeatures};
	const hid_t state_memory = require_id(H5Screate_simple(2, state_dims, nullptr), "state memory");
	require_h5(H5Dread(impl_->states, H5T_NATIVE_UINT8, state_memory, state_space, H5P_DEFAULT, state_tensor.data_ptr<std::uint8_t>()), "read state rows");
	H5Sclose(state_memory);
	H5Sclose(state_space);

	for (const auto &[dataset, type, destination] : {
	         std::tuple<hid_t, hid_t, void *>{impl_->moves, H5T_NATIVE_INT16, move_tensor.data_ptr<std::int16_t>()},
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
	return {
	    std::move(state_tensor),
	    std::move(move_tensor),
	    std::move(value_tensor),
	};
}

// Read one physical HDF5 range while preserving state, move, and value alignment.
SupervisedBatch SupervisedH5::read_contiguous(std::int64_t begin, std::int64_t count, bool pinned_memory) const {
	if (begin < 0 || count <= 0 || begin > impl_->info.length - count)
		throw std::out_of_range("contiguous HDF5 row range");
	const hsize_t batch = static_cast<hsize_t>(count);
	auto state_options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
	auto move_options = torch::TensorOptions().dtype(torch::kInt16).device(torch::kCPU);
	auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
	if (pinned_memory) {
		state_options = state_options.pinned_memory(true);
		move_options = move_options.pinned_memory(true);
		float_options = float_options.pinned_memory(true);
	}
	auto state_tensor = torch::empty({count, kStateFeatures}, state_options);
	auto move_tensor = torch::empty({count}, move_options);
	auto value_tensor = torch::empty({count}, float_options);

	const hid_t state_space = require_id(H5Dget_space(impl_->states), "get state range");
	const hsize_t state_start[] = {static_cast<hsize_t>(begin), 0};
	const hsize_t state_count[] = {batch, kStateFeatures};
	require_h5(H5Sselect_hyperslab(state_space, H5S_SELECT_SET, state_start, nullptr, state_count, nullptr), "select state range");
	const hid_t state_memory = require_id(H5Screate_simple(2, state_count, nullptr), "create state range memory");
	require_h5(H5Dread(impl_->states, H5T_NATIVE_UINT8, state_memory, state_space, H5P_DEFAULT, state_tensor.data_ptr<std::uint8_t>()), "read state range");
	H5Sclose(state_memory);
	H5Sclose(state_space);

	for (const auto &[dataset, type, destination] : {
	         std::tuple<hid_t, hid_t, void *>{impl_->moves, H5T_NATIVE_INT16, move_tensor.data_ptr<std::int16_t>()},
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
	return {
	    std::move(state_tensor),
	    std::move(move_tensor),
	    std::move(value_tensor),
	};
}

// Stream position-evaluation JSONL records into the Melano schema.
void preprocess_jsonl(const PreprocessOptions &options) {
	if (options.input != std::filesystem::path{"-"} && options.input.extension() == ".zst") {
		throw std::invalid_argument("compressed JSONL input must be streamed with zstdcat and --input -");
	}

	std::ifstream file;
	std::istream *input = &std::cin;
	if (options.input != std::filesystem::path{"-"}) {
		file.open(options.input);
		if (!file) {
			throw std::runtime_error("evaluation JSONL not found: " + options.input.string());
		}
		input = &file;
	}

	H5Writer writer(options);
	std::vector<PackedState> states;
	std::vector<std::uint16_t> moves;
	std::vector<float> values;
	const auto buffer_capacity = static_cast<std::size_t>(std::max(1, options.chunk_size));
	states.reserve(buffer_capacity);
	moves.reserve(buffer_capacity);
	values.reserve(buffer_capacity);
	auto flush = [&] {
		writer.append(states, moves, values);
		states.clear();
		moves.clear();
		values.clear();
	};

	std::cout << "preprocess start: input=" << options.input.string() << " output=" << options.output.string() << " arch_type=" << kArchType << std::endl;
	std::int64_t records = 0;
	std::int64_t accepted_records = 0;
	std::int64_t skipped_records = 0;
	std::string line;
	while ((options.max_positions < 0 || accepted_records < options.max_positions) && std::getline(*input, line)) {
		++records;
		try {
			if (line.empty()) {
				throw std::invalid_argument("empty record");
			}
			const auto record = nlohmann::json::parse(line);
			if (!record.is_object() || !record.contains("fen") || !record.at("fen").is_string()) {
				throw std::invalid_argument("record has no FEN");
			}
			chess::Board board(complete_fen(record.at("fen").get<std::string>()));
			const auto target = evaluation_target(record, board);
			const auto state = encode_state(board);
			const auto move = static_cast<std::uint16_t>(move_to_index(target.move, board.sideToMove()));
			states.push_back(state);
			moves.push_back(move);
			values.push_back(target.value);
			++accepted_records;
			if (states.size() >= buffer_capacity) {
				flush();
			}
			if (options.log_every > 0 && accepted_records % options.log_every == 0) {
				std::cout << "preprocess progress: positions=" << accepted_records << " skipped=" << skipped_records << std::endl;
			}
		} catch (const std::exception &error) {
			++skipped_records;
			if (skipped_records <= 8) {
				std::cerr << "preprocess skipped record " << records << ": " << error.what() << std::endl;
			}
		}
	}
	flush();
	writer.finish(records, skipped_records);
	std::cout << "preprocess summary: records=" << records << " positions=" << accepted_records << " skipped=" << skipped_records
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
	const auto parameters = model->parameters();
	torch::optim::AdamW optimizer(parameters, torch::optim::AdamWOptions(options.learning_rate).weight_decay(options.weight_decay));
	TrainingGraph training_graph;
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
			chunk.states = shuffled_rows(chunk.states, permutation, device.is_cuda());
			chunk.moves = shuffled_rows(chunk.moves, permutation, device.is_cuda());
			chunk.values = shuffled_rows(chunk.values, permutation, device.is_cuda());

			for (std::int64_t begin = 0; begin < chunk_rows; begin += options.batch_size) {
				const auto count = std::min<std::int64_t>(options.batch_size, chunk_rows - begin);
				auto states = chunk.states.narrow(0, begin, count);
				auto moves = chunk.moves.narrow(0, begin, count);
				auto values = chunk.values.narrow(0, begin, count);
				const double learning_rate = scheduled_learning_rate(options.learning_rate, global_step + 1, warmup_steps);
				set_learning_rate(optimizer, learning_rate);
				auto step = training_graph.run(model, optimizer, states, moves, values, device, options.precision, options.value_weight, count == options.batch_size);
				if (options.grad_clip > 0.0) {
					clip_gradient_norm_async(parameters, options.grad_clip);
				}
				optimizer.step();
				model->center_geometry_templates();

				++global_step;
				if (options.log_every > 0 && (global_step == 1 || global_step % options.log_every == 0)) {
					auto metrics = torch::stack({step.policy_loss.detach(), step.value_bce.detach(), step.loss.detach()}).to(torch::kCPU).contiguous();
					auto metric_values = metrics.accessor<float, 1>();
					std::cout << "train step: epoch=" << epoch << " global_step=" << global_step << " policy=" << metric_values[0] << " value=" << metric_values[1]
					          << " loss=" << metric_values[2] << " lr=" << learning_rate << std::endl;
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
	}
	std::cout << "training finished: " << options.output.string() << std::endl;
}

} // namespace melano
