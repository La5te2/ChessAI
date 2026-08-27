// Implements Gadus PGN parsing, schema-checked HDF5 I/O, and supervised training.

#include "gadus/dataset.hpp"
#include "gadus/checkpoint.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <future>
#include <hdf5.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <regex>
#include <stdexcept>
#include <torch/csrc/autograd/autograd.h>
#include <torch/optim.h>
#include <tuple>
#include <utility>

namespace gadus {

namespace {

constexpr std::int64_t kValueWeightProbeInterval = 500;
constexpr double kValueGradientRatio = 0.5;
constexpr double kValueHeadGradientClip = 1.0;
constexpr double kValueWeightSmoothing = 0.08;
constexpr double kMinValueWeight = 0.2;
constexpr double kMaxValueWeight = 2.0;
constexpr double kGradientEpsilon = 1e-12;
constexpr double kConflictMargin = 1e-3;
constexpr std::int64_t kOutputPriorSampleRows = 1 << 20;

struct GradientBalanceStats {
	double policy_squared_norm = 0.0;
	double value_squared_norm = 0.0;
	double inner_product = 0.0;
};

GradientBalanceStats shared_gradient_stats(const torch::Tensor &policy_loss, const torch::Tensor &value_loss, const Model &model) {
	const auto parameters = model->backbone->parameters();
	const auto policy_gradients = torch::autograd::grad({policy_loss}, parameters, {}, true, false, false);
	const auto value_gradients = torch::autograd::grad({value_loss}, parameters, {}, true, false, false);
	auto totals = torch::zeros({3}, torch::TensorOptions().dtype(torch::kFloat64).device(policy_loss.device()));
	for (std::size_t index = 0; index < parameters.size(); ++index) {
		const auto policy = policy_gradients[index].detach().to(torch::kFloat32);
		const auto value = value_gradients[index].detach().to(torch::kFloat32);
		totals.add_(torch::stack({policy.square().sum(), value.square().sum(), (policy * value).sum()}).to(torch::kFloat64));
	}
	totals = totals.to(torch::kCPU).contiguous();
	const auto values = totals.accessor<double, 1>();
	return {values[0], values[1], values[2]};
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
	// Create a fresh Gadus file with extensible, chunked architecture-specific datasets.
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
		const bool lichess_eval = options.source == "lichess-eval";
		write_int_attribute(file_, "has_cmt", lichess_eval ? 1 : options.has_comments);
		if (!lichess_eval && options.has_comments) {
			write_string_attribute(file_, "comment_eval_perspective", "white");
			write_string_attribute(file_, "comment_value_transform", "tanh(side_to_move_pawn_score/3)");
		} else if (lichess_eval) {
			write_string_attribute(file_, "evaluation_source", "lichess_cloud_eval");
			write_string_attribute(file_, "evaluation_perspective", "white");
			write_string_attribute(file_, "evaluation_value_transform", "tanh(side_to_move_centipawn_score/300)");
		}
		states_ = create_dataset(
		    "states", {0, kStatePlanes, 8}, {H5S_UNLIMITED, kStatePlanes, 8}, {static_cast<hsize_t>(std::max(1, options.chunk_size)), kStatePlanes, 8}, H5T_STD_U8LE);
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

	// Append one aligned block of packed states, action ids, and values.
	void append(const std::vector<PackedState> &states, const std::vector<std::uint16_t> &moves, const std::vector<float> &values) {
		if (states.empty())
			return;
		if (states.size() != moves.size() || states.size() != values.size()) {
			throw std::runtime_error("preprocess buffers have mismatched lengths");
		}
		const hsize_t count = states.size();
		const hsize_t old = size_;
		const hsize_t next = old + count;
		extend(states_, {next, kStatePlanes, 8});
		extend(moves_, {next});
		extend(values_, {next});

		write_slice(states_, H5T_NATIVE_UINT8, states.data(), {old, 0, 0}, {count, kStatePlanes, 8});
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

	// Record source-line counters for a position-oriented evaluation export.
	void finish_evaluations(std::int64_t source_records, std::int64_t skipped_records) {
		write_int_attribute(file_, "games", 0);
		write_int_attribute(file_, "positions", static_cast<std::int64_t>(size_));
		write_int_attribute(file_, "source_records", source_records);
		write_int_attribute(file_, "skipped_records", skipped_records);
		write_int_attribute(file_, "skipped_games", 0);
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

// Lichess cloud evaluations use White's point of view; convert cp to the existing target scale.
float lichess_cp_value(int centipawns, chess::Color turn) {
	const double white = static_cast<double>(centipawns) / 100.0;
	const double side = turn == chess::Color::WHITE ? white : -white;
	return static_cast<float>(std::tanh(side / 3.0));
}

// Convert a White-perspective mate sign to an exact side-to-move endpoint.
float lichess_mate_value(int mate, chess::Color turn) {
	const float white = mate > 0 ? 1.0F : -1.0F;
	return turn == chess::Color::WHITE ? white : -white;
}

// Lichess FEN keys omit move counters, which chess-library expects in its full FEN parser.
std::string complete_lichess_fen(const std::string &fen) {
	const auto fields = static_cast<int>(std::count(fen.begin(), fen.end(), ' ')) + 1;
	if (fields == 4)
		return fen + " 0 1";
	if (fields == 6)
		return fen;
	throw std::invalid_argument("FEN must contain four or six fields");
}

// Match both ordinary UCI and UCI_Chess960 notation against the generated legal moves.
chess::Move parse_lichess_pv_move(const chess::Board &board, std::string_view token) {
	for (const auto &move : legal_moves(board)) {
		if (chess::uci::moveToUci(move) == token || chess::uci::moveToUci(move, true) == token)
			return move;
	}
	throw std::invalid_argument("PV starts with an illegal move: " + std::string(token));
}

struct LichessTarget {
	chess::Move move;
	float value = 0.0F;
};

// Select the deepest evaluation, breaking equal depths by searched nodes, then use its first PV.
LichessTarget lichess_target(const nlohmann::json &record, const chess::Board &board) {
	if (!record.contains("evals") || !record.at("evals").is_array())
		throw std::invalid_argument("record has no evaluation array");
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
	if (selected == nullptr)
		throw std::invalid_argument("record has no principal variation");

	const auto &pv = selected->at("pvs").front();
	if (!pv.is_object() || !pv.contains("line") || !pv.at("line").is_string())
		throw std::invalid_argument("selected principal variation has no line");
	const std::string line = pv.at("line").get<std::string>();
	const auto separator = line.find(' ');
	const auto first_move_length = separator == std::string::npos ? line.size() : separator;
	const std::string_view first_move(line.data(), first_move_length);
	if (first_move.empty())
		throw std::invalid_argument("selected principal variation is empty");

	float value = 0.0F;
	if (pv.contains("cp") && pv.at("cp").is_number_integer()) {
		value = lichess_cp_value(pv.at("cp").get<int>(), board.sideToMove());
	} else if (pv.contains("mate") && pv.at("mate").is_number_integer() && pv.at("mate").get<int>() != 0) {
		value = lichess_mate_value(pv.at("mate").get<int>(), board.sideToMove());
	} else {
		throw std::invalid_argument("selected principal variation has no cp or mate score");
	}
	return {parse_lichess_pv_move(board, first_move), value};
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
	// Bind parser callbacks to one Gadus writer and option set.
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

	// Encode the pre-move state and use the previous post-move comment as its V target.
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
			game_states_.push_back(encode_state(board_));
			game_moves_.push_back(static_cast<std::uint16_t>(hdf5_action_index(move)));
			game_values_.push_back(options_.has_comments ? comment_value(previous_comment_, board_.sideToMove()) : result_value(result_, board_.sideToMove()));
			if (comment_score_white(std::string(comment)).has_value())
				game_has_eval_ = true;
			board_.makeMove(move);
			previous_comment_ = std::string(comment);
		} catch (const std::exception &error) {
			game_invalid_ = true;
			if (parse_warning_count_ < 8) {
				std::cerr << "Gadus preprocess rejected game after SAN '" << san << "': " << error.what() << std::endl;
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
		if (rank == 3) {
			const hsize_t start[] = {static_cast<hsize_t>(index), 0, 0};
			const hsize_t count[] = {1, kStatePlanes, 8};
			require_h5(H5Sselect_hyperslab(space, H5S_SELECT_OR, start, nullptr, count, nullptr), "select state rows");
		} else {
			const hsize_t start[] = {static_cast<hsize_t>(index)};
			const hsize_t count[] = {1};
			require_h5(H5Sselect_hyperslab(space, H5S_SELECT_OR, start, nullptr, count, nullptr), "select scalar rows");
		}
	}
}

} // namespace

struct SupervisedH5::Impl {
	// Open required datasets and reject any non-Gadus schema before reading data.
	explicit Impl(const std::filesystem::path &path) {
		file = require_id(H5Fopen(path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), "open supervised data");
		info.arch_type = read_string_attribute(file, "arch_type");
		info.state_encoding = read_string_attribute(file, "state_encoding");
		info.move_encoding = read_string_attribute(file, "move_encoding");
		info.target_schema = read_string_attribute(file, "target_schema");
		info.has_comments = static_cast<int>(read_int_attribute(file, "has_cmt"));
		if (info.arch_type != kArchType || info.state_encoding != kStateEncoding || info.move_encoding != kMoveEncoding || info.target_schema != kTargetSchema) {
			throw std::runtime_error("HDF5 schema does not match the Gadus architecture");
		}
		states = require_id(H5Dopen2(file, "states", H5P_DEFAULT), "open states");
		moves = require_id(H5Dopen2(file, "moves", H5P_DEFAULT), "open moves");
		values = require_id(H5Dopen2(file, "values", H5P_DEFAULT), "open values");
		const hid_t space = require_id(H5Dget_space(states), "get states shape");
		hsize_t dimensions[3]{};
		if (H5Sget_simple_extent_ndims(space) != 3) {
			throw std::runtime_error("states must have rank 3");
		}
		H5Sget_simple_extent_dims(space, dimensions, nullptr);
		H5Sclose(space);
		if (dimensions[1] != kStatePlanes || dimensions[2] != 8) {
			throw std::runtime_error("states must have shape [N,18,8]");
		}
		info.length = static_cast<std::int64_t>(dimensions[0]);
		if (info.length <= 0)
			throw std::runtime_error("supervised HDF5 is empty");
		const hid_t properties = require_id(H5Dget_create_plist(states), "get states creation properties");
		if (H5Pget_layout(properties) == H5D_CHUNKED) {
			hsize_t chunk_dimensions[3]{};
			if (H5Pget_chunk(properties, 3, chunk_dimensions) != 3 || chunk_dimensions[0] == 0) {
				H5Pclose(properties);
				throw std::runtime_error("states has an invalid HDF5 chunk shape");
			}
			info.chunk_rows = static_cast<std::int64_t>(chunk_dimensions[0]);
		} else {
			info.chunk_rows = info.length;
		}
		H5Pclose(properties);
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

// Read sorted HDF5 rows into owned tensors; training is order-invariant within a batch.
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
	std::vector<std::uint16_t> moves(batch);
	auto state_options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
	auto move_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
	auto value_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
	if (pinned_memory) {
		state_options = state_options.pinned_memory(true);
		move_options = move_options.pinned_memory(true);
		value_options = value_options.pinned_memory(true);
	}
	auto state_tensor = torch::empty({static_cast<std::int64_t>(batch), kStatePlanes, 8}, state_options);
	auto move_tensor = torch::empty({static_cast<std::int64_t>(batch)}, move_options);
	auto value_tensor = torch::empty({static_cast<std::int64_t>(batch)}, value_options);

	const hid_t state_space = require_id(H5Dget_space(impl_->states), "get states selection");
	select_rows(state_space, indices, 3);
	const hsize_t state_dims[] = {batch, kStatePlanes, 8};
	const hid_t state_memory = require_id(H5Screate_simple(3, state_dims, nullptr), "state memory");
	require_h5(H5Dread(impl_->states, H5T_NATIVE_UINT8, state_memory, state_space, H5P_DEFAULT, state_tensor.data_ptr<std::uint8_t>()), "read state rows");
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
	const auto *state_data = state_tensor.data_ptr<std::uint8_t>();
	for (std::size_t index = 0; index < moves.size(); ++index) {
		const bool white_to_move = state_data[index * kStatePlanes * 8 + 12 * 8] != 0;
		move_destination[index] = canonical_action_index(moves[index], white_to_move ? chess::Color::WHITE : chess::Color::BLACK);
	}

	return {
	    std::move(state_tensor),
	    std::move(move_tensor),
	    std::move(value_tensor),
	};
}

// Read one physical HDF5 range with three hyperslabs instead of hundreds of random rows.
SupervisedBatch SupervisedH5::read_contiguous(std::int64_t begin, std::int64_t count, bool pinned_memory) const {
	if (begin < 0 || count <= 0 || begin > impl_->info.length - count)
		throw std::out_of_range("contiguous HDF5 row range");
	const hsize_t batch = static_cast<hsize_t>(count);
	std::vector<std::uint16_t> moves(batch);
	auto state_options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
	auto move_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
	auto value_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
	if (pinned_memory) {
		state_options = state_options.pinned_memory(true);
		move_options = move_options.pinned_memory(true);
		value_options = value_options.pinned_memory(true);
	}
	auto state_tensor = torch::empty({count, kStatePlanes, 8}, state_options);
	auto move_tensor = torch::empty({count}, move_options);
	auto value_tensor = torch::empty({count}, value_options);

	const hid_t state_space = require_id(H5Dget_space(impl_->states), "get states range");
	const hsize_t state_start[] = {static_cast<hsize_t>(begin), 0, 0};
	const hsize_t state_count[] = {batch, kStatePlanes, 8};
	require_h5(H5Sselect_hyperslab(state_space, H5S_SELECT_SET, state_start, nullptr, state_count, nullptr), "select states range");
	const hid_t state_memory = require_id(H5Screate_simple(3, state_count, nullptr), "create states range memory");
	require_h5(H5Dread(impl_->states, H5T_NATIVE_UINT8, state_memory, state_space, H5P_DEFAULT, state_tensor.data_ptr<std::uint8_t>()), "read states range");
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
	const auto *state_data = state_tensor.data_ptr<std::uint8_t>();
	for (std::size_t index = 0; index < moves.size(); ++index) {
		const bool white_to_move = state_data[index * kStatePlanes * 8 + 12 * 8] != 0;
		move_destination[index] = canonical_action_index(moves[index], white_to_move ? chess::Color::WHITE : chess::Color::BLACK);
	}

	return {
	    std::move(state_tensor),
	    std::move(move_tensor),
	    std::move(value_tensor),
	};
}

// Stream PGN input through the visitor and finalize a fresh Gadus dataset.
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

// Stream position-oriented Lichess cloud evaluations into the Gadus supervised schema.
void preprocess_lichess_evaluations(const PreprocessOptions &options) {
	if (options.input != std::filesystem::path{"-"} && options.input.extension() == ".zst") {
		throw std::invalid_argument("compressed Lichess input must be streamed with zstdcat and --input -");
	}

	std::ifstream file;
	std::istream *input = &std::cin;
	if (options.input != std::filesystem::path{"-"}) {
		file.open(options.input);
		if (!file)
			throw std::runtime_error("evaluation JSONL not found: " + options.input.string());
		input = &file;
	}

	PreprocessOptions writer_options = options;
	writer_options.source = "lichess-eval";
	H5Writer writer(writer_options);
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

	std::cout << "Lichess evaluation preprocess start: input=" << options.input.string() << " output=" << options.output.string() << " arch_type=" << kArchType << std::endl;
	std::int64_t source_records = 0;
	std::int64_t accepted_records = 0;
	std::int64_t skipped_records = 0;
	std::string line;
	while ((options.max_positions < 0 || accepted_records < options.max_positions) && std::getline(*input, line)) {
		++source_records;
		const auto state_count = states.size();
		const auto move_count = moves.size();
		const auto value_count = values.size();
		try {
			if (line.empty())
				throw std::invalid_argument("empty record");
			const auto record = nlohmann::json::parse(line);
			if (!record.is_object() || !record.contains("fen") || !record.at("fen").is_string()) {
				throw std::invalid_argument("record has no FEN");
			}
			chess::Board board(complete_lichess_fen(record.at("fen").get<std::string>()));
			const auto target = lichess_target(record, board);
			const auto state = encode_state(board);
			const auto move = static_cast<std::uint16_t>(hdf5_action_index(target.move));
			states.push_back(state);
			moves.push_back(move);
			values.push_back(target.value);
			++accepted_records;
			if (states.size() >= buffer_capacity)
				flush();
			if (options.log_every > 0 && accepted_records % options.log_every == 0) {
				std::cout << "Lichess evaluation preprocess progress: positions=" << accepted_records << " skipped=" << skipped_records << std::endl;
			}
		} catch (const std::exception &error) {
			states.resize(state_count);
			moves.resize(move_count);
			values.resize(value_count);
			++skipped_records;
			if (skipped_records <= 8) {
				std::cerr << "Lichess evaluation preprocess skipped record " << source_records << ": " << error.what() << std::endl;
			}
		}
	}
	flush();
	writer.finish_evaluations(source_records, skipped_records);
	std::cout << "Lichess evaluation preprocess summary: records=" << source_records << " positions=" << accepted_records << " skipped=" << skipped_records
	          << " output=" << options.output.string() << std::endl;
}

ValueWeightController::ValueWeightController(double initial_weight) : value_(initial_weight) {
	if (!std::isfinite(initial_weight) || initial_weight < kMinValueWeight || initial_weight > kMaxValueWeight) {
		throw std::invalid_argument("value weight must be in [0.2, 2]");
	}
}

double ValueWeightController::update(double policy_squared_norm, double value_squared_norm, double inner_product) {
	if (!std::isfinite(policy_squared_norm) || !std::isfinite(value_squared_norm) || !std::isfinite(inner_product) || policy_squared_norm <= kGradientEpsilon ||
	    value_squared_norm <= kGradientEpsilon) {
		return value_;
	}

	double target = kValueGradientRatio * std::sqrt(policy_squared_norm) / (std::sqrt(value_squared_norm) + kGradientEpsilon);
	target = std::clamp(target, kMinValueWeight, kMaxValueWeight);
	double lower = kMinValueWeight;
	double upper = kMaxValueWeight;
	if (inner_product < 0.0) {
		lower = std::max(lower, (-inner_product / value_squared_norm) * (1.0 + kConflictMargin));
		upper = std::min(upper, (policy_squared_norm / -inner_product) * (1.0 - kConflictMargin));
		if (lower > upper) {
			return value_;
		}
		target = std::clamp(target, lower, upper);
	}

	const double smoothed = std::exp((1.0 - kValueWeightSmoothing) * std::log(value_) + kValueWeightSmoothing * std::log(target));
	value_ = std::clamp(smoothed, kMinValueWeight, kMaxValueWeight);
	return value_;
}

double ValueWeightController::value() const noexcept {
	return value_;
}

namespace {

struct OutputTargetStatistics {
	torch::Tensor action_counts;
	double mean_value = 0.0;
};

OutputTargetStatistics collect_output_target_statistics(const SupervisedH5 &data) {
	auto action_counts = torch::zeros({kActionSize}, torch::kInt64);
	double value_sum = 0.0;
	std::int64_t sampled_rows = 0;
	const auto total_chunks = (data.info().length + data.info().chunk_rows - 1) / data.info().chunk_rows;
	const auto sampled_chunks = std::min(total_chunks, (kOutputPriorSampleRows + data.info().chunk_rows - 1) / data.info().chunk_rows);
	for (std::int64_t sample = 0; sample < sampled_chunks; ++sample) {
		const auto chunk_index = ((2 * sample + 1) * total_chunks) / (2 * sampled_chunks);
		const auto begin = chunk_index * data.info().chunk_rows;
		const auto count = std::min(data.info().chunk_rows, data.info().length - begin);
		auto batch = data.read_contiguous(begin, count);
		action_counts.add_(torch::bincount(batch.moves, {}, kActionSize));
		value_sum += batch.values.sum().item<double>();
		sampled_rows += count;
	}
	return {std::move(action_counts), value_sum / static_cast<double>(sampled_rows)};
}

} // namespace

// Optimize a newly initialized model with the joint supervised objective.
void train_supervised(const TrainOptions &options) {
	ValueWeightController value_weight(options.value_weight);
	torch::manual_seed(static_cast<std::int64_t>(options.seed));
	const auto device = resolve_device(options.device);
	validate_compute_precision(options.precision, device);
	SupervisedH5 data(options.data);
	ArchitectureInfo architecture{options.channels, options.blocks};
	Model model(architecture.channels, architecture.blocks);
	std::cout << "initializing output priors from stratified target sample" << std::endl;
	const auto target_statistics = collect_output_target_statistics(data);
	model->initialize_output_priors(target_statistics.action_counts, target_statistics.mean_value);
	std::cout << "output priors ready: mean_value=" << target_statistics.mean_value << std::endl;
	model->to(device);
	for (auto &parameter : model->parameters()) {
		if (parameter.dim() == 4) {
			parameter.set_data(parameter.contiguous(torch::MemoryFormat::ChannelsLast));
		}
	}
	model->train();
	auto value_head_parameters = model->value_head->parameters();
	torch::optim::AdamW optimizer(model->parameters(), torch::optim::AdamWOptions(options.learning_rate).weight_decay(options.weight_decay));

	std::mt19937_64 rng(options.seed);
	std::int64_t global_step = 0;
	bool stop = false;
	std::cout << "training start: data=" << options.data.string() << " out=" << options.output.string() << " arch_type=" << kArchType << " device=" << device.str()
	          << " epochs=" << options.epochs << " batch_size=" << options.batch_size << " max_steps=" << options.max_steps
	          << " precision=" << compute_precision_name(options.precision) << std::endl;
	std::cout << "model ready: channels=" << architecture.channels << " blocks=" << architecture.blocks << " parameters=" << parameter_count(model) << std::endl;
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
			permutation = permutation.to(device, true);
			chunk.states = chunk.states.to(device, true).index_select(0, permutation);
			chunk.moves = chunk.moves.to(device, true).index_select(0, permutation);
			chunk.values = chunk.values.to(device, true).index_select(0, permutation);

			for (std::int64_t begin = 0; begin < chunk_rows; begin += options.batch_size) {
				const auto count = std::min<std::int64_t>(options.batch_size, chunk_rows - begin);
				auto states = decode_states_device(chunk.states.narrow(0, begin, count), device).contiguous(torch::MemoryFormat::ChannelsLast);
				auto moves = chunk.moves.narrow(0, begin, count);
				auto values = chunk.values.narrow(0, begin, count);
				optimizer.zero_grad();

				torch::Tensor logits;
				torch::Tensor predicted;
				{
					AutocastGuard autocast(options.precision, device);
					std::tie(logits, predicted) = model->forward(states);
				}
				predicted = predicted.to(torch::kFloat32);
				logits = logits.to(torch::kFloat32);
				auto policy_loss = torch::nn::functional::cross_entropy(logits, moves);
				auto value_loss = torch::mse_loss(predicted.squeeze(1), values);
				if (global_step == 0 || (global_step + 1) % kValueWeightProbeInterval == 0) {
					const auto stats = shared_gradient_stats(policy_loss, value_loss, model);
					value_weight.update(stats.policy_squared_norm, stats.value_squared_norm, stats.inner_product);
				}
				auto loss = policy_loss + value_weight.value() * value_loss;
				loss.backward();
				torch::nn::utils::clip_grad_norm_(value_head_parameters, kValueHeadGradientClip);
				optimizer.step();
				model->project_relation_residuals();

				++global_step;
				++batches;
				metric_totals.add_(torch::stack({policy_loss.detach(), value_loss.detach()}));
				if (options.log_every > 0 && (global_step == 1 || global_step % options.log_every == 0)) {
					auto metrics = torch::stack({policy_loss.detach(), value_loss.detach(), loss.detach()}).to(torch::kCPU).contiguous();
					auto metric_values = metrics.accessor<float, 1>();
					std::cout << "train step: epoch=" << epoch << " global_step=" << global_step;
					std::cout << " policy=" << metric_values[0] << " value_weight=" << value_weight.value();
					std::cout << " value=" << metric_values[1] << " loss=" << metric_values[2] << std::endl;
				}
				if (options.save_every > 0 && global_step % options.save_every == 0) {
					save_checkpoint_atomic(options.output, model, architecture);
					std::cout << "checkpoint saved: path=" << options.output.string() << " global_step=" << global_step << std::endl;
				}
				if (options.max_steps > 0 && global_step >= options.max_steps) {
					stop = true;
					break;
				}
			}
		}
		save_checkpoint_atomic(options.output, model, architecture);
		auto epoch_metrics = metric_totals.to(torch::kCPU).contiguous();
		auto epoch_values = epoch_metrics.accessor<float, 1>();
		std::cout << "epoch=" << epoch << ", steps=" << global_step;
		std::cout << ", policy=" << epoch_values[0] / std::max<std::int64_t>(1, batches);
		std::cout << ", value=" << epoch_values[1] / std::max<std::int64_t>(1, batches) << std::endl;
	}
	std::cout << "training finished: " << options.output.string() << std::endl;
}

} // namespace gadus
