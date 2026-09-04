#include "eleginus/game.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <hdf5.h>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
	struct Options {
		std::filesystem::path input;
		std::filesystem::path output;
		std::size_t chunk = 16384;
		std::uint64_t limit = 0;
		std::uint64_t logEvery = 100000;
		unsigned compression = 1;
	};

	class H5Writer {
	public:
		H5Writer(const std::filesystem::path &path, std::size_t chunk, unsigned compression)
			: file(check(H5Fcreate(path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), "create HDF5 file")) {
			states = create("states", {0, eleginus::packedBoardSize}, {H5S_UNLIMITED, eleginus::packedBoardSize},
				{chunk, eleginus::packedBoardSize}, H5T_STD_U8LE, compression);
			centipawns = create("centipawns", {0}, {H5S_UNLIMITED}, {chunk}, H5T_STD_I32LE, compression);
			attribute("state_bytes", static_cast<std::uint64_t>(eleginus::packedBoardSize));
			stringAttribute("state_encoding", "chess_compact24");
			stringAttribute("target_schema", "white_centipawns");
		}

		~H5Writer() { close(); }

		void close() noexcept {
			if (states >= 0) H5Dclose(states);
			if (centipawns >= 0) H5Dclose(centipawns);
			if (file >= 0) H5Fclose(file);
			states = -1;
			centipawns = -1;
			file = -1;
		}

		void append(const std::vector<eleginus::PackedBoard> &boards, const std::vector<std::int32_t> &scores) {
			if (boards.empty()) return;
			if (boards.size() != scores.size()) throw std::logic_error("preprocess buffers have different lengths");
			const hsize_t count = boards.size();
			extend(states, {rows + count, eleginus::packedBoardSize});
			extend(centipawns, {rows + count});
			write(states, H5T_NATIVE_UINT8, boards.data(), {rows, 0}, {count, eleginus::packedBoardSize});
			write(centipawns, H5T_NATIVE_INT32, scores.data(), {rows}, {count});
			rows += count;
		}

		std::uint64_t size() const noexcept { return rows; }

	private:
		static hid_t check(hid_t value, std::string_view action) {
			if (value < 0) throw std::runtime_error(std::string(action) + " failed");
			return value;
		}

		static void require(herr_t value, std::string_view action) {
			if (value < 0) throw std::runtime_error(std::string(action) + " failed");
		}

		hid_t create(const char *name, const std::vector<hsize_t> &initial, const std::vector<hsize_t> &maximum,
			const std::vector<hsize_t> &chunks, hid_t type, unsigned compression) {
			const hid_t space = check(H5Screate_simple(static_cast<int>(initial.size()), initial.data(), maximum.data()), "create dataspace");
			const hid_t properties = check(H5Pcreate(H5P_DATASET_CREATE), "create dataset properties");
			require(H5Pset_chunk(properties, static_cast<int>(chunks.size()), chunks.data()), "set HDF5 chunk");
			if (compression != 0) {
				require(H5Pset_shuffle(properties), "enable HDF5 shuffle");
				require(H5Pset_deflate(properties, compression), "enable HDF5 compression");
			}
			const hid_t dataset = H5Dcreate2(file, name, type, space, H5P_DEFAULT, properties, H5P_DEFAULT);
			H5Pclose(properties);
			H5Sclose(space);
			return check(dataset, "create dataset");
		}

		static void extend(hid_t dataset, const std::vector<hsize_t> &shape) {
			require(H5Dset_extent(dataset, shape.data()), "extend dataset");
		}

		static void write(hid_t dataset, hid_t type, const void *data, const std::vector<hsize_t> &start, const std::vector<hsize_t> &shape) {
			const hid_t fileSpace = check(H5Dget_space(dataset), "open dataset space");
			require(H5Sselect_hyperslab(fileSpace, H5S_SELECT_SET, start.data(), nullptr, shape.data(), nullptr), "select dataset slice");
			const hid_t memorySpace = check(H5Screate_simple(static_cast<int>(shape.size()), shape.data(), nullptr), "create memory space");
			require(H5Dwrite(dataset, type, memorySpace, fileSpace, H5P_DEFAULT, data), "write dataset slice");
			H5Sclose(memorySpace);
			H5Sclose(fileSpace);
		}

		void attribute(const char *name, std::uint64_t value) {
			const hid_t space = check(H5Screate(H5S_SCALAR), "create attribute space");
			const hid_t item = check(H5Acreate2(file, name, H5T_STD_U64LE, space, H5P_DEFAULT, H5P_DEFAULT), "create attribute");
			require(H5Awrite(item, H5T_NATIVE_UINT64, &value), "write attribute");
			H5Aclose(item);
			H5Sclose(space);
		}

		void stringAttribute(const char *name, std::string_view value) {
			const hid_t type = check(H5Tcopy(H5T_C_S1), "copy string type");
			require(H5Tset_size(type, value.size()), "set string size");
			const hid_t space = check(H5Screate(H5S_SCALAR), "create string space");
			const hid_t item = check(H5Acreate2(file, name, type, space, H5P_DEFAULT, H5P_DEFAULT), "create string attribute");
			require(H5Awrite(item, type, value.data()), "write string attribute");
			H5Aclose(item);
			H5Sclose(space);
			H5Tclose(type);
		}

		hid_t file = -1;
		hid_t states = -1;
		hid_t centipawns = -1;
		hsize_t rows = 0;
	};

	std::uint64_t unsignedValue(std::string_view text, std::string_view name) {
		std::uint64_t value = 0;
		const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
		if (error != std::errc{} || end != text.data() + text.size()) throw std::invalid_argument("invalid " + std::string(name));
		return value;
	}

	Options parse(int argc, char **argv) {
		Options options;
		for (int i = 1; i < argc; ++i) {
			const std::string_view key = argv[i];
			auto next = [&]() -> std::string_view {
				if (++i >= argc) throw std::invalid_argument("missing value after " + std::string(key));
				return argv[i];
			};
			if (key == "--input" || key == "--data") options.input = next();
			else if (key == "--output" || key == "--out") options.output = next();
			else if (key == "--chunk-rows") options.chunk = static_cast<std::size_t>(unsignedValue(next(), key));
			else if (key == "--max-positions") options.limit = unsignedValue(next(), key);
			else if (key == "--log-every") options.logEvery = unsignedValue(next(), key);
			else if (key == "--compression") options.compression = static_cast<unsigned>(unsignedValue(next(), key));
			else if (key == "--help") {
				std::cout << "Usage: preprocess --input <positions.jsonl|-> --output <positions.eleginus.h5>\n"
					<< "  --chunk-rows <n> --max-positions <n|0=all> --compression <0..9> --log-every <n>\n";
				std::exit(0);
			} else {
				throw std::invalid_argument("unknown option: " + std::string(key));
			}
		}
		if (options.input.empty() || options.output.empty()) throw std::invalid_argument("--input and --output are required");
		if (options.chunk == 0) throw std::invalid_argument("--chunk-rows must be positive");
		if (options.compression > 9) throw std::invalid_argument("--compression must be between 0 and 9");
		return options;
	}

	std::string completeFen(const std::string &fen) {
		const int fields = static_cast<int>(std::count(fen.begin(), fen.end(), ' ')) + 1;
		if (fields == 4) return fen + " 0 1";
		if (fields == 6) return fen;
		throw std::invalid_argument("FEN must contain four or six fields");
	}

	bool legalPosition(const chess::Board &board) {
		constexpr std::uint64_t edgeRanks = 0xFF000000000000FFULL;
		for (int color = 0; color < 2; ++color) {
			const auto side = static_cast<chess::Color>(color);
			const int kings = std::popcount(board.pieces(chess::PieceType::KING, side).getBits());
			const int pawns = std::popcount(board.pieces(chess::PieceType::PAWN, side).getBits());
			if (kings != 1 || pawns > 8 || (board.pieces(chess::PieceType::PAWN, side).getBits() & edgeRanks) != 0) return false;

			int total = 0;
			int promotions = 0;
			constexpr std::array<int, 6> original{{8, 2, 2, 2, 1, 1}};
			for (int type = 0; type < 6; ++type) {
				const auto piece = chess::PieceType(static_cast<chess::PieceType::underlying>(type));
				const int count = std::popcount(board.pieces(piece, side).getBits());
				total += count;
				if (type > 0 && type < 5) promotions += std::max(0, count - original[static_cast<std::size_t>(type)]);
			}
			if (total > 16 || pawns + promotions > 8) return false;
		}

		const auto previous = ~board.sideToMove();
		return !board.isAttacked(board.kingSq(previous), board.sideToMove());
	}

	std::int32_t centipawns(const nlohmann::json &record) {
		if (!record.contains("evals") || !record.at("evals").is_array()) throw std::invalid_argument("record has no evaluation array");
		const nlohmann::json *selected = nullptr;
		int bestDepth = std::numeric_limits<int>::min();
		std::int64_t bestNodes = std::numeric_limits<std::int64_t>::min();
		for (const auto &evaluation : record.at("evals")) {
			if (!evaluation.is_object() || !evaluation.contains("pvs") || !evaluation.at("pvs").is_array() || evaluation.at("pvs").empty()) continue;
			const int depth = evaluation.value("depth", -1);
			const std::int64_t nodes = evaluation.value("knodes", std::int64_t{-1});
			if (selected == nullptr || depth > bestDepth || (depth == bestDepth && nodes > bestNodes)) {
				selected = &evaluation;
				bestDepth = depth;
				bestNodes = nodes;
			}
		}
		if (selected == nullptr) throw std::invalid_argument("record has no usable evaluation");
		const auto &pv = selected->at("pvs").front();
		if (!pv.is_object() || !pv.contains("cp") || !pv.at("cp").is_number_integer()) {
			throw std::invalid_argument("selected evaluation has no centipawn score");
		}
		const auto value = pv.at("cp").get<std::int64_t>();
		if (value < std::numeric_limits<std::int32_t>::min() || value > std::numeric_limits<std::int32_t>::max()) {
			throw std::out_of_range("centipawn score exceeds int32 range");
		}
		return static_cast<std::int32_t>(value);
	}
} // namespace

int main(int argc, char **argv) {
	try {
		const auto options = parse(argc, argv);
		if (!options.output.parent_path().empty()) std::filesystem::create_directories(options.output.parent_path());
		std::ifstream file;
		std::istream *input = &std::cin;
		if (options.input != "-") {
			file.open(options.input);
			if (!file) throw std::runtime_error("cannot open input: " + options.input.string());
			input = &file;
		}

		const auto temporary = options.output.string() + ".tmp";
		std::filesystem::remove(temporary);
		H5Writer writer(temporary, options.chunk, options.compression);
		std::vector<eleginus::PackedBoard> boards;
		std::vector<std::int32_t> scores;
		boards.reserve(options.chunk);
		scores.reserve(options.chunk);

		std::uint64_t lines = 0;
		std::uint64_t skipped = 0;
		std::string line;
		while (std::getline(*input, line) && (options.limit == 0 || writer.size() + boards.size() < options.limit)) {
			++lines;
			try {
				const auto record = nlohmann::json::parse(line);
				if (!record.is_object() || !record.contains("fen") || !record.at("fen").is_string()) throw std::invalid_argument("record has no FEN");
				chess::Board board;
				if (!board.setFen(completeFen(record.at("fen").get<std::string>()))) throw std::invalid_argument("invalid FEN");
				if (!legalPosition(board)) throw std::invalid_argument("illegal chess position");
				(void)eleginus::legalmoves(board);
				const auto score = centipawns(record);
				boards.push_back(eleginus::packBoard(board));
				scores.push_back(score);
			} catch (const std::exception &) {
				++skipped;
			}

			if (boards.size() == options.chunk) {
				writer.append(boards, scores);
				boards.clear();
				scores.clear();
			}
			if (options.logEvery != 0 && lines % options.logEvery == 0) {
				std::cout << "preprocess step: lines=" << lines << " positions=" << writer.size() + boards.size() << " skipped=" << skipped << '\n';
			}
		}
		writer.append(boards, scores);
		const auto positions = writer.size();
		writer.close();

		std::filesystem::remove(options.output);
		std::filesystem::rename(temporary, options.output);
		std::cout << "preprocess complete: lines=" << lines << " positions=" << positions << " skipped=" << skipped << " output=" << options.output.string() << '\n';
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "preprocess error: " << error.what() << '\n';
		return 1;
	}
}
