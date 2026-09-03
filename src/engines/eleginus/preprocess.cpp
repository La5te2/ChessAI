// Converts generic position JSONL into compact Eleginus supervised data.

#include "chess.hpp"
#include "eleginus/formula.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <hdf5.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>
#ifdef _WIN32
	#define NOMINMAX
	#define WIN32_LEAN_AND_MEAN
	#include <windows.h>
#endif

namespace {

	constexpr hsize_t planeCount = 4;

	struct Options {
		std::filesystem::path input = "data/positions.jsonl";
		std::filesystem::path output = "data/eleginus.h5";
		std::int64_t limit = -1;
		int chunk = 16384;
		int compression = 1;
		int log = 1000000;
	};

	struct Position {
		std::array<std::uint64_t, planeCount> pieces{};
		std::uint8_t state = 0;
		float value = 0.5F;
	};

	void require(herr_t status, const std::string &operation) {
		if (status < 0) throw std::runtime_error("HDF5 operation failed: " + operation);
	}

	hid_t requireId(hid_t id, const std::string &operation) {
		if (id < 0) throw std::runtime_error("HDF5 operation failed: " + operation);
		return id;
	}

	void stringAttribute(hid_t object, const char *name, const std::string &value) {
		const hid_t space = requireId(H5Screate(H5S_SCALAR), "create attribute space");
		const hid_t type = requireId(H5Tcopy(H5T_C_S1), "copy attribute type");
		require(H5Tset_size(type, value.size() + 1), "set attribute size");
		const hid_t attribute = requireId(H5Acreate2(object, name, type, space, H5P_DEFAULT, H5P_DEFAULT), "create attribute");
		require(H5Awrite(attribute, type, value.c_str()), "write attribute");
		H5Aclose(attribute);
		H5Tclose(type);
		H5Sclose(space);
	}

	void integerAttribute(hid_t object, const char *name, std::int64_t value) {
		const hid_t space = requireId(H5Screate(H5S_SCALAR), "create attribute space");
		const hid_t attribute = requireId(H5Acreate2(object, name, H5T_STD_I64LE, space, H5P_DEFAULT, H5P_DEFAULT), "create attribute");
		require(H5Awrite(attribute, H5T_NATIVE_INT64, &value), "write attribute");
		H5Aclose(attribute);
		H5Sclose(space);
	}

	class Writer {
	public:
		Writer(const std::filesystem::path &path, const Options &options) : compression(options.compression) {
			file = requireId(H5Fcreate(path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), "create output file");
			stringAttribute(file, "arch_type", "eleginus");
			stringAttribute(file, "state_encoding", "piece-code-bitplanes");
			stringAttribute(file, "target_schema", "white-expected-score");
			stringAttribute(file, "score_perspective", "white");
			stringAttribute(file, "value_transform", "sigmoid(white_centipawn_score/150)");
			integerAttribute(file, "formula_count", static_cast<std::int64_t>(eleginus::kFormulaCount));
			const hsize_t rows = static_cast<hsize_t>(options.chunk);
			pieces = dataset("pieces", {0, planeCount}, {H5S_UNLIMITED, planeCount}, {rows, planeCount}, H5T_STD_U64LE);
			states = dataset("states", {0}, {H5S_UNLIMITED}, {rows}, H5T_STD_U8LE);
			values = dataset("values", {0}, {H5S_UNLIMITED}, {rows}, H5T_IEEE_F32LE);
		}

		~Writer() {
			if (pieces >= 0) H5Dclose(pieces);
			if (states >= 0) H5Dclose(states);
			if (values >= 0) H5Dclose(values);
			if (file >= 0) H5Fclose(file);
		}

		void append(const std::vector<Position> &positions) {
			if (positions.empty()) return;
			std::vector<std::uint64_t> packed(positions.size() * planeCount);
			std::vector<std::uint8_t> state(positions.size());
			std::vector<float> value(positions.size());
			for (std::size_t i = 0; i < positions.size(); ++i) {
				std::copy(positions[i].pieces.begin(), positions[i].pieces.end(), packed.begin() + static_cast<std::ptrdiff_t>(i * planeCount));
				state[i] = positions[i].state;
				value[i] = positions[i].value;
			}
			const hsize_t count = positions.size();
			const hsize_t next = size + count;
			extend(pieces, {next, planeCount});
			extend(states, {next});
			extend(values, {next});
			write(pieces, H5T_NATIVE_UINT64, packed.data(), {size, 0}, {count, planeCount});
			write(states, H5T_NATIVE_UINT8, state.data(), {size}, {count});
			write(values, H5T_NATIVE_FLOAT, value.data(), {size}, {count});
			size = next;
		}

		void finish(std::int64_t records, std::int64_t skipped) {
			integerAttribute(file, "positions", static_cast<std::int64_t>(size));
			integerAttribute(file, "records", records);
			integerAttribute(file, "skipped_records", skipped);
			require(H5Fflush(file, H5F_SCOPE_GLOBAL), "flush output file");
		}

	private:
		hid_t dataset(const char *name, const std::vector<hsize_t> &initial, const std::vector<hsize_t> &maximum,
			const std::vector<hsize_t> &chunk, hid_t type) const {
			const hid_t space = requireId(H5Screate_simple(static_cast<int>(initial.size()), initial.data(), maximum.data()), name);
			const hid_t properties = requireId(H5Pcreate(H5P_DATASET_CREATE), name);
			require(H5Pset_chunk(properties, static_cast<int>(chunk.size()), chunk.data()), name);
			if (compression > 0) {
				require(H5Pset_shuffle(properties), "enable shuffle filter");
				require(H5Pset_deflate(properties, static_cast<unsigned>(compression)), "enable deflate filter");
			}
			const hid_t result = requireId(H5Dcreate2(file, name, type, space, H5P_DEFAULT, properties, H5P_DEFAULT), name);
			H5Pclose(properties);
			H5Sclose(space);
			return result;
		}

		static void extend(hid_t target, const std::vector<hsize_t> &dimensions) {
			require(H5Dset_extent(target, dimensions.data()), "extend dataset");
		}

		static void write(hid_t target, hid_t type, const void *data, const std::vector<hsize_t> &start, const std::vector<hsize_t> &count) {
			const hid_t destination = requireId(H5Dget_space(target), "get dataset space");
			require(H5Sselect_hyperslab(destination, H5S_SELECT_SET, start.data(), nullptr, count.data(), nullptr), "select append range");
			const hid_t source = requireId(H5Screate_simple(static_cast<int>(count.size()), count.data(), nullptr), "create memory space");
			require(H5Dwrite(target, type, source, destination, H5P_DEFAULT, data), "append dataset");
			H5Sclose(source);
			H5Sclose(destination);
		}

		int compression;
		hid_t file = -1;
		hid_t pieces = -1;
		hid_t states = -1;
		hid_t values = -1;
		hsize_t size = 0;
	};

	std::string completeFen(const std::string &fen) {
		const int fields = static_cast<int>(std::count(fen.begin(), fen.end(), ' ')) + 1;
		if (fields == 4) return fen + " 0 1";
		if (fields == 6) return fen;
		throw std::invalid_argument("FEN must contain four or six fields");
	}

	float sigmoid(double x) {
		if (x >= 0) return static_cast<float>(1.0 / (1.0 + std::exp(-x)));
		const double e = std::exp(x);
		return static_cast<float>(e / (1.0 + e));
	}

	float target(const nlohmann::json &record) {
		if (!record.contains("evals") || !record.at("evals").is_array()) throw std::invalid_argument("record has no evaluation array");
		const nlohmann::json *selected = nullptr;
		int bestDepth = -1;
		std::int64_t bestNodes = -1;
		for (const auto &evaluation : record.at("evals")) {
			if (!evaluation.is_object() || !evaluation.contains("pvs") || !evaluation.at("pvs").is_array() || evaluation.at("pvs").empty()) continue;
			const int depth = evaluation.value("depth", -1);
			const std::int64_t nodes = evaluation.value("knodes", std::int64_t{-1});
			if (!selected || depth > bestDepth || (depth == bestDepth && nodes > bestNodes)) {
				selected = &evaluation;
				bestDepth = depth;
				bestNodes = nodes;
			}
		}
		if (!selected) throw std::invalid_argument("record has no principal variation");
		const auto &pv = selected->at("pvs").front();
		if (!pv.is_object()) throw std::invalid_argument("selected principal variation is invalid");
		if (pv.contains("cp") && pv.at("cp").is_number_integer()) return sigmoid(pv.at("cp").get<double>() / 150.0);
		if (pv.contains("mate") && pv.at("mate").is_number_integer()) {
			const int mate = pv.at("mate").get<int>();
			if (mate != 0) return mate > 0 ? 1.0F : 0.0F;
		}
		throw std::invalid_argument("selected principal variation has no cp or mate score");
	}

	Position encode(const nlohmann::json &record) {
		if (!record.is_object() || !record.contains("fen") || !record.at("fen").is_string()) throw std::invalid_argument("record has no FEN");
		chess::Board board(completeFen(record.at("fen").get<std::string>()));
		for (const auto color : {chess::Color::WHITE, chess::Color::BLACK}) {
			if (board.pieces(chess::PieceType::PAWN, color).count() > 8) throw std::invalid_argument("FEN contains more than eight pawns for one side");
		}
		Position result;
		for (int square = 0; square < 64; ++square) {
			const auto piece = board.at(chess::Square(square));
			if (piece == chess::Piece::NONE) continue;
			const int type = static_cast<int>(piece.type().internal());
			if (type < 0 || type >= 6) throw std::invalid_argument("FEN contains an invalid piece type");
			const unsigned code = 1U + static_cast<unsigned>(type) + (piece.color() == chess::Color::BLACK ? 6U : 0U);
			for (unsigned bit = 0; bit < planeCount; ++bit) {
				if (code & (1U << bit)) result.pieces[bit] |= 1ULL << square;
			}
		}
		if (board.sideToMove() == chess::Color::BLACK) result.state |= 1U;
		for (int color = 0; color < 2; ++color) {
			for (int wing = 0; wing < 2; ++wing) {
				const auto side = wing == 0 ? chess::Board::CastlingRights::Side::KING_SIDE : chess::Board::CastlingRights::Side::QUEEN_SIDE;
				if (board.castlingRights().has(static_cast<chess::Color>(color), side)) result.state |= static_cast<std::uint8_t>(1U << (1 + 2 * color + wing));
			}
		}
		result.value = target(record);
		return result;
	}

	Options parse(int argc, char **argv) {
		Options options;
		for (int i = 1; i < argc; ++i) {
			const std::string key = argv[i];
			if (key == "--help") {
				std::cout << "usage: preprocess --input <path|-> --output <data.h5> [options]\n";
				std::cout << "  --max-positions <n> --chunk-size <n> --compression-level <0..9>\n";
				std::cout << "  --log-every <accepted positions>\n";
				std::exit(0);
			}
			if (++i == argc) throw std::invalid_argument("missing value after " + key);
			const std::string value = argv[i];
			if (key == "--input") options.input = value;
			else if (key == "--output") options.output = value;
			else if (key == "--max-positions") options.limit = std::stoll(value);
			else if (key == "--chunk-size") options.chunk = std::stoi(value);
			else if (key == "--compression-level") options.compression = std::stoi(value);
			else if (key == "--log-every") options.log = std::stoi(value);
			else throw std::invalid_argument("unknown option: " + key);
		}
		if (options.input.empty() || options.output.empty() || options.limit == 0 || options.limit < -1 || options.chunk < 1 || options.compression < 0 ||
			options.compression > 9 || options.log < 0) {
			throw std::invalid_argument("invalid preprocessing options");
		}
		return options;
	}

	void replace(const std::filesystem::path &temporary, const std::filesystem::path &output) {
		#ifdef _WIN32
		if (!MoveFileExW(temporary.c_str(), output.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
			throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), "cannot replace preprocessed data");
		}
		#else
		std::filesystem::rename(temporary, output);
		#endif
	}

	void preprocess(const Options &options) {
		if (options.input != std::filesystem::path{"-"} && options.input.extension() == ".zst") {
			throw std::invalid_argument("compressed JSONL must be streamed through standard input with --input -");
		}
		std::ifstream file;
		std::istream *input = &std::cin;
		if (options.input != std::filesystem::path{"-"}) {
			file.open(options.input);
			if (!file) throw std::runtime_error("JSONL input not found: " + options.input.string());
			input = &file;
		}
		if (!options.output.parent_path().empty()) std::filesystem::create_directories(options.output.parent_path());
		auto temporary = options.output;
		temporary += ".tmp";
		std::error_code ignored;
		std::filesystem::remove(temporary, ignored);
		try {
			std::int64_t records = 0, accepted = 0, skipped = 0;
			{
				Writer writer(temporary, options);
				std::vector<Position> buffer;
				buffer.reserve(static_cast<std::size_t>(options.chunk));
				std::cout << "preprocess start: input=" << options.input.string() << " output=" << options.output.string() << std::endl;
				std::string line;
				while ((options.limit < 0 || accepted < options.limit) && std::getline(*input, line)) {
					++records;
					try {
						if (line.empty()) throw std::invalid_argument("empty record");
						buffer.push_back(encode(nlohmann::json::parse(line)));
						++accepted;
						if (buffer.size() == static_cast<std::size_t>(options.chunk)) {
							writer.append(buffer);
							buffer.clear();
						}
						if (options.log > 0 && accepted % options.log == 0) {
							std::cout << "preprocess progress: positions=" << accepted << " skipped=" << skipped << std::endl;
						}
					} catch (const std::exception &error) {
						++skipped;
						if (skipped <= 8) std::cerr << "preprocess skipped record " << records << ": " << error.what() << std::endl;
					}
				}
				writer.append(buffer);
				if (accepted == 0) throw std::runtime_error("no valid positions were found");
				writer.finish(records, skipped);
			}
			replace(temporary, options.output);
			std::cout << "preprocess summary: records=" << records << " positions=" << accepted << " skipped=" << skipped;
			std::cout << " output=" << options.output.string() << std::endl;
		} catch (...) {
			std::filesystem::remove(temporary, ignored);
			throw;
		}
	}

} // namespace

int main(int argc, char **argv) {
	try {
		preprocess(parse(argc, argv));
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "preprocess error: " << error.what() << std::endl;
		return 1;
	}
}
