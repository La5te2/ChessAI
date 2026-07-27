// Validates and decodes the compressed indexed-piece archive embedded by CMake.
#include "graphics/archive.hpp"
#include <zlib.h>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace gadidae::graphics {

#ifndef _WIN32
extern "C" {
extern const std::uint8_t gadidae_piece_archive_start[];
extern const std::uint8_t gadidae_piece_archive_end[];
}
#endif

namespace {

constexpr std::size_t maximum_archive_size = 256U * 1024U * 1024U;
constexpr unsigned short piece_archive_resource = 101;

/// Returns the compressed bytes linked into the current executable.
std::span<const std::uint8_t> embedded_archive() {
#ifdef _WIN32
	const HMODULE module = GetModuleHandleW(nullptr);
	const HRSRC resource = FindResourceW(
		module, MAKEINTRESOURCEW(piece_archive_resource), MAKEINTRESOURCEW(10));
	if(resource == nullptr) {
		return {};
	}
	const HGLOBAL loaded = LoadResource(module, resource);
	const DWORD size = SizeofResource(module, resource);
	const void *data = loaded == nullptr ? nullptr : LockResource(loaded);
	return data == nullptr
		? std::span<const std::uint8_t>()
		: std::span<const std::uint8_t>(
			  static_cast<const std::uint8_t *>(data), size);
#else
	return {gadidae_piece_archive_start,
			static_cast<std::size_t>(
				gadidae_piece_archive_end - gadidae_piece_archive_start)};
#endif
}

class ArchiveReader {
public:
	explicit ArchiveReader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}

	/// Reads a little-endian arithmetic value without unaligned pointer casts.
	template<typename Value>
	Value read() {
		static_assert(std::is_arithmetic_v<Value>);
		const auto data = take(sizeof(Value));
		Value value{};
		std::memcpy(&value, data.data(), sizeof(Value));
		return value;
	}

	/// Reads one UTF-8 style name.
	std::string read_string(std::size_t size) {
		const auto data = take(size);
		return {reinterpret_cast<const char *>(data.data()), data.size()};
	}

	/// Reports whether every decompressed byte belongs to a parsed record.
	bool finished() const {
		return offset_ == bytes_.size();
	}

private:
	/// Advances through one validated byte range.
	std::span<const std::uint8_t> take(std::size_t size) {
		if(size > bytes_.size() - offset_) {
			throw std::runtime_error("truncated piece archive");
		}
		const auto result = bytes_.subspan(offset_, size);
		offset_ += size;
		return result;
	}

	std::span<const std::uint8_t> bytes_;
	std::size_t offset_ = 0;
};

/// Inflates the zlib wrapper after checking its declared upper bound.
std::vector<std::uint8_t> decompress_archive(
	std::span<const std::uint8_t> compressed) {
	if(compressed.size() < 12 ||
	   std::memcmp(compressed.data(), "GPCZ", 4) != 0) {
		throw std::runtime_error("invalid compressed piece archive");
	}
	std::uint64_t raw_size = 0;
	std::memcpy(&raw_size, compressed.data() + 4, sizeof(raw_size));
	if(raw_size > maximum_archive_size ||
	   raw_size > std::numeric_limits<uLongf>::max()) {
		throw std::runtime_error("piece archive is too large");
	}
	std::vector<std::uint8_t> raw(static_cast<std::size_t>(raw_size));
	uLongf destination_size = static_cast<uLongf>(raw.size());
	const int status = uncompress(
		raw.data(), &destination_size, compressed.data() + 12,
		static_cast<uLong>(compressed.size() - 12));
	if(status != Z_OK || destination_size != raw.size()) {
		throw std::runtime_error("cannot decompress piece archive");
	}
	return raw;
}

/// Parses all style and mesh records from the uncompressed payload.
std::vector<ArchivedPieceStyle> decode_archive() {
	const auto compressed = embedded_archive();
	if(compressed.empty()) {
		return {};
	}
	const auto raw = decompress_archive(compressed);
	ArchiveReader reader(raw);
	if(reader.read_string(4) != "GPS1") {
		throw std::runtime_error("invalid piece archive payload");
	}
	const auto style_count = reader.read<std::uint32_t>();
	std::vector<ArchivedPieceStyle> styles;
	styles.reserve(style_count);
	for(std::uint32_t style_index = 0; style_index < style_count; ++style_index) {
		ArchivedPieceStyle style;
		style.name = reader.read_string(reader.read<std::uint16_t>());
		for(auto &mesh : style.pieces) {
			const auto vertex_count = reader.read<std::uint32_t>();
			const auto index_count = reader.read<std::uint32_t>();
			if(vertex_count > std::numeric_limits<std::uint16_t>::max() ||
			   index_count % 3 != 0) {
				throw std::runtime_error("invalid piece mesh dimensions");
			}
			mesh.vertices.resize(vertex_count);
			for(auto &vertex : mesh.vertices) {
				vertex.x = reader.read<float>();
				vertex.y = reader.read<float>();
				vertex.color = reader.read<std::uint32_t>();
			}
			mesh.indices.resize(index_count);
			for(auto &index : mesh.indices) {
				index = reader.read<std::uint16_t>();
				if(index >= vertex_count) {
					throw std::runtime_error("piece mesh index is out of range");
				}
			}
		}
		styles.push_back(std::move(style));
	}
	if(!reader.finished()) {
		throw std::runtime_error("piece archive contains trailing data");
	}
	return styles;
}

} // namespace

const std::vector<ArchivedPieceStyle> &archived_piece_styles() {
	static const std::vector<ArchivedPieceStyle> styles = [] {
		try {
			return decode_archive();
		} catch(const std::exception &) {
			// Fixed styles remain usable when an embedded custom archive is corrupt.
			return std::vector<ArchivedPieceStyle>{};
		}
	}();
	return styles;
}

} // namespace gadidae::graphics
