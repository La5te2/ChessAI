// Draws procedural and pre-tessellated chess-piece styles compiled into the executable.
#include "graphics/pieces.hpp"
#include "graphics/archive.hpp"
#include "piece.inc"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace gadidae::graphics {
namespace {

constexpr std::uint8_t fill_command = 0;
constexpr std::uint8_t stroke_command = 1;
constexpr std::size_t pieces_per_style = 12;
constexpr std::array<std::string_view, 5> built_in_style_names = {"vector", "rhosgfx", "chessnut", "spatial", "cburnett"};

/// Maps one fixed SVG-backed style name to its first compiled piece index.
std::optional<std::size_t> fixed_style_offset(std::string_view style) {
	if (style == "rhosgfx") {
		return 0;
	}
	if (style == "chessnut") {
		return pieces_per_style;
	}
	if (style == "spatial") {
		return pieces_per_style * 2;
	}
	return std::nullopt;
}

/// Maps chess metadata to the white-then-black KQRBNP source ordering.
std::optional<std::size_t> piece_offset(const chess::Piece &piece) {
	std::size_t offset = 0;
	switch (piece.type().internal()) {
	case chess::PieceType::underlying::KING:
		offset = 0;
		break;
	case chess::PieceType::underlying::QUEEN:
		offset = 1;
		break;
	case chess::PieceType::underlying::ROOK:
		offset = 2;
		break;
	case chess::PieceType::underlying::BISHOP:
		offset = 3;
		break;
	case chess::PieceType::underlying::KNIGHT:
		offset = 4;
		break;
	case chess::PieceType::underlying::PAWN:
		offset = 5;
		break;
	default:
		return std::nullopt;
	}
	if (piece.color() == chess::Color::BLACK) {
		offset += 6;
	}
	return offset;
}

/// Converts normalized piece-local geometry into one screen-space coordinate.
ImVec2 screen_point(const EmbeddedPieceVertex &vertex, ImVec2 center, float size) {
	return {
	    center.x + vertex.x * size,
	    center.y + vertex.y * size,
	};
}

/// Emits one pre-triangulated fill from any compiled piece vertex array.
template <std::size_t VertexCount>
void draw_fill(ImDrawList *draw, const EmbeddedPieceCommand &command, const std::array<EmbeddedPieceVertex, VertexCount> &vertices, ImVec2 center, float size) {
	draw->PrimReserve(static_cast<int>(command.count), static_cast<int>(command.count));
	const ImVec2 white_pixel = ImGui::GetFontTexUvWhitePixel();
	const auto end = command.first + command.count;
	for (std::uint32_t index = command.first; index < end; ++index) {
		const auto &vertex = vertices[index];
		draw->PrimVtx(screen_point(vertex, center, size), white_pixel, vertex.color);
	}
}

/// Draws one pre-sampled stroke from any compiled piece vertex array.
template <std::size_t VertexCount>
void draw_stroke(ImDrawList *draw, const EmbeddedPieceCommand &command, const std::array<EmbeddedPieceVertex, VertexCount> &vertices, ImVec2 center, float size) {
	thread_local std::vector<ImVec2> points;
	points.clear();
	points.reserve(command.count);
	const auto end = command.first + command.count;
	for (std::uint32_t index = command.first; index < end; ++index) {
		points.push_back(screen_point(vertices[index], center, size));
	}
	draw->AddPolyline(points.data(), static_cast<int>(points.size()), command.color, command.closed ? ImDrawFlags_Closed : ImDrawFlags_None, std::max(1.0F, command.width * size));
}

/// Draws one piece from a standalone compiled geometry set.
template <std::size_t VertexCount, std::size_t CommandCount>
void draw_geometry(ImDrawList *draw, const std::array<EmbeddedPieceVertex, VertexCount> &vertices, const std::array<EmbeddedPieceCommand, CommandCount> &commands,
    const EmbeddedPieceGeometry &geometry, ImVec2 center, float size) {
	const auto command_end = geometry.first + geometry.count;
	for (std::uint32_t index = geometry.first; index < command_end; ++index) {
		const auto &command = commands[index];
		if (command.kind == fill_command) {
			draw_fill(draw, command, vertices, center, size);
		} else if (command.kind == stroke_command) {
			draw_stroke(draw, command, vertices, center, size);
		}
	}
}

/// Emits one imported mesh through ImGui's indexed primitive path.
void draw_archived_mesh(ImDrawList *draw, const ArchivedPieceMesh &mesh, ImVec2 center, float size) {
	if (mesh.vertices.empty() || mesh.indices.empty()) {
		return;
	}
	draw->PrimReserve(static_cast<int>(mesh.indices.size()), static_cast<int>(mesh.vertices.size()));
	const ImDrawIdx base = static_cast<ImDrawIdx>(draw->_VtxCurrentIdx);
	for (const auto index : mesh.indices) {
		draw->PrimWriteIdx(static_cast<ImDrawIdx>(base + index));
	}
	const ImVec2 white_pixel = ImGui::GetFontTexUvWhitePixel();
	for (const auto &vertex : mesh.vertices) {
		draw->PrimWriteVtx({center.x + vertex.x * size, center.y + vertex.y * size}, white_pixel, vertex.color);
	}
}

} // namespace

std::size_t piece_style_count() {
	return built_in_style_names.size() + archived_piece_styles().size();
}

std::string_view piece_style_name(std::size_t index) {
	if (index < built_in_style_names.size()) {
		return built_in_style_names[index];
	}
	const auto generated_index = index - built_in_style_names.size();
	const auto &archived = archived_piece_styles();
	return generated_index < archived.size() ? std::string_view(archived[generated_index].name) : std::string_view("vector");
}

bool piece_style_available(std::string_view name) {
	for (std::size_t index = 0; index < piece_style_count(); ++index) {
		if (piece_style_name(index) == name) {
			return true;
		}
	}
	return false;
}

bool draw_compiled_piece(ImDrawList *draw, std::string_view style, const chess::Piece &piece, ImVec2 center, float size) {
	const auto local_piece = piece_offset(piece);
	if (!local_piece || style == "vector") {
		return false;
	}
	if (style == "cburnett") {
		draw_geometry(draw, cburnett_piece_vertices, cburnett_piece_commands, cburnett_piece_geometries[*local_piece], center, size);
		return true;
	}
	for (const auto &archived : archived_piece_styles()) {
		if (style == archived.name) {
			draw_archived_mesh(draw, archived.pieces[*local_piece], center, size);
			return true;
		}
	}
	const auto style_index = fixed_style_offset(style);
	if (!style_index) {
		return false;
	}
	draw_geometry(draw, embedded_piece_vertices, embedded_piece_commands, embedded_piece_geometries[*style_index + *local_piece], center, size);
	return true;
}

} // namespace gadidae::graphics
