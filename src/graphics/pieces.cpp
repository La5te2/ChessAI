// Draws pre-tessellated SVG geometry without runtime assets or SVG libraries.
#include "graphics/pieces.hpp"
#include "piece.inc"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace gadidae::graphics {
namespace {

constexpr std::uint8_t fill_command = 0;
constexpr std::uint8_t stroke_command = 1;
constexpr std::size_t pieces_per_style = 12;

/// Maps one SVG-backed style to its first compiled piece index.
std::optional<std::size_t> style_offset(PieceStyle style) {
	switch(style) {
	case PieceStyle::Rhosgfx:
		return 0;
	case PieceStyle::Chessnut:
		return pieces_per_style;
	case PieceStyle::Spatial:
		return pieces_per_style * 2;
	case PieceStyle::Vector:
	case PieceStyle::Imported:
		return std::nullopt;
	}
	return std::nullopt;
}

/// Maps chess metadata to the white-then-black KQRBNP source ordering.
std::optional<std::size_t> piece_offset(const chess::Piece &piece) {
	std::size_t offset = 0;
	switch(piece.type().internal()) {
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
	if(piece.color() == chess::Color::BLACK) {
		offset += 6;
	}
	return offset;
}

/// Converts normalized piece-local geometry into one screen-space coordinate.
ImVec2 screen_point(const EmbeddedPieceVertex &vertex,
					ImVec2 center, float size) {
	return {
		center.x + vertex.x * size,
		center.y + vertex.y * size,
	};
}

/// Emits one pre-triangulated fill from any compiled piece vertex array.
template<std::size_t VertexCount>
void draw_fill(ImDrawList *draw, const EmbeddedPieceCommand &command,
			   const std::array<EmbeddedPieceVertex, VertexCount> &vertices,
			   ImVec2 center, float size) {
	draw->PrimReserve(static_cast<int>(command.count),
					  static_cast<int>(command.count));
	const ImVec2 white_pixel = ImGui::GetFontTexUvWhitePixel();
	const auto end = command.first + command.count;
	for(std::uint32_t index = command.first; index < end; ++index) {
		const auto &vertex = vertices[index];
		draw->PrimVtx(screen_point(vertex, center, size), white_pixel, vertex.color);
	}
}

/// Draws one pre-sampled stroke from any compiled piece vertex array.
template<std::size_t VertexCount>
void draw_stroke(ImDrawList *draw, const EmbeddedPieceCommand &command,
				 const std::array<EmbeddedPieceVertex, VertexCount> &vertices,
				 ImVec2 center, float size) {
	thread_local std::vector<ImVec2> points;
	points.clear();
	points.reserve(command.count);
	const auto end = command.first + command.count;
	for(std::uint32_t index = command.first; index < end; ++index) {
		points.push_back(screen_point(vertices[index], center, size));
	}
	draw->AddPolyline(points.data(), static_cast<int>(points.size()), command.color,
					  command.closed ? ImDrawFlags_Closed : ImDrawFlags_None,
					  std::max(1.0F, command.width * size));
}

} // namespace

std::string_view piece_style_name(PieceStyle style) {
	switch(style) {
	case PieceStyle::Vector:
		return "vector";
	case PieceStyle::Rhosgfx:
		return "rhosgfx";
	case PieceStyle::Chessnut:
		return "chessnut";
	case PieceStyle::Spatial:
		return "spatial";
	case PieceStyle::Imported:
		return imported_piece_style_name;
	}
	return "vector";
}

std::optional<PieceStyle> parse_piece_style(std::string_view name) {
	for(const auto style : {PieceStyle::Vector, PieceStyle::Rhosgfx,
						   PieceStyle::Chessnut, PieceStyle::Spatial,
						   PieceStyle::Imported}) {
		if(name == piece_style_name(style)) {
			return style;
		}
	}
	return std::nullopt;
}

bool imported_piece_available() {
	return imported_piece_geometries.size() == pieces_per_style;
}

bool draw_compiled_piece(ImDrawList *draw, PieceStyle style,
						 const chess::Piece &piece, ImVec2 center, float size) {
	const auto local_piece = piece_offset(piece);
	if(!local_piece) {
		return false;
	}
	if(style == PieceStyle::Imported) {
		if(!imported_piece_available()) {
			return false;
		}
		const auto &geometry = imported_piece_geometries[*local_piece];
		const auto command_end = geometry.first + geometry.count;
		for(std::uint32_t index = geometry.first; index < command_end; ++index) {
			const auto &command = imported_piece_commands[index];
			if(command.kind == fill_command) {
				draw_fill(draw, command, imported_piece_vertices, center, size);
			} else if(command.kind == stroke_command) {
				draw_stroke(draw, command, imported_piece_vertices, center, size);
			}
		}
		return true;
	}
	const auto style_index = style_offset(style);
	if(!style_index) {
		return false;
	}
	const auto &geometry =
		embedded_piece_geometries[*style_index + *local_piece];
	const auto command_end = geometry.first + geometry.count;
	for(std::uint32_t index = geometry.first; index < command_end; ++index) {
		const auto &command = embedded_piece_commands[index];
		if(command.kind == fill_command) {
			draw_fill(draw, command, embedded_piece_vertices, center, size);
		} else if(command.kind == stroke_command) {
			draw_stroke(draw, command, embedded_piece_vertices, center, size);
		}
	}
	return true;
}

} // namespace gadidae::graphics
