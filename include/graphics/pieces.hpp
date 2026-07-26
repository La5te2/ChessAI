// Defines selectable chess-piece styles and draws their compiled vector geometry.
#pragma once
#include <chess.hpp>
#include <imgui.h>
#include <optional>
#include <string_view>

namespace gadidae::graphics {

/// Identifies one user-selectable chess-piece presentation.
enum class PieceStyle {
	Vector,
	Rhosgfx,
	Chessnut,
	Spatial,
	Imported,
};

/// Returns the stable lowercase name stored in settings and accepted by the CLI.
std::string_view piece_style_name(PieceStyle style);

/// Parses one stable style name without silently selecting an unrelated style.
std::optional<PieceStyle> parse_piece_style(std::string_view name);

/// Reports whether piece.inc currently contains one imported 12-piece style.
bool imported_piece_available();

/// Draws compiled vector geometry and returns false for the procedural style.
bool draw_compiled_piece(ImDrawList *draw, PieceStyle style,
						 const chess::Piece &piece, ImVec2 center, float size);

} // namespace gadidae::graphics
