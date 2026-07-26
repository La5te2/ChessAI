// Defines selectable chess-piece styles and draws their compiled vector geometry.
#pragma once
#include <chess.hpp>
#include <imgui.h>
#include <cstddef>
#include <string_view>

namespace gadidae::graphics {

/// Returns the number and stable names of every style compiled into the executable.
std::size_t piece_style_count();
std::string_view piece_style_name(std::size_t index);

/// Reports whether a stable style name is compiled into the executable.
bool piece_style_available(std::string_view name);

/// Draws compiled vector geometry and returns false for the procedural style.
bool draw_compiled_piece(ImDrawList *draw, std::string_view style,
						 const chess::Piece &piece, ImVec2 center, float size);

} // namespace gadidae::graphics
