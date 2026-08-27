// Loads indexed chess-piece meshes from the compressed archive embedded in Gadidae.
#pragma once
#include <array>
#include <cstdint>
#include <imgui.h>
#include <string>
#include <vector>

namespace gadidae::graphics {

struct ArchivedPieceVertex {
	float x;
	float y;
	ImU32 color;
};

struct ArchivedPieceMesh {
	std::vector<ArchivedPieceVertex> vertices;
	std::vector<std::uint16_t> indices;
};

struct ArchivedPieceStyle {
	std::string name;
	std::array<ArchivedPieceMesh, 12> pieces;
};

/// Decompresses and validates the embedded archive once on first use.
const std::vector<ArchivedPieceStyle> &archived_piece_styles();

} // namespace gadidae::graphics
