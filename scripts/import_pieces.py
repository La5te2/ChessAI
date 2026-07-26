"""Pre-triangulate one 12-piece SVG style into src/graphics/piece.inc."""

from __future__ import annotations

import argparse
import math
import re
import xml.etree.ElementTree as et
from pathlib import Path

from shapely import constrained_delaunay_triangles
from shapely.geometry import GeometryCollection, LineString, LinearRing, Polygon
from svgelements import Path as SvgPath
from svgelements import SVG, Shape


PIECES = ("wK", "wQ", "wR", "wB", "wN", "wP", "bK", "bQ", "bR", "bB", "bN", "bP")
BEGIN = "// GADIDAE_IMPORTED_PIECES_BEGIN"
END = "// GADIDAE_IMPORTED_PIECES_END"
HEX_COLOR = re.compile(r"^#([0-9a-fA-F]{3,8})$")


def parser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Compile one 12-piece SVG set into Gadidae piece.inc geometry."
	)
	result.add_argument("--input", required=True, type=Path)
	result.add_argument(
		"--name",
		default="imported",
		help="Stable lowercase style name stored in GUI settings.",
	)
	result.add_argument(
		"--output",
		type=Path,
		default=Path("src/graphics/piece.inc"),
	)
	result.add_argument(
		"--curve-step",
		type=float,
		default=1.5,
		help="Approximate source units between sampled curve points.",
	)
	return result


def source_path(directory: Path, stem: str) -> Path:
	path = directory / f"{stem}.svg"
	if not path.is_file():
		raise FileNotFoundError(f"missing {path}")
	return path


def svg_frame(path: Path) -> tuple[float, float, float, float]:
	root = et.parse(path).getroot()
	view_box = root.attrib.get("viewBox", "").replace(",", " ").split()
	if len(view_box) == 4:
		return tuple(float(value) for value in view_box)

	def dimension(name: str) -> float:
		text = root.attrib.get(name, "0")
		match = re.match(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text)
		return float(match.group(0)) if match else 0.0

	width = dimension("width")
	height = dimension("height")
	if width <= 0 or height <= 0:
		raise ValueError(f"{path} requires viewBox or positive width/height")
	return 0.0, 0.0, width, height


def packed_color(value: object, fallback: int | None) -> int | None:
	text = str(value).strip()
	if not text or text.lower() in {"none", "null"}:
		return None
	if text.startswith("url("):
		return fallback
	match = HEX_COLOR.match(text)
	if not match:
		return fallback
	digits = match.group(1)
	if len(digits) in {3, 4}:
		digits = "".join(character * 2 for character in digits)
	if len(digits) == 6:
		digits += "ff"
	if len(digits) != 8:
		return fallback
	red, green, blue, alpha = (
		int(digits[index : index + 2], 16)
		for index in range(0, 8, 2)
	)
	return (alpha << 24) | (blue << 16) | (green << 8) | red


def sampled_subpaths(shape: Shape, curve_step: float) -> list[tuple[list[tuple[float, float]], bool]]:
	path = SvgPath(shape)
	path.reify()
	result: list[tuple[list[tuple[float, float]], bool]] = []
	for subpath in path.as_subpaths():
		points: list[tuple[float, float]] = []
		for segment in subpath.segments():
			length = float(segment.length())
			if length <= 0:
				continue
			steps = min(1024, max(1, int(math.ceil(length / curve_step))))
			for index in range(steps + 1):
				point = segment.point(index / steps)
				sampled = (float(point.real), float(point.imag))
				if not points or math.dist(points[-1], sampled) > 1e-7:
					points.append(sampled)
		if len(points) < 2:
			continue
		closed = math.dist(points[0], points[-1]) <= max(1e-5, curve_step * 0.25)
		if closed:
			points.pop()
		if len(points) >= 2:
			result.append((points, closed))
	return result


def normalized(point: tuple[float, float], frame: tuple[float, float, float, float]) -> tuple[float, float]:
	left, top, width, height = frame
	scale = max(width, height)
	return (
		(point[0] - left - width * 0.5) / scale,
		(point[1] - top - height * 0.5) / scale,
	)


def fill_geometry(rings: list[list[tuple[float, float]]]) -> object:
	geometry: object = GeometryCollection()
	for ring in rings:
		if len(ring) < 3:
			continue
		polygon = Polygon(ring)
		if not polygon.is_valid:
			polygon = polygon.buffer(0)
		if polygon.is_empty:
			continue
		geometry = polygon if geometry.is_empty else geometry.symmetric_difference(polygon)
	return geometry


def append_triangles(
	geometry: object,
	color: int,
	frame: tuple[float, float, float, float],
	vertices: list[tuple[float, float, int]],
) -> int:
	first_vertex = len(vertices)
	for triangle in constrained_delaunay_triangles(geometry).geoms:
		for x, y in list(triangle.exterior.coords)[:3]:
			nx, ny = normalized((x, y), frame)
			vertices.append((nx, ny, color))
	return len(vertices) - first_vertex


def stroke_geometry(
	points: list[tuple[float, float]],
	closed: bool,
	width: float,
	linecap: str,
	linejoin: str,
) -> object:
	if len(points) < 2 or width <= 0:
		return GeometryCollection()
	line = LinearRing(points) if closed else LineString(points)
	cap_style = {"round": 1, "butt": 2, "square": 3}.get(linecap, 1)
	join_style = {"round": 1, "miter": 2, "bevel": 3}.get(linejoin, 1)
	return line.buffer(
		width * 0.5,
		quad_segs=6,
		cap_style=cap_style,
		join_style=join_style,
	)


def append_piece(
	path: Path,
	curve_step: float,
	vertices: list[tuple[float, float, int]],
	commands: list[tuple[int, int, int, float, int, bool]],
) -> tuple[int, int]:
	frame = svg_frame(path)
	first_command = len(commands)
	white_piece = path.stem.startswith("w")
	default_fill = 0xFFF2F2F2 if white_piece else 0xFF202020
	default_stroke = 0xFF202020 if white_piece else 0xFFF2F2F2
	for element in SVG.parse(str(path)).elements():
		if not isinstance(element, Shape):
			continue
		subpaths = sampled_subpaths(element, curve_step)
		fill = packed_color(getattr(element, "fill", None), default_fill)
		if fill is not None:
			# SVG fills implicitly close every open subpath. Strokes retain their
			# original open/closed state, which matters for the Cburnett knights.
			rings = [points for points, _closed in subpaths if len(points) >= 3]
			geometry = fill_geometry(rings)
			first_vertex = len(vertices)
			# Constrained triangulation preserves concave boundaries and holes.
			count = append_triangles(geometry, fill, frame, vertices)
			if count:
				commands.append(
					(first_vertex, count, 0, 0.0, 0, False)
				)
		stroke = packed_color(getattr(element, "stroke", None), None)
		if stroke is not None:
			try:
				width = float(getattr(element, "stroke_width", 1.0))
			except (TypeError, ValueError):
				width = 1.0
			values = getattr(element, "values", {})
			linecap = str(values.get("stroke-linecap", "round")).lower()
			linejoin = str(values.get("stroke-linejoin", "round")).lower()
			for points, closed in subpaths:
				first_vertex = len(vertices)
				geometry = stroke_geometry(
					points,
					closed,
					width,
					linecap,
					linejoin,
				)
				count = append_triangles(
					geometry,
					stroke or default_stroke,
					frame,
					vertices,
				)
				if count:
					commands.append((first_vertex, count, 0, 0.0, 0, False))
	return first_command, len(commands) - first_command


def array_block(
	style_name: str,
	vertices: list[tuple[float, float, int]],
	commands: list[tuple[int, int, int, float, int, bool]],
	geometries: list[tuple[int, int]],
) -> str:
	vertex_lines = [
		f"\t{{{x:.7f}F, {y:.7f}F, 0x{color:08x}U}},"
		for x, y, color in vertices
	]
	command_lines = [
		f"\t{{{first}U, {count}U, 0x{color:08x}U, {width:.7f}F, {kind}U, "
		f"{str(closed).lower()}}},"
		for first, count, color, width, kind, closed in commands
	]
	geometry_lines = [
		f"\t{{{first}U, {count}U}}," for first, count in geometries
	]
	return "\n".join(
		(
			BEGIN,
			f'inline constexpr char imported_piece_style_name[] = "{style_name}";',
			f"inline constexpr std::array<EmbeddedPieceVertex, {len(vertices)}> imported_piece_vertices = {{{{",
			*vertex_lines,
			"}};",
			f"inline constexpr std::array<EmbeddedPieceCommand, {len(commands)}> imported_piece_commands = {{{{",
			*command_lines,
			"}};",
			f"inline constexpr std::array<EmbeddedPieceGeometry, {len(geometries)}> imported_piece_geometries = {{{{",
			*geometry_lines,
			"}};",
			END,
		)
	)


def replace_block(document: str, block: str) -> str:
	pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.DOTALL)
	updated, count = pattern.subn(block, document)
	if count != 1:
		raise RuntimeError("piece.inc does not contain exactly one imported-piece block")
	return updated


def main() -> None:
	args = parser().parse_args()
	if args.curve_step <= 0:
		raise ValueError("--curve-step must be positive")
	style_name = args.name.strip().lower()
	if not re.fullmatch(r"[a-z][a-z0-9-]*", style_name):
		raise ValueError("--name must be a lowercase identifier such as cburnett")
	vertices: list[tuple[float, float, int]] = []
	commands: list[tuple[int, int, int, float, int, bool]] = []
	geometries = [
		append_piece(
			source_path(args.input, piece),
			args.curve_step,
			vertices,
			commands,
		)
		for piece in PIECES
	]
	document = args.output.read_text(encoding="utf-8")
	args.output.write_text(
		replace_block(
			document,
			array_block(style_name, vertices, commands, geometries),
		),
		encoding="utf-8",
		newline="\n",
	)
	print(
		"piece style embedded: "
		f"input={args.input} output={args.output} "
		f"vertices={len(vertices)} commands={len(commands)}"
	)


if __name__ == "__main__":
	main()
