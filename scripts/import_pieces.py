"""Add or update compiled SVG piece styles in src/graphics/piece.inc."""

from __future__ import annotations

import argparse
import math
import re
import shutil
import xml.etree.ElementTree as et
from dataclasses import dataclass
from pathlib import Path

from shapely import constrained_delaunay_triangles
from shapely.geometry import GeometryCollection, LineString, LinearRing, Polygon
from svgelements import Matrix
from svgelements import Path as SvgPath
from svgelements import SVG, Shape


PIECES = ("wK", "wQ", "wR", "wB", "wN", "wP", "bK", "bQ", "bR", "bB", "bN", "bP")
BEGIN = "// GADIDAE_GENERATED_PIECES_BEGIN"
END = "// GADIDAE_GENERATED_PIECES_END"
RESERVED_STYLES = {"vector", "rhosgfx", "chessnut", "spatial", "cburnett"}
HEX_COLOR = re.compile(r"^#([0-9a-fA-F]{3,8})$")
URL_PAINT = re.compile(r"^url\(\s*#([^)]+)\s*\)$")


@dataclass(frozen=True)
class ColorStop:
	offset: float
	red: int
	green: int
	blue: int
	alpha: int


@dataclass(frozen=True)
class GradientCoordinate:
	value: float
	percentage: bool

	def resolve(self, origin: float, extent: float, user_space: bool) -> float:
		if user_space:
			return origin + self.value * extent if self.percentage else self.value
		return self.value


@dataclass(frozen=True)
class LinearPaint:
	x1: GradientCoordinate
	y1: GradientCoordinate
	x2: GradientCoordinate
	y2: GradientCoordinate
	stops: tuple[ColorStop, ...]
	transform: Matrix
	user_space: bool

	def color_at(
		self,
		x: float,
		y: float,
		bounds: tuple[float, float, float, float],
		frame: tuple[float, float, float, float],
	) -> int:
		"""Interpolate one SVG linear gradient at an unnormalized source point."""
		if self.user_space:
			left, top, width, height = frame
			x1 = self.x1.resolve(left, width, True)
			y1 = self.y1.resolve(top, height, True)
			x2 = self.x2.resolve(left, width, True)
			y2 = self.y2.resolve(top, height, True)
		else:
			min_x, min_y, max_x, max_y = bounds
			width = max(max_x - min_x, 1e-9)
			height = max(max_y - min_y, 1e-9)
			x1 = min_x + self.x1.resolve(0.0, width, False) * width
			y1 = min_y + self.y1.resolve(0.0, height, False) * height
			x2 = min_x + self.x2.resolve(0.0, width, False) * width
			y2 = min_y + self.y2.resolve(0.0, height, False) * height
		start = self.transform.point_in_matrix_space((x1, y1))
		end = self.transform.point_in_matrix_space((x2, y2))
		dx, dy = end.x - start.x, end.y - start.y
		length_squared = dx * dx + dy * dy
		position = 0.0 if length_squared <= 1e-12 else (
			(x - start.x) * dx + (y - start.y) * dy
		) / length_squared
		position = min(1.0, max(0.0, position))
		left_stop = self.stops[0]
		right_stop = self.stops[-1]
		for candidate in self.stops[1:]:
			if position <= candidate.offset:
				right_stop = candidate
				break
			left_stop = candidate
		span = max(1e-9, right_stop.offset - left_stop.offset)
		amount = min(1.0, max(0.0, (position - left_stop.offset) / span))
		channels = [
			round(left + (right - left) * amount)
			for left, right in zip(
				(
					left_stop.red,
					left_stop.green,
					left_stop.blue,
					left_stop.alpha,
				),
				(
					right_stop.red,
					right_stop.green,
					right_stop.blue,
					right_stop.alpha,
				),
			)
		]
		red, green, blue, alpha = channels
		return (alpha << 24) | (blue << 16) | (green << 8) | red


def parser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Add one 12-piece SVG set to Gadidae compiled geometry."
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
		"--catalog",
		type=Path,
		default=Path("src/graphics/pieces"),
		help="Source catalog retained for regenerating every added style.",
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


def gradient_coordinate(value: str) -> GradientCoordinate:
	text = value.strip()
	if text.endswith("%"):
		return GradientCoordinate(float(text[:-1]) / 100.0, True)
	return GradientCoordinate(float(text), False)


def declarations(value: str) -> dict[str, str]:
	result: dict[str, str] = {}
	for item in value.split(";"):
		if ":" not in item:
			continue
		name, content = item.split(":", 1)
		result[name.strip()] = content.strip()
	return result


def gradient_definitions(path: Path) -> dict[str, LinearPaint]:
	"""Parse SVG linear gradients, including inherited stops and transforms."""
	root = et.parse(path).getroot()
	nodes = {
		node.attrib["id"]: node
		for node in root.iter()
		if node.tag.rsplit("}", 1)[-1] == "linearGradient" and "id" in node.attrib
	}
	resolved: dict[str, LinearPaint] = {}

	def resolve(name: str, active: set[str]) -> LinearPaint:
		if name in resolved:
			return resolved[name]
		if name in active:
			raise ValueError(f"{path} contains a cyclic gradient reference: {name}")
		node = nodes[name]
		href = next(
			(
				value.lstrip("#")
				for key, value in node.attrib.items()
				if key.rsplit("}", 1)[-1] == "href"
			),
			"",
		)
		base = resolve(href, active | {name}) if href in nodes else None
		attributes = dict(node.attrib)
		stops: list[ColorStop] = []
		for stop in node:
			if stop.tag.rsplit("}", 1)[-1] != "stop":
				continue
			values = declarations(stop.attrib.get("style", ""))
			values.update(stop.attrib)
			color = packed_color(values.get("stop-color", "#000"), 0xFF000000)
			if color is None:
				continue
			opacity = min(1.0, max(0.0, float(values.get("stop-opacity", "1"))))
			offset_text = values.get("offset", "0").strip()
			offset = (
				float(offset_text[:-1]) / 100.0
				if offset_text.endswith("%")
				else float(offset_text)
			)
			stops.append(
				ColorStop(
					min(1.0, max(0.0, offset)),
					color & 0xFF,
					(color >> 8) & 0xFF,
					(color >> 16) & 0xFF,
					round(((color >> 24) & 0xFF) * opacity),
				)
			)
		if not stops and base is not None:
			stops = list(base.stops)
		if not stops:
			stops = [
				ColorStop(0.0, 0, 0, 0, 255),
				ColorStop(1.0, 0, 0, 0, 255),
			]
		stops.sort(key=lambda stop: stop.offset)

		def inherited(attribute: str, default: str) -> str:
			if attribute in attributes:
				return attributes[attribute]
			if base is None:
				return default
			coordinate = getattr(base, attribute)
			return (
				f"{coordinate.value * 100.0}%"
				if coordinate.percentage
				else str(coordinate.value)
			)

		units = attributes.get(
			"gradientUnits",
			"userSpaceOnUse" if base is not None and base.user_space else "objectBoundingBox",
		)
		transform_text = attributes.get("gradientTransform")
		transform = (
			Matrix(transform_text)
			if transform_text
			else (base.transform if base is not None else Matrix())
		)
		paint = LinearPaint(
			gradient_coordinate(inherited("x1", "0%")),
			gradient_coordinate(inherited("y1", "0%")),
			gradient_coordinate(inherited("x2", "100%")),
			gradient_coordinate(inherited("y2", "0%")),
			tuple(stops),
			transform,
			units == "userSpaceOnUse",
		)
		resolved[name] = paint
		return paint

	for name in nodes:
		resolve(name, set())
	return resolved


def paint_value(element: Shape, name: str) -> object:
	"""Return the source paint before svgelements resolves unsupported gradients."""
	values = getattr(element, "values", {})
	if name in values:
		return values[name]
	return getattr(element, name, None)


def resolved_paint(
	element: Shape,
	name: str,
	gradients: dict[str, LinearPaint],
	fallback: int | None,
) -> int | LinearPaint | None:
	raw = paint_value(element, name)
	match = URL_PAINT.match(str(raw).strip())
	if match:
		return gradients.get(match.group(1), fallback)
	return packed_color(raw, fallback)


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
	paint: int | LinearPaint,
	frame: tuple[float, float, float, float],
	vertices: list[tuple[float, float, int]],
) -> int:
	first_vertex = len(vertices)
	bounds = geometry.bounds
	for triangle in constrained_delaunay_triangles(geometry).geoms:
		for x, y in list(triangle.exterior.coords)[:3]:
			nx, ny = normalized((x, y), frame)
			color = (
				paint
				if isinstance(paint, int)
				else paint.color_at(x, y, bounds, frame)
			)
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
	gradients = gradient_definitions(path)
	first_command = len(commands)
	white_piece = path.stem.startswith("w")
	default_fill = 0xFFF2F2F2 if white_piece else 0xFF202020
	default_stroke = 0xFF202020 if white_piece else 0xFFF2F2F2
	for element in SVG.parse(str(path)).elements():
		if not isinstance(element, Shape):
			continue
		subpaths = sampled_subpaths(element, curve_step)
		# svgelements resolves an unsupported url(#gradient) paint to black.
		# Reading the raw value preserves the URL so packed_color can use the
		# side-specific fallback instead of silently recoloring white pieces.
		fill = resolved_paint(element, "fill", gradients, default_fill)
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
		stroke = resolved_paint(element, "stroke", gradients, None)
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
	styles: list[tuple[str, int]],
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
	style_lines = [
		f'\t{{"{name}", {geometry_first}U}},'
		for name, geometry_first in styles
	]
	return "\n".join(
		(
			BEGIN,
			f"inline constexpr std::array<EmbeddedPieceVertex, {len(vertices)}> generated_piece_vertices = {{{{",
			*vertex_lines,
			"}};",
			f"inline constexpr std::array<EmbeddedPieceCommand, {len(commands)}> generated_piece_commands = {{{{",
			*command_lines,
			"}};",
			f"inline constexpr std::array<EmbeddedPieceGeometry, {len(geometries)}> generated_piece_geometries = {{{{",
			*geometry_lines,
			"}};",
			f"inline constexpr std::array<EmbeddedPieceStyle, {len(styles)}> generated_piece_styles = {{{{",
			*style_lines,
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
	if style_name in RESERVED_STYLES:
		raise ValueError(f"{style_name} is already a built-in style")

	# Retain normalized SVG sources so later imports can regenerate every
	# previously added style into one compact compiled geometry catalog.
	for piece in PIECES:
		source_path(args.input, piece)
	target = args.catalog / style_name
	target.mkdir(parents=True, exist_ok=True)
	for piece in PIECES:
		shutil.copy2(source_path(args.input, piece), target / f"{piece}.svg")

	styles: list[tuple[str, int]] = []
	vertices: list[tuple[float, float, int]] = []
	commands: list[tuple[int, int, int, float, int, bool]] = []
	geometries: list[tuple[int, int]] = []
	for directory in sorted(path for path in args.catalog.iterdir() if path.is_dir()):
		name = directory.name.lower()
		if not re.fullmatch(r"[a-z][a-z0-9-]*", name):
			raise ValueError(f"invalid style directory name: {directory.name}")
		if name in RESERVED_STYLES:
			raise ValueError(f"generated style conflicts with built-in style: {name}")
		styles.append((name, len(geometries)))
		for piece in PIECES:
			geometries.append(
				append_piece(
					source_path(directory, piece),
					args.curve_step,
					vertices,
					commands,
				)
			)
	document = args.output.read_text(encoding="utf-8")
	args.output.write_text(
		replace_block(
			document,
			array_block(styles, vertices, commands, geometries),
		),
		encoding="utf-8",
		newline="\n",
	)
	print(
		"piece style embedded: "
		f"input={args.input} output={args.output} "
		f"styles={len(styles)} vertices={len(vertices)} commands={len(commands)}"
	)


if __name__ == "__main__":
	main()
