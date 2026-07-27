"""Convert one SVG piece set into Gadidae's compressed geometry archive."""

from __future__ import annotations

import argparse
import math
import os
import re
import struct
import xml.etree.ElementTree as et
import zlib
from dataclasses import dataclass
from pathlib import Path

from shapely import constrained_delaunay_triangles
from shapely.geometry import GeometryCollection, LineString, LinearRing, Polygon
from svgelements import Matrix
from svgelements import Path as SvgPath
from svgelements import SVG, Shape


PIECES = ("wK", "wQ", "wR", "wB", "wN", "wP", "bK", "bQ", "bR", "bB", "bN", "bP")
RESERVED_STYLES = {"vector", "rhosgfx", "chessnut", "spatial", "cburnett"}
HEX_COLOR = re.compile(r"^#([0-9a-fA-F]{3,8})$")
URL_PAINT = re.compile(r"^url\(\s*#([^)]+)\s*\)$")
COMPRESSED_MAGIC = b"GPCZ"
ARCHIVE_MAGIC = b"GPS1"
MAX_ARCHIVE_SIZE = 256 * 1024 * 1024


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


@dataclass
class PackedMesh:
	vertices: list[tuple[float, float, int]]
	indices: list[int]


@dataclass
class PackedStyle:
	name: str
	pieces: list[PackedMesh]


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
		default=Path("src/graphics/pieces.gpack"),
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


def append_piece(path: Path, curve_step: float) -> PackedMesh:
	"""Triangulate one SVG and deduplicate its exact normalized vertices."""
	frame = svg_frame(path)
	gradients = gradient_definitions(path)
	triangle_vertices: list[tuple[float, float, int]] = []
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
			# Constrained triangulation preserves concave boundaries and holes.
			append_triangles(geometry, fill, frame, triangle_vertices)
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
				geometry = stroke_geometry(
					points,
					closed,
					width,
					linecap,
					linejoin,
				)
				append_triangles(
					geometry,
					stroke or default_stroke,
					frame,
					triangle_vertices,
				)

	vertices: list[tuple[float, float, int]] = []
	indices: list[int] = []
	seen: dict[bytes, int] = {}
	for x, y, color in triangle_vertices:
		# Keep the old generator's seven-decimal normalization before storing
		# float32 coordinates, so migration does not change visible geometry.
		key = struct.pack("<ffI", round(x, 7), round(y, 7), color)
		index = seen.get(key)
		if index is None:
			index = len(vertices)
			if index >= 65536:
				raise ValueError(f"{path} exceeds the uint16 vertex limit")
			px, py, packed = struct.unpack("<ffI", key)
			vertices.append((px, py, packed))
			seen[key] = index
		indices.append(index)
	return PackedMesh(vertices, indices)


def read_exact(view: memoryview, offset: int, size: int) -> tuple[memoryview, int]:
	"""Read one bounded archive slice."""
	end = offset + size
	if size < 0 or end > len(view):
		raise ValueError("truncated piece archive")
	return view[offset:end], end


def decode_archive(path: Path) -> list[PackedStyle]:
	"""Load the existing compressed archive for replace-or-append imports."""
	if not path.exists():
		return []
	data = path.read_bytes()
	if len(data) < 12 or data[:4] != COMPRESSED_MAGIC:
		raise ValueError(f"{path} is not a Gadidae piece archive")
	raw_size = struct.unpack_from("<Q", data, 4)[0]
	if raw_size > MAX_ARCHIVE_SIZE:
		raise ValueError(f"{path} expands beyond {MAX_ARCHIVE_SIZE} bytes")
	raw = zlib.decompress(data[12:])
	if len(raw) != raw_size or raw[:4] != ARCHIVE_MAGIC:
		raise ValueError(f"{path} has an invalid payload")
	view = memoryview(raw)
	offset = 4
	if offset + 4 > len(view):
		raise ValueError("truncated piece archive")
	style_count = struct.unpack_from("<I", view, offset)[0]
	offset += 4
	styles: list[PackedStyle] = []
	for _ in range(style_count):
		if offset + 2 > len(view):
			raise ValueError("truncated piece archive")
		name_size = struct.unpack_from("<H", view, offset)[0]
		offset += 2
		name_data, offset = read_exact(view, offset, name_size)
		name = bytes(name_data).decode("utf-8")
		pieces: list[PackedMesh] = []
		for _piece in PIECES:
			if offset + 8 > len(view):
				raise ValueError("truncated piece archive")
			vertex_count, index_count = struct.unpack_from("<II", view, offset)
			offset += 8
			if vertex_count > 65535 or index_count % 3:
				raise ValueError("invalid piece mesh dimensions")
			vertex_data, offset = read_exact(view, offset, vertex_count * 12)
			index_data, offset = read_exact(view, offset, index_count * 2)
			vertices = [
				struct.unpack_from("<ffI", vertex_data, index * 12)
				for index in range(vertex_count)
			]
			indices = list(struct.unpack(f"<{index_count}H", index_data))
			if indices and max(indices) >= vertex_count:
				raise ValueError("piece mesh index is out of range")
			pieces.append(PackedMesh(vertices, indices))
		styles.append(PackedStyle(name, pieces))
	if offset != len(view):
		raise ValueError("piece archive contains trailing data")
	return styles


def encode_archive(styles: list[PackedStyle]) -> bytes:
	"""Serialize styles deterministically, then compress the complete payload."""
	raw = bytearray(ARCHIVE_MAGIC)
	raw.extend(struct.pack("<I", len(styles)))
	for style in styles:
		name = style.name.encode("utf-8")
		if len(name) > 65535 or len(style.pieces) != len(PIECES):
			raise ValueError(f"invalid style record: {style.name}")
		raw.extend(struct.pack("<H", len(name)))
		raw.extend(name)
		for mesh in style.pieces:
			if len(mesh.vertices) > 65535 or len(mesh.indices) % 3:
				raise ValueError(f"invalid mesh in style: {style.name}")
			raw.extend(struct.pack("<II", len(mesh.vertices), len(mesh.indices)))
			for vertex in mesh.vertices:
				raw.extend(struct.pack("<ffI", *vertex))
			raw.extend(struct.pack(f"<{len(mesh.indices)}H", *mesh.indices))
	if len(raw) > MAX_ARCHIVE_SIZE:
		raise ValueError(f"piece archive exceeds {MAX_ARCHIVE_SIZE} bytes")
	return COMPRESSED_MAGIC + struct.pack("<Q", len(raw)) + zlib.compress(raw, 9)


def atomic_write(path: Path, data: bytes) -> None:
	"""Replace the archive only after the complete new payload is durable."""
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(f"{path.name}.tmp_{os.getpid()}")
	try:
		temporary.write_bytes(data)
		os.replace(temporary, path)
	finally:
		temporary.unlink(missing_ok=True)


def main() -> None:
	args = parser().parse_args()
	if args.curve_step <= 0:
		raise ValueError("--curve-step must be positive")
	style_name = args.name.strip().lower()
	if not re.fullmatch(r"[a-z][a-z0-9-]*", style_name):
		raise ValueError("--name must be a lowercase identifier such as cburnett")
	if style_name in RESERVED_STYLES:
		raise ValueError(f"{style_name} is already a built-in style")

	for piece in PIECES:
		source_path(args.input, piece)
	style = PackedStyle(
		style_name,
		[append_piece(source_path(args.input, piece), args.curve_step) for piece in PIECES],
	)
	styles = decode_archive(args.output)
	for index, current in enumerate(styles):
		if current.name == style_name:
			styles[index] = style
			break
	else:
		styles.append(style)
	archive = encode_archive(styles)
	atomic_write(args.output, archive)
	vertex_count = sum(len(mesh.vertices) for mesh in style.pieces)
	index_count = sum(len(mesh.indices) for mesh in style.pieces)
	print(
		"piece style archived: "
		f"input={args.input} output={args.output} "
		f"styles={len(styles)} vertices={vertex_count} indices={index_count} "
		f"archive_bytes={len(archive)}"
	)


if __name__ == "__main__":
	main()
