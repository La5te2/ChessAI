#!/usr/bin/env python3
"""Render the learned Gadus square relations from one checkpoint."""

from __future__ import annotations

import argparse
import re
import struct
import tempfile
import uuid
import zipfile
import zlib
from datetime import datetime
from functools import cache
from pathlib import Path

import torch


OUTPUT_DIRECTORY = Path("data")
COLOR_EPSILON = 1e-30
DISPLACEMENT_WIDTH = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render every learned Gadus relation group from one checkpoint."
    )
    parser.add_argument("--model", required=True, type=Path, help="Gadus checkpoint")
    parser.add_argument(
        "--source",
        default="e4",
        help="source square shown in every heatmap (default: e4)",
    )
    return parser.parse_args()


def square_index(name: str) -> int:
    if not re.fullmatch(r"[a-h][1-8]", name):
        raise ValueError("--source must be a square from a1 through h8")
    return (int(name[1]) - 1) * 8 + ord(name[0]) - ord("a")


def natural_key(text: str):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def require(state: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    try:
        return state[key]
    except KeyError as error:
        nearby = [name for name in state if key.split(".")[-1] in name]
        raise RuntimeError(f"checkpoint is missing {key}; nearby keys: {nearby[:8]}") from error


def load_state(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(path)
    archive = torch.jit.load(str(path), map_location="cpu")
    root = getattr(archive, "model", archive)
    state = dict(root.state_dict())
    if not state:
        raise RuntimeError("checkpoint contains no model state")
    return state


def relation_prefixes(state: dict[str, torch.Tensor]) -> list[str]:
    suffix = ".relation_coefficients"
    prefixes = sorted(
        (name[: -len(suffix)] for name in state if name.endswith(suffix)),
        key=natural_key,
    )
    if not prefixes:
        raise RuntimeError("checkpoint contains no Gadus relation blocks")
    return prefixes


@cache
def displacement_geometry() -> tuple[torch.Tensor, torch.Tensor]:
    index = torch.empty((64, 64), dtype=torch.int64)
    scale = torch.empty((64, 64), dtype=torch.float64)
    for target in range(64):
        target_rank, target_file = divmod(target, 8)
        for source in range(64):
            source_rank, source_file = divmod(source, 8)
            rank_delta = target_rank - source_rank
            file_delta = target_file - source_file
            index[target, source] = (
                (rank_delta + 7) * DISPLACEMENT_WIDTH + file_delta + 7
            )
            support = (8 - abs(rank_delta)) * (8 - abs(file_delta))
            scale[target, source] = support**-0.5
    return index, scale


def fused_relation(state: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    coefficients = require(state, f"{prefix}.relation_coefficients")
    residual = require(state, f"{prefix}.relation_residual")
    if coefficients.ndim != 2 or coefficients.shape[1] != 225:
        raise RuntimeError(
            f"checkpoint relation coefficients have unsupported shape {tuple(coefficients.shape)}"
        )
    index, scale = displacement_geometry()
    relation = coefficients.index_select(1, index.reshape(-1)).reshape(-1, 64, 64) * scale
    return (relation + residual).detach().to(dtype=torch.float64, device="cpu")


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def heat_color(value: float) -> tuple[int, int, int]:
    amount = min(1.0, abs(value))
    target = (188, 35, 50) if value >= 0.0 else (35, 90, 170)
    return tuple(round(255 + amount * (channel - 255)) for channel in target)


def write_relation_sheet(
    path: Path,
    relations: list[torch.Tensor],
    source: int,
) -> None:
    block_count = len(relations)
    group_count = int(relations[0].shape[0])
    if any(tuple(item.shape) != (group_count, 64, 64) for item in relations):
        raise RuntimeError("relation blocks have inconsistent shapes")

    cell = 10
    margin = 4
    label_height = 13
    gap = 4
    tile_width = 2 * margin + 8 * cell
    tile_height = 2 * margin + label_height + 8 * cell
    width = group_count * tile_width + (group_count - 1) * gap
    height = block_count * tile_height + (block_count - 1) * gap
    pixels = bytearray([238, 238, 238] * width * height)

    glyphs = {
        "0": ("111", "101", "101", "101", "111"),
        "1": ("010", "110", "010", "010", "111"),
        "2": ("111", "001", "111", "100", "111"),
        "3": ("111", "001", "111", "001", "111"),
        "4": ("101", "101", "111", "001", "001"),
        "5": ("111", "100", "111", "001", "111"),
        "6": ("111", "100", "111", "101", "111"),
        "7": ("111", "001", "010", "010", "010"),
        "8": ("111", "101", "111", "101", "111"),
        "9": ("111", "101", "111", "001", "111"),
        ":": ("0", "1", "0", "1", "0"),
    }

    def paint(x: int, y: int, color: tuple[int, int, int]) -> None:
        offset = (y * width + x) * 3
        pixels[offset : offset + 3] = bytes(color)

    def draw_label(text: str, x0: int, y0: int) -> None:
        cursor = x0
        for character in text:
            glyph = glyphs[character]
            glyph_width = len(glyph[0])
            for y, line in enumerate(glyph):
                for x, bit in enumerate(line):
                    if bit == "1":
                        for yy in range(2):
                            for xx in range(2):
                                paint(cursor + x * 2 + xx, y0 + y * 2 + yy, (55, 55, 55))
            cursor += glyph_width * 2 + 2

    for block_index, block in enumerate(relations):
        for group_index, matrix in enumerate(block):
            tile_x = group_index * (tile_width + gap)
            tile_y = block_index * (tile_height + gap)
            draw_label(
                f"{block_index + 1}:{group_index + 1}",
                tile_x + margin,
                tile_y + 1,
            )
            values = matrix[:, source].reshape(8, 8)
            maximum = max(float(values.abs().max()), COLOR_EPSILON)
            board_y = tile_y + margin + label_height
            for rank_from_top in range(8):
                rank = 7 - rank_from_top
                for file in range(8):
                    color = heat_color(float(values[rank, file]) / maximum)
                    x0 = tile_x + margin + file * cell
                    y0 = board_y + rank_from_top * cell
                    for y in range(y0, y0 + cell):
                        for x in range(x0, x0 + cell):
                            paint(x, y, color)

            source_rank, source_file = divmod(source, 8)
            x0 = tile_x + margin + source_file * cell
            y0 = board_y + (7 - source_rank) * cell
            for offset in range(cell):
                paint(x0 + offset, y0, (35, 35, 35))
                paint(x0 + offset, y0 + cell - 1, (35, 35, 35))
                paint(x0, y0 + offset, (35, 35, 35))
                paint(x0 + cell - 1, y0 + offset, (35, 35, 35))

    raw = b"".join(
        b"\x00" + pixels[y * width * 3 : (y + 1) * width * 3]
        for y in range(height)
    )
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(raw, level=9))
        + png_chunk(b"IEND", b"")
    )
    path.write_bytes(payload)


def main() -> None:
    args = parse_args()
    source = square_index(args.source)
    state = load_state(args.model)
    prefixes = relation_prefixes(state)
    relations = [fused_relation(state, prefix) for prefix in prefixes]

    run_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    archive_path = OUTPUT_DIRECTORY / f"{run_id}-{timestamp}.zip"

    with tempfile.TemporaryDirectory(prefix="gadus-visual-") as temporary:
        image_path = Path(temporary) / "relations.png"
        write_relation_sheet(image_path, relations, source)
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as output:
            output.write(image_path, arcname=image_path.name)

    print(
        f"rendered model={args.model} source={args.source} "
        f"blocks={len(relations)} groups={relations[0].shape[0]}"
    )
    print(f"archive={archive_path}")


if __name__ == "__main__":
    main()
