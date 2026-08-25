#!/usr/bin/env python3
"""Bootstrap a fixed Gadus square-relation dictionary from trained checkpoints."""

from __future__ import annotations

import argparse
import json
import re
import struct
import tempfile
import uuid
import zipfile
import zlib
from datetime import datetime
from pathlib import Path

import torch


SVD_ITERATIONS = 4
BASIS_EPSILON = 1e-30
OUTPUT_DIRECTORY = Path("data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap a fixed Gadus relation dictionary from one or more checkpoints."
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        type=Path,
        help="Gadus checkpoint; repeat to combine independently trained relation matrices",
    )
    parser.add_argument(
        "--dimension",
        required=True,
        type=int,
        help="relation-dictionary dimension H to produce",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.model:
        raise ValueError("at least one --model checkpoint is required")
    if not 1 <= args.dimension <= 64:
        raise ValueError("--dimension must lie in [1, 64]")


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def natural_key(text: str):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def require(state: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    try:
        return state[key]
    except KeyError as error:
        nearby = [name for name in state if key.split(".")[-1] in name]
        raise RuntimeError(f"checkpoint is missing {key}; nearby keys: {nearby[:8]}") from error


def load_state(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(path)
    archive = torch.jit.load(str(path), map_location=device)
    root = getattr(archive, "model", archive)
    state = dict(root.state_dict())
    if not state:
        raise RuntimeError("checkpoint contains no model state")
    return state


def relation_prefixes(state: dict[str, torch.Tensor]) -> list[str]:
    suffix = ".relation_coefficients"
    prefixes = sorted(
        (name[: -len(suffix)] for name in state if name.endswith(suffix)), key=natural_key
    )
    if not prefixes:
        raise RuntimeError("checkpoint contains no Gadus relation blocks")
    for prefix in prefixes:
        for field in ("relation_coefficients", "relation_residual", "relation_basis"):
            require(state, f"{prefix}.{field}")
    return prefixes


def fused_relation(state: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    coefficients = require(state, f"{prefix}.relation_coefficients")
    basis = require(state, f"{prefix}.relation_basis")
    residual = require(state, f"{prefix}.relation_residual")
    return torch.einsum("gh,hqp->gqp", coefficients, basis) + residual


def orthonormal_span(rows: torch.Tensor, tolerance: float = 1e-10) -> torch.Tensor:
    rows = rows.to(dtype=torch.float64, device="cpu")
    gram = rows @ rows.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    largest = float(eigenvalues[-1].clamp_min(0.0)) if eigenvalues.numel() else 0.0
    keep = eigenvalues > max(tolerance, largest * tolerance)
    if not bool(keep.any()):
        return torch.empty((0, rows.shape[1]), dtype=torch.float64)
    return (eigenvectors[:, keep].T @ rows) / eigenvalues[keep].sqrt().unsqueeze(1)


def normalize_checkpoint_rows(rows: torch.Tensor) -> torch.Tensor:
    rows = rows.detach().to(dtype=torch.float32, device="cpu")
    if rows.ndim != 2 or rows.shape[1] != 4096:
        raise RuntimeError(f"relation samples must have shape [N,4096], got {tuple(rows.shape)}")
    denominator = max(float(torch.linalg.vector_norm(rows)), BASIS_EPSILON)
    return rows / denominator


def parameter_rows(state: dict[str, torch.Tensor], prefixes: list[str]) -> torch.Tensor:
    return torch.cat(
        [fused_relation(state, prefix).detach().reshape(-1, 4096).cpu() for prefix in prefixes],
        dim=0,
    )


def optimal_basis(
    rows: torch.Tensor,
    dimension: int,
    device: torch.device,
):
    if rows.shape[0] == 0:
        raise RuntimeError("the fitting objective contains no relation samples")
    rank_limit = min(rows.shape)
    if dimension > rank_limit:
        raise ValueError(
            f"--dimension {dimension} exceeds the sampled objective rank limit {rank_limit}"
        )
    count = dimension
    matrix = rows.to(device)
    exact = rows.shape[0] <= 512 or count == rank_limit
    if exact:
        _, singular, right = torch.linalg.svd(matrix, full_matrices=False)
        basis = right[:count]
        method = "exact_svd"
    else:
        _, singular, right_columns = torch.pca_lowrank(
            matrix, q=count, center=False, niter=SVD_ITERATIONS
        )
        basis = right_columns.T
        method = "randomized_svd"
    basis = basis.detach().to(dtype=torch.float64, device="cpu")
    basis = torch.linalg.qr(basis.T, mode="reduced").Q.T
    for index in range(basis.shape[0]):
        pivot = int(basis[index].abs().argmax())
        if float(basis[index, pivot]) < 0.0:
            basis[index].neg_()
    return (
        basis,
        singular.detach().double().cpu(),
        method,
    )


def energy_curve(rows: torch.Tensor, basis: torch.Tensor) -> list[float]:
    rows = rows.detach().to(dtype=torch.float64, device="cpu")
    total = float(rows.square().sum())
    if total <= 0.0:
        return [0.0] * basis.shape[0]
    coordinates = rows @ basis.T
    return (coordinates.square().sum(0).cumsum(0) / total).tolist()


def basis_overlap(left: torch.Tensor, right: torch.Tensor, count: int) -> float:
    count = min(count, left.shape[0], right.shape[0])
    if count == 0:
        return 0.0
    overlap = left[:count] @ right[:count].T
    return float(overlap.square().sum() / count)


def effective_rank(singular: torch.Tensor) -> float:
    eigenvalues = singular.square()
    captured = float(eigenvalues.sum())
    denominator = float(eigenvalues.square().sum())
    return captured * captured / denominator if denominator > 0.0 else 0.0


def fixed_basis(state: dict[str, torch.Tensor], prefixes: list[str]) -> torch.Tensor:
    return (
        state[f"{prefixes[0]}.relation_basis"]
        .detach()
        .to(dtype=torch.float64, device="cpu")
        .reshape(-1, 4096)
    )


def resource_estimates(
    points: set[int], prefixes: list[str], state: dict[str, torch.Tensor]
):
    total_groups = sum(int(fused_relation(state, prefix).shape[0]) for prefix in prefixes)
    result = {}
    for count in sorted(points):
        if count <= 0:
            continue
        result[str(count)] = {
            "trainable_coefficients": total_groups * count,
            "shared_dense_basis_floats": 4096 * count,
            "shared_dense_basis_bytes_fp32": 4096 * count * 4,
            "per_block_repeated_basis_bytes_fp32": 4096 * count * 4 * len(prefixes),
            "relation_composition_multiply_adds": total_groups * count * 4096,
        }
    return result


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


def write_basis_sheet(path: Path, basis: torch.Tensor) -> None:
    """Render all e4 source-square slices in a four-column contact sheet."""
    basis = basis.detach().double().cpu()
    source = 3 * 8 + 4
    columns = 4
    rows = (basis.shape[0] + columns - 1) // columns
    cell, margin, label_height, gap = 20, 4, 13, 6
    tile_width = 2 * margin + 8 * cell
    tile_height = 2 * margin + label_height + 8 * cell
    width = columns * tile_width + (columns - 1) * gap
    height = rows * tile_height + (rows - 1) * gap
    pixels = bytearray([238, 238, 238] * width * height)

    digits = {
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
    }

    def paint(x: int, y: int, color: tuple[int, int, int]) -> None:
        offset = (y * width + x) * 3
        pixels[offset : offset + 3] = bytes(color)

    def draw_number(text: str, x0: int, y0: int) -> None:
        for index, character in enumerate(text):
            for y, line in enumerate(digits[character]):
                for x, bit in enumerate(line):
                    if bit == "1":
                        for yy in range(2):
                            for xx in range(2):
                                paint(x0 + index * 8 + x * 2 + xx, y0 + y * 2 + yy, (55, 55, 55))

    for index, matrix in enumerate(basis):
        tile_row, tile_column = divmod(index, columns)
        tile_x = tile_column * (tile_width + gap)
        tile_y = tile_row * (tile_height + gap)
        draw_number(str(index + 1), tile_x + margin, tile_y + 1)
        values = matrix[:, source].reshape(8, 8)
        maximum = max(float(values.abs().max()), BASIS_EPSILON)
        board_y = tile_y + margin + label_height
        for target_rank_from_top in range(8):
            target_rank = 7 - target_rank_from_top
            for target_file in range(8):
                color = heat_color(float(values[target_rank, target_file]) / maximum)
                x0 = tile_x + margin + target_file * cell
                y0 = board_y + target_rank_from_top * cell
                for y in range(y0, y0 + cell):
                    for x in range(x0, x0 + cell):
                        paint(x, y, color)

    raw = b"".join(
        b"\x00" + pixels[y * width * 3 : (y + 1) * width * 3] for y in range(height)
    )
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(raw, level=9))
        + png_chunk(b"IEND", b"")
    )
    path.write_bytes(payload)


def write_cpp_header(path: Path, basis: torch.Tensor) -> None:
    """Write the exact float32 basis in the model's [basis][target][source] order."""
    basis = basis.detach().to(dtype=torch.float32, device="cpu")
    dimension = int(basis.shape[0])

    def literal(value: torch.Tensor) -> str:
        text = f"{float(value):.9g}"
        if "." not in text and "e" not in text:
            text += ".0"
        return text + "F"

    lines = [
        "#pragma once",
        "",
        "#include <array>",
        "#include <cstddef>",
        "",
        "namespace gadus::generated {",
        "",
        f"inline constexpr std::size_t kRelationBasisDimension = {dimension};",
        "inline constexpr std::array<float, kRelationBasisDimension * 64 * 64>",
        "kRelationBasis = {",
    ]
    for index in range(dimension):
        lines.append(f"\t// Basis {index + 1}: rows are target squares q; columns are source squares p.")
        for target in range(64):
            values = ", ".join(
                literal(basis[index, target, source]) for source in range(64)
            )
            lines.append(f"\t{values},")
    lines.extend(["};", "", "}  // namespace gadus::generated", ""])
    path.write_text("\n".join(lines), encoding="ascii")


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = resolve_device()

    parameter_sources = []
    current_spans = []
    reference_state = None
    reference_prefixes = None
    for path in args.model:
        state = load_state(path, device)
        prefixes = relation_prefixes(state)
        if reference_prefixes is not None and prefixes != reference_prefixes:
            raise RuntimeError(f"checkpoint {path} has a different relation-block layout")
        reference_state = state if reference_state is None else reference_state
        reference_prefixes = prefixes if reference_prefixes is None else reference_prefixes
        parameters = parameter_rows(state, prefixes)
        parameter_sources.append(parameters)
        current_spans.append(orthonormal_span(fixed_basis(state, prefixes)))
        print(f"loaded model={path} relation_rows={parameters.shape[0]}")

    normalized_sources = [normalize_checkpoint_rows(rows) for rows in parameter_sources]
    fit_rows = torch.cat(normalized_sources, dim=0)
    basis, singular, decomposition_method = optimal_basis(fit_rows, args.dimension, device)
    orthogonality_error = float(
        (
            basis @ basis.T
            - torch.eye(basis.shape[0], dtype=torch.float64)
        )
        .abs()
        .max()
    )
    fit_curve = energy_curve(fit_rows, basis)
    checkpoint_curves = [energy_curve(rows, basis) for rows in parameter_sources]
    current_explained = [
        energy_curve(rows, current)[-1]
        for rows, current in zip(parameter_sources, current_spans)
    ]

    individual_bases = []
    for rows in normalized_sources:
        own_basis, _, _ = optimal_basis(rows, args.dimension, device)
        individual_bases.append(own_basis)
    stability_curve = None
    if len(individual_bases) > 1:
        stability_curve = [
            min(basis_overlap(basis, own, count) for own in individual_bases)
            for count in range(1, basis.shape[0] + 1)
        ]
    dimension = args.dimension
    checkpoints = [
        {
            "automatic_basis_parameter_explained": checkpoint_curves[index][dimension - 1],
            "current_basis_parameter_explained": current_explained[index],
        }
        for index in range(len(args.model))
    ]

    points = {1, 2, 4, 8, 16, 32, 64, dimension}
    resources = resource_estimates(points, reference_prefixes, reference_state)
    run_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    archive_path = OUTPUT_DIRECTORY / f"{run_id}-{timestamp}.zip"

    with tempfile.TemporaryDirectory(prefix="gadus-bases-") as temporary:
        directory = Path(temporary)
        summary_path = directory / "summary.json"
        metrics_path = directory / "metrics.json"
        basis_path = directory / "basis.pt"
        header_path = directory / "basis.hpp"
        image_path = directory / "bases.png"

        selected_resources = resources[str(dimension)]
        current_resources = resources[str(int(current_spans[0].shape[0]))]
        summary = {
            "dimension": dimension,
            "fit_explained": fit_curve[dimension - 1],
            "stability": (
                stability_curve[dimension - 1] if stability_curve is not None else None
            ),
            "checkpoints": checkpoints,
            "numerics": {
                "orthogonality_error": orthogonality_error,
            },
            "cost": {
                "current_basis": {
                    "trainable_coefficients": current_resources["trainable_coefficients"],
                    "basis_bytes_fp32": current_resources["shared_dense_basis_bytes_fp32"],
                    "composition_multiply_adds": current_resources[
                        "relation_composition_multiply_adds"
                    ],
                },
                "automatic_basis": {
                    "trainable_coefficients": selected_resources["trainable_coefficients"],
                    "basis_bytes_fp32": selected_resources["shared_dense_basis_bytes_fp32"],
                    "composition_multiply_adds": selected_resources[
                        "relation_composition_multiply_adds"
                    ],
                },
            },
            "image": {
                "file": image_path.name,
                "source_square": "e4",
                "columns": 4,
                "count": dimension,
            },
        }

        overlap_by_h = None
        if len(individual_bases) > 1:
            overlap_by_h = {
                str(count): [
                    basis_overlap(basis, own, count) for own in individual_bases
                ]
                for count in range(1, dimension + 1)
            }
        metrics = {
            "run_id": run_id,
            "created_at": timestamp,
            "device": str(device),
            "checkpoint_count": len(args.model),
            "normalization": "unit Frobenius norm per checkpoint",
            "dimension": dimension,
            "decomposition_method": decomposition_method,
            "singular_values": singular.tolist(),
            "effective_rank_of_computed_spectrum": effective_rank(singular),
            "fit_explained_curve": fit_curve,
            "checkpoint_parameter_curves": checkpoint_curves,
            "current_basis_parameter_explained": current_explained,
            "stability_curve": stability_curve,
            "checkpoint_subspace_overlap_by_h": overlap_by_h,
            "resource_estimates": resources,
            "image_source_square": "e4",
        }

        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        torch.save({
            "basis": basis.reshape(dimension, 64, 64).float(),
            "singular_values": singular.float(),
            "dimension": dimension,
        }, basis_path)
        write_cpp_header(header_path, basis.reshape(dimension, 64, 64))
        write_basis_sheet(image_path, basis.reshape(dimension, 64, 64))

        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as output:
            for path in sorted(directory.iterdir()):
                output.write(path, arcname=path.name)

    print(
        f"designed dimension={dimension} checkpoints={len(args.model)} "
        f"orthogonality_error={orthogonality_error:.3e}"
    )
    print(
        "fit explained: "
        + " ".join(
            f"H={k}:{fit_curve[k - 1]:.4f}" for k in sorted(points) if k <= len(fit_curve)
        )
    )
    for index, item in enumerate(checkpoints, start=1):
        print(
            f"checkpoint={index} automatic@H="
            f"{item['automatic_basis_parameter_explained']:.4f} "
            f"current={item['current_basis_parameter_explained']:.4f}"
        )
    print(f"archive={archive_path}")


if __name__ == "__main__":
    main()
