from __future__ import annotations

import argparse
import hashlib
import math
import struct
import sys
from array import array
from collections import Counter
from pathlib import Path

ELEGINUS_MAGIC = b"ELEGINUS"
ELEGINUS_TYPE_ID = 3
ELEGINUS_HEADER = struct.Struct("=8sIIIIIIQ")

ARCHITECTURES = {
    1: {
        "name": "gadus",
        "heads": "policy, value",
        "fields": ("type_id", "channels", "blocks", "action_size"),
    },
    2: {
        "name": "melano",
        "heads": "policy, value",
        "fields": ("type_id", "channels", "blocks", "action_size"),
    },
}


def load_torch():
    try:
        import torch
    except Exception as error:
        raise SystemExit(
            f"check requires PyTorch, but {sys.executable} cannot import torch: {error}"
        ) from error
    return torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_fields(module) -> tuple[dict[str, int], dict[str, object]]:
    parameters = dict(module.named_parameters())
    if parameters:
        raise ValueError(
            "checkpoint arch metadata must be registered as buffers, not parameters"
        )
    fields = dict(module.named_buffers())
    if "type_id" not in fields:
        raise ValueError("checkpoint arch is missing fields: type_id")
    type_tensor = fields["type_id"]
    if type_tensor.numel() != 1:
        raise ValueError("checkpoint arch field is not scalar: type_id")
    type_id = int(type_tensor.detach().cpu().item())
    specification = ARCHITECTURES.get(type_id)
    if specification is None:
        raise ValueError(f"checkpoint has unknown architecture type_id: {type_id}")
    required_fields = specification["fields"]
    missing = [name for name in required_fields if name not in fields]
    if missing:
        raise ValueError(f"checkpoint arch is missing fields: {', '.join(missing)}")
    unexpected = sorted(set(fields) - set(required_fields))
    if unexpected:
        raise ValueError(f"checkpoint arch has unexpected fields: {', '.join(unexpected)}")

    values = {}
    for name in required_fields:
        tensor = fields[name]
        if tensor.numel() != 1:
            raise ValueError(f"checkpoint arch field is not scalar: {name}")
        values[name] = int(tensor.detach().cpu().item())
    return values, specification


def tensor_bytes(tensors) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def all_finite(tensors, torch) -> bool:
    for tensor in tensors:
        value = tensor.detach()
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all().item():
            return False
    return True


def format_mib(byte_count: int) -> str:
    return f"{byte_count / (1024 * 1024):.2f} MiB"


def inspect_eleginus(path: Path) -> dict[str, object] | None:
    with path.open("rb") as stream:
        header = stream.read(ELEGINUS_HEADER.size)
        if header[:8] != ELEGINUS_MAGIC:
            return None
        if len(header) != ELEGINUS_HEADER.size:
            raise ValueError("truncated Eleginus checkpoint header")
        _, type_id, n, z, width, layers, count, _ = ELEGINUS_HEADER.unpack(header)
        if type_id != ELEGINUS_TYPE_ID or n == 0 or z != n + 16 or width == 0 or layers == 0:
            raise ValueError("inconsistent Eleginus dimensions")
        expected = n + (n + z) * width + layers * (width * width + width)
        if count != expected or path.stat().st_size != ELEGINUS_HEADER.size + count * 4:
            raise ValueError("Eleginus checkpoint size does not match its parameter dimensions")
        weights = array("f")
        weights.fromfile(stream, count)

    tensors = 3 + 2 * layers
    return {
        "architecture": "eleginus",
        "heads": "dynamic formula value",
        "arch": {"type_id": type_id, "formulas": n, "inputs": z, "width": width, "layers": layers},
        "model_children": "v, E, U, G, b",
        "parameters": count,
        "trainable_parameters": count,
        "parameter_tensors": tensors,
        "buffers": 0,
        "tensor_memory": count * 4,
        "dtypes": f"float32 ({tensors})",
        "devices": "cpu",
        "finite": all(math.isfinite(weight) for weight in weights),
    }


def inspect_libtorch(path: Path) -> dict[str, object]:
    torch = load_torch()
    try:
        archive = torch.jit.load(str(path), map_location="cpu")
    except Exception as error:
        raise ValueError("file is not a readable Gadidae LibTorch checkpoint") from error

    children = dict(archive.named_children())
    if set(children) != {"model", "arch"}:
        names = ", ".join(sorted(children)) or "<empty>"
        raise ValueError(f"checkpoint top level must contain only model and arch; found: {names}")

    arch, specification = scalar_fields(children["arch"])
    model = children["model"]
    model_children = tuple(name for name, _ in model.named_children())
    expected_children = specification.get("model_children")
    if expected_children is not None and set(model_children) != set(expected_children):
        names = ", ".join(model_children) or "<empty>"
        raise ValueError(
            f"{specification['name']} model has unexpected children: {names}"
        )
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    tensors = parameters + buffers
    dtype_counts = Counter(str(tensor.dtype).removeprefix("torch.") for tensor in tensors)
    devices = sorted({str(tensor.device) for tensor in tensors})

    return {
        "architecture": specification["name"],
        "heads": specification["heads"],
        "arch": arch,
        "model_children": ", ".join(model_children) or "none",
        "parameters": sum(tensor.numel() for tensor in parameters),
        "trainable_parameters": sum(
            tensor.numel() for tensor in parameters if tensor.requires_grad
        ),
        "parameter_tensors": len(parameters),
        "buffers": len(buffers),
        "tensor_memory": tensor_bytes(tensors),
        "dtypes": ", ".join(
            f"{name} ({count})" for name, count in sorted(dtype_counts.items())
        ),
        "devices": ", ".join(devices) or "none",
        "finite": all_finite(tensors, torch),
    }


def inspect_model(path: Path) -> dict[str, object]:
    eleginus = inspect_eleginus(path)
    return eleginus if eleginus is not None else inspect_libtorch(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a Gadidae checkpoint and detect its architecture."
    )
    parser.add_argument("model", type=Path, metavar="checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.model.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"model not found: {path}")

    info = inspect_model(path)
    print(f"model: {path}")
    print(f"file_size: {format_mib(path.stat().st_size)}")
    print(f"sha256: {sha256(path)}")
    print(f"architecture: {info['architecture']}")
    print(f"heads: {info['heads']}")
    for name, value in info["arch"].items():
        print(f"arch.{name}: {value}")
    print(f"model_children: {info['model_children']}")
    print(f"parameters: {info['parameters']}")
    print(f"trainable_parameters: {info['trainable_parameters']}")
    print(f"parameter_tensors: {info['parameter_tensors']}")
    print(f"buffers: {info['buffers']}")
    print(f"tensor_memory: {format_mib(int(info['tensor_memory']))}")
    print(f"dtypes: {info['dtypes']}")
    print(f"devices: {info['devices']}")
    print(f"finite: {str(info['finite']).lower()}")


if __name__ == "__main__":
    main()
