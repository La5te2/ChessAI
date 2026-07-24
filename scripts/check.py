from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

try:
    import torch
except ImportError as error:
    raise SystemExit("check requires Python PyTorch for reading LibTorch checkpoints") from error


ARCHITECTURES = {
    1: ("gadus", "policy, value"),
    2: ("melano", "policy, value, advantage"),
}
REQUIRED_ARCH_FIELDS = ("type_id", "channels", "blocks", "action_size")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_fields(module) -> dict[str, int]:
    parameters = dict(module.named_parameters())
    if parameters:
        raise ValueError(
            "checkpoint arch metadata must be registered as buffers, not parameters"
        )
    fields = dict(module.named_buffers())
    missing = [name for name in REQUIRED_ARCH_FIELDS if name not in fields]
    if missing:
        raise ValueError(f"checkpoint arch is missing fields: {', '.join(missing)}")
    unexpected = sorted(set(fields) - set(REQUIRED_ARCH_FIELDS))
    if unexpected:
        raise ValueError(f"checkpoint arch has unexpected fields: {', '.join(unexpected)}")

    values = {}
    for name in REQUIRED_ARCH_FIELDS:
        tensor = fields[name]
        if tensor.numel() != 1:
            raise ValueError(f"checkpoint arch field is not scalar: {name}")
        values[name] = int(tensor.detach().cpu().item())
    return values


def tensor_bytes(tensors) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def all_finite(tensors) -> bool:
    for tensor in tensors:
        value = tensor.detach()
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all().item():
            return False
    return True


def format_mib(byte_count: int) -> str:
    return f"{byte_count / (1024 * 1024):.2f} MiB"


def inspect_model(path: Path) -> dict[str, object]:
    try:
        archive = torch.jit.load(str(path), map_location="cpu")
    except Exception as error:
        raise ValueError("file is not a readable current LibTorch checkpoint") from error

    children = dict(archive.named_children())
    if set(children) != {"model", "arch"}:
        names = ", ".join(sorted(children)) or "<empty>"
        raise ValueError(f"checkpoint top level must contain only model and arch; found: {names}")

    arch = scalar_fields(children["arch"])
    architecture, heads = ARCHITECTURES.get(
        arch["type_id"], (f"unknown(type_id={arch['type_id']})", "unknown")
    )
    model = children["model"]
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    tensors = parameters + buffers
    dtype_counts = Counter(str(tensor.dtype).removeprefix("torch.") for tensor in tensors)
    devices = sorted({str(tensor.device) for tensor in tensors})

    return {
        "architecture": architecture,
        "heads": heads,
        "channels": arch["channels"],
        "blocks": arch["blocks"],
        "action_size": arch["action_size"],
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
        "finite": all_finite(tensors),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display basic information about a current Gadidae LibTorch checkpoint."
    )
    parser.add_argument("--model", required=True, type=Path)
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
    print(f"channels: {info['channels']}")
    print(f"blocks: {info['blocks']}")
    print(f"action_size: {info['action_size']}")
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
