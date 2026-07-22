from __future__ import annotations

import argparse
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import torch
    from torch import nn
except ImportError as error:
    raise SystemExit(
        "transit requires Python PyTorch for reading historical .pth files"
    ) from error


GADUS_ARCH_NAMES = {"gadus", "resnet_pv_linear"}
GADUS_ACTION_SIZE = 64 * 73


class GadusResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.relu(inputs + self.block(inputs))


class GadusModel(nn.Module):
    def __init__(self, channels: int, blocks: int) -> None:
        super().__init__()
        backbone: list[nn.Module] = [
            nn.Conv2d(18, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        ]
        backbone.extend(GadusResidualBlock(channels) for _ in range(blocks))
        self.backbone = nn.Sequential(*backbone)
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, GADUS_ACTION_SIZE),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(inputs)
        return self.policy_head(features), self.value_head(features)


class ArchitectureArchive(nn.Module):
    def __init__(self, type_id: int, channels: int, blocks: int, action_size: int) -> None:
        super().__init__()
        self.register_buffer("type_id", torch.tensor(type_id, dtype=torch.int64))
        self.register_buffer("channels", torch.tensor(channels, dtype=torch.int64))
        self.register_buffer("blocks", torch.tensor(blocks, dtype=torch.int64))
        self.register_buffer("action_size", torch.tensor(action_size, dtype=torch.int64))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


class GadusCheckpointArchive(nn.Module):
    def __init__(self, model: GadusModel, channels: int, blocks: int) -> None:
        super().__init__()
        self.model = model
        self.arch = ArchitectureArchive(1, channels, blocks, GADUS_ACTION_SIZE)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(inputs)


def load_historical_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], Mapping[str, Any]]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(
            "input is not a supported historical Python checkpoint or pure state_dict"
        ) from error

    arch: Mapping[str, Any] = {}
    if isinstance(payload, Mapping) and isinstance(payload.get("model"), Mapping):
        state = payload["model"]
        raw_arch = payload.get("arch", {})
        if raw_arch is not None and not isinstance(raw_arch, Mapping):
            raise ValueError("checkpoint arch must be a mapping")
        arch = raw_arch or {}
    elif isinstance(payload, Mapping) and payload and all(
        isinstance(value, torch.Tensor) for value in payload.values()
    ):
        state = payload
    else:
        raise ValueError("checkpoint must contain model and arch, or be a pure state_dict")

    tensors = {str(name): value.detach().cpu() for name, value in state.items()}
    if not tensors or not all(isinstance(value, torch.Tensor) for value in tensors.values()):
        raise ValueError("model state contains non-tensor values")
    return tensors, arch


def infer_gadus(state: Mapping[str, torch.Tensor], arch: Mapping[str, Any]) -> tuple[int, int]:
    declared = str(arch.get("type", "")).strip().lower()
    stem = state.get("backbone.0.weight")
    policy = state.get("policy_head.4.weight")
    looks_like_gadus = (
        stem is not None
        and stem.ndim == 4
        and tuple(stem.shape[1:]) == (18, 3, 3)
        and policy is not None
        and policy.ndim == 2
        and policy.shape[0] == GADUS_ACTION_SIZE
    )
    if declared and declared not in GADUS_ARCH_NAMES:
        raise ValueError(f"unsupported historical architecture: {declared}")
    if not looks_like_gadus:
        raise ValueError("parameter layout does not match Gadus")

    channels = int(stem.shape[0])
    block_indices = sorted(
        int(match.group(1))
        for name in state
        if (match := re.fullmatch(r"backbone\.(\d+)\.block\.0\.weight", name))
    )
    if not block_indices or block_indices != list(range(3, 3 + len(block_indices))):
        raise ValueError("Gadus residual block sequence is incomplete")
    blocks = len(block_indices)

    for field, inferred in (("channels", channels), ("blocks", blocks)):
        if field in arch and int(arch[field]) != inferred:
            raise ValueError(
                f"declared arch {field}={arch[field]} does not match parameters ({inferred})"
            )
    if "action_size" in arch and int(arch["action_size"]) != GADUS_ACTION_SIZE:
        raise ValueError("declared action_size does not match Gadus")
    return channels, blocks


def validate_tensors(state: Mapping[str, torch.Tensor]) -> None:
    for name, tensor in state.items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
            raise ValueError(f"model tensor contains NaN or infinity: {name}")


def save_gadus(
    state: Mapping[str, torch.Tensor], output: Path, channels: int, blocks: int
) -> None:
    model = GadusModel(channels, blocks)
    model.load_state_dict(state, strict=True)
    model.eval()
    archive = torch.jit.script(GadusCheckpointArchive(model, channels, blocks))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp_{os.getpid()}")
    try:
        torch.jit.save(archive, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a historical parameter checkpoint into the current model/arch archive."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    if source == output:
        raise ValueError("input and output must be different paths")
    if not source.is_file():
        raise FileNotFoundError(f"input checkpoint not found: {source}")

    state, arch = load_historical_checkpoint(source)
    channels, blocks = infer_gadus(state, arch)
    validate_tensors(state)
    save_gadus(state, output, channels, blocks)
    print("transit finished")
    print(f"input: {source}")
    print(f"output: {output}")
    print("arch: gadus")
    print(f"channels: {channels}")
    print(f"blocks: {blocks}")
    print(f"parameters: {sum(tensor.numel() for tensor in state.values())}")


if __name__ == "__main__":
    main()
