import os
from typing import Any, Dict

import torch
import torch.nn as nn

from checkpoint_io import ensure_parent
from config import INPUT_CHANNELS, NUM_ACTIONS


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return torch.relu(x + self.block(x))


class ChessNet(nn.Module):
    """Residual policy/value network used by training, MCTS and self-learning."""

    def __init__(self, channels=128, blocks=10, action_size=NUM_ACTIONS):
        super().__init__()
        self.channels = int(channels)
        self.blocks = int(blocks)
        self.action_size = int(action_size)

        self.backbone = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            *[ResidualBlock(channels) for _ in range(blocks)],
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, action_size),
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

    def arch(self) -> Dict[str, Any]:
        return {
            "channels": self.channels,
            "blocks": self.blocks,
            "action_size": self.action_size,
        }

    def forward(self, x):
        z = self.backbone(x)
        return self.policy_head(z), self.value_head(z)


def checkpoint_arch(checkpoint):
    if isinstance(checkpoint, dict):
        arch = checkpoint.get("arch")
        if isinstance(arch, dict):
            return arch
        extra = checkpoint.get("extra")
        if isinstance(extra, dict) and isinstance(extra.get("arch"), dict):
            return extra["arch"]
    return {}


def checkpoint_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict", "state_dict", "net", "network"):
            state = checkpoint.get(key)
            if isinstance(state, dict):
                return state
    return checkpoint


def clean_state_dict_keys(state):
    if not isinstance(state, dict):
        return state
    cleaned = {}
    for key, value in state.items():
        key = key.replace("module.", "", 1) if key.startswith("module.") else key
        # The current checkpoint format contains only policy/value parameters.
        if key.startswith("gate_head."):
            continue
        cleaned[key] = value
    return cleaned


def make_model_from_checkpoint(checkpoint, device="cpu"):
    arch = checkpoint_arch(checkpoint)
    model = ChessNet(
        channels=int(arch.get("channels", 128)),
        blocks=int(arch.get("blocks", 10)),
        action_size=int(arch.get("action_size", NUM_ACTIONS)),
    ).to(device)
    state = clean_state_dict_keys(checkpoint_state(checkpoint))
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        print("warning: unexpected checkpoint keys ignored:", list(incompatible.unexpected_keys)[:10])
    if incompatible.missing_keys:
        print("warning: missing checkpoint keys:", list(incompatible.missing_keys)[:10])
    return model


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model = make_model_from_checkpoint(checkpoint, device=device)
    model.eval()
    return model


def checkpoint_metadata(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        return 0, 0, {}
    epoch = int(checkpoint.get("epoch", 0) or 0)
    global_step = int(checkpoint.get("global_step", 0) or 0)
    extra = checkpoint.get("extra") or {}
    if not isinstance(extra, dict):
        extra = {"source_extra": extra}
    return epoch, global_step, extra


def save_model(path, model, optimizer=None, epoch=None, extra=None, global_step=None):
    """Write the unified checkpoint format atomically.

    optimizer is accepted for call compatibility but is intentionally not saved.
    """
    obj = {
        "model": model.state_dict(),
        "arch": model.arch() if hasattr(model, "arch") else {},
        "epoch": int(epoch or 0),
        "global_step": int(global_step or 0),
        "extra": extra or {},
    }

    ensure_parent(path)
    tmp = f"{path}.tmp_{os.getpid()}"
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
