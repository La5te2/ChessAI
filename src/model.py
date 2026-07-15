import os
from typing import Any, Dict

import torch
import torch.nn as nn

from checkpoint_io import ensure_parent
from config import INPUT_CHANNELS, NUM_ACTIONS

RESNET_PV_LINEAR = "resnet_pv_linear"
RESNET_PV_PLANE = "resnet_pv_plane"
DEFAULT_ARCH_TYPE = RESNET_PV_LINEAR
SUPPORTED_ARCH_TYPES = {
    RESNET_PV_LINEAR,
    RESNET_PV_PLANE,
}
POLICY_PLANES = 73
BOARD_SQUARES = 64


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


class PlanePolicyHead(nn.Module):
    def __init__(self, channels: int, action_size: int):
        super().__init__()
        if int(action_size) != BOARD_SQUARES * POLICY_PLANES:
            raise ValueError(
                f"plane policy requires action_size={BOARD_SQUARES * POLICY_PLANES}, "
                f"got {action_size}"
            )
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.logits = nn.Conv2d(32, POLICY_PLANES, 1)

    def forward(self, x):
        # Conv output is [batch, plane, rank, file]. The move encoder uses
        # index = from_square * 73 + plane, where from_square = rank * 8 + file.
        y = self.logits(self.features(x))
        return y.permute(0, 2, 3, 1).contiguous().view(y.shape[0], -1)


def normalize_arch_type(arch_type: Any) -> str:
    value = str(arch_type or DEFAULT_ARCH_TYPE).strip().lower()
    if value in {"", "default", "linear", "resnet", "resnet_pv"}:
        value = RESNET_PV_LINEAR
    elif value in {"plane", "policy_plane", "resnet_plane"}:
        value = RESNET_PV_PLANE
    if value not in SUPPORTED_ARCH_TYPES:
        raise ValueError(
            f"unsupported model arch type {arch_type!r}; "
            f"expected one of {sorted(SUPPORTED_ARCH_TYPES)}"
        )
    return value


class ChessNet(nn.Module):
    """Residual policy/value network used by training, MCTS and self-learning."""

    def __init__(
        self,
        channels=128,
        blocks=10,
        action_size=NUM_ACTIONS,
        arch_type=DEFAULT_ARCH_TYPE,
    ):
        super().__init__()
        self.arch_type = normalize_arch_type(arch_type)
        self.channels = int(channels)
        self.blocks = int(blocks)
        self.action_size = int(action_size)

        self.backbone = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            *[ResidualBlock(channels) for _ in range(blocks)],
        )
        if self.arch_type == RESNET_PV_LINEAR:
            self.policy_head = nn.Sequential(
                nn.Conv2d(channels, 32, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(32 * 8 * 8, action_size),
            )
            self.policy_head_type = "linear"
        elif self.arch_type == RESNET_PV_PLANE:
            self.policy_head = PlanePolicyHead(channels, action_size)
            self.policy_head_type = "plane"
        else:
            raise AssertionError(f"unhandled arch type: {self.arch_type}")
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
            "type": self.arch_type,
            "backbone": "resnet",
            "policy_head": self.policy_head_type,
            "value_head": "mlp",
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


def infer_arch_type_from_state(state) -> str:
    if isinstance(state, dict):
        if any(str(key).startswith("policy_head.logits.") for key in state):
            return RESNET_PV_PLANE
    return RESNET_PV_LINEAR


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


def create_model(
    arch_type=DEFAULT_ARCH_TYPE,
    channels=128,
    blocks=10,
    action_size=NUM_ACTIONS,
    device=None,
):
    model = ChessNet(
        channels=int(channels),
        blocks=int(blocks),
        action_size=int(action_size),
        arch_type=arch_type,
    )
    if device is not None:
        model = model.to(device)
    return model


def make_model_from_checkpoint(checkpoint, device="cpu"):
    arch = checkpoint_arch(checkpoint)
    state = clean_state_dict_keys(checkpoint_state(checkpoint))
    arch_type = normalize_arch_type(
        arch.get("type")
        or infer_arch_type_from_state(state)
    )
    model = create_model(
        arch_type=arch_type,
        channels=int(arch.get("channels", 128)),
        blocks=int(arch.get("blocks", 10)),
        action_size=int(arch.get("action_size", NUM_ACTIONS)),
        device=device,
    )
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
