import os
from typing import Any, Dict

import torch
import torch.nn as nn

from checkpoint_io import ensure_parent
from architectures import (
    DEFAULT_ARCH_TYPE,
    RESNET_PVA_GAD,
    RESNET_PV_LINEAR,
    SUPPORTED_ARCH_TYPES,
    action_size_for_arch,
    move_encoding_for_arch,
    normalize_arch_type,
    state_encoding_for_arch,
)
from move_codecs import (
    BOARD_SQUARES,
    SD_64X64_UP9_ACTION_SIZE,
)
from state_codecs import GAD_STATE_FEATURES, get_state_codec

GAD_TOKEN_COUNT = BOARD_SQUARES + 1
GAD_GLOBAL_TOKEN_INDEX = 0
GAD_SQUARE_TOKEN_OFFSET = 1
GEOMETRY_RELATIONS = 32


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


def attention_heads_for_channels(channels: int) -> int:
    for heads in (8, 4, 2):
        if channels % heads == 0:
            return heads
    return 1


def square_geometry_relation(source: int, target: int) -> int:
    source_rank, source_file = divmod(source, 8)
    target_rank, target_file = divmod(target, 8)
    dr = target_rank - source_rank
    dc = target_file - source_file
    adr, adc = abs(dr), abs(dc)
    if source == target:
        value = 0
    elif dr == 0:
        value = 1 + adc - 1
    elif dc == 0:
        value = 8 + adr - 1
    elif adr == adc:
        value = 15 + adr - 1
    elif (adr, adc) in ((1, 2), (2, 1)):
        value = 22
    elif max(adr, adc) == 1:
        value = 23
    else:
        value = 24 + min(6, adr + adc - 2)
    return value + 1


def build_geometry_relation_ids() -> torch.Tensor:
    relation = torch.zeros(GAD_TOKEN_COUNT, GAD_TOKEN_COUNT, dtype=torch.long)
    for source in range(BOARD_SQUARES):
        for target in range(BOARD_SQUARES):
            relation[
                source + GAD_SQUARE_TOKEN_OFFSET,
                target + GAD_SQUARE_TOKEN_OFFSET,
            ] = square_geometry_relation(source, target)
    return relation


class GeometryAttentionBlock(nn.Module):
    """Residual geometry attention with static and position-dependent relation bias."""

    def __init__(self, channels: int):
        super().__init__()
        heads = attention_heads_for_channels(int(channels))
        if channels % heads != 0:
            raise ValueError(f"channels={channels} must be divisible by heads={heads}")
        self.channels = int(channels)
        self.heads = int(heads)
        self.head_dim = int(channels) // int(heads)
        self.position = nn.Parameter(torch.zeros(1, GAD_TOKEN_COUNT, channels))
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.out = nn.Linear(channels, channels)
        self.relation_bias = nn.Embedding(GEOMETRY_RELATIONS, heads)
        self.register_buffer("relation_ids", build_geometry_relation_ids(), persistent=False)
        self.dynamic_relation = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, heads * GEOMETRY_RELATIONS),
        )
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, tokens):
        batch, token_count, channels = tokens.shape
        if token_count != GAD_TOKEN_COUNT:
            raise RuntimeError(f"expected {GAD_TOKEN_COUNT} GAD tokens, got {token_count}")
        tokens = tokens + self.position
        attn_input = self.norm1(tokens)
        qkv = self.qkv(attn_input).view(batch, GAD_TOKEN_COUNT, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        bias = self.relation_bias(self.relation_ids).permute(2, 0, 1).unsqueeze(0)
        dynamic = self.dynamic_relation(tokens[:, GAD_GLOBAL_TOKEN_INDEX]).view(
            batch,
            self.heads,
            GEOMETRY_RELATIONS,
        )
        dynamic_bias = dynamic[:, :, self.relation_ids.reshape(-1)].view(
            batch,
            self.heads,
            GAD_TOKEN_COUNT,
            GAD_TOKEN_COUNT,
        )
        scores = scores + bias + dynamic_bias
        attn = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, GAD_TOKEN_COUNT, channels)
        tokens = tokens + self.out(attn_out)
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens


class SourceDestinationActionHead(nn.Module):
    def __init__(self, channels: int, action_size: int, zero_to_projection: bool = False):
        super().__init__()
        if int(action_size) != SD_64X64_UP9_ACTION_SIZE:
            raise ValueError(
                f"source-destination action head requires action_size={SD_64X64_UP9_ACTION_SIZE}, "
                f"got {action_size}"
            )
        self.norm = nn.LayerNorm(channels)
        self.from_proj = nn.Linear(channels, channels)
        self.to_proj = nn.Linear(channels, channels)
        self.underpromotion = nn.Linear(channels, 9)
        if zero_to_projection:
            nn.init.zeros_(self.to_proj.weight)
            nn.init.zeros_(self.to_proj.bias)
            nn.init.zeros_(self.underpromotion.weight)
            nn.init.zeros_(self.underpromotion.bias)

    def forward(self, x):
        tokens = x
        if tokens.dim() != 3 or tokens.shape[1] != BOARD_SQUARES:
            raise RuntimeError(f"expected square tokens [batch, {BOARD_SQUARES}, channels], got {tuple(tokens.shape)}")
        tokens = self.norm(tokens)
        from_tokens = self.from_proj(tokens)
        to_tokens = self.to_proj(tokens)
        from_to = torch.matmul(from_tokens, to_tokens.transpose(1, 2)) / (from_tokens.shape[-1] ** 0.5)
        underpromotion = self.underpromotion(tokens)
        return torch.cat(
            [
                from_to.contiguous().view(tokens.shape[0], BOARD_SQUARES * BOARD_SQUARES),
                underpromotion.contiguous().view(tokens.shape[0], BOARD_SQUARES * 9),
            ],
            dim=1,
        )


class DuelingAdvantageHead(nn.Module):
    def __init__(self, channels: int, action_size: int):
        super().__init__()
        self.action_head = SourceDestinationActionHead(
            channels,
            action_size,
            zero_to_projection=True,
        )

    def forward(self, x):
        advantage = self.action_head(x)
        advantage = advantage - advantage.mean(dim=1, keepdim=True)
        return torch.tanh(advantage)


class TokenValueHead(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.value = nn.Sequential(
            nn.Linear(channels, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        pooled = self.norm(x[:, GAD_GLOBAL_TOKEN_INDEX])
        return self.value(pooled)


class GADStateEmbedding(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.piece = nn.Embedding(13, channels)
        self.square = nn.Embedding(BOARD_SQUARES, channels)
        self.side = nn.Embedding(2, channels)
        self.castling = nn.Embedding(16, channels)
        self.ep_file = nn.Embedding(9, channels)
        self.global_token = nn.Parameter(torch.zeros(1, 1, channels))
        self.register_buffer(
            "square_indices",
            torch.arange(BOARD_SQUARES, dtype=torch.long),
            persistent=False,
        )

    def forward(self, state):
        if state.dim() != 2 or state.shape[1] != GAD_STATE_FEATURES:
            raise RuntimeError(f"expected GAD state [batch, {GAD_STATE_FEATURES}], got {tuple(state.shape)}")
        state = state.long()
        pieces = state[:, :BOARD_SQUARES].clamp(0, 12)
        side = state[:, 64].clamp(0, 1)
        castling = state[:, 65].clamp(0, 15)
        ep_file = state[:, 66].clamp(0, 8)

        global_context = (
            self.side(side)
            + self.castling(castling)
            + self.ep_file(ep_file)
        )
        square_tokens = (
            self.piece(pieces)
            + self.square(self.square_indices).unsqueeze(0)
            + global_context.unsqueeze(1)
        )
        global_token = self.global_token.expand(state.shape[0], -1, -1) + global_context.unsqueeze(1)
        return torch.cat([global_token, square_tokens], dim=1)


class ResidualBackboneModel(nn.Module):
    def __init__(
        self,
        arch_type: str,
        channels=128,
        blocks=10,
        action_size=None,
    ):
        super().__init__()
        self.arch_type = arch_type
        self.state_encoding = state_encoding_for_arch(arch_type)
        self.input_channels = get_state_codec(self.state_encoding).input_channels
        self.channels = int(channels)
        self.blocks = int(blocks)
        self.action_size = int(action_size_for_arch(arch_type) if action_size is None else action_size)
        self.backbone = nn.Sequential(
            nn.Conv2d(self.input_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            *[ResidualBlock(channels) for _ in range(blocks)],
        )

    def arch(self) -> Dict[str, Any]:
        return {
            "type": self.arch_type,
            "backbone": "resnet",
            "policy_head": self.policy_head_type,
            "value_head": self.value_head_type,
            "state_encoding": self.state_encoding,
            "move_encoding": move_encoding_for_arch(self.arch_type),
            "input_channels": self.input_channels,
            "channels": self.channels,
            "blocks": self.blocks,
            "action_size": self.action_size,
        }

    def forward_heads(self, x) -> Dict[str, torch.Tensor]:
        z = self.backbone(x)
        policy_logits = self.policy_head(z)
        value = self.value_head(z)
        return {
            "policy_logits": policy_logits,
            "value": value,
        }

    def forward(self, x):
        heads = self.forward_heads(x)
        return heads["policy_logits"], heads["value"]


class ResNetPVLinearModel(ResidualBackboneModel):
    def __init__(self, channels=128, blocks=10, action_size=None):
        super().__init__(
            arch_type=RESNET_PV_LINEAR,
            channels=channels,
            blocks=blocks,
            action_size=action_size,
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, self.action_size),
        )
        self.policy_head_type = "linear"
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
        self.value_head_type = "mlp"


class ResNetPVAGadModel(nn.Module):
    def __init__(
        self,
        channels=128,
        blocks=10,
        action_size=None,
        attention_blocks=None,
    ):
        super().__init__()
        self.arch_type = RESNET_PVA_GAD
        self.state_encoding = state_encoding_for_arch(self.arch_type)
        self.channels = int(channels)
        self.blocks = max(1, int(blocks if attention_blocks is None else attention_blocks))
        self.action_size = int(
            action_size_for_arch(self.arch_type)
            if action_size is None
            else action_size
        )
        self.state_embedding = GADStateEmbedding(self.channels)
        self.trunk = nn.Sequential(
            *[
                GeometryAttentionBlock(self.channels)
                for _ in range(self.blocks)
            ]
        )
        self.policy_head = SourceDestinationActionHead(self.channels, self.action_size)
        self.policy_head_type = "source_destination"
        self.value_head = TokenValueHead(self.channels)
        self.value_head_type = "global_token_mlp"
        self.advantage_head = DuelingAdvantageHead(self.channels, self.action_size)
        self.advantage_head_type = "dueling_advantage_source_destination"

    def arch(self) -> Dict[str, Any]:
        return {
            "type": self.arch_type,
            "backbone": "residual_geometry_transformer",
            "state_encoding": self.state_encoding,
            "move_encoding": move_encoding_for_arch(self.arch_type),
            "policy_head": self.policy_head_type,
            "value_head": self.value_head_type,
            "advantage_head": self.advantage_head_type,
            "attention": "static_dynamic_geometry_relation_bias",
            "dueling": True,
            "channels": self.channels,
            "blocks": self.blocks,
            "action_size": self.action_size,
        }

    def forward_heads(self, x) -> Dict[str, torch.Tensor]:
        tokens = self.trunk(self.state_embedding(x))
        square_tokens = tokens[:, GAD_SQUARE_TOKEN_OFFSET:, :]
        return {
            "policy_logits": self.policy_head(square_tokens),
            "value": self.value_head(tokens),
            "advantages": self.advantage_head(square_tokens),
        }

    def forward(self, x):
        heads = self.forward_heads(x)
        return heads["policy_logits"], heads["value"]


def make_resnet_pv_linear(
    channels=128,
    blocks=10,
    action_size=None,
    attention_blocks=None,
):
    return ResNetPVLinearModel(
        channels=channels,
        blocks=blocks,
        action_size=action_size,
    )


def make_resnet_pva_gad(
    channels=128,
    blocks=10,
    action_size=None,
    attention_blocks=None,
):
    return ResNetPVAGadModel(
        channels=channels,
        blocks=blocks,
        action_size=action_size,
        attention_blocks=attention_blocks,
    )


MODEL_FACTORIES = {
    RESNET_PV_LINEAR: make_resnet_pv_linear,
    RESNET_PVA_GAD: make_resnet_pva_gad,
}


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
        if any(str(key).startswith("policy_head.from_proj.") for key in state):
            return RESNET_PVA_GAD
        if any(str(key).startswith("advantage_head.action_head.") for key in state):
            return RESNET_PVA_GAD
        if any(str(key).startswith("attention.") and ".relation_bias." in str(key) for key in state):
            return RESNET_PVA_GAD
        if any(str(key).startswith("attention.") for key in state):
            raise ValueError(
                "checkpoint has geometry attention parameters but no advantage_head parameters"
            )
        if any(str(key).startswith("q_head.") for key in state):
            raise ValueError("checkpoint action-value head is not part of the current model set")
        if any(str(key).startswith("policy_head.4.") for key in state):
            return RESNET_PV_LINEAR
        if any(str(key).startswith("policy_head.logits.") for key in state):
            raise ValueError(
                "checkpoint looks like an unsupported retired policy head without "
                "GAD advantage parameters"
            )
    raise ValueError("cannot infer model architecture from checkpoint state dict")


def resolve_arch_type(
    arch: Dict[str, Any],
    state,
    infer_unknown_arch: bool = False,
) -> str:
    raw_arch_type = arch.get("type") if isinstance(arch, dict) else None
    if raw_arch_type:
        try:
            return normalize_arch_type(raw_arch_type)
        except ValueError:
            if not infer_unknown_arch:
                raise
            return infer_arch_type_from_state(state)
    return infer_arch_type_from_state(state)


def clean_state_dict_keys(state):
    if not isinstance(state, dict):
        return state
    cleaned = {}
    for key, value in state.items():
        key = key.replace("module.", "", 1) if key.startswith("module.") else key
        cleaned[key] = value
    return cleaned


def create_model(
    arch_type=DEFAULT_ARCH_TYPE,
    channels=128,
    blocks=10,
    action_size=None,
    attention_blocks=None,
    device=None,
):
    arch_type = normalize_arch_type(arch_type)
    factory = MODEL_FACTORIES[arch_type]
    model = factory(
        channels=int(channels),
        blocks=int(blocks),
        action_size=None if action_size is None else int(action_size),
        attention_blocks=attention_blocks,
    )
    if device is not None:
        model = model.to(device)
    return model


def make_model_from_checkpoint(
    checkpoint,
    device="cpu",
    infer_unknown_arch: bool = False,
):
    arch = checkpoint_arch(checkpoint)
    state = clean_state_dict_keys(checkpoint_state(checkpoint))
    arch_type = resolve_arch_type(
        arch,
        state,
        infer_unknown_arch=infer_unknown_arch,
    )
    model = create_model(
        arch_type=arch_type,
        channels=int(arch.get("channels", 128)),
        blocks=int(arch.get("blocks", 10)),
        action_size=int(arch["action_size"]) if "action_size" in arch else None,
        attention_blocks=arch.get("attention_blocks"),
        device=device,
    )
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        print("warning: unexpected checkpoint keys ignored:", list(incompatible.unexpected_keys)[:10])
    if incompatible.missing_keys:
        print("warning: missing checkpoint keys:", list(incompatible.missing_keys)[:10])
    return model


def load_model(path, device="cpu", infer_unknown_arch: bool = False):
    checkpoint = torch.load(path, map_location=device)
    model = make_model_from_checkpoint(
        checkpoint,
        device=device,
        infer_unknown_arch=infer_unknown_arch,
    )
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


def save_model(path, model, epoch=None, extra=None, global_step=None):
    """Write the unified checkpoint format atomically."""
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
