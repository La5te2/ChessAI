from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

from move_codecs import (
    MOVE_ENCODING_AZ_64X73,
    MOVE_ENCODING_SD_64X64_UP9,
    get_move_codec,
)
from state_codecs import (
    STATE_ENCODING_RESNET_PVA_GAD_SQUARE_TOKENS,
    STATE_ENCODING_RESNET_PV_LINEAR_18_PLANES,
)


RESNET_PV_LINEAR = "resnet_pv_linear"
RESNET_PVA_GAD = "resnet_pva_gad"
DEFAULT_ARCH_TYPE = RESNET_PV_LINEAR


@dataclass(frozen=True)
class ArchitectureSpec:
    name: str
    state_encoding: str
    move_encoding: str
    supervised_datasets: Tuple[str, ...]
    target_schema: str
    default_has_cmt: int


ARCHITECTURES = {
    RESNET_PV_LINEAR: ArchitectureSpec(
        name=RESNET_PV_LINEAR,
        state_encoding=STATE_ENCODING_RESNET_PV_LINEAR_18_PLANES,
        move_encoding=MOVE_ENCODING_AZ_64X73,
        supervised_datasets=("states", "moves", "values"),
        target_schema="pv_supervised",
        default_has_cmt=1,
    ),
    RESNET_PVA_GAD: ArchitectureSpec(
        name=RESNET_PVA_GAD,
        state_encoding=STATE_ENCODING_RESNET_PVA_GAD_SQUARE_TOKENS,
        move_encoding=MOVE_ENCODING_SD_64X64_UP9,
        supervised_datasets=(
            "states",
            "moves",
            "values",
            "adv_moves",
            "adv_values",
        ),
        target_schema="pva_minimax_dueling",
        default_has_cmt=1,
    ),
}

SUPPORTED_ARCH_TYPES = frozenset(ARCHITECTURES)


def normalize_arch_type(arch_type: Any) -> str:
    value = DEFAULT_ARCH_TYPE if arch_type is None else str(arch_type).strip()
    if value not in ARCHITECTURES:
        raise ValueError(
            f"unsupported model arch type {arch_type!r}; "
            f"expected one of {sorted(SUPPORTED_ARCH_TYPES)}"
        )
    return value


def architecture_spec(arch_type: Any) -> ArchitectureSpec:
    return ARCHITECTURES[normalize_arch_type(arch_type)]


def move_encoding_for_arch(arch_type: Any) -> str:
    return architecture_spec(arch_type).move_encoding


def state_encoding_for_arch(arch_type: Any) -> str:
    return architecture_spec(arch_type).state_encoding


def action_size_for_arch(arch_type: Any) -> int:
    return get_move_codec(move_encoding_for_arch(arch_type)).action_size
