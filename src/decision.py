from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import chess
import numpy as np
import torch

from architectures import RESNET_PVA_GAD, RESNET_PV_LINEAR, architecture_spec
from move_codecs import get_move_codec
from state_codecs import get_state_codec


@dataclass
class NetworkOutput:
    policy_logits: torch.Tensor
    value: torch.Tensor
    expansion_payload: Any = None


class DecisionProfile:
    name = "base"
    arch_type = ""

    @property
    def spec(self):
        return architecture_spec(self.arch_type)

    @property
    def state_codec(self):
        return get_state_codec(self.spec.state_encoding)

    @property
    def move_codec(self):
        return get_move_codec(self.spec.move_encoding)

    def evaluate_tensor(self, model, tensor: torch.Tensor) -> NetworkOutput:
        logits, value = model(tensor)
        return NetworkOutput(policy_logits=logits, value=value)

    def output_payload_to_numpy(self, output: NetworkOutput):
        return None

    def merge_payloads(self, payloads: List[Any]):
        return None

    def payload_for_index(self, payloads, index: int):
        return None

    def child_profile_state(self, expansion_payload, move: chess.Move) -> Dict[str, Any]:
        return {}

    def unvisited_q_from_parent(self, parent_fpu: float, child) -> float:
        return float(parent_fpu)

    def root_row_fields(self, child) -> Dict[str, Any]:
        return {}


class ResNetPVLinearDecision(DecisionProfile):
    name = "resnet_pv_linear_mcts"
    arch_type = RESNET_PV_LINEAR


class ResNetPVAGadDecision(DecisionProfile):
    name = "resnet_pva_gad_mcts"
    arch_type = RESNET_PVA_GAD

    def evaluate_tensor(self, model, tensor: torch.Tensor) -> NetworkOutput:
        heads = model.forward_heads(tensor)
        advantages = heads.get("advantages")
        if advantages is None:
            raise RuntimeError("resnet_pva_gad decision profile received invalid model heads")
        return NetworkOutput(
            policy_logits=heads["policy_logits"],
            value=heads["value"],
            expansion_payload=advantages,
        )

    def output_payload_to_numpy(self, output: NetworkOutput):
        if output.expansion_payload is None:
            raise RuntimeError("resnet_pva_gad decision profile received no payload")
        return output.expansion_payload.detach().float().cpu().numpy()

    def merge_payloads(self, payloads: List[Any]):
        if not payloads:
            return None
        return np.concatenate(payloads, axis=0).astype(np.float32, copy=False)

    def payload_for_index(self, payloads, index: int):
        if payloads is None:
            raise RuntimeError("resnet_pva_gad decision profile received no batch payload")
        return payloads[index]

    def child_profile_state(self, expansion_payload, move: chess.Move) -> Dict[str, Any]:
        if expansion_payload is None:
            raise RuntimeError("resnet_pva_gad decision profile received no action payload")
        index = self.move_codec.move_to_index(move)
        if index >= len(expansion_payload):
            raise RuntimeError(f"resnet_pva_gad payload missing action index {index}")
        return {"adv": float(np.clip(float(expansion_payload[index]), -1.0, 1.0))}

    def unvisited_q_from_parent(self, parent_fpu: float, child) -> float:
        try:
            return float(parent_fpu + child.profile_state["adv"])
        except KeyError as exc:
            raise RuntimeError("resnet_pva_gad child missing adv") from exc

    def root_row_fields(self, child) -> Dict[str, Any]:
        if child is None or "adv" not in child.profile_state:
            return {}
        return {"adv": float(child.profile_state["adv"])}


PROFILES_BY_ARCH: Dict[str, DecisionProfile] = {
    RESNET_PV_LINEAR: ResNetPVLinearDecision(),
    RESNET_PVA_GAD: ResNetPVAGadDecision(),
}


def model_arch_type(model) -> str:
    if not hasattr(model, "arch"):
        raise RuntimeError("model must expose arch() for decision profile selection")
    arch = model.arch()
    arch_type = arch.get("type") if isinstance(arch, dict) else None
    if arch_type not in PROFILES_BY_ARCH:
        raise RuntimeError(f"no decision profile registered for arch type {arch_type!r}")
    return str(arch_type)


def profile_for_model(model) -> DecisionProfile:
    return PROFILES_BY_ARCH[model_arch_type(model)]
