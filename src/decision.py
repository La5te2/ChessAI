from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

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


class InferenceProfile:
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
        raise NotImplementedError

    def output_payload_to_numpy(self, output: NetworkOutput):
        raise NotImplementedError

    def merge_payloads(self, payloads: List[Any]):
        raise NotImplementedError

    def payload_for_index(self, payloads, index: int):
        raise NotImplementedError


class ResNetPVLinearInference(InferenceProfile):
    name = "resnet_pv_linear_inference"
    arch_type = RESNET_PV_LINEAR

    def evaluate_tensor(self, model, tensor: torch.Tensor) -> NetworkOutput:
        logits, value = model(tensor)
        return NetworkOutput(policy_logits=logits, value=value)

    def output_payload_to_numpy(self, output: NetworkOutput):
        return None

    def merge_payloads(self, payloads: List[Any]):
        return None

    def payload_for_index(self, payloads, index: int):
        return None


class ResNetPVAGadInference(InferenceProfile):
    name = "resnet_pva_gad_inference"
    arch_type = RESNET_PVA_GAD

    def evaluate_tensor(self, model, tensor: torch.Tensor) -> NetworkOutput:
        heads = model.forward_heads(tensor)
        advantages = heads.get("advantages")
        if advantages is None:
            raise RuntimeError("resnet_pva_gad inference profile received invalid model heads")
        return NetworkOutput(
            policy_logits=heads["policy_logits"],
            value=heads["value"],
            expansion_payload=advantages,
        )

    def output_payload_to_numpy(self, output: NetworkOutput):
        if output.expansion_payload is None:
            raise RuntimeError("resnet_pva_gad inference profile received no payload")
        return output.expansion_payload.detach().float().cpu().numpy()

    def merge_payloads(self, payloads: List[Any]):
        if not payloads:
            return None
        return np.concatenate(payloads, axis=0).astype(np.float32, copy=False)

    def payload_for_index(self, payloads, index: int):
        if payloads is None:
            raise RuntimeError("resnet_pva_gad inference profile received no batch payload")
        return payloads[index]


INFERENCE_PROFILES_BY_ARCH: Dict[str, InferenceProfile] = {
    RESNET_PV_LINEAR: ResNetPVLinearInference(),
    RESNET_PVA_GAD: ResNetPVAGadInference(),
}


def model_arch_type(model) -> str:
    if not hasattr(model, "arch"):
        raise RuntimeError("model must expose arch() for inference profile selection")
    arch = model.arch()
    arch_type = arch.get("type") if isinstance(arch, dict) else None
    if arch_type not in INFERENCE_PROFILES_BY_ARCH:
        raise RuntimeError(f"no inference profile registered for arch type {arch_type!r}")
    return str(arch_type)


def inference_profile_for_model(model) -> InferenceProfile:
    return INFERENCE_PROFILES_BY_ARCH[model_arch_type(model)]
