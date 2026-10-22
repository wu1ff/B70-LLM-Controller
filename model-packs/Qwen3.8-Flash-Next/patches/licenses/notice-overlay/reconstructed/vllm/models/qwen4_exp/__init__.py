# Modified for the B70 Flash-Next DFlash2 campaign; notice export 2026-10-22.
# DFlash2 loading, proposal/verification/state handling, Disk PLE, native dispatch and serving integration; per-file changes are recorded in production/patch-chain and archive/reconnaissance/CAMPAIGN.md.
# Implementation attribution: Wulf integration and Astra campaign records; see licenses/NOTICE.md.
# This notice adds no license grant and does not replace upstream notices.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp model package."""

from typing import TYPE_CHECKING, Any

from .common.hyperconnection import (
    GatedResidual,
    GroupedGemmaRMSNorm,
    HyperConnectionBase,
    HyperConnectionConfig,
)

if TYPE_CHECKING:
    from .nvidia.model import (
        Qwen4ExpForCausalLM,
        Qwen4ExpForConditionalGeneration,
    )
    from .nvidia.mtp import Qwen4ExpMTP


def __getattr__(name: str) -> Any:
    if name in {
        "Qwen4ExpForCausalLM",
        "Qwen4ExpForConditionalGeneration",
        "Qwen4ExpMTP",
    }:
        from vllm.platforms import current_platform

        if current_platform.is_tpu():
            raise NotImplementedError("Qwen4Exp currently supports CUDA and ROCm only")
        if current_platform.is_rocm():
            from .amd.model import (
                Qwen4ExpForCausalLM,
                Qwen4ExpForConditionalGeneration,
            )
            from .amd.mtp import Qwen4ExpMTP
        else:
            from .nvidia.model import (
                Qwen4ExpForCausalLM,
                Qwen4ExpForConditionalGeneration,
            )
            from .nvidia.mtp import Qwen4ExpMTP

        return {
            "Qwen4ExpForCausalLM": Qwen4ExpForCausalLM,
            "Qwen4ExpForConditionalGeneration": (Qwen4ExpForConditionalGeneration),
            "Qwen4ExpMTP": Qwen4ExpMTP,
        }[name]
    raise AttributeError(name)


__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HyperConnectionBase",
    "HyperConnectionConfig",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpMTP",
]
