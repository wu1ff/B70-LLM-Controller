# Modified for the B70 Flash-Next DFlash2 campaign; notice export 2026-10-22.
# DFlash2 loading, proposal/verification/state handling, Disk PLE, native dispatch and serving integration; per-file changes are recorded in production/patch-chain and archive/reconnaissance/CAMPAIGN.md.
# Implementation attribution: Wulf integration and Astra campaign records; see licenses/NOTICE.md.
# This notice adds no license grant and does not replace upstream notices.
# SPDX-License-Identifier: Apache-2.0
"""Native HC fusion and direct-token W4A16 MoE verification."""

from pathlib import Path
import os

import torch

_directory = Path(__file__).parent
torch.ops.load_library(str(_directory / "libastra_hc.so"))
torch.ops.load_library(str(_directory / "libastra_direct.so"))


def prepare_hc_weight(weight):
    if (os.getenv("VLLM_XPU_ASTRA_HC_FUSION", "1") == "1"
            and weight.device.type == "xpu" and weight.dtype == torch.bfloat16
            and weight.shape == (10240, 320)):
        # The native kernel addresses each stream directly; retain only a view.
        return weight.detach()
    return None


def hc_up_mix(lora, weight, normalized):
    output = torch.empty((lora.shape[0], 2560), device=lora.device, dtype=lora.dtype)
    torch.ops.astra_hc_v3.up(lora, weight, normalized, output)
    return output


def direct_moe(module, output, hidden, weights, ids, expert_map, activation_scales):
    if not (
        os.getenv("VLLM_XPU_ASTRA_DIRECT_MOE", "1") == "1"
        and hidden.shape == (8, 2560) and hidden.dtype == torch.bfloat16
        and module.recipe == "int4" and module.inter_size == 640
        and module.num_experts == 128 and module.activation == "silu"
        and module.w13_bias is None and module.w2_bias is None
        and not module.gemm1_clamp_limit and activation_scales is None
        and ids.shape == (8, 10) and weights.dtype == torch.float32
    ):
        return False
    mapping = module.expert_map if expert_map is None else expert_map
    if mapping is None or mapping.numel() != 512:
        return False
    torch.ops.astra_direct_v2.forward(
        hidden, module.w13, module.gemm1_wei_scales, module.w2,
        module.gemm2_wei_scales, ids, weights, mapping, output,
    )
    return True
