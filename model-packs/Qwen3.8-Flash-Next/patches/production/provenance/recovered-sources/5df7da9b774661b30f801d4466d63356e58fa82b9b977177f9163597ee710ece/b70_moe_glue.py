# SPDX-License-Identifier: Apache-2.0
"""Exact fused MoE glue for B70 DFlash2 decode widths (experiment; env-gated).

For a Qwen4Exp sparse MoE block at 8/16/24/32 rows:
  * router gate + shared gate_up (+ silu_and_mul) + shared expert gate run in one
    native launch (b70_exact_moe_v1.front);
  * the shared down projection and sigmoid(expert_gate) * out run in one native
    launch (b70_exact_moe_v1.back).
Both reproduce the image's oneDNN/vllm arithmetic bit-for-bit (K=2560: four chained
FP32 DPAS K-quarters summed left to right; K=160 and N=1: one chained accumulation;
silu/sigmoid element formulas verified over the whole BF16 domain). Routing, routed
experts, the routed+shared add and the TP all-reduce are untouched.
"""
from pathlib import Path

import torch

WIDTHS = (8, 16, 24, 32)
_loaded = False


def _load():
    global _loaded
    if not _loaded:
        import vllm_xpu_kernels

        torch.ops.load_library(str(Path(vllm_xpu_kernels.__file__).with_name("libb70_exact_moe_v1.so")))
        _loaded = True


def _eligible_x(x):
    return (
        x.dim() == 2
        and x.shape[0] in WIDTHS
        and x.shape[1] == 2560
        and x.dtype == torch.bfloat16
        and x.device.type == "xpu"
        and x.is_contiguous()
    )


def install(block):
    """Bind instance-level fast paths on one block; returns False if not applicable."""
    if getattr(block, "_b70_moe_glue", None) is not None:
        return True
    gate, se = getattr(block, "gate", None), getattr(block, "shared_expert", None)
    if gate is None or se is None:
        return False
    try:
        wr, wgu = gate.weight, se.gate_up_proj.weight
        wd, we = se.down_proj.weight, se.expert_gate.weight
    except AttributeError:
        return False
    if not (
        tuple(wr.shape) == (512, 2560) and tuple(wgu.shape) == (320, 2560)
        and tuple(wd.shape) == (2560, 160) and tuple(we.shape) == (1, 2560)
        and all(w.dtype == torch.bfloat16 and w.is_contiguous() and w.device.type == "xpu"
                for w in (wr, wgu, wd, we))
        and getattr(gate, "bias", None) is None
        and getattr(se.gate_up_proj, "bias", None) is None
        and getattr(se.down_proj, "bias", None) is None
        and getattr(se.expert_gate, "bias", None) is None
        and type(se.gate_up_proj.quant_method).__name__ == "UnquantizedLinearMethod"
        and type(se.down_proj.quant_method).__name__ == "UnquantizedLinearMethod"
        and type(se.act_fn).__name__ == "SiluAndMul"
        and getattr(se.down_proj, "reduce_results", False) is False
    ):
        return False
    _load()
    state = {}
    orig_gate, orig_se = gate.forward, se.forward
    ret = getattr(gate, "_return", None)

    def gate_forward(x):
        if _eligible_x(x):
            logits, act, egate = torch.ops.b70_exact_moe_v1.front(x, wr, wgu, we)
            state.clear()
            state.update(x=x, act=act, egate=egate)
            return ret(logits) if ret is not None else logits
        state.clear()
        return orig_gate(x)

    def se_forward(x):
        if state.get("x") is x:
            act, egate = state["act"], state["egate"]
            state.clear()
            return torch.ops.b70_exact_moe_v1.back(act, wd, egate, x, False)
        state.clear()
        return orig_se(x)

    gate.forward = gate_forward
    se.forward = se_forward
    block._b70_moe_glue = dict(gate=(gate, orig_gate), se=(se, orig_se), state=state)
    return True


def uninstall(block):
    g = getattr(block, "_b70_moe_glue", None)
    if g is None:
        return
    for mod, _ in (g["gate"], g["se"]):
        mod.__dict__.pop("forward", None)
    block._b70_moe_glue = None


def wrap_block_forward(forward):
    """Install the fast path lazily on first call of each target MoE block."""
    def wrapped(self, hidden_states, *args, **kwargs):
        if getattr(self, "_b70_moe_glue", None) is None and not getattr(self, "_b70_moe_glue_checked", False):
            self._b70_moe_glue_checked = True
            install(self)
        glue = getattr(self, "_b70_moe_glue", None)
        if glue is not None:
            # A stash may only be consumed within the block call that produced it
            # (defensive: block inputs can be fixed-address tensors across steps).
            glue["state"].clear()
        return forward(self, hidden_states, *args, **kwargs)
    wrapped.__wrapped__ = forward
    return wrapped
