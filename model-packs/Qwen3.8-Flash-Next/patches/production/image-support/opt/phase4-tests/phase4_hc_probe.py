"""Compare occupancy variants against the retained ordered-K arithmetic."""

import json
from pathlib import Path
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _hc(
    x,
    w,
    y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    XS: tl.constexpr,
    WS: tl.constexpr,
    RM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr = 128,
):
    base = tl.program_id(0) * RM
    rows = base + tl.arange(0, 16)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    acc = tl.zeros((16, BN), tl.float32)
    for step in range(triton.cdiv(K, BK)):
        kk = step * BK + inner
        a = tl.load(
            x + rows[:, None] * XS + kk[None, :],
            (rows[:, None] < M) & (rows[:, None] < base + RM) & (kk[None, :] < K),
            other=0,
        )
        b = tl.load(
            w + cols[None, :] * WS + kk[:, None],
            (cols[None, :] < V) & (kk[:, None] < K),
            other=0,
        )
        acc += tl.dot(a, b)
    tl.store(
        y + rows[:, None] * N + cols[None, :],
        acc,
        (rows[:, None] < M) & (rows[:, None] < base + RM) & (cols[None, :] < N),
    )


def project(x, w, v, rm=1, bn=32, warps=4):
    m, k = x.shape
    n = w.shape[0]
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    _hc[(triton.cdiv(m, rm), triton.cdiv(n, bn))](
        x,
        w,
        y,
        m,
        n,
        k,
        v,
        x.stride(0),
        w.stride(0),
        rm,
        bn,
        num_warps=warps,
        num_stages=2,
    )
    return y


def run(worker, config):
    from vllm.models.qwen4_exp.nvidia.b70_hc_gemm import ordered_hc_gemm

    module = next(
        m for m in worker.model_runner.model.modules() if hasattr(m, "_down_gemm")
    )
    w = module.input_mix_weight_down_block_inject.weight
    v = module.lora_rank + module.hc_count
    torch.manual_seed(176)
    records = []
    for m in (1, 4):
        x = (torch.randn(m, w.shape[1], device=w.device) * 0.2).bfloat16()
        reference = ordered_hc_gemm(x, w, v)
        for rm, bn, warps in [
            (16, 32, 4),
            (1, 32, 4),
            (1, 16, 4),
            (1, 32, 8),
            (2, 16, 4),
            (4, 16, 4),
        ]:
            fn = lambda: project(x, w, v, rm, bn, warps)
            actual = fn()
            equal = torch.equal(reference, actual)
            for _ in range(3):
                fn()
            a = torch.xpu.Event(enable_timing=True)
            b = torch.xpu.Event(enable_timing=True)
            a.record()
            for _ in range(50):
                fn()
            b.record()
            b.synchronize()
            records.append(
                dict(
                    m=m,
                    rm=rm,
                    bn=bn,
                    warps=warps,
                    bitwise_equal=equal,
                    ms=a.elapsed_time(b) / 50,
                )
            )
    Path(f"/work/phase4-audit/hc-variants-rank-{worker.rank}.json").write_text(
        json.dumps(records, indent=2)
    )
    return records
