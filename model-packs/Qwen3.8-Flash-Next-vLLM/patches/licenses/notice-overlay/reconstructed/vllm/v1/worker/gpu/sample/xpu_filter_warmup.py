# Modified for the B70 Flash-Next DFlash2 campaign; notice export 2026-10-22.
# DFlash2 loading, proposal/verification/state handling, Disk PLE, native dispatch and serving integration; per-file changes are recorded in production/patch-chain and archive/reconnaissance/CAMPAIGN.md.
# Implementation attribution: Wulf integration and Astra campaign records; see licenses/NOTICE.md.
# This notice adds no license grant and does not replace upstream notices.
"""Bounded initialized warmup for the XPU DFlash2 sampling filter."""
import time
import torch
from vllm.logger import init_logger

@torch.inference_mode()
def warmup_filter(runner):
    spec = runner.speculative_config
    if (spec is None or spec.method != "dflash"
        or runner.model_config.hf_text_config.model_type != "qwen4_exp_text"):
        return
    from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton
    started = time.perf_counter()
    before = torch.xpu.memory_allocated()
    # Dedicated initialized tensors; no live request, KV, RNG or sampler state.
    logits = torch.arange(runner.vocab_size, device=runner.device,
                          dtype=torch.float32).unsqueeze(0) / runner.vocab_size
    k = torch.full((1,), 20, device=runner.device, dtype=torch.int32)
    p = torch.full((1,), .95, device=runner.device, dtype=torch.float32)
    for topk, topp in ((k, p), (k, None), (None, p)):
        apply_top_k_top_p_triton(logits.clone(), topk, topp)
    torch.xpu.synchronize()
    del logits, k, p
    init_logger(__name__).info(
        "Initialized XPU filter warmup: %.3f s, retained tensor bytes=%d",
        time.perf_counter() - started, torch.xpu.memory_allocated() - before)
