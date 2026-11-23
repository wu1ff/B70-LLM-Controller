# SPDX-License-Identifier: Apache-2.0
# Modified from Wulf's frozen MTP5 XPU staging for v0.33 with Codex.
"""Per-builder XPU GDN staging with actual lengths and maximum reservations.

The ALIGN gather follows vLLM PR 38020. Mixed, prefill and unsupported
layouts use the released builder, including its group-specific checkpoints.
"""

import numpy as np
from vllm.logger import init_logger
_reported = False

from vllm.triton_utils import tl, triton


@triton.jit
def _stage(
    BT,
    SEQ,
    QSL,
    ACC,
    STATE,
    MASK,
    TOKENS,
    OUT_QSL,
    OUT_ACC,
    BT_STRIDE: tl.constexpr,
    BT_COL: tl.constexpr,
    SEQ_STRIDE: tl.constexpr,
    QSL_STRIDE: tl.constexpr,
    ACC_STRIDE: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    ACTIVE: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    TILE: tl.constexpr,
):
    x = tl.arange(0, TILE)
    row = x // K
    col = x % K
    seq = tl.load(SEQ + row * SEQ_STRIDE, mask=row < ACTIVE, other=0)
    start = tl.maximum((seq - 1) // BLOCK_SIZE, 0)
    state = tl.load(
        BT + row * BT_STRIDE + (start + col) * BT_COL, mask=row < ACTIVE, other=0
    )
    tl.store(STATE + row * STATE_STRIDE + col, state, mask=row < N)
    tl.store(TOKENS + x, x, mask=x < NUM_TOKENS)
    tl.store(MASK + x, x < ACTIVE, mask=x < N)
    acc = tl.load(ACC + x * ACC_STRIDE, mask=x < ACTIVE, other=1)
    tl.store(OUT_ACC + x, acc, mask=x < N)
    q = tl.load(QSL + tl.minimum(x, ACTIVE) * QSL_STRIDE, mask=x <= N, other=0)
    tl.store(OUT_QSL + x, q, mask=x <= N)


def eligible(b, m, d):
    """Accept DFlash actual widths 1–8 with trailing zero-length padding."""
    if (
        not b.use_spec_decode
        or d is None
        or d.device.type != "cpu"
        or b.vllm_config.cache_config.mamba_cache_mode != "align"
    ):
        return None
    n = m.num_reqs
    if (
        n == 0
        or n > b.decode_cudagraph_max_bs
        or b.kv_cache_spec.num_speculative_blocks < b.num_spec
        or m.block_table_tensor.shape[0] != n
        or m.seq_lens.shape[0] != n
        or m.block_table_tensor.device != m.query_start_loc.device
        or m.seq_lens.device != m.query_start_loc.device
    ):
        return None
    dc = d.numpy()
    q = m.query_start_loc_cpu.numpy()
    if len(dc) != n or len(q) != n + 1:
        return None
    active = int(np.count_nonzero(dc >= 0))
    query_len = int(dc[0]) + 1 if active else 0
    if (
        active == 0
        or active * query_len > b.decode_cudagraph_max_bs
        or active * query_len > m.num_actual_tokens
        or not 1 <= query_len <= b.num_spec + 1
        or not np.all(dc[:active] == query_len - 1)
        or not np.all(dc[active:] < 0)
        or not np.all(np.diff(q[: active + 1]) == query_len)
        or not np.all(q[active:] == active * query_len)
        or q[0] != 0
    ):
        return None
    return active, query_len


def try_build_uniform_spec_metadata(b, m, num_accepted_tokens, draft_counts_cpu):
    shape = eligible(b, m, draft_counts_cpu)
    if shape is None:
        return None
    active, query_len = shape
    global _reported
    if not _reported:
        _reported = True
        init_logger(__name__).info(
            "DFlash per-builder GDN metadata active: reserved=%d actual_width=%d requests=%d",
            b.num_spec + 1, query_len, m.num_reqs)
    assert num_accepted_tokens is not None
    assert num_accepted_tokens.shape[0] >= active
    assert num_accepted_tokens.device == m.query_start_loc.device
    n = m.num_reqs
    # State columns always follow maximum depth, including at actual K0/K3.
    reserved = b.num_spec + 1
    nt = active * query_len
    _stage[(1,)](
        m.block_table_tensor,
        m.seq_lens,
        m.query_start_loc,
        num_accepted_tokens,
        b.spec_state_indices_tensor,
        b.spec_sequence_masks,
        b.spec_token_indx,
        b.spec_query_start_loc,
        b.num_accepted_tokens,
        m.block_table_tensor.stride(0),
        m.block_table_tensor.stride(1),
        m.seq_lens.stride(0),
        m.query_start_loc.stride(0),
        num_accepted_tokens.stride(0),
        b.spec_state_indices_tensor.stride(0),
        b.kv_cache_spec.block_size,
        reserved,
        n,
        active,
        nt,
        triton.next_power_of_2(max(n * reserved, n + 1)),
        num_warps=4,
    )
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    # This branch has no prefill rows, so checkpoint fields keep their released
    # None defaults. No metadata or state indices are shared across groups.
    return GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=active,
        num_spec_decode_tokens=nt,
        num_actual_tokens=m.num_actual_tokens,
        spec_query_start_loc=b.spec_query_start_loc[: n + 1],
        spec_state_indices_tensor=b.spec_state_indices_tensor[:n],
        spec_sequence_masks=b.spec_sequence_masks[:n],
        spec_sequence_masks_cpu=draft_counts_cpu >= 0,
        spec_token_indx=b.spec_token_indx[:nt],
        non_spec_token_indx=b.non_spec_token_indx[:0],
        num_accepted_tokens=b.num_accepted_tokens[:n],
        uniform_spec_sequence_length=query_len,
    )
