# Retain GDN states N blocks below the replay boundary

- **Purpose:** Retain GDN states N blocks below the replay boundary.
- **Category:** correctness.
- **Pinned upstream base:** vLLM 0.30.0 XPU image `vllm/vllm-openai-xpu@sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9` (`gced6857af`; extracted `image-source/vllm`) plus patches 001–005 and the vLLM half of 006 (the 1.0.2 tree).
- **Files affected:** `vllm/v1/core/kv_cache_coordinator.py`; `vllm/v1/core/sched/scheduler.py`.
- **Applied runtime stage:** Not in a published image. Proposed stage: `b70-offload`, on top of `gdn-index64-c1`.
- **Currently shipping:** No. Not in `ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0`; proposed. It runs on our 4× Arc Pro B70 host in a tree built from 001–006 plus this series (github.com/Lumnus/b70-flash-next, release `0.30.0-b70.1`).
- **Technical explanation:** `B70_OFFLOAD_GDN_BACKSTEP=N` (default 0 = off) also retains the GDN states at the N block boundaries below the replay boundary, so a divergence within the last N blocks hits on the first revisit too. Cost per request: N × one GDN state set (116,195,328 B for Qwen3.8-Flash-Next's 36 GDN groups) in the CPU tier, and N evictable GDN blocks per group on the GPU. We serve N=1 together with 014.
- **Reproduction/application:** Apply `015-offload-gdn-backstep.patch` after 014 from the vLLM tree root (`patch -p1 --fuzz=0`).
- **Retained provenance:** github.com/Lumnus/vllm, branch `b70/v0.30.0`, commit `80aacb5725` (`0014c-b70-offload-gdn-backstep`); exported as `patches/vllm/0014c-b70-offload-gdn-backstep.patch` in github.com/Lumnus/b70-flash-next. This file is the same change rewritten as a plain unified diff against the 1.0.2 tree; applying the proposed patches to that tree gives files byte-identical to ours.
