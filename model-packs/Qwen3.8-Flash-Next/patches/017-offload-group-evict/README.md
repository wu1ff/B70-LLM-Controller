# Backport of vllm#51787 (apply only together with 018)

- **Purpose:** Backport of vllm#51787 (apply only together with 018).
- **Category:** correctness.
- **Pinned upstream base:** vLLM 0.30.0 XPU image `vllm/vllm-openai-xpu@sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9` (`gced6857af`; extracted `image-source/vllm`) plus patches 001–005 and the vLLM half of 006 (the 1.0.2 tree).
- **Files affected:** `.../offloading/scheduler.py`; `vllm/v1/kv_offload/base.py`; `vllm/v1/kv_offload/cpu/manager.py`; `vllm/v1/kv_offload/cpu/policies/{arc,base,lru}.py`; `vllm/v1/kv_offload/tiering/manager.py`.
- **Applied runtime stage:** Not in a published image. Proposed stage: `b70-offload`, on top of `gdn-index64-c1`.
- **Currently shipping:** No. Not in `ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0`; proposed. It runs on our 4× Arc Pro B70 host in a tree built from 001–006 plus this series (github.com/Lumnus/b70-flash-next, release `0.30.0-b70.1`).
- **Technical explanation:** Backport of upstream vllm#51787 (request-scoped recency, tail-before-head eviction across all KV groups of a request). On its own it is **not** gated; 018 moves it behind `B70_OFFLOAD_GROUP_EVICT=1` and restores the upstream LRU/ARC policies byte for byte when the switch is unset. Apply 017 and 018 together, or neither. It goes away when the vLLM base moves past #51787.
- **Reproduction/application:** Apply `017-offload-group-evict.patch` after 016 from the vLLM tree root (`patch -p1 --fuzz=0`).
- **Retained provenance:** github.com/Lumnus/vllm, branch `b70/v0.30.0`, commit `11a65f29a1` (`0014e-b70-offload-group-evict-51787`); exported as `patches/vllm/0014e-b70-offload-group-evict-51787.patch` in github.com/Lumnus/b70-flash-next. This file is the same change rewritten as a plain unified diff against the 1.0.2 tree; applying the proposed patches to that tree gives files byte-identical to ours.
