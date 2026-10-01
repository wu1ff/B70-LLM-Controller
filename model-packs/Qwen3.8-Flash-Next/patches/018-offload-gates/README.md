# Gate 017 behind B70_OFFLOAD_GROUP_EVICT; cheap TRACE=1

- **Purpose:** Gate 017 behind B70_OFFLOAD_GROUP_EVICT; cheap TRACE=1.
- **Category:** correctness.
- **Pinned upstream base:** vLLM 0.30.0 XPU image `vllm/vllm-openai-xpu@sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9` (`gced6857af`; extracted `image-source/vllm`) plus patches 001–005 and the vLLM half of 006 (the 1.0.2 tree).
- **Files affected:** `.../offloading/b70_offload.py`; `.../offloading/scheduler.py`; `vllm/v1/kv_offload/cpu/manager.py`; `vllm/v1/kv_offload/cpu/policies/{arc,lru}.py`; `.../policies/{arc,lru}_group_evict.py` (new); CPU tests in `offload-tests.patch` (not applied to the image).
- **Applied runtime stage:** Not in a published image. Proposed stage: `b70-offload`, on top of `gdn-index64-c1`.
- **Currently shipping:** No. Not in `ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0`; proposed. It runs on our 4× Arc Pro B70 host in a tree built from 001–006 plus this series (github.com/Lumnus/b70-flash-next, release `0.30.0-b70.1`).
- **Technical explanation:** Makes 017 opt-in: unset restores upstream v0.30.0 eviction (the #51787 classes move to `*_group_evict.py` and are selected only with `B70_OFFLOAD_GROUP_EVICT=1`, off in what we serve). Also makes `B70_OFFLOAD_TRACE=1` cheap (no per-key counters). Ungated remainder: the request context records key positions (write-only with the switch unset).
- **Reproduction/application:** Apply `018-offload-gates.patch` after 017 from the vLLM tree root (`patch -p1 --fuzz=0`).
- **Retained provenance:** github.com/Lumnus/vllm, branch `b70/v0.30.0`, commit `bd64e11aef` (`0014f-b70-offload-gates`); exported as `patches/vllm/0014f-b70-offload-gates.patch` in github.com/Lumnus/b70-flash-next. This file is the same change rewritten as a plain unified diff against the 1.0.2 tree; applying the proposed patches to that tree gives files byte-identical to ours.
