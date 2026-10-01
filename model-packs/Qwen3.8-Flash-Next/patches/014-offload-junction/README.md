# Hybrid offload miss: set a shared-prefix junction

- **Purpose:** Hybrid offload miss: set a shared-prefix junction.
- **Category:** correctness.
- **Pinned upstream base:** vLLM 0.30.0 XPU image `vllm/vllm-openai-xpu@sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9` (`gced6857af`; extracted `image-source/vllm`) plus patches 001–005 and the vLLM half of 006 (the 1.0.2 tree).
- **Files affected:** `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`.
- **Applied runtime stage:** Not in a published image. Proposed stage: `b70-offload`, on top of `gdn-index64-c1`.
- **Currently shipping:** No. Not in `ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0`; proposed. It runs on our 4× Arc Pro B70 host in a tree built from 001–006 plus this series (github.com/Lumnus/b70-flash-next, release `0.30.0-b70.1`).
- **Technical explanation:** Opt-in with `B70_OFFLOAD_JUNCTION=1`. On a hybrid model with the default `prefix_cache_retention_interval=0`, a request stores one GDN state, at `round_down(L-1, block)`. A revisit whose suffix changes before that boundary finds the full-attention chunks but no GDN state at or below them, and one group returning 0 makes the whole lookup return 0. The offload path never sets a shared-prefix junction (only a GPU prefix-cache hit does), so the recompute stores its GDN state inside the changed suffix again and the prefix misses on every revisit. With the switch on, the connector sets `request.shared_prefix_boundary` where the GDN group cut the hit (never lowering an existing one); the core scheduler already honours it. The first revisit misses, later revisits hit.
- **Reproduction/application:** Apply `014-offload-junction.patch` after 013 from the vLLM tree root (`patch -p1 --fuzz=0`).
- **Retained provenance:** github.com/Lumnus/vllm, branch `b70/v0.30.0`, commit `0369d95b15` (`0014b-b70-offload-junction`); exported as `patches/vllm/0014b-b70-offload-junction.patch` in github.com/Lumnus/b70-flash-next. This file is the same change rewritten as a plain unified diff against the 1.0.2 tree; applying the proposed patches to that tree gives files byte-identical to ours.
