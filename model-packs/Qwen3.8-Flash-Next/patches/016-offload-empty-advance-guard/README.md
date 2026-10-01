# Do not skip past write-pending keys (vllm#56795)

- **Purpose:** Do not skip past write-pending keys (vllm#56795).
- **Category:** correctness.
- **Pinned upstream base:** vLLM 0.30.0 XPU image `vllm/vllm-openai-xpu@sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9` (`gced6857af`; extracted `image-source/vllm`) plus patches 001–005 and the vLLM half of 006 (the 1.0.2 tree).
- **Files affected:** `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`.
- **Applied runtime stage:** Not in a published image. Proposed stage: `b70-offload`, on top of `gdn-index64-c1`.
- **Currently shipping:** No. Not in `ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0`; proposed. It runs on our 4× Arc Pro B70 host in a tree built from 001–006 plus this series (github.com/Lumnus/b70-flash-next, release `0.30.0-b70.1`).
- **Technical explanation:** Opt-in with `B70_OFFLOAD_EMPTY_ADVANCE_GUARD=1`. Guard for vllm#56795 (the stored index advances past keys still write-pending under another request). Not the cause of the miss above, and **off in what we serve**; it is in the series because 017 and 018 were written on top of it. Unset: upstream behaviour.
- **Reproduction/application:** Apply `016-offload-empty-advance-guard.patch` after 015 from the vLLM tree root (`patch -p1 --fuzz=0`).
- **Retained provenance:** github.com/Lumnus/vllm, branch `b70/v0.30.0`, commit `b714309e9e` (`0014d-b70-offload-empty-advance-guard`); exported as `patches/vllm/0014d-b70-offload-empty-advance-guard.patch` in github.com/Lumnus/b70-flash-next. This file is the same change rewritten as a plain unified diff against the 1.0.2 tree; applying the proposed patches to that tree gives files byte-identical to ours.
