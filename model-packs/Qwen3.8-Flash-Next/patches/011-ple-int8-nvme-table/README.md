# INT8 PLE table served from NVMe

- **Purpose:** INT8 PLE table served from NVMe.
- **Category:** resource.
- **Pinned upstream base:** vLLM 0.30.0 XPU image `vllm/vllm-openai-xpu@sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9` (`gced6857af`; extracted `image-source/vllm`) plus patches 001–005 and the vLLM half of 006 (the 1.0.2 tree).
- **Files affected:** `vllm/models/qwen4_exp/nvidia/model_state.py`; `vllm/models/qwen4_exp/nvidia/ngram_embedding.py`; `vllm/models/qwen4_exp/nvidia/ple_nvme.py` (new); `vllm/v1/worker/gpu/model_runner.py`; CPU tests in `ple-int8-nvme-tests.patch` (not applied to the image).
- **Applied runtime stage:** Not in a published image. Proposed stage: `b70-ple`, on top of `gdn-index64-c1`.
- **Currently shipping:** No. Not in `ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0`; proposed. It runs on our 4× Arc Pro B70 host in a tree built from 001–006 plus this series (github.com/Lumnus/b70-flash-next, release `0.30.0-b70.1`).
- **Technical explanation:** Opt-in with `B70_PLE_INT8_NVME=1` (needs 010). The INT8 table stays on disk; each TP rank keeps a pinned row cache (`B70_PLE_INT8_NVME_CACHE_GIB`, total over all ranks, default 8) with CLOCK eviction. A host hook in `execute_model`, before the graph replay and on real batches only, hashes the batch's n-grams on the host (bit-identical to the device function; boot self-test), resolves them against the cache, reads misses with `O_DIRECT`, and launches the existing gather into the static prefetch buffer. No host I/O runs inside a captured region. Each rank serves only its own rows. Differs from our `0013` only by leaving out its one KV-offload log-line hunk in `gpu_worker.py`, which moved to 019.
- **Reproduction/application:** Apply `011-ple-int8-nvme-table.patch` after 010 from the vLLM tree root (`patch -p1 --fuzz=0`).
- **Retained provenance:** github.com/Lumnus/vllm, branch `b70/v0.30.0`, commit `bd0c807e9e` (`0013-b70-ple-int8-nvme-table`), with one log-line hunk in `gpu_worker.py` moved between 011 and 019 (see above). Applying all proposed patches to the 1.0.2 tree gives files byte-identical to ours.
