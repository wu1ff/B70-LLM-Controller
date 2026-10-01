# PLE table in FP8 (dependency of 010)

- **Purpose:** PLE table in FP8 (dependency of 010).
- **Category:** resource.
- **Pinned upstream base:** vLLM 0.30.0 XPU image `vllm/vllm-openai-xpu@sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9` (`gced6857af`; extracted `image-source/vllm`) plus patches 001–005 and the vLLM half of 006 (the 1.0.2 tree).
- **Files affected:** `vllm/models/qwen4_exp/nvidia/ngram_embedding.py`.
- **Applied runtime stage:** Not in a published image. Proposed stage: `b70-ple`, on top of `gdn-index64-c1`.
- **Currently shipping:** No. Not in `ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0`; proposed. It runs on our 4× Arc Pro B70 host in a tree built from 001–006 plus this series (github.com/Lumnus/b70-flash-next, release `0.30.0-b70.1`).
- **Technical explanation:** Opt-in with `B70_PLE_FP8=1` + `B70_PLE_FP8_PATH`. Keeps the table as FP8 E4M3 with one global scale in the pinned slabs (47.7 GiB over 4 ranks) and dequantises only the gathered rows. **Included because 010 is built on its loader (safetensors reader, row copy, layout check), not as a recommended mode:** FP8 measured 2.65 % relative L2 error against BF16, four times INT8's, and we do not serve it. Switch unset: 008's path, unchanged.
- **Reproduction/application:** Apply `009-ple-fp8-pinned-table.patch` after 008 from the vLLM tree root (`patch -p1 --fuzz=0`).
- **Retained provenance:** github.com/Lumnus/vllm, branch `b70/v0.30.0`, commit `7a2f0c95ae` (`0007-b70-ple-fp8-pinned-table`); exported as `patches/vllm/0007-b70-ple-fp8-pinned-table.patch` in github.com/Lumnus/b70-flash-next. This file is the same change rewritten as a plain unified diff against the 1.0.2 tree; applying the proposed patches to that tree gives files byte-identical to ours.
