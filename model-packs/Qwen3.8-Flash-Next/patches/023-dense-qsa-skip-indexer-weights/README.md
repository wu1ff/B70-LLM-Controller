# Dense QSA: skip indexer tensors at load

- **Purpose:** Dense QSA: skip indexer tensors at load.
- **Category:** correctness.
- **Pinned upstream base:** vLLM 0.30.0 XPU image `vllm/vllm-openai-xpu@sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9` (`gced6857af`; extracted `image-source/vllm`) plus patches 001–005 and the vLLM half of 006 (the 1.0.2 tree).
- **Files affected:** `vllm/models/qwen4_exp/nvidia/model.py`.
- **Applied runtime stage:** Not in a published image. Proposed stage: `b70-load`, on top of `gdn-index64-c1`.
- **Currently shipping:** No. Not in `ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0`; proposed. It runs on our 4× Arc Pro B70 host in a tree built from 001–006 plus this series (github.com/Lumnus/b70-flash-next, release `0.30.0-b70.1`).
- **Technical explanation:** On dense-QSA configs no indexer module is built, but some checkpoints still ship `*.self_attn.indexer.*` tensors inside mixed shards, and the loader then fails with "There is no module or parameter named …". Example: the calibrated AWQ export `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16@0939125` with 007's dense-QSA serve config. This skips those tensors the same way the PLE `hashstats_`/`token_lookup` columns are skipped. Ungated; inert on checkpoints without indexer tensors, including the pinned `devan-carlin` revision.
- **Reproduction/application:** Apply `023-dense-qsa-skip-indexer-weights.patch` after 006 (vLLM half) from the vLLM tree root (`patch -p1 --fuzz=0`).
- **Retained provenance:** github.com/Lumnus/vllm, branch `b70/v0.30.0`, commit `01abfaaa54` (`0019-b70-qwen4exp-dense-qsa-skip-indexer`); exported as `patches/vllm/0019-b70-qwen4exp-dense-qsa-skip-indexer.patch` in github.com/Lumnus/b70-flash-next. This file is the same change rewritten as a plain unified diff against the 1.0.2 tree; applying the proposed patches to that tree gives files byte-identical to ours.
