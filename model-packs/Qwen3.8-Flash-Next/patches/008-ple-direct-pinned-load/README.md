# PLE direct-to-pinned load

- **Purpose:** PLE direct-to-pinned load.
- **Category:** resource.
- **Pinned upstream base:** vLLM 0.30.0 XPU image `vllm/vllm-openai-xpu@sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9` (`gced6857af`; extracted `image-source/vllm`) plus patches 001–005 and the vLLM half of 006 (the 1.0.2 tree).
- **Files affected:** `vllm/models/qwen4_exp/nvidia/ngram_embedding.py`.
- **Applied runtime stage:** Not in a published image. Proposed stage: `b70-ple`, on top of `gdn-index64-c1`.
- **Currently shipping:** No. Not in `ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0`; proposed. It runs on our 4× Arc Pro B70 host in a tree built from 001–006 plus this series (github.com/Lumnus/b70-flash-next, release `0.30.0-b70.1`).
- **Technical explanation:** Opt-in with `B70_PLE_DIRECT_PINNED=1`. With 002 each TP rank loads its PLE shard twice at once: the weight loader fills a 23.8 GiB pageable shard from the memory-mapped table, then copies it into the two pinned slabs before freeing it. On four ranks that is ~191 GiB of non-evictable host memory at the boot peak. With the switch on, the slabs are filled straight from the memory map (this rank's rows only, vocabulary-padding rows zeroed) and the pageable shard is never written. Measured boot peak on 4 ranks with the BF16 table: ~191 GiB → ~158 GiB. Logs VmRSS/MemAvailable before and after. Switch unset: 002's path, unchanged.
- **Reproduction/application:** Apply `008-ple-direct-pinned-load.patch` after 006 (vLLM half) from the vLLM tree root (`patch -p1 --fuzz=0`).
- **Retained provenance:** github.com/Lumnus/vllm, branch `b70/v0.30.0`, commit `57679613d7` (`0006-b70-ple-direct-pinned-load`); exported as `patches/vllm/0006-b70-ple-direct-pinned-load.patch` in github.com/Lumnus/b70-flash-next. This file is the same change rewritten as a plain unified diff against the 1.0.2 tree; applying the proposed patches to that tree gives files byte-identical to ours.
