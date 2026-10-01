# NVMe PLE: native reader and prefill lookahead

- **Purpose:** NVMe PLE: native reader and prefill lookahead.
- **Category:** performance.
- **Pinned upstream base:** vLLM 0.30.0 XPU image `vllm/vllm-openai-xpu@sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9` (`gced6857af`; extracted `image-source/vllm`) plus patches 001–005 and the vLLM half of 006 (the 1.0.2 tree).
- **Files affected:** `vllm/models/qwen4_exp/nvidia/model_state.py`; `vllm/models/qwen4_exp/nvidia/ngram_embedding.py`; `vllm/models/qwen4_exp/nvidia/ple_nvme.py`.
- **Applied runtime stage:** Not in a published image. Proposed stage: `b70-ple`, on top of `gdn-index64-c1`.
- **Currently shipping:** No. Not in `ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0`; proposed. It runs on our 4× Arc Pro B70 host in a tree built from 001–006 plus this series (github.com/Lumnus/b70-flash-next, release `0.30.0-b70.1`).
- **Technical explanation:** Two knobs on top of 011, both default to 011's behaviour. `B70_PLE_INT8_NVME_READER=native` uses a C thread pool that reads a whole batch in one call with the GIL released (compiled with `gcc` at first use into `B70_PLE_INT8_NVME_NATIVE_DIR`, default `$TMPDIR/b70-ple-nvme`; the base image has gcc); `uring` uses io_uring via raw syscalls. `B70_PLE_INT8_NVME_LOOKAHEAD=1` reads the next prefill chunk's rows in a background thread per rank. A 1,024-token prefill chunk on a trigram rank reads in 63 ms (py), 31 ms (uring), 23 ms (native). We serve `native`, lookahead off.
- **Reproduction/application:** Apply `012-ple-int8-nvme-lookahead.patch` after 011 from the vLLM tree root (`patch -p1 --fuzz=0`).
- **Retained provenance:** github.com/Lumnus/vllm, branch `b70/v0.30.0`, commit `41a455c5a1` (`0013b-b70-ple-int8-nvme-lookahead`); exported as `patches/vllm/0013b-b70-ple-int8-nvme-lookahead.patch` in github.com/Lumnus/b70-flash-next. This file is the same change rewritten as a plain unified diff against the 1.0.2 tree; applying the proposed patches to that tree gives files byte-identical to ours.
