> Historical Vanilla stage. Current Disk uses the cumulative source and exact native recipes in [../disk/](../disk/README.md). The old shipping identity and application instructions below describe Vanilla; they are not the current Disk build contract.

# PLE FULL graph capture

- **Purpose:** PLE FULL graph capture.
- **Category:** correctness.
- **Pinned upstream base:** vLLM 0.30.0 XPU image `vllm/vllm-openai-xpu@sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9` (`gced6857af`; extracted `image-source/vllm`) plus patches 001–002.
- **Files affected:** `vllm/models/qwen4_exp/nvidia/ngram_embedding.py`.
- **Applied runtime stage:** `full-c1`.
- **Currently shipping:** Yes, in `ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0` (`sha256:85512b52c09fa660a2e7fe441417129e7c47fac727fd85f66ccea6b65e0a9122`).
- **Technical explanation:** During XPU graph capture, run lookup on the capture stream and skip the side-stream wait. Outside capture, keep the side-stream path. This permits `FULL_AND_PIECEWISE` without the Level Zero cross-stream event rejection.
- **Reproduction/application:** Apply `003-ple-full-graph-capture.patch` after 002 from the vLLM tree root.
- **Retained provenance:** `build/full-c1/manifest.json`; `build/full-c1/overlay/vllm/models/qwen4_exp/nvidia/ngram_embedding.py`.
