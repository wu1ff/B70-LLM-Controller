> Historical Vanilla stage. Current Disk uses the cumulative source and exact native recipes in [../disk/](../disk/README.md). The old shipping identity and application instructions below describe Vanilla; they are not the current Disk build contract.

# Deterministic HC K-split

- **Purpose:** Deterministic HC K-split.
- **Category:** correctness.
- **Pinned upstream base:** vLLM 0.30.0 XPU image `vllm/vllm-openai-xpu@sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9` (`gced6857af`; extracted `image-source/vllm`) plus patches 001–003.
- **Files affected:** `vllm/models/qwen4_exp/nvidia/hyperconnection.py`; `vllm/model_executor/layers/linear.py`.
- **Applied runtime stage:** `hcsplit-c1`.
- **Currently shipping:** Yes, in `ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0` (`sha256:85512b52c09fa660a2e7fe441417129e7c47fac727fd85f66ccea6b65e0a9122`).
- **Technical explanation:** Prepare contiguous halves of the HC down-projection weight after loading, before graph capture. On XPU at row widths above 64, sum two half-K BF16 linear operations; narrow widths use the stock GEMM. This removes the wide-width accumulation race that seeded replay corruption.
- **Reproduction/application:** Apply `004-hc-ksplit.patch` after 003 from the vLLM tree root.
- **Retained provenance:** `build/hcsplit-c1/manifest.json`; `build/hcsplit-c1/overlay/`; `docs/HC_DOWN_GEMM_FIX_RECON.md`.
