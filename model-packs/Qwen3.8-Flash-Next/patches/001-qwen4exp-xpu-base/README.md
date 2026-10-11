> Historical Vanilla stage. Current Disk uses the cumulative source and exact native recipes in [../disk/](../disk/README.md). The old shipping identity and application instructions below describe Vanilla; they are not the current Disk build contract.

# Qwen4Exp XPU backend gate

- **Purpose:** Qwen4Exp XPU backend gate.
- **Category:** platform.
- **Pinned upstream base:** vLLM 0.30.0 XPU image `vllm/vllm-openai-xpu@sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9` (`gced6857af`; extracted `image-source/vllm`).
- **Files affected:** `vllm/models/qwen4_exp/__init__.py`.
- **Applied runtime stage:** `base-c5`.
- **Currently shipping:** Yes, in `ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0` (`sha256:85512b52c09fa660a2e7fe441417129e7c47fac727fd85f66ccea6b65e0a9122`).
- **Technical explanation:** Narrow the backend rejection to TPU so the pinned XPU path uses the available Qwen4Exp backend. Stock model plumbing already exists in this pin; no other Base model source edit ships.
- **Reproduction/application:** Apply `001-qwen4exp-xpu-base.patch` from the vLLM tree root.
- **Retained provenance:** `build/manifest.json`; `build/overlay/vllm/models/qwen4_exp/__init__.py`.
