> Historical Vanilla stage. Current Disk uses the cumulative source and exact native recipes in [../disk/](../disk/README.md). The old shipping identity and application instructions below describe Vanilla; they are not the current Disk build contract.

# Final Astra GDN speculative metadata

- **Purpose:** Final Astra GDN speculative metadata.
- **Category:** performance.
- **Pinned upstream base:** vLLM 0.30.0 XPU image `vllm/vllm-openai-xpu@sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9` (`gced6857af`; extracted `image-source/vllm`) plus patches 001–004.
- **Files affected:** `vllm/v1/attention/backends/gdn_attn.py`; `gdn_spec_metadata.py`; `gdn_spec_metadata_batched.py`; `vllm/v1/worker/gpu/attn_utils.py`.
- **Applied runtime stage:** `astra-cumulative / mtp3-c1`.
- **Currently shipping:** Yes, in `ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0` (`sha256:85512b52c09fa660a2e7fe441417129e7c47fac727fd85f66ccea6b65e0a9122`).
- **Technical explanation:** The final cumulative overlay provides per-builder uniform staging, an all-compatible-group launch, and dispatch eligibility with stock fallback. It retains each builder’s graph-static output buffers and group-specific state IDs, with layout and pointer invalidation guards. Without speculative decode, the fast path stays inert in Base mode.
- **Reproduction/application:** Apply `gdn-spec-metadata.patch` after 004 from the vLLM tree root. New modules are represented in the unified patch.
- **Retained provenance:** `build/astra-cumulative/manifest.json`; `build/astra-cumulative/source_delta.patch`; `build/astra-cumulative/overlay/`; `build/gdn-index64/PROMOTION.json`.
