> Historical Vanilla stage. Current Disk uses the cumulative source and exact native recipes in [../disk/](../disk/README.md). The old shipping identity and application instructions below describe Vanilla; they are not the current Disk build contract.

# Pinned-host PLE table

- **Purpose:** Pinned-host PLE table.
- **Category:** resource.
- **Pinned upstream base:** vLLM 0.30.0 XPU image `vllm/vllm-openai-xpu@sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9` (`gced6857af`; extracted `image-source/vllm`).
- **Files affected:** `vllm/models/qwen4_exp/nvidia/ngram_embedding.py`.
- **Applied runtime stage:** `base-c5`.
- **Currently shipping:** Yes, in `ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0` (`sha256:85512b52c09fa660a2e7fe441417129e7c47fac727fd85f66ccea6b65e0a9122`).
- **Technical explanation:** Select pinned-host PLE on XPU with `PLE_TABLE_PATH`; load the external in-revision table through the native shard loader; split large host shards into two pinned slabs; use slab-aware UVA lookup and XPU streams. The compressed-tensors ignore-list path keeps the PLE unquantized.
- **Reproduction/application:** Apply `002-pinned-host-ple.patch` after 001 from the vLLM tree root.
- **Retained provenance:** `build/manifest.json`; `build/overlay/vllm/models/qwen4_exp/nvidia/ngram_embedding.py`; `docs/BASE_BRINGUP.md`.
