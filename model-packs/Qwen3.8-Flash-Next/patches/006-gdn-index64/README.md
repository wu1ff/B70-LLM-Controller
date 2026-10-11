> Historical Vanilla stage. Current Disk uses the cumulative source and exact native recipes in [../disk/](../disk/README.md). The old shipping identity and application instructions below describe Vanilla; they are not the current Disk build contract.

# GDN state-offset index64

- **Purpose:** GDN state-offset index64.
- **Category:** correctness.
- **Pinned upstream base:** vllm-xpu-kernels `0.1.14.1` commit `6d92b1bfbf32767ecda8e819613eb151e70030ad`, plus the pinned vLLM tree.
- **Files affected:** `csrc/xpu/gdn_attn/causal_conv1d.hpp`; `csrc/xpu/gdn_attn/xe_2/chunk_causal_conv1d_xe2.hpp`; `csrc/xpu/gdn_attn/xe_2/chunk_causal_conv1d_tiled_xe2.hpp`; `vllm/_xpu_ops.py`; `gdn_index64.cpp`.
- **Applied runtime stage:** `gdn-index64-c1 / runtime-authority`.
- **Currently shipping:** Yes, in `ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0` (`sha256:85512b52c09fa660a2e7fe441417129e7c47fac727fd85f66ccea6b65e0a9122`).
- **Technical explanation:** Seven state-offset products cast the valid state ID to 64 bits before multiplication. Valid ID 5042 crossed the old signed 32-bit product limit; the ID itself was valid. A private correction library registers the fixed operator and vLLM dispatch selects it.
- **Reproduction/application:** Apply `gdn-index64-kernels.patch` to the pinned kernel commit and `gdn-index64-vllm.patch` to the vLLM tree after 005; see `BUILD.md`.
- **Retained provenance:** `build/gdn-index64/manifest.json`; `build/gdn-index64/PROMOTION.json`; `build/gdn-index64/source_delta.patch`; `build/gdn-index64/vllm_delta.patch`.
