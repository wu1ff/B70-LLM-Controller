# Flash-Next B70 runtime source patches

These are the exact retained source deltas used by the published Qwen3.8-Flash-Next B70 runtime. The production image remains:

```text
ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0
sha256:85512b52c09fa660a2e7fe441417129e7c47fac727fd85f66ccea6b65e0a9122
```

The vLLM base is the official `vllm/vllm-openai-xpu` v0.30.0 amd64 image at `sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9` (`gced6857af`; retained pristine `image-source/vllm`). The native base is `vllm-xpu-kernels` 0.1.14.1 at commit `6d92b1bfbf32767ecda8e819613eb151e70030ad`. The runtime’s installed kernel package is 0.1.14.1 with torch 2.13.0+xpu.

| Patch | Purpose | Category | Source base | Shipping |
| --- | --- | --- | --- | --- |
| [001](001-qwen4exp-xpu-base/README.md) | Qwen4Exp XPU gate | platform | vLLM 0.30.0 XPU | Yes |
| [002](002-pinned-host-ple/README.md) | External, two-slab pinned-host PLE | resource | vLLM 0.30.0 XPU after 001 | Yes |
| [003](003-ple-full-graph-capture/README.md) | FULL capture-stream correction | correctness | vLLM after 002 | Yes |
| [004](004-hc-ksplit/README.md) | Deterministic wide HC down GEMM | correctness | vLLM after 003 | Yes |
| [005](005-gdn-spec-metadata/README.md) | Final cumulative GDN metadata staging | performance | vLLM after 004 | Yes |
| [006](006-gdn-index64/README.md) | Seven native index-width fixes, private operator, dispatch | correctness | kernel commit 6d92b1b; vLLM after 005 | Yes |
| [007](007-serve-packaging/README.md) | Dense-QSA config and serve preparator | correctness | pinned model config / unified runtime | Yes |
| [Shim](b70-residency-shim/README.md) | B70 peer-residency preload | resource | standalone C / pinned parent image | Yes |

Apply vLLM patches 001–005 in order to the pristine vLLM tree, then the vLLM dispatch patch in 006. Apply the native patch in 006 separately to the pinned kernel source. Apply the model config patch in 007 to the pinned model config. Build the native correction library and shim from their retained C/C++ source and recipes. The image lineage was base-c5 → full-c1 → hcsplit-c1 → astra-cumulative/mtp3-c1 → gdn-index64-c1, then the 007 packaging layer with the pinned serving config and entrypoint. Both Base and MTP3 use the final unified image.

The patches were checked by applying them to clean retained source copies in lineage order and comparing the resulting files byte-for-byte with the production overlays. Native patch reversal and reapplication were checked against the retained pinned tree. `SHA256SUMS` covers all other files in this directory, in sorted order. Rejected experiments, temporary probes, observers, and diagnostic-only patches are intentionally omitted.

## Proposed: INT8 PLE table, optionally served from NVMe (008–012)

| Patch | Purpose | Category | Source base | Shipping |
| --- | --- | --- | --- | --- |
| [008](008-ple-direct-pinned-load/README.md) | PLE direct-to-pinned load | resource | vLLM 1.0.2 tree, after 006 (vLLM half) | No (proposed) |
| [009](009-ple-fp8-pinned-table/README.md) | PLE table in FP8 (dependency of 010) | resource | vLLM 1.0.2 tree, after 008 | No (proposed) |
| [010](010-ple-int8-rowscale-table/README.md) | PLE table in INT8, one scale per row | resource | vLLM 1.0.2 tree, after 009 | No (proposed) |
| [011](011-ple-int8-nvme-table/README.md) | INT8 PLE table served from NVMe | resource | vLLM 1.0.2 tree, after 010 | No (proposed) |
| [012](012-ple-int8-nvme-lookahead/README.md) | NVMe PLE: native reader and prefill lookahead | performance | vLLM 1.0.2 tree, after 011 | No (proposed) |

Apply 008–012 in order after the vLLM half of 006. Every path is opt-in by environment and off by default; with no `B70_PLE_*` variable set the runtime takes 002's path unchanged. 009 (FP8) is carried as a code dependency of 010 and is not a recommended mode.

What we serve: `B70_PLE_DIRECT_PINNED=1`, `B70_PLE_INT8=1`, `B70_PLE_INT8_PATH=<table>`, and, to keep the table on NVMe, `B70_PLE_INT8_NVME=1`, `B70_PLE_INT8_NVME_CACHE_GIB=8`, `B70_PLE_INT8_NVME_READER=native`. `PLE_TABLE_PATH` stays set as the boot cross-check reference. The INT8 table is a derived file: build it once from the checkpoint's `ple_table_qwen4exp.pt` with `010-ple-int8-rowscale-table/build_int8_ple.py` (`build`, then `verify`). It is not part of the Hugging Face revision, so the pack cannot download it; that needs a decision on how to ship or build derived files.
