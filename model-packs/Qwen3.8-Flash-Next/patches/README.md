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

## Proposed: CPU KV offload fixes for the hybrid model (013–019)

| Patch | Purpose | Category | Source base | Shipping |
| --- | --- | --- | --- | --- |
| [013](013-offload-trace/README.md) | KV-offload instrumentation | observability | vLLM 1.0.2 tree, after 006 (vLLM half) | No (proposed) |
| [014](014-offload-junction/README.md) | Hybrid offload miss: set a shared-prefix junction | correctness | vLLM 1.0.2 tree, after 013 | No (proposed) |
| [015](015-offload-gdn-backstep/README.md) | Retain GDN states N blocks below the replay boundary | correctness | vLLM 1.0.2 tree, after 014 | No (proposed) |
| [016](016-offload-empty-advance-guard/README.md) | Do not skip past write-pending keys (vllm#56795) | correctness | vLLM 1.0.2 tree, after 015 | No (proposed) |
| [017](017-offload-group-evict/README.md) | Backport of vllm#51787 (apply only together with 018) | correctness | vLLM 1.0.2 tree, after 016 | No (proposed) |
| [018](018-offload-gates/README.md) | Gate 017 behind B70_OFFLOAD_GROUP_EVICT; cheap TRACE=1 | correctness | vLLM 1.0.2 tree, after 017 | No (proposed) |
| [019](019-offload-chunked-pinned-pool/README.md) | Split the pinned CPU KV pool when one allocation is too large | resource | vLLM 1.0.2 tree, after 018 | No (proposed) |

Apply 013–019 in order after the vLLM half of 006 (they do not depend on 008–012). They only matter with the CPU KV offload tier (`--kv-offloading-size <GiB> --kv-offloading-backend native`), which the pack does not enable today. Apply 017 and 018 together or not at all: 017 alone changes the eviction policy ungated.

What we serve, in addition to the offload flags: `B70_OFFLOAD_TRACE=1`, `B70_OFFLOAD_JUNCTION=1`, `B70_OFFLOAD_GDN_BACKSTEP=1`; `B70_OFFLOAD_EMPTY_ADVANCE_GUARD` and `B70_OFFLOAD_GROUP_EVICT` unset. 019 is not gated: a pool that fits in one pinned allocation is unchanged; a larger one (≥ ~31 GiB per rank on a B70) is split into chunks instead of failing at boot.

Size the pool per rank as a power of two (`--kv-offloading-size` ÷ TP) and keep torch's default pinned-memory rounding. With `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1024` (exact pinned sizes) the offload copy segfaulted (`xpuAsyncMemcpyBatch`) in 2 of 2 runs under load on torch 2.13; with the default rounding, 0 of 2. Adding `pinned_max_cached_size_mb:1024` keeps the host-RAM use of the copy cache bounded; we run with that alone.
