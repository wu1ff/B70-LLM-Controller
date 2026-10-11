# Flash-Next B70 cumulative Disk source

The current distributed Disk runtime is `ghcr.io/wu1ff/qwen38-flashnext-b70:disk`, qualified RC Docker ID `sha256:b13fed6dd54051395f603409ada440c74afcbd5331810eee3acd62315d7787e3`, with verified immutable acquisition reference `ghcr.io/wu1ff/qwen38-flashnext-b70@sha256:b13fed6dd54051395f603409ada440c74afcbd5331810eee3acd62315d7787e3`. Base and MTP3 use this one GHCR version. The predecessor local `qwen38-flashnext:disk` and Vanilla source/image history are preserved for local recovery; the superseded Vanilla GHCR versions have been removed after replacement verification.

[Disk](disk/README.md) is the complete accepted source/build export: vendor-to-Disk overlays, native sources/headers/dispatchers, Wulf's residency shim, SYCL-TLA, dependency locks, licenses, qualified Base/MTP3 receipts and the accepted installed checksum manifest. It includes Disk BF16 mmap/staging, conditional Base scheduling, corrected GDN partial verification, all three accepted MTP3 optimization rounds, guarded MoE and native tiled attention reduction. Historical Vanilla patches alone are insufficient.

`disk/SHA256SUMS` checks the source package; root `SHA256SUMS` covers every other regular file below this directory, including that nested manifest. Final source is already patched; historical diffs document lineage and must not be reapplied. `disk/manifests/accepted.json` records 3,118 installed identities and `runtime-lock.json` locks the four native outputs.

All 15 Intel compiler **package binaries are excluded** from this public export. Their exact hashes/acquisition location remain locked; the reproducible build wrapper accepts a separately obtained authorized cache. No checkpoint or accepted prebuilt native library/object is included. Licensing/provenance findings are recorded in [disk/NOTICE.md](disk/NOTICE.md); old permission/co-authorship assumptions are superseded by the completed source audit. The technical mechanisms and qualification bounds are in the existing root [recipe](../RUNTIME_RECIPE.md).

---

# Historical Vanilla 001–007 runtime source patches

These retained deltas describe published Vanilla through pack1.0.2. Its GitHub release and source remain preserved. The former GHCR image below was intentionally superseded and cannot be downloaded there; rollback requires a retained local image:

```text
ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0
sha256:85512b52c09fa660a2e7fe441417129e7c47fac727fd85f66ccea6b65e0a9122
```

The vLLM base is the official `vllm/vllm-openai-xpu` v0.30.0 amd64 image at `sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9` (`gced6857af`; retained pristine `image-source/vllm`). The native base is `vllm-xpu-kernels` 0.1.14.1 at commit `6d92b1bfbf32767ecda8e819613eb151e70030ad`. The runtime’s installed kernel package is 0.1.14.1 with torch 2.13.0+xpu.

| Patch | Purpose | Category | Source base | Historical Vanilla shipping |
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
