# Flash-Next B70 cumulative Disk source

[Disk](disk/README.md) is the canonical source/build package for the Intel AutoRound dynamic MTP5 pack 1.0.4. It retains the GDN/MoE guards, residency shim, Disk BF16 PLE staging, APC/ALIGN, multimodal and tool support. Intel-only preparation validates and maps the PLE safetensor and installs the packaged qualified frontend. The root pack manifest and recipe identify the verified immutable GHCR image.

`disk/SHA256SUMS` checks all source-package inputs. This directory's `SHA256SUMS` covers every other regular file recursively, including the nested checksum manifest. `disk/manifests/accepted.json` locks 3130 installed payload entries; source, compiler, model and native locks accompany it. Only the dynamic MTP5 TP4/262144 profile is qualified for this image. Historical 1.0.3 acceptance anchors are retained in `disk/provenance/historical-1.0.3.json`.

No checkpoint, prebuilt native object/library or Intel compiler package is distributed in this source export. Obtain the 15 locked compiler packages separately from the recorded official Intel pool or an authorized cache. See [notices](disk/NOTICE.md) and the existing root [recipe](../RUNTIME_RECIPE.md). The historical patches below document lineage and must not be reapplied over the cumulative source.

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
