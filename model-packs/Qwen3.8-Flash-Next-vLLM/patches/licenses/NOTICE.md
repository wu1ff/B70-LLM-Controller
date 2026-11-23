# v0.33 release scope

The following published v0.32 attribution/notice text is retained verbatim as a
historical record. All its accompanying notice and source-access assets remain
in this package. Current accepted v0.33 sources and provenance are in
[production](../production/BUNDLE.md); compiler build inputs are oneAPI
2026.0.0.20260331, while inherited Intel/SYCL runtime is 2026.1.1. The old
`compiler-input/` and `accepted-vendor/` trees describe the previously inspected
v0.32 stack; they do not attest every v0.33 binary. Existing per-file copyrights,
SPDX and terms in the accepted source package are unchanged.

The historical 57-file annotated export and its original mapping are retained as
[distribution-source.tar.gz](../archive/v032-source-delivery/distribution-source.tar.gz)
and [source-export.json](../archive/v032-source-delivery/manifests/source-export.json).
References below to old production notice-export tools/delivery describe the
published 1.0.5 package, not the new ten-patch v0.33 production subtree. No new
clearance, author identity, license grant or compliance infrastructure is asserted.

---

# DFlash2 source notices and attribution

This pack reconstructs the accepted B70 Flash-Next DFlash2 runtime. Retain all
original SPDX, copyright, license, NOTICE and embedded third-party terms. The
Controller's MIT license does not relicense these sources or model assets.

- Wulf: Disk/checkpoint/frontend/serving integration and residency shim work,
  retained in campaign and historical attribution records.
- Astra: DFlash2 implementation, native/graph/ownership/prefill and serving campaign
  work identified in the retained records. No new personal author or release
  co-author is inferred.
- vLLM and vllm-xpu-kernels: retain Apache-2.0 and applicable per-file notices;
  native files with Intel BSD notices retain those exact terms.
- Intel SYCL-TLA at `87f6850680a580654b9ea2c80dbc01aeb36ad231`: retain its BSD
  and embedded per-file third-party terms in the complete pinned source archive.
- Qwen: baked frontend/configuration assets retain Qwen Community License 1.0.
  Intel AutoRound target and incoai DFlash2 model cards/revisions are retained
  under `models/`. The draft card states Apache-2.0. Weights are acquired separately.
- Historical KaloyanNaumov/Lumnus research/configuration influence and independent
  implementation attribution remain in `historical-controller/` and
  `historical-1.0.4/`; those records are historical, not current executable clearance.

`accepted-vendor/` preserves notices extracted from the accepted stack;
`compiler-input/` preserves readable 2026.1.1 notices/designation records. No
compiler executables, checkpoint weights or native library binaries are in this
source archive. Historical Torch 2.13/Intel 2026.0 records do not attest the current
Torch 2.14/Intel 2026.1.1 stack.

The exact accepted source is the reconstruction comparison authority. Separate
comment-only copies in `notice-overlay/` add prominent changed-file notices to
all fifteen previously missing upstream vLLM headers, the modified kernel wrapper,
three corrected GDN headers, and custom campaign Python/native sources. Original
source/patch bytes and existing notices are unchanged. `notice-export.json`
records each input/output hash and exact insertion; `production/tools/notice_export.py`
checks the insertion and Python AST equality before creating an export. This
source-only derivative is not the frozen image and makes no native binary claim.

[Source access](SOURCE_ACCESS.md) and [distribution status](SOURCE_DISTRIBUTION.md)
remain part of this notice package. No source offer, new third-party license grant,
Intel recipient agreement or general distribution clearance is made here.

The 57 reviewed copies are now incorporated in
`production/distribution-source.tar.gz` (paths relative to patches/), with their
own source-export manifest and exact accepted-to-annotated mapping. They do not
replace accepted comparison or build inputs. Executable notice placement and
historical archival-original notice treatment remain separate release decisions;
see `source-delivery/archival-notice-review.json`. Production coverage does not
constitute blanket clearance of the complete experimental archive.
