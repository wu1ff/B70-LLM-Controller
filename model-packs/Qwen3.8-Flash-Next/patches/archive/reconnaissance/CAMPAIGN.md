# Complete DFlash2 campaign source ledger

This ledger describes retained source and evidence, not new optimization work. Paths below are relative to `/home/wulf/Desktop/qwen3.8-flash-next-dflash2-v032` unless absolute. Original bytes are preserved in `originals/`; full path/hash/target inventories are under `inventories/`. The workspace itself has no enclosing Git repository. Its source checkouts do, and their commits/working-tree status are recorded.

## Pinned foundations

| Input | Revision or immutable identity |
| --- | --- |
| vLLM v0.32.0 | `5df63a7203cdb9328d824a27b682e77c7b9928be` |
| official XPU vendor image | `vllm/vllm-openai-xpu@sha256:9dd913e9036633d55ace17265cbfc29f48b9b85fe4bc8b129debcf7e1b269ce7` |
| Intel llm-scaler branch reference | `f266b606be293c79c39012d47935d464ff68a513` |
| Intel v0.31 base / reconstructed tree | `db9527a46873454610df6dbedf79a36d6bf1a7f6` / `92b365c00c93d5183506bf9494b3e699451755a1` |
| kernel reference base / reconstructed Intel tree | `ddf336d86e3c8602888572a3502f951abd51df12` / `1ce13e70894b11165f66a357d512517bf81c7bf6` |
| SYCL-TLA | `87f6850680a580654b9ea2c80dbc01aeb36ad231` |
| compiler image | `intel/omix:0.4.0-devel-ubuntu26.04@sha256:59cef13f8e87275846cc08e01d32b49082694a5c01d842d2e845552cbc8c199f` |
| target / draft revisions | `4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897` / `a22a1744e85c17364dcc54a6a420af7135d4c7f6` |

The current official dependency image supplies Torch **2.14.0+xpu**, Triton XPU **3.8.0**, Transformers **5.19.0**, XPU kernels **0.1.15.4**, and Intel runtimes **2026.1.1**. `evidence/accepted-python-distributions.json` and `accepted-dpkg-status.txt` record installed versions without importing model/GPU code. Intel's reference manifest does not attest the exact source commit used to build the stock kernel wheel; the immutable vendor image pins its binary bytes instead. Do not relabel the reference source as attested stock-wheel source.

## Original campaign patches and actual application order

All names in this table refer to original files under `patches/`. These are **incremental** changes in the verified order below. Historical documentation sometimes describes the combined result as cumulative; concatenating that result with already-applied snapshots would double-apply changes. These patches alone do not compile or deliver the native companions.

| Order | Original patch | Role in accepted runtime / supersession |
| ---: | --- | --- |
| 1 | `target-v032.patch` | XPU target route, Intel checkpoint adaptation, Disk PLE/BF16 HC, completed-layer auxiliary features. No external drafter yet. |
| 2 | `dflash2-v032.patch` | Real BF16 DFlash2 loading/configuration, realized conditional selector probability cache, support/rejection/recovery and state handling; ordered HC math. Initial serial C1 verifier is later replaced. |
| 3 | `verify-c1-v032.patch` | One width-four target forward; canonical one-row native XCCL communication to preserve established BF16 arithmetic. Serial replay remains a test reference. |
| 4 | `perf-c1-v032.patch` | Grouped row communications and 97 stateless HC graphs. Initial fixed-width leaf-graph scope is superseded/extended later. |
| 5 | `fullwidth-c1-v032.patch` | Eight-position target verification/seven proposals, whole-target guarded replay, fused metadata and depth-aware state staging. |
| 6 | `deep-c1-v032.patch` | Packed rank-quarter reduction, deterministic split-K HC projection; later HC implementations extend this dispatch. |
| 7 | `mtp7-gap-v032.patch` | Fixed-split target attention and target graph changes used in the matched MTP7 investigation. |
| 8 | `astra500-v032.patch` | Deterministic prefill gates, C1–C4 target graphs, 1024 PLE capacity with bounded clearing, native W4 MoE dispatch. Requires `libastra_moe.so`. |
| 9 | `astra500-structural.patch` | Direct-token MoE and native HC-up/gate/mix; requires native structural sources. Old direct whole-expert path remains imported/fallback material after ownership changes. |
| 10 | `astra500-cycle.patch` | Guarded vocab-parallel greedy verification; ordinary sampled/logprob/constraint paths remain. |
| 11 | `astra500-ownership.patch` | Cyclic ownership of intact quantization groups, remap/gather plus C1 and C2–C4 consumers. Requires both groups and batch native libraries. |
| 12 | `astra500-beyond.patch` | Target-consumer-only graph staging and smaller prefill expert workspace. Later prefill consumers replace selected dispatch ranges, but its library is still loaded. |
| — | `astra500-consumers.patch` | **Rejected branch after beyond-c1.** Cooperative fused expert producer/consumer implementation; no useful endpoint gain. Never apply to production. |
| 13 | `astra500-dominant.patch` | Selected C1 W4 layout mode 1. Original diff uses `build/<context>/payload/` paths: apply with **`git apply -p3`** to a package-root tree, not the normal `-p1`. Native `w4_layout.cpp`/`layout_gemm.hpp` are separate. |
| 14 | `prefill-serving.patch` | Removes redundant row specialization, short expert tile, bounded stateless short-HC capture and unfinished-prefill draft context-only path; selected image is **prefill-v5**, not v1–v4. |
| 15 | `long-prefill.patch` | Large native HC down projection and eight-column expert gather sharing. Existing C1/short/concurrent dispatch remains. |
| 16 | `integrated-serving.patch` | Spare-budget short admission with progress/slot protection; shared top-k/top-p specialization plus bounded initialized-tensor startup warmup. Selected recipe is **`build/integrated-v2/Dockerfile`**, tag **integrated-serving-v1**. |

Additional produced diagnostics: `evidence/hc-v032-v104.diff` compares implementations; `evidence/phase4/incomplete-patch-detected.patch` is a **superseded incomplete patch**, not a production alternative. Preserve both. The campaign therefore has **17 named runtime patches + two diagnostic/incomplete diffs**, all indexed with full SHA256 and targets. The broader census includes 96 original patch/diff paths, encompassing published legacy and upstream reference material. A separately retrieved pinned drafter capture patch is external reference material, bringing the indexed reference census to 97 paths.

`reconstruct.py` replayed the **16 selected patches**, with a copy of the retained vendor `fused_moe_interface.py` as the kernel Python baseline. All patches apply without rejects. The result exactly matches **28 changed/added vLLM implementation files** and **four custom/modified kernel Python files** extracted from the accepted container. The extra `vllm/_version.py` is vendor-generated version metadata (`0.32.0`, commit `g5df63a720`) and is preserved separately. Source matches do not assert a fresh binary/image rebuild. `docker diff` found no changed installed `.py` or `.so` files in that container.

The verified accepted image layer prefixes follow phase1-c5 → phase2-c6 → phase3-c2 → phase4-c1 → fullwidth-c1 → deep-seven-c1 → mtp7-gap → astra500-retained → structural-c2 → cycle-c2 → ownership-c3 → beyond-c1 → dominant-c1 → prefill-v5 → long-prefill-v1 → integrated-serving-v1. `inventories/image-lineage.json` supplies every full image ID and branch verdict; test and rejected image tags are not inferred to be ancestors.

## All custom native libraries physically in the accepted payload

Preserve even imported fallback libraries when reproducing the accepted payload. A library being present or loaded is distinct from its dispatch being selected for an ordinary request. Exact full binary hashes and matching retained output paths are in `native-binary-matches.json`; binary copies under `accepted-extract/` are local comparison evidence, **not source-package contents**.

| Accepted library | Exact source family and build recipe | Selected use |
| --- | --- | --- |
| `libgdn_index64.so` | `build/native/gdn_index64.cpp`, `build/native/source/`, `build/native/build.sh`, `build/Dockerfile.native` | Native conv address safety; links unchanged vendor delta-rule library. |
| `libl0_peer_residency_shim.so` | `sources/release-1.0.4/native/l0_peer_residency_shim.c`; gcc command in `build/Dockerfile` | Required preloaded residency shim. |
| `libastra_moe.so` | `build/astra500-native/astra_moe.cpp`, `build.sh`, `build/Dockerfile.astra500-native` | Native grouped GEMM and fallback infrastructure; source SHA256 `38dd2f12da21a143a6cfa7050d630e460954cfa06499c672d7023b554cd08baf`. |
| `libastra_hc.so` | `build/astra500-structural-hc/astra_hc_v3.cpp`, `fused_hc_v3.hpp`, `build-v3.sh`, `build/Dockerfile.astra500-structural-hc-v3` | Small HC up/mix. Output `libastra_hc_v3.so` is installed under this shorter name. |
| `libastra_direct.so` | `build/astra500-structural-moe/astra_direct.cpp` (= v2), `direct_gemm.hpp`, `build.sh`, structural-moe Dockerfile | Imported older whole-expert fallback; exact output is retained as `out/libastra_direct_v2.so`. |
| `libastra_groups.so` | `build/astra500-ownership-groups/{astra_direct.cpp,direct_gemm.hpp,ownership_remap.cpp,ownership_gather.cpp,build.sh}`, ownership-groups Dockerfile | Ownership preparation/general fallback. Accepted output is `out-v2/libastra_groups.so`. |
| `libastra_batch.so` | `build/astra500-ownership-batch/{astra_batch.cpp,direct_gemm.hpp,build.sh}`, ownership-batch Dockerfile | 16/24/32-row C2–C4 consumer. |
| `libbeyond_prefill.so` | `build/astra500-beyond-prefill/{beyond_prefill.cpp,direct_gemm_v2.hpp,build.sh}`, beyond-prefill Dockerfile | Still eagerly loaded; selected <=1024-row dispatch is now covered by newer consumers. |
| `libdominant_w4_layout.so` | `build/astra500-dominant-native/{w4_layout.cpp,layout_gemm.hpp,build.sh}`, dominant-native Dockerfile | Eight-row C1 mode 1; accepted output `out-layout-fixed/`. |
| `libserving_prefill.so` | `build/prefill-native/{prefill_small.cpp,direct_gemm_prefill.hpp,build.sh}`, prefill-native Dockerfile | Irregular short rows through 256 after C1/C2–C4 dispatch. |
| `liblong_hc.so` | `build/long-hc-selected/{hc.cpp,hc_tile.hpp,build.sh}`, long-hc-selected Dockerfile | HC K=10240, N=336/320, rows 257–1024. |
| `liblong_prefill_v3.so` | `build/long-prefill-native/{prefill_small.cpp,direct_gemm.hpp,build.sh}`, long-prefill-native Dockerfile | Expert consumer at rows 257–1024; v3 gather. |

All 81 native/recipe path/hash entries checked against the retained campaign manifests match. This checks retained inputs; it is not a claim that all native objects have been rebuilt independently during Phase 1. The twelve-family map includes the shim whose accepted binary is identified in earlier qualification evidence; the other eleven have matching retained build outputs.

Common sources: **238 files** in the Intel-patched kernel header/reference tree used by the MoE/HC builders; **184 files** in the GDN builder tree; **5703 files** in the pinned SYCL-TLA tree. Across 34 retained source/TLA roots there are exactly three distinct tree hashes. The GDN tree differs from kernel base in precisely the three conv headers (`causal_conv1d.hpp`, `chunk_causal_conv1d_xe2.hpp`, `chunk_causal_conv1d_tiled_xe2.hpp`). Their int64 changes must accompany the wrapper. The common 238-file tree contains additional Intel reference sources: those files' presence is not proof that every Intel kernel is compiled or shipped. Full file locks and deltas are in `native-dependency-trees.json` and `native-header-deltas.json`.

## Complete native experimental archive and reasons

`custom-native-and-build.json` lists **113** immediate custom source/build files; `FILE_CATALOG.md` lists every basename/path/hash. This includes successful, intermediate and rejected variants. `campaign-files.json` also indexes Python implementations in retained payload snapshots, generators, diagnostics, tests, scripts and external Benchmarking source material. The five handcrafted runtime helper Python files under `runtime/work/astra500/` are separately preserved. Generated TorchInductor cache modules are compiler products, not hand-authored experimental implementations.

| Family / all retained variants | Disposition and evidence |
| --- | --- |
| `astra500-structural-native/astra_ar4.cpp`, `_v2` through `_v9`, build script | Rejected custom TP4 exchange. Corrected peer reads/writes were slower than native communication; unsafe notification/polling variants stopped before endpoint use. `evidence/astra500/rr-cycle-before.md`. |
| `astra500-structural-hc/` base, v2 and v3 CPP/header/build scripts + generator | v3 selected; earlier layouts superseded. Native fused HC projection/gate/mix eventually improved small-width work. Same structural report/manifests. |
| `astra500-structural-moe/` base, v1/v2/v3 CPP/header variants | v2-equivalent current generic files selected. Initial occupancy loss and other layouts superseded. Preserve all explicit versions. |
| `astra500-cycle-hc/` base/v2 down CPP/header, generator | Rejected TP-sharded, K-partition/fused-activation and persistent HC alternatives: added communication or insufficient endpoint benefit/changed yield. `rr-ownership-before.md`. |
| `astra500-cycle-gdn/` CPP, prepared conv/recurrent headers, generator | Rejected GDN cross-operation fusion: passed checks but core and endpoint were slower; output trajectory changed. Same report. |
| `astra500-ownership-exec/`, `ownership-groups/`, `ownership-batch/` | Initial ownership consumer superseded by accepted v2 group remap/gather and compact batch path. FP32 partial exchange and whole-expert striping did not win; original invalid 256-budget development comparisons are explicitly superseded. `rr-beyond-before.md`. |
| `astra500-ownership-clock/` clock/trace/legacy trace + build | Diagnostic GPU timestamp/launch instrumentation only; absent from production dispatch. Keep authorship and source. |
| `astra500-beyond-gdn/` CPP/project header/build | Rejected GDN projection/convolution fusion. A 36-layer chain saved about 0.42 ms, but rounding changed trajectory/yield and useful coding fell 270.18 → 257.39 tok/s. `rr-consumers-before.md`, `beyond-experiments.json`; comparator repair evidence is retained separately. |
| `astra500-beyond-prefill/` current + v1 CPP and two direct GEMM headers | v2 implementation historically selected, later superseded in supported dispatch ranges; binary remains loaded. |
| `astra500-consumers-native/` CPP/header current, v1 and v2 | Cooperative expert producer/consumer branch rejected despite exact outputs: no useful complete-request gain. `rr-dominant-before.md`, `consumers-result.json`. Preserve comparator repair and stale-KV replay failure evidence. |
| `astra500-dominant-native/` W4 base/dual/profile/layout CPP and five headers, three build scripts | Layout mode 1 selected; alternative layouts/profile/dual implementations unselected. The final manifest explicitly inventories all variants; preserve it without treating every source as selected. |
| `prefill-native/` small + medium CPP/headers/build scripts | Small eight-row tile selected. Sixteen-row/medium choices and padded/full-prefill graphs did not justify adoption. Prefill final report in external Benchmarking directory. |
| `long-hc-native/`, `long-hc-selected/` | Selected native HC exact; other native policies and Triton tile choices slower. Preserve source variants and external `hc-v1.cpp`. |
| `long-prefill-native/`, `long-prefill-token/` and external `native-v1.cpp`, `native-v2.cpp` | Column-sharing gather selected. Initial equality-failing layout rejected; corrected token-order layout was exact but slower for the full expert operation. External long-prefill `rr-final.md`. |
| `integrated-native/` HC/expert CPP/headers/build script and `integrated-v1/` | **Excluded expanded 2048/1664-capacity prototype.** Passed checks and improved long TTFT but changed coding/sampled trajectories and lost useful coding rate. No expanded native/PLE payload in accepted image. Integrated final report. |

Other rejected implementations remain in Python/tests/payload contexts: width-four raw BF16 block collectives failed serial equivalence; draft-query capture did not win; smaller split schedules, alternative attention/HC/dense GEMMs and FP32/gather collectives were slower; full/segmented prefill graphs passed eventual exact state checks but regressed endpoint performance; budget 4096, cap 512, exclusive and extra-decode cadence trials had unfavorable serving tradeoffs. Raw failure/selection evidence is preserved under `evidence/`, `runtime/work/` and the named external Benchmarking directories. These are archive dispositions, not invitations to resume tuning.

## Upstream changes that replace historical patches

The per-path decisions in `evidence/source-selection.json` are authoritative historical research. v0.32 already supplies Qwen4Exp/Transformers registration, MRV2 anchor/proposal rows and logits, speculative accepted offsets, GDN actual-query-length and reserved-depth semantics, compact/ragged recurrence, accepted-state selection and PLE prefix recovery. Do not reapply the old MTP-specific metadata/partial-verification/scheduler fast paths simply because they existed in Disk 1.0.4. The remaining native conv address products still required the separate index64 build.

Intel's six v0.31 patch files are retained with original hashes. Their sym_int4/FP16, MTP, async TP4, EP-disabled and QSA assumptions are not selected wholesale for this dense AutoRound BF16 target. Intel `0008-vllm-gdn-compile.patch` fixes its stream-keyed eager-workspace path and is unnecessary for the selected upstream compilation-visible allocation path. The original Flash-Next drafter's **SGLang** capture patch is saved separately at `evidence/qwen38-dflash2-capture.patch`; its TP1/CP1 assertion does not implement the campaign's verified TP4 completed-HC-feature interface.

## Attribution and source-delivery retention

Keep Wulf's Disk/checkpoint/shim integration, Astra campaign implementation records, Intel/vLLM/kernel/TLA copyrights and the pinned incoai drafter/model-card attribution. Original campaign patches mostly contain plain diffs without author trailers; do not invent personal authors, contributor permission requirements or release co-author trailers. Retain original headers, original filenames, source revisions, original copyright/SPDX text and explicit modification provenance. SYCL-TLA-derived native code includes embedded Intel/BSD notices; other files carry Apache identifiers. Keep these per-file distinctions.

The published Disk records acknowledge KaloyanNaumov/Lumnus research/configuration influence and report no identifiable incorporated executable source from that contribution audit. Preserve that historical finding and ledger rather than converting it into new authorship. The Controller's MIT license does not relicense runtime, frontend or model material. Preserve all relevant old notices/source-delivery findings through the layout replacement; update exact accepted-version findings separately as described in `REPRODUCTION.md`.
