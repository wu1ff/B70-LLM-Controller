# Archived Flash-Next runtime recipe — snapshot before the Disk rewrite

Archived 2026-10-11. This file preserves the previous release audit and complete
Vanilla v0.26/v0.30 campaign, including superseded build/launch instructions,
measurements, image ledgers and provisional pack decisions. Use the current
[Disk runtime recipe](../RUNTIME_RECIPE.md) for reproduction and serving.

The text below is historical. “Current”, “final”, pending-release and open-issue
claims retain their original scope and dates. Workspace-relative evidence paths
and live endpoints are archival breadcrumbs; those local records are not
included in the public pack. The snapshot is preserved verbatim except that
relative links to the distributed patch sources gain `../` for this location.
An exact original-byte copy remains in the development rewrite evidence.

---

# Qwen3.8-Flash-Next on four B70s — current Disk runtime recipe

2026-10-11, America/Phoenix. Pack **1.0.3** is the qualified Disk release, retaining the tool-choice-none fix. The corrected RC is published as GHCR `:disk`; the [public release](https://github.com/wu1ff/B70-LLM-Controller/releases/tag/qwen3.8-flash-next-v1.0.3) distributes the final model-pack archive, cumulative source/build export and SHA256 checksum. This section defines the current runtime; the complete Vanilla campaign below is preserved as dated history. Historical “final”, “shipping”, pinned-slab, MBT256, graph/performance and pending-APC statements apply to their recorded stages, not current Disk.

## Current image and dependency identities

| Item | Identity |
| --- | --- |
| Qualified RC local name | `qwen38-flashnext:disk-rc` |
| Published GHCR tag | `ghcr.io/wu1ff/qwen38-flashnext-b70:disk` |
| Qualified RC Docker `.Id` / verified GHCR platform manifest | `sha256:b13fed6dd54051395f603409ada440c74afcbd5331810eee3acd62315d7787e3` |
| Verified immutable acquisition reference | `ghcr.io/wu1ff/qwen38-flashnext-b70@sha256:b13fed6dd54051395f603409ada440c74afcbd5331810eee3acd62315d7787e3` |
| Preserved predecessor local name / ID | `qwen38-flashnext:disk` / `sha256:fad9d4cfae29a4fc3abd70e7339e4f2784c3589c988f7ca3f6751ffb46806316` |
| Distinct RC OCI config digest | `sha256:2399885f6fbc59cdb602fe869350ddd80e085f62da25d55859d683b5e3450baf` |
| Official vendor amd64 dependency | `sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9` |
| Official vendor multi-platform index | `sha256:fc0e112afb64e3a06fe8daff34652435822a629412f38efce8f0f67a46636b8d` |
| Checkpoint revision | `devan-carlin/Qwen3.8-Flash-Next-W4A16@40b8f18df4d4a32cb6e687a51c78207e5e438522` |
| Native kernel base | `vllm-xpu-kernels 0.1.14.1`, commit `6d92b1bfbf32767ecda8e819613eb151e70030ad` |
| SYCL-TLA | `87f6850680a580654b9ea2c80dbc01aeb36ad231` |
| Compiler | Intel oneAPI DPC++ `2026.0.0-947`; all 15 exact packages locked |
| Framework | vendor Python3.12, vLLM0.30.0+xpu, Torch2.13.0+xpu, triton-xpu3.7.2, GCC13.3 |

The Controller's existing `digest` means expected Docker `.Id`; its optional `registry` means an immutable acquisition reference. This release sets `registry` to the actual verified immutable GHCR reference following push, anonymous clean pull and Controller acquisition. Launching uses the verified ID, never a mutable tag. On this containerd image store `.Id` equals the local platform manifest; it is distinct from the config digest. A classic store may report a config digest and fail the existing contract. The remote manifest/config bytes match the RC, and all 22 layers and the clean pulled ID were verified.

## Cumulative accepted implementation and native dependencies

The distributed [Disk source directory](../patches/disk/README.md) contains the final 20-file source/config overlay, complete native sources/headers/dispatchers, shim, dependency locks, upstream notices, historical diffs and the 3,118-entry accepted installed manifest. Source files are already patched. Historical Vanilla 001–007 alone cannot reproduce Disk; do not apply the latest optimization onto an incomplete parent or reapply diffs onto final sources.

The implementation retains Vanilla's XPU Qwen4Exp gate, dense-QSA serving configuration, deterministic wide HC down-GEMM correction, graph stream correction, speculative/GDN metadata and peer residency policy. Disk adds BF16 mapped-table transport and ownership, conditional Base scheduling, corrected GDN partial-verification and shared Mamba metadata, followed by all three accepted optimization rounds:

| Stage | Accepted behavior / local image identity |
| --- | --- |
| Consolidated Disk baseline | mmap/staging, scheduler, GDN/APC; `9277fa7cb868c5cf1209d555f78e3056df772c145d918ec4900f259e5d9e4fb1` |
| Round 1 intermediate | corrected eager draft prefill and guarded native MoE; `b1944496d759faa1377b830c7b5f1c7a70c7ab256d5bb7d8f5270f6d0c1245a1` |
| Round 1 final | accepted MBT1024, KV-only draft prefill, startup warmup; `0240ef70806268eeb73aa5aecc00a905e844e2c310a7c4e942f7c3fa46b7dc65` |
| Round 2 final | fused live metadata and compact device 1/32/64 split plan; `40f663a776e195e72f8df33a533690872a0a6a7e40317b60db3f3c1f0080873e` |
| Round 3 final | native four-way tiled attention reducer and CCL LL65536; `e6d5862d5db8753f436048b9e88ad818756ca883d7629653a664d1448a915b60` |
| Independent package reconstruction | same accepted installed payload, normal serving entrypoint; `fad9d4cfae29a4fc3abd70e7339e4f2784c3589c988f7ca3f6751ffb46806316` |

The native payload hashes are:

| Installed library | SHA256 |
| --- | --- |
| `libgdn_index64.so` | `e79c1108ceb865282f40cc9376999c140074ba0a3eec21b529927342e94de2dc` |
| `libmoe_remap_guard.so` | `55636e569464e19b2c8820ad9c122a406bb5af9f8870a98aa7c9882da02868bc` |
| `libb70_attn_reduce.so` | `ce5f80804d0aef575f4b888f83a87410e86879d34eeae49de09dd06a8d0743c5` |
| `libl0_peer_residency_shim.so` | `ae2c82f549d97393268cbd7b90dba7cf38fd00dbccae54f090b1a545e5613b3c` |

The corrected GDN library supersedes the historical index64-only binary. The MoE guard checks global and mapped local expert IDs before offset/map access and avoids negative excluded-row pointers; valid-row arithmetic and atomic ordering are retained. During initial source packaging an attention snapshot overwrote the guarded remapper with pristine source. The final portable package restores the guard last and verifies its six direct inputs in `native/moe-inputs.sha256`. Independent reproduction then matched all native bytes, ELF sections and SPIR-V modules. Do not substitute the earlier failed package source.

The native reducer is selected only for the qualified q4/single-logical-sequence/h6/kv1/d256/BF16/832-page shape and eligible contiguous strides. Unsupported shapes retain native fallback. Per-query 1/32/64 plans are generated fresh on device for each replay; the corrected compact list preserves short normalization. The reducer preserves per-coordinate accumulation and uses four 64-thread workgroups per head. Its ABI-compatible library is required when its gate selects it. Scratch is approximately 0.75MiB partials plus12KiB sums/maxima per eligible call/rank; this is not a whole-runtime residency estimate.

## Reproducible source build

Both Docker stages use the exact official vendor dependency, with no accepted, experimental or Vanilla parent. Vendor Torch/XPU/SYCL/Level Zero/XCCL/UCX libraries are inherited and locked in `manifests/vendor-python.json` and `vendor-dpkg.tsv`. Native source/base, compiler and SYCL-TLA inventories are retained beside them. The vendor runtime stack is not rebuilt.

Acquire the pinned official vendor image or use its trusted saved archive:

```bash
docker pull --platform linux/amd64 vllm/vllm-openai-xpu@sha256:fc0e112afb64e3a06fe8daff34652435822a629412f38efce8f0f67a46636b8d
docker image inspect --format '{{.Id}}' vllm/vllm-openai-xpu@sha256:fc0e112afb64e3a06fe8daff34652435822a629412f38efce8f0f67a46636b8d
```

Verify the expected vendor `.Id` above. Obtain the 15 exact Intel package filenames from the official pool recorded in `patches/disk/manifests/compiler-lock.json`, separately under their applicable terms. **Compiler binaries are excluded from this public source export**; the original local reproduction archive is not cleared for redistribution as a whole. The source-only wrapper takes an authorized cache and verifies every hash:

```bash
cd patches/disk
sha256sum -c SHA256SUMS
python3 build.py --compiler-dir /absolute/authorized/compiler-cache --tag qwen38-flashnext:disk-fresh-test --evidence /absolute/build-evidence
python3 verify.py qwen38-flashnext:disk-fresh-test --output /absolute/build-evidence/recheck.json
```

The wrapper refuses existing tags and never overwrites qualified Disk. It materializes compiler bytes in a disposable context and runs `DOCKER_BUILDKIT=0 docker build --pull=false --no-cache --network=none`. Extraction remains `/toolchain/oneapi` and native sources `/build`; native scripts preserve C++20/O3/NDEBUG/SYCL spir64 flags, extensions, dispatchers, link libraries and rpath. Linux amd64, Python3.11+, Docker's available legacy builder and adequate compiler RAM/storage are required. No GPU, model, host compiler, resolver or network is used inside the build. Compiler/source build tools do not enter the final image.

The original independent build and archive reimport matched **3,118/3,118** accepted installed entries, with zero missing/extra/nonmatching files. The inventory excludes bytecode and is not complete filesystem/layer/build-metadata identity. Rebuilt image IDs can differ. The source-only wrapper is checked without an unnecessary rebuild; its materialized Docker/native inputs must equal the accepted independent input manifest before a future build.

The source export retains prominent Wulf modification comments on 21 changed upstream files. The wrapper now preserves all 15 Python comments in the runtime overlay and reconstructs only native compiler inputs inside its disposable build context. Qualified source/payload locks remain historical anchors; `verify.py` requires the notice-bearing sources and unchanged native/config files. The smaller `patches/disk/build-notice-image.py` recipe builds only the notice layer over the qualified image, without native compilation. Its reproducible candidate is `qwen38-flashnext:disk-rc`, `sha256:b13fed6dd54051395f603409ada440c74afcbd5331810eee3acd62315d7787e3`. [Disk license/source delivery](../patches/disk/licenses/README.md) records the remaining delivery/recipient-term requirements. The corrected RC has now passed focused Base/MTP3 serving qualification and the authorized GHCR replacement; the original local parent remains preserved.

## Verified GHCR replacement and model-pack release

On 2026-10-11 the corrected linux/amd64 RC was pushed to the existing `ghcr.io/wu1ff/qwen38-flashnext-b70:disk` using `docker push --platform linux/amd64`. GHCR returned manifest `sha256:b13fed6dd54051395f603409ada440c74afcbd5331810eee3acd62315d7787e3`. Direct anonymous reads of tag and immutable manifest matched the local manifest byte-for-byte; config `sha256:2399885f6fbc59cdb602fe869350ddd80e085f62da25d55859d683b5e3450baf` also matched byte-for-byte. All 22 compressed layer descriptors were present with their exact sizes.

An empty, separate containerd image store and anonymous Docker client exercised the unchanged Controller `runtime.AcquireOwned` path with the declared tag and ID initially missing. It pulled `ghcr.io/wu1ff/qwen38-flashnext-b70@sha256:b13fed6dd54051395f603409ada440c74afcbd5331810eee3acd62315d7787e3`, returned `Downloaded`, verified Docker `.Id` `sha256:b13fed6dd54051395f603409ada440c74afcbd5331810eee3acd62315d7787e3` and recorded ownership; subsequent acquisition returned `Reused`. The pulled configuration, uncompressed layer identities and complete installed payload matched the RC (3,118/3,118 entries and all 15 required Python modification notices). No serving test or restart was needed for registry verification.

After all remote/pull/payload gates passed, only the three positively identified superseded Vanilla versions of this exact package were deleted: tagged index `85512b52…` (version 1302351140), its linux/amd64 manifest `761f2154…` (1302351115), and its attestation `cd591ee8…` (1302351099). Final GitHub package/version reads and anonymous registry tag reads show one current version `sha256:b13fed6dd54051395f603409ada440c74afcbd5331810eee3acd62315d7787e3` (1369918807), with the sole tag `disk`. The package was preserved. There is no second GHCR image for rollback.

`runtime.image` is `ghcr.io/wu1ff/qwen38-flashnext-b70:disk`, `runtime.digest` remains the qualified RC Docker ID, and `runtime.registry` is the verified immutable reference. Original local Vanilla, predecessor Disk, archives, source history and the stopped recovery container remain retained. The running Controller-managed RC MTP3 container retains its ID, PID, start time and restart count. Verification-daemon initialization briefly removed the host Docker bridge; it was restored without changing or restarting the serving container, and local/LAN health returned HTTP 200.

The final pack retains the verified runtime identity and all eight qualified launch configurations. Release archives use sorted file membership, regular mode0644, epoch timestamps, zero ownership and nameless gzip. The final archive is rebuilt from the public pack source, extracted and compared byte-for-byte, and checked by the actual Controller catalog/import and launch-rendering implementation. The release’s `SHA256SUMS` and public catalog identify those final bytes; earlier provisional checksums are superseded.

The complete accepted Disk sources, native build recipes, dependency locks and prepared notices ship under `patches/disk/`. Historical source-delivery and recipient-term investigation records are preserved as dated evidence; this pack release does not rewrite them or third-party licenses. No image rebuild, registry republication or additional inference campaign is part of model-pack publication.

## Disk BF16 PLE ownership and memory

`Qwen4ExpPLEDiskBF16Embedding.load_mmap/prepare` maps the original table as protected CPU file-backed storage, gathers only TP-owned rows into bounded pinned host staging and uploads before forward/graph replay. `ModelState.prepare_inputs` supplies fresh ngram IDs and buffers before FULL replay. No whole-table pinned slabs or device-resident PLE weights are allocated. MBT1024 uses 5MiB/rank,20MiB/TP4 staging. BF16 precision, checkpoint hashes and original table are retained; no FP8/INT8 conversion or contributor NVMe reader ships.

The map can populate a substantial reclaimable OS page cache. Staging size is not total host memory or a minimum-RAM guarantee. Runtime allocations, caches, checkpoint mapping and storage latency remain relevant. Idle Docker memory is not peak/endurance memory. Natural-cache qualification did not establish storage-cold or long-term memory behavior.

## Exact Base and MTP3 contracts

The authorities are distributed `patches/disk/profiles/base.json` and `mtp3.json`, retained byte-for-byte from the qualified package. The Controller maps their settings into existing runtime → model → mode → profile blocks; duplicate environment keys are rejected. The model declaration is identical to `manifests/external-model.json`, including all31 files/179841735233 bytes.

Shared: one image, TP4+EP, BF16 W4A16, four slots, MBT1024, contexts32768/65536/131072/262144, `B70_PLE_DISK_BF16=1`, XPU graphs, spawn, FLAT and affinity0,1,2,3, CCL temporary buffers, offline/HF/cache settings, original shim preload, read-only `/dev/dri/by-path` and checkpoint binds. Vision limit is image2/video0. Keep tool/reasoning parser pair `qwen3_xml`/`qwen3`, automatic tool choice, graph metrics and `--exclude-tools-when-tool-choice-none`. APC and Mamba ALIGN/832-page behavior remain inherited defaults. There are no research mounts or developer-server flags.

| Setting | Base | MTP3 |
| --- | --- | --- |
| Utilization | 0.85 | 0.91 |
| Speculation | None | method=mtp,3 tokens,use_local_argmax_reduction=true |
| Captures, FULL_AND_PIECEWISE | [1,2,4,8,256,512,1024] | [1,2,4,8,64,128,256,512,832,1024] |
| Conditional scheduler | `B70_CONDITIONAL_PREFILL_TOKEN_CAP=256` | Absent |
| Accepted MTP gates | Absent | `B70_MTP_SKIP_PREFILL_TAIL`, `B70_PLE_MTP_PREFILL_1024`, `B70_MTP_DRAFT_PREFILL_EAGER`, `B70_MTP_PREFILL_KV_ONLY`, `B70_XPU_POST_CAPTURE_WARMUP`, `B70_XPU_SPEC_SPLIT_PLAN`, `B70_XPU_SPEC_META_FUSED`, `B70_XPU_TILED_REDUCER` all1 |
| CCL LL threshold override | Absent | `CCL_SYCL_ALLREDUCE_LL_THRESHOLD=65536` |

Base's scheduler lowers the aggregate budget to256 when any running request reaches decode; otherwise it remains1024. The policy is disabled for speculation. MTP3 uses the fixed1024 policy. Context selection changes only `--max-model-len`; lifecycle name/labels/access/port are Controller-owned. The existing development validator checks all eight exact environments/semantic flags/mounts and can consume actual Controller-rendered Docker arguments.

## Accepted qualification, tuning and limits

Final reconstructed-image standalone qualification covered both normal-entrypoint modes at native context: health/stability/zero restarts, BF16 Disk PLE and shim, vision/tools/reasoning/none, exact mode-specific APC/ALIGN boundaries, mixed four-slot serving with no preemption, finite sampled logprobs, natural/tool/length termination, native262112+32 cold/APC with >260000 repeat hits, post-long short coding and first-token fingerprints, disconnect/reuse, and positive depth3 acceptance/native reducer mapping in MTP3. Readiness was274.46s Base/292.23s MTP3, n=1 each. These broad receipts describe the preserved predecessor; the corrected RC passed focused Controller Base/MTP3 qualification with readiness265.3s/294.4s, exact native mappings, deterministic completion, MTP3 streaming/stopping and depth3 acceptance. The comment-only equivalence retains the earlier broad qualification bounds.

The README summarizes retained final-image n=1 natural coding rates (Base45.48–50.24tok/s; MTP3108.77–122.21tok/s across64K/160K cold/reused APC). Different outputs and small samples limit interpretation. Accepted combined GPU tuning preserved outputs in matched fixtures and improved warm generation about1.8–2.9%; repeat/trajectory caveats remain. The original long Three.js sessions are dated Round1/2 evidence, not final-image benchmarks. Active tuning concluded on diminishing measured return; this integration neither tunes nor repeats that campaign.

Keep these concrete limits: synthetic startup-audit nonfinite cause remains unresolved in historical evidence; broader numerical/reduction-order equivalence is bounded by differential tests; natural model outputs can fail tasks; cache/first-use and storage-cold/endurance behavior vary; four-slot functional checks do not prove throughput/fairness; native262112+32 does not prove sustained ceiling-length generation. Lower contexts are exact render selections, without eight independent runtime qualification campaigns.

Local qualification authorities in the development workspace are `runtime/rr-final-disk-qualification-20261010/REPORT.md`, `runtime/disk-package/evidence/native-moe-resolution/`, `DISK_MTP3_MODEL_PACK_INTEGRATION.md` and the accepted Round1/2/3 reports referenced there. Current integration evidence is consolidated under `runtime/rr-disk-modelpack-integration-20261011/`; `rr.md` reports actual managed lifecycle and final endpoint without duplicating raw receipts here.

## Archive recovery and release boundary

Original qualified package artifacts, retained in `runtime/disk-package/`:

| Archive | Bytes / SHA256 |
| --- | --- |
| `qwen38-flashnext-disk.tar.zst` | 4306594130 / `4c4c9c177ded2cd09d49013aea83f821e6ba1dd1b371fec3143f9a4242aea2e6` |
| `qwen38-flashnext-disk-source.tar.gz` | 602201848 / `e2bfcf65f30590039871228b9bc02803de0dfc835ca49fac3cde4d89649a738a` |

The image archive carries alias `qwen38-flashnext:disk-reproduced-20261010`. Verify its hash, load with `zstd -dc qwen38-flashnext-disk.tar.zst | docker load`, inspect the imported ID and assign `qwen38-flashnext:disk` only if absent and the expected ID matches. Preserve existing aliases and Vanilla. The original source archive contains compiler binaries and is retained local build evidence; publish the prepared source-only model-pack archive instead unless package redistribution rights are established separately.

For managed recovery, stop the active runtime through Runtime → Stop, then select the desired installed pack/mode and Start. Manual Vanilla rollback uses retained1.0.2 source pack and image `sha256:85512b52c09fa660a2e7fe441417129e7c47fac727fd85f66ccea6b65e0a9122`; import locally and choose its version-labelled target. Updates can retire older packs/unshared owned images, so save both before promotion. There is no automatic rollback or catalog downgrade.

If managed Disk startup fails, retain logs/inspect/readiness evidence, stop the failed managed container, verify endpoint/GPU availability and use qualified `patches/disk/launch.py mtp3 --model-dir /absolute/checkpoint --port 8000 --execute` to restore packaged standalone serving where possible. This starts the normal qualified entrypoint on the immutable local image alias. Never substitute an experimental image silently.

Provenance follows the completed contributor audit: no identifiable KaloyanNaumov/Lumnus executable source incorporated; historical research/configuration influence remains acknowledged. Preserve original messages/trailers as records without assigning them to new commits. Wulf owns the shim; no external permission blocker exists. The actual distributed vLLM/kernel/SYCL-TLA/vendor material keeps its notices; compiler binaries are excluded. The final public release and clean Controller acquisition record close [issue#9](https://github.com/wu1ff/B70-LLM-Controller/issues/9). Its historical discussion and individual PR#4–#8 dispositions remain preserved. The historical Vanilla1.0.2 source/release is retained; its retired GHCR image is unavailable, so recovery requires a saved local image.

---

# Historical Vanilla development and pack releases (v0.26 through provisional tool-choice fix)

Everything below is retained campaign history. Its image identities, whole-table pinned PLE, graph policies, benchmarks, current/final labels, validators and pending investigations describe their original dates. The current Disk implementation and supported contract above supersede those operational instructions.

# Qwen3.8-Flash-Next (Qwen4Exp) on Intel Arc Pro B70 — the v0.30 runtime recipe

This is the technical history of the vLLM 0.30 runtime that serves Qwen3.8-Flash-Next
W4A16 on 4× Intel Arc Pro B70 here: where the runtime came from, what was broken or
missing at each step, what changed, why, and what each change produced — in the order
the changes actually stack into the promoted Base authority, followed by the
post-freeze MTP speculative-decoding campaign that builds on it.

It is written for someone who wants to understand this runtime or investigate the same
source areas, so it carries breadcrumbs (files, functions, kernels, thresholds, workspace
evidence). It is not an SHA-by-SHA rebuild tutorial and not a dependency-install guide;
the campaign journal receipts (`docs/*.md`) and build manifests (`build/*/manifest.json`)
are the authority for any single number quoted here. No benchmark was rerun to write
this document — every number below is retained evidence.

Stage categories are labeled because they have different shelf lives: foundation/rebase,
correctness, feature support, performance tuning, reliability/containment, launch-contract
change. Everything except the graph policy, the capture list, and the PLE-table
environment variable is baked into the promoted image.

## Where this runtime came from: the v0.26 closure

The predecessor campaign served the same checkpoint on a heavily-modified vLLM 0.26
stack. Its frozen Base authority was `local/qwen38-flashnext-b70:phase2a-rc2`
(sha256 `42c1a58ab031…`): TP4+EP, Inductor piecewise compile + XPU-graph capture at
sizes 1..64, **p512 decode 45.30 tok/s median** (≈ +590% over its eager lane). That
number — 45.3020 / 45.4162 / 45.1523 tok/s at p512/p4096/p8192 — remains the retained
v0.26 comparison authority and is quoted as such below. Detail archaeology lives in
`../qwen3.8-flash-next/v026-reference/` (INDEX + `retired/V026-CAMPAIGN-SUMMARY.md`);
the facts that drove the rebase decision:

- **MTP could not be made stable.** A correctness candidate existed (70.2% acceptance,
  PLE spec path proven offline 452/452 bitwise), but a terminal single-rank compute
  stall reproduced in *every* execution posture — compile-on/graph-on, either off,
  and fully eager. The PTI victim-rank trace (§41) showed the stuck kernel was
  **submitted, started, and never retired**: the victim's own `oneccl_allreduce_pcie`
  **RingTransmit** instance of the previous all-reduce. Verdict: engine-level
  (oneCCL/XPU compute-queue) defect, not addressable inside the v0.26 tree.
- **APC was understood but blocked.** The root cause (PLE rolling state living outside
  mamba save/migrate/restore) was reconstructed and a fix was built and offline-exact
  (36/36), but qualification was blocked by an accelerated-prefill corruption: at
  btk-1024 with compile+graphs, prompts collapsed to `!!!!`, while the fully-eager
  discriminator ran 6/6 clean — corruption tied to the compiled/graph accelerated path,
  not to eager large chunks.
- **Base perf was at its floor.** The §21–§26 micro-optimization campaign (PLE loop,
  GDN island slimming, sampler, replay orchestration, drain) all rejected at ≈ +0.4%
  cap; C1 floor ≈ 49 tok/s.

Operator stop 2026-09-24: no further v0.26 fixes. The decision was a **clean rebase to
the official vLLM 0.30 XPU image** — pin the public stack, carry forward only what the
new stack provably lacks — rather than continue patching a private 0.26 tree whose
remaining defects were below the framework layer.

## The v0.30 authority stack (pinned, re-verified live)

Official image `vllm/vllm-openai-xpu` tag `v0.30.0`:

| Component | Pinned identity |
|---|---|
| Multi-arch index digest | `sha256:fc0e112afb64e3a06fe8daff34652435822a629412f38efce8f0f67a46636b8d` |
| linux/amd64 manifest = image id | `sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9` |
| vLLM | `0.30.0+xpu` |
| vllm-xpu-kernels | `0.1.14.1` (`_C`, `_xpu_C`, `_moe_C`, `_vllm_fa2_C`; `libgdn_attn_kernels_xe_2.so`, `libgrouped_gemm_xe_2.so`, `libmhc_kernels_xe_2.so`, `xpumem_allocator`) |
| PyTorch | `2.13.0+xpu` |
| Triton / triton-xpu | `3.7.2+xpu` |
| Transformers | `5.16.1` |
| compressed-tensors | `0.17.0` |
| oneCCL / intel-pti | `2022.0.0` / `0.17.0` |

Receipt: `evidence/image-inspect-v030-amd64.json`; extracted stock tree under
`image-source/vllm/`. Known gotchas that shaped everything after: the image WORKDIR
(`/workspace/vllm`) shadows site-packages for ad-hoc python; ENTRYPOINT is `vllm serve`;
and the model tree lives at `vllm/models/<arch>/{common,nvidia,amd}/` (no
`model_executor/models/qwen4_exp/`).

## The chain at a glance

1. Static gap audit on the pinned 0.30 tree (foundation)
2. XPU arch gate + dense-QSA config surgery (correctness / feature support)
3. PLE table: pinned-host XPU port with 2-slab split (feature support — the hard blocker)
4. B70 residency shim, rebuilt in-image (reliability/containment)
5. Core Base correctness qualification (correctness)
6. Vision — launch-contract change only (feature support)
7. Full-runtime retention + eager C1 baseline (correctness/control posture)
8. Breakable PIECEWISE XPU graphs (performance tuning, launch-only)
9. FULL graphs via the PLE capture-stream fix (correctness + performance, 1 file)
10. Prefill graph investigation: PIECEWISE-256 corruption, width brackets, btk-48 rejected (correctness)
11. M=256 corruption localization: segment → collectives exonerated → HC gate-mix (correctness)
12. HC down-GEMM accumulation race isolated (correctness)
13. Deterministic width-gated K-split (correctness fix, 2 files)
14. Final M=256 qualification + promotion as the frozen Base authority (freeze)

→ the promoted Base runtime. Base tuning is **closed**.

15. Native v0.30 MTP recon — feasibility, flags-only (foundation for the MTP lane)
16. MTP1 eager + accelerated correctness, then the performance LOSS receipt (correctness + decision)
17. MTP3 legality audit + first receipt — the depth win (feasibility + performance)
18. Step-cost profile — GDN speculative-metadata hotspot (profiling)
19. Independent Astra verification and correction of the profile (measurement audit)
20. Retained metadata optimizations 1+2 (performance, 4 source files)
21. Final MTP3 receipt + qualification as the `mtp3-c1` authority (freeze of the MTP lane)
22. Deep-prefill DEVICE_LOST investigation — APC/ALIGN exonerated, graph/workspace localization superseded by the root cause (investigation, closed)
23. GDN convolution state-index 64-bit correction (correctness fix — in the unified image)
24. Unified runtime promotion: one corrected image serves Base (speculation off) and MTP3 (depth 3) (freeze)

Model: `devan-carlin/Qwen3.8-Flash-Next-W4A16` @ revision
`40b8f18df4d4a32cb6e687a51c78207e5e438522` (local read-only snapshot; 17 shards,
77.4 GB safetensors; arch `Qwen4ExpForConditionalGeneration`: 48 layers = 36 GDN
linear-attention + 12 full-attention, GQA 24q/2kv head_dim 256, MoE 512 experts top-10
with EP, PLE at layer 1 (0-based), hc_count 4, vocab 248,320, native max position
262,144; served here under a 32K contract). A revision mismatch is a stop condition.

---

## Static gap audit — what 0.30 already shipped (foundation)

**What I did.** Before building anything, a file:line audit of the extracted stock tree
against the checkpoint (`RECON.md`, first pass 2026-09-23, independently re-verified
2026-09-24 against the live pinned image). Upstream issue #54595's meta-claim — "the
XPU kernels were already there" — was verified and extended: on v0.30 the compute this
model needs is present *and registered first-class XPU upstream*:

- **Qwen4Exp model + config, registered natively** (`vllm/models/qwen4_exp/`, registry
  and `transformers_utils/config.py`) — the entire 0.26-phase0 donor port is obsolete.
- **W4A16 dense linears** — `XPUwNa16LinearKernel` registered for XPU; our pack-quantized
  int4 group-128 format matches exactly (`weight_packed` + `weight_scale`).
- **W4A16 MoE** — `XPUExpertsWNA16` → `xpu_fused_moe(is_int4=True)` →
  `libgrouped_gemm_xe_2.so`, with EP `expert_map` handling.
- **GDN linear attention** — `gdn_attention_core_xpu` → `torch.ops._xpu_C.gdn_attention`
  → `libgdn_attn_kernels_xe_2.so`.
- **Flash Attention XPU** — `_vllm_fa2_C` varlen path, preferred by the XPU platform.
- **TP4 + EP** — served this exact checkpoint at v0.26; same machinery upstream.

What remained was small and enumerable:

1. **XPU arch gate** — `qwen4_exp/__init__.py:30` raised `NotImplementedError` on XPU;
   the nvidia backend behind the `else` branch is what XPU needs.
2. **PLE table storage** — the 102.4 GB table cannot be device-resident (≈26 GB/rank →
   OOM); the stock pinned-host class is CUDA-flavored and CUDA-gated. The one hard blocker.
3. **Dense-QSA config surgery** — the checkpoint ships `indexer_n_heads: 4` in config
   but **zero indexer weights**, so stock `use_qsa=True` would construct QSA with no
   weights. Dense `Qwen3NextAttention` is *required*, via config surgery, not merely
   acceptable. (`--hf-overrides` is unusable: it shallow-replaces `text_config`.)
4. **Vision retained, not removed** — 333 real `model.visual.*` ViT tensors ship in the
   checkpoint; the text-only lane needed a load-time drop, and vision later needed
   nothing but launch flags (see the vision stage).

This audit is why the final image is only **four overlaid vLLM files on the official
image** (arch gate, PLE host path, PLE capture fix, HC K-split) plus a shim — no pip
changes, no rebuilds of any package.

## The XPU gate and the dense-QSA serve directory (correctness / feature support)

**What I changed.** Two things, both minimal:

- `vllm/models/qwen4_exp/__init__.py`: the raise narrows from `(is_xpu() or is_tpu())`
  to `is_tpu()` — XPU falls through to the nvidia backend, whose compute is the
  first-class XPU machinery above. (~2 lines.)
- A **serve directory** `serve/flashnext-w4a16-serve/`: a patched `config.json` with the
  five `indexer_*` keys removed from `text_config` (QSA config validation is
  all-or-none, so all five had to go; `indexer_n_heads` absent ⇒ dense attention), plus
  relative symlinks to the untouched snapshot shards/tokenizer/PLE `.pt`. The original
  snapshot is read-only and untouched. `vision_config` and all visual tensors are
  preserved (vision deferred, not removed).

Boot-discovered along the way and folded into the PLE file: the
`from_quant_config` rejection of compressed-tensors PLE layers (the ignore list is
compressed-tensors regex, not Fp8Config), and the WORKDIR checkout-shadowing failure
(fix: `WORKDIR /opt/venv` + mirror overlay files into `/workspace/vllm`).

## The PLE table: 102.4 GB pinned-host on XPU (feature support)

**What was wrong.** The PLE (persistent latent embedding) n-gram table is a single
`[320,001,536, 160]` bf16 tensor — 102.4 GB — living only in the external asset
`ple_table_qwen4exp.pt` (128 `ngram_embedding.shard_*` mmap tensors in exactly the
row-split layout `load_weights` natively consumes). Stock XPU selection would make it
device-resident ≈26 GB/rank → OOM on 32 GB cards; the stock pinned-host class is
CUDA-only (`torch.cuda.Stream` etc.), and the Engram route that would select it is
rejected on non-CUDA.

**What I changed** (one file, `ngram_embedding.py`, class
`Qwen4ExpPLEPinnedHostEmbedding`, XPU-gated; CUDA paths untouched):

- **Selection**: on XPU, `PLE_TABLE_PATH` set → pinned-host class.
- **Loader**: `torch.load(PLE_TABLE_PATH, mmap=True, weights_only=True)` → the one mmap
  tensor sliced into the 128 `split_ngram_parts` virtual shards → each through the
  *native* `weight_loader(checkpoint_start=…)` path (TP-owned rows only). No repack.
- **The 2^34-byte pin ceiling** (boot-discovered): torch 2.13.0+xpu pins at most 2^34 B
  per allocation; above it `pin_memory=True` **silently** returns pageable memory and
  the UVA-view op segfaults (16 GiB pins, 17 GiB does not; explicit `.pin_memory()`
  above the ceiling segfaults). The 23.8 GiB/rank TP shard is therefore loaded into a
  transient pageable copy and re-materialized as **2 pinned slabs × 40,000,192 rows
  (12.8 GiB each)** per rank, each with a `get_xpu_view_from_cpu_tensor` UVA view;
  every slab asserts `is_pinned()` (fail-closed on silent fallback).
- **Slab-aware lookup kernel**: the Triton pinned-UVA lookup gained `slab_start/slab_end`
  store-mask args; out-of-slab rows are not stored and the output is pre-zeroed so
  non-owned rows are ETP all-reduce zero-contributions (stock semantics). The CUDA
  single-view path is unchanged.
- **Stream compat**: `torch.xpu` Stream/current_stream/stream + `record_stream`.

**What that gave me.** Zero device residency for the table — all 4 ranks log
`Loaded PLE table … 320,001,536/320,001,536 rows … storage=pinned-host slabs`;
weights stay 18.0–18.7 GiB/rank on device. Correctness evidence: slab-dispatch kernel
bitwise-exact 3/3 standalone (2 slabs, out-of-range ids, large vocab offsets) plus a
4.77 GiB-slab deep-offset (>2^31 element) exactness test before the qualifying boot;
and every later needle/vision battery exercises the real path. The 2^34 pin ceiling and
the WORKDIR shadowing are both upstream-reportable (deferred).

## The B70 residency shim (reliability/containment — hard requirement)

**What was wrong.** On multi-B70 serving the Level Zero runtime mirrors large
non-oneCCL device allocations into host memory. With the pinned PLE slabs ≈95 GB
legitimately resident, an unshimmed boot would additionally demand ≈195 GB+ of
unevictable host memory on a 247 GB machine — the peer mirror for ~82–84 GB **per
worker** of device allocations.

**What I changed.** Nothing new: the qualified shim source (`Desktop/shim/
l0_peer_residency_shim.c`, sha256 `f79e1bba…e1f10c` — the qualified tracer receipt)
was rebuilt **inside the pinned parent image** (gcc 13.3, `-O2 -fPIC -shared -Werror`)
and installed at `/opt/b70-residency-shim/`. The rebuild is deterministic: the result
(sha256 `ae2c82f5…53b3c`) is **byte-identical** to the qualified production `.so` hash
in `Desktop/shim/rrshim.md`. Activated by launch contract:
`LD_PRELOAD=/opt/b70-residency-shim/libl0_peer_residency_shim.so`.

**What that gave me.** Live counters in every qualified boot, e.g.
`alloc_calls=352 tracked=352 live=339 owner_allowed=352 peer_preserved=159
peer_suppressed=843 peer_bytes=84,393,590,784` — the same profile as the production
receipt, ~84 GB of peer residency suppressed per worker, host MemAvailable ≈95 GB at
idle. To be precise about what it is not: it does not offload anything, does not touch
oneCCL, and does not fix the driver — it contains the pathology at the loader boundary.
**This shim is a hard runtime requirement**, not an optional extra: without it the
qualified boot does not fit host memory.

## Core Base correctness (correctness)

The first qualified candidate (`base-c5`, single-COPY overlay of the two files above +
shim on the official image) ran the correctness-first eager posture (`--enforce-eager`,
graphs and compile off):

- Shorts battery (greedy, 8 prompts): **8/8 coherent, factually correct**.
- Long-prefill needle: exact `ZEPHYR-77` recall at 23,795 and 31,652 prompt tokens
  (the near-ceiling depth of the 32K contract).
- PLE pinned-host complete on 4/4 ranks; shim active; health 200 throughout.
- `sudo xpu-smi` 4/4 `Device State: normal`; full-log scan zero
  Traceback/SIGSEGV/DEVICE_LOST/UR_RESULT_ERROR.

Failed candidates c1–c4 (WORKDIR shadowing; compressed-tensors PLE rejection; the 2^34
pin ceiling; a stale device ref) are retained in `build/manifest.json` and
`docs/BASE_BRINGUP.md` §6. Evidence: `runtime/base-c5/evidence/`.

## Vision — a launch-contract change only (feature support — hard requirement)

**What was missing.** Nothing in the source. A pre-change audit of the installed vision
path (`docs/VISION_BRINGUP.md` §1) found: the 333 checkpoint visual tensors map 1:1 to
the vLLM module tree (checkpoint already stores stacked `attn.qkv`); Qwen4Exp is *not*
a multimodal prefix-LM here — image tokens are attended causally in the LM, so the
stock XPU Triton mm-prefix fallback never engages and no bidirectional mask is lost;
the ViT encoder defaults to FLASH_ATTN on XPU (the earlier stack's
`--mm-encoder-attn-backend TORCH_SDPA` oneDNN failure remains a do-not-use); mrope
stays interleaved once `--language-model-only` is dropped.

**What I changed.** Launch flags only, on the unchanged `base-c5` image: remove
`--language-model-only`, set `--limit-mm-per-prompt '{"image":2,"video":0}'`, remove
`--skip-mm-profiling`. Zero source, zero image changes.

**What that gave me.** 333 visual tensors loaded (+0.31 GiB/rank, "Model loading took
18.04 GiB"), ViT executed at init on stock FLASH_ATTN (head-72 varlen), and the
qualification: **7/7 deterministic image-understanding cases correct and materially
image-grounded** (object, color, left/right, above/below, counting, **OCR "ZEBRA-42"
exact**, image+reasoning); **8/8 text-regression completions clean** on the same server;
zero crash markers; 4/4 devices normal. Vision memory cost at this posture: the encoder
activation reserve is carved from the same 0.85 utilization, shrinking KV
497,316 → 416,047 tokens (eager posture) — the concurrency-relevant price of vision,
recorded, not tuned. **Vision is a hard runtime requirement of the final contract**
(image modality; video untested, `video:0`).

## The eager C1 baseline (correctness/control posture)

Before any acceleration, one server ran the full retention battery end-to-end (shorts
8/8, needle 3/3 exact at 17,878/23,795/31,652, vision 7/7, post-work text clean) and
the first C1 decode baseline on the v0.30 eager posture (temp 0.7 / top_p 0.8 / top_k
20 / seed 42 / ignore_eos / g512, 3 runs per depth, exact-count prompts):

| Workload | decode tok/s (median of 3) |
|---|---|
| p512  | **7.3059** |
| p4096 | **7.2319** |
| p8192 | **7.2665** |

Decode is flat vs prompt length (decode-bound at C1; prompt length moves TTFT only —
cold TTFTs ≈ 0.32 / 3.02 / 3.58 s). These are not production numbers; they are the
**correctness/control posture** every acceleration step below is measured against, and
the proof the later speedups are dispatch effects, not numerics changes. The same
receipt first recorded that v0.30's stock default enables the built-in prefix cache
(hit rates 51.7→54.8% in engine stats) — see the prefix-caching section for status.
Ledger: `runtime/vision-b1/evidence/c1-eager-baseline/`.

## Breakable PIECEWISE graphs — the first acceleration (performance tuning, launch-only)

**What was wrong.** The eager posture pays the full host dispatch chain per forward.
The stock v0.30 accelerated design for this architecture is *breakable XPU graphs
without torch.compile*: `Qwen4ExpForConditionalGeneration` is in
`DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES`, which auto-enables
`VLLM_USE_BREAKABLE_CUDAGRAPH=1` and forces compile mode NONE — the stateful ops
(GDN core, PLE short-conv/prefetch, attention) are `@eager_break_during_capture`
breakpoints that re-run eagerly between captured segment graphs.

**What I changed.** Launch contract only: remove `--enforce-eager` /
`TORCH_COMPILE_DISABLE`, set `VLLM_XPU_ENABLE_XPU_GRAPH=1`, and select
`-cc '{"cudagraph_mode":"PIECEWISE"}'` — on the unchanged image. Capture sizes stay the
stock-derived `[1,2,4,8]` (ceiling = `max_num_seqs × decode_query_len × 2` = 8).

**What that gave me.** PIECEWISE 4/4 captured; mode NONE confirmed (no Dynamo/Inductor
anywhere in the accelerated path); graph memory profiled inside the 0.85 budget
(KV 416,638 vs 416,047 eager, −0.2%); full battery exact (shorts 8/8, needle 3/3,
vision 7/7); and decode:

| Workload | PIECEWISE median | vs eager | vs v0.26 authority |
|---|---|---|---|
| p512  | **56.7443** | 7.76× | +25.2% (45.3020) |
| p4096 | **56.8566** | 7.86× | +25.2% (45.4162) |
| p8192 | **56.2983** | 7.75× | +24.8% (45.1523) |

TTFT unchanged (0.32/2.91/3.54 s) — the win is decode dispatch, exactly as designed.
Older, wider graph captures from v0.26 were **not** a lever here: the stock capture
ceiling is decode-oriented by construction, and widening is a different axis (below).
Ledger: `runtime/accel-b2/evidence/c1-accel-baseline/`.

## FULL graphs — the PLE capture-stream fix (correctness + performance, 1 file)

**What was wrong.** The stock `FULL_AND_PIECEWISE` boot died during FULL capture at
`ngram_embedding.py:658`: the PLE pinned-host prefetch **side-stream join**
(`current_stream().wait_stream(_prefetch_stream)`) is a cross-stream `Event.wait`, and
under FULL capture the eager-break decorator deliberately runs the op *inside* the L0
command-graph build, which rejects events outside the graph
(`RuntimeError: Event dependency from handler::depends_on does not correspond to a
node within the graph` — nondeterministic: 2 of 3 FULL descriptors captured first).
PIECEWISE was immune only because its breakpoints run outside capture.

**What I changed** (`full-c1`, one file, effective delta = 1 helper + 2 branches, XPU
class only): pre-fix probes proved `torch.xpu.is_current_stream_capturing()` is a
reliable per-call gate and that the pinned-UVA slab-lookup kernel itself captures and
replays bitwise-exact. So `start_prefetch` runs the lookup **on the capture stream**
when capturing (captured into the graph), and `_finalize_prefetch` skips the join. The
eager/PIECEWISE side-stream overlap is untouched — the branch fires only under FULL
capture.

**What that gave me.** FULL 3/3 + PIECEWISE 4/4 captured cleanly; the
nondeterministic accept/reject window is eliminated *by construction* (no cross-stream
join is ever recorded); bounded replay stress clean; battery exact; and decode:

| Workload | FULL median | vs PIECEWISE | vs eager |
|---|---|---|---|
| p512  | **57.9601** | +2.14% | 7.93× |
| p4096 | **57.9404** | +1.91% | 8.02× |
| p8192 | **57.5025** | +2.14% | 7.92× |

TTFT unchanged (0.3228/2.9859/3.5651 s) — prefill was still eager, which is the next
stage's subject. FULL tier covers uniform decode ≤4 (size 8 dropped above
`max_num_seqs`); fallback behavior outside capture unchanged. Image `full-c1`
(sha256 `7c8a35df…`) became the pre-K-split authority and **remains the retained
fallback**. Receipt: `docs/FULL_GRAPH_LANE.md`, ledger
`runtime/full-c1/evidence/c1-full-lane/`.

## Prefill graphs — the corruption and the width brackets (correctness)

Decode was graphed; prefill was not, **by capture-size arithmetic**: every chunked
prefill step is a 256-token batch (`max_num_batched_tokens=256`), dispatch keys on
exact padded `num_tokens`, and 256 ∉ [1..8] ⇒ every prefill chunk ran eager at
~111–187 ms/chunk, host-bound (the same per-forward host chain eager decode paid;
`docs/PREFILL_GRAPH_RECON.md`). The PIECEWISE tier is the designed mixed tier — one
launch flag would move chunks into it. So the obvious experiment ran, and failed
instructively:

- **PIECEWISE-256 (launch-only)**: capture 5/5 including 256, dispatch flipped exactly
  as predicted (116 exact-256 replays + padded sub-256 tails) — and **every prefill
  that traversed the 256-entry graph beyond trivial depth destroyed model state**:
  needle 3/3 depths and vision 7/7 all collapsed to a `duct` token loop, silently
  (health 200, 4/4 devices normal, no NaN). The prefix-cache-hit re-run reproduced the
  identical loop ⇒ the corruption is KV state written *during* graph-replayed prefill;
  decode graphs and every ≤8 dispatch stayed exact. **Rejected immediately**
  (`docs/PREFILL_256_TEST.md`, evidence `runtime/prefill-c1/`).
- A static recon narrowed the fault to the frozen per-size segment graphs themselves:
  everything Python-visible is fresh-by-construction at replay and byte-exact at
  width ≤ 8; the only width-specific component is the captured kernel set itself
  (`docs/PREFILL_REPLAY_CORRUPTION_RECON.md`).
- **Width-64 discriminator**: exact `64→64` replay clean (needle PASS through 19
  align-clip replays) — but **padded-to-64 replay corrupts**, with a silent-onset
  poisoning law: the first padded dispatch looks clean and plants persistent state;
  every later padded dispatch corrupts, including re-dispatch of a previously-clean
  prompt. Two corruption axes established: padding boundary ∈ (8,64], exact-width
  boundary ∈ (64,256] (`docs/PREFILL_WIDTH64_TEST.md`).
- **Width-32 and width-48 brackets**: every padded-32 and padded-48 dispatch class
  clean, deterministic, no onset, no poisoning; exact re-dispatch byte-identical after
  padding. Padding interval tightened to **(48,64]**; exact interval stays **(64,256]**
  (`docs/PREFILL_WIDTH32_TEST.md`, `docs/PREFILL_WIDTH48_TEST.md`).
- **btk-48 (chunk at the largest proven-safe replay width)**: legal, captured 5/5,
  correctness fully exact (needle 3/3 at 31,652 through ~660 chunked 48-width steps,
  vision 7/7), **zero eager prefill dispatches all session** — and TTFT got materially
  **worse**: +42.5% / +25.3% / +39.7% at p512/p4096/p8192 (decode flat, noise). The
  mechanism is compute-bound the wrong way: 48-token chunks lose more MoE/attention
  efficiency than replay saves in dispatch — 0.608 vs 0.435 ms/token; a replay width
  must approach ~192+ to break even, and the proven envelope makes that unreachable by
  bisection. **Rejected; the safe-width lever closed** (`docs/SAFE_WIDTH_PREFILL_TEST.md`,
  evidence `runtime/btk48-c1/`).

Conclusion that redirected the campaign: efficient TTFT required **fixing M=256 replay
correctness itself**, not shrinking the chunk width. (Older wider captures were never a
C1 lever: the v0.26 corruption lived at compiled chunks >256, and v0.30 mode NONE has no
compiled ranges at all — the corruption here turned out to live somewhere else entirely,
next section.)

## M=256 corruption localization — from segment to kernel family (correctness)

The causal narrowing, each step on the untouched `full-c1` authority with disposable
instrumented sibling lanes:

1. **Isolated primitives looked clean.** Every production per-rank primitive — W4A16
   dense GEMM, BF16/oneDNN GEMM, W4A16 grouped MoE, router, HC Triton norms, glue,
   mixed-width graphs sharing one pool — replayed **bitwise-exact at M=256**. (One
   genuine width-gated defect surfaced: `int4_gemm_w4a16` races at K ≥ 6144 — but no
   TP4 per-rank shape reaches it; TP1/TP2 geometries only.) So the corruption lived
   above single primitives (`docs/M256_REPLAY_ISOLATION.md`).
2. **First real divergence: layer-0 post-GDN segment.** Boundary fingerprinting of all
   51 eager-break boundaries localized the first divergent tensor to the
   `hidden_states` stream entering PLE `_short_conv`, written by the segment between
   the L0 GDN core and PLE finalize — a few bf16 ULPs on replay #1, chunk 1, exact-256,
   all four ranks, amplifying monotonically to model-output rel 1.8e-3
   (`docs/M256_SEGMENT_DIFFERENTIAL.md`).
3. **Collectives exonerated.** In-segment probes around both layer-0 all-reduces
   (GDN `out_proj` AR, MoE AR) proved the first collective is bitwise-exact on both
   sides and the second only propagates already-diverged inputs — **no equal-input/
   different-output event anywhere**. The first execution-order divergence moved to
   the **HC gate-mix output** (`mlp_hyper_connection.combine_and_mix` → `block_input`),
   and the eager lane itself was shown to vary boot-to-boot in exactly that chain
   (`docs/M256_COLLECTIVE_DIFFERENTIAL.md`).
4. **Layer-0 eager break moved the divergence to layer 1.** Breaking only the layer-0
   HC chain out of the graph restored layer 0 bitwise — and the first divergence
   relocated to the layer-1 `attn_hyper_connection.mix` gate-mix, same ULP signature.
   The racy behavior is therefore a property of the **HC gate-mix kernel family
   executing at M=256**, not of any one layer; 96 instances all execute it
   (`docs/M256_HC_BREAK_TEST.md`).

## The HC down-GEMM accumulation race (correctness — the root cause)

Offline incremental isolation of the gate-mix chain pinned it to one stage
(`docs/HC_GATE_MIX_ISOLATION.md`):

- **The unit**: the merged skinny BF16 down GEMM of every hyper-connection
  `GatedResidual` — `input_mix_weight_down_block_inject`, weight `[336, 10240]`
  (320 lora + 4 block-inject + 12 zero pad), i.e. **K=10240 → N=336** — executed as
  plain oneDNN `F.linear` on XPU (HC projections are dense BF16 and **rank-replicated**,
  so K=10240 is never TP-split: the shape is production-reachable in **every HC
  instance**, every layer, all 96 merged instances plus the final mixer's `[320,10240]`).
- **The behavior**: value-identical inputs produce outputs differing by sparse 1-ULP
  bf16 flips — 62–63 of 64 runs differ at M=256, and over a 512-run stress the stock
  GEMM emits **~395–508 truly distinct outputs** (essentially every execution re-samples
  the accumulation draw). **M=64 is clean; M=128 is already racy** (a correction to the
  initial expectation of a 128 clean control). The race survives device syncs, host
  sleeps, interleaving, and pinned output buffers ⇒ **intra-kernel
  scheduling-dependent accumulation order** (split-K/atomics class). Flips land only in
  the lora columns 0–319 — matching the serving fingerprint where `block_input`
  diverged while `injection` stayed bitwise-equal. **Eager itself races** — this was
  never simply a graph bug; graph replay just re-samples the same distribution, which
  is why replay-vs-eager diverged at M=256 while M≤64 replay was bitwise-clean, and
  why cross-boot eager variation existed all along.
- **The layout arm was worse**: the N-contiguous weight form (`W.t().contiguous().t()`,
  the upstream "ab-weights matmul is not run-to-run bitwise reproducible" path behind
  `VLLM_XPU_FORCE_N_CONTIG_WEIGHT`) races at **every** width including M=64 where stock
  is clean — rejected; that env stays off, now with evidence
  (`docs/HC_DOWN_GEMM_FIX_RECON.md`).

This also retro-explains the width brackets: the K-threshold of the race is in
(7680, 10240] — the padding/exact intervals were the shadow of a per-execution draw
that only becomes visible where widths exceed 64.

## The deterministic width-gated K-split (correctness fix — in the promoted image)

**What I changed** (`hcsplit-c1`, two files on `full-c1`; XPU-only; CUDA untouched):

```python
# GatedResidual._down_gemm(xn) — XPU and row width > 64:
F.linear(xn[..., :5120], W1) + F.linear(xn[..., 5120:], W2)
```

- **Contiguous K=5120 halves prepared once at load time** (`hc_ksplit_prepare()` marker
  hooked into `UnquantizedLinearMethod.process_weights_after_loading`, XPU branch) —
  the first execution of any width *is* its graph capture, so lazy prep would allocate
  inside capture; the loader guarantees prep runs in the eager profile run ~50 s before
  capture (verified live by the engagement log).
- **Width > 64 gating**: widths ≤ 64 — all decode graphs and the M1 FULL tier — keep the
  stock single GEMM; M=64 is proven clean and decode numerics are untouched. Branch
  conditions are shape/attribute reads only, so capture bakes one arm per graph.
- **No FP32 partials**: two bf16 GEMMs and a bf16 add — the expected BF16 split-K
  semantics, mean ≈1 ULP vs FP64 reference (~4× stock's single-rounding), inside the
  absolute envelope the stock race itself spans run-to-run; no bias mechanism; 0
  nonfinites everywhere.
- Covers all 96 merged `[336,10240]` instances **plus the final mixer's `[320,10240]`**
  (the race is N-independent; the mixer runs at width 256 in every prefill chunk — its
  K-split arm was separately proven offline before serving: stock racy 63/64, K-split
  0/64 ×3 protocols, 0/512 deep, rep1==eager1 bitwise).

**Offline determinism record** (`docs/HC_DOWN_GEMM_FIX_RECON.md`,
`runtime/hcfix-c1/`): **0 flips in 64 runs at M=64/128/256** in eager fixed-address,
eager fresh-clone, and graph capture+replay (replay1 == eager1 bitwise), one unique
output fingerprint always, both GPUs, and **0 flips in the 512-run M=256 deep stress**
— against the stock control's 511–512/512. Cost: +4.3–4.7 µs (~+11%) per HC down GEMM
in graph mode — invisible at serving scale, because determinism is what makes the
256-prefill graph capturable at all.

## Final M=256 qualification and promotion (freeze)

Four fresh boots of the candidate (two eager-sized `[1,2,4,8]`, two replay-sized
`[1,2,4,8,256]`), retained colldiff instrumentation, token-id-exact diagnostic prompts:

- **Causal gate: 7/7 comparisons × 4 ranks × 50 probe slots bitwise-equal** —
  eager↔replay, cross-boot eager↔eager, cross-boot replay↔replay, and a replication —
  entry and model output bitwise-equal, 0 nonfinites. The predecessor's divergence at
  the L0 HC boundary is **gone and did not relocate**; stock's documented cross-boot
  eager variance is gone.
- **Dispatch proof** (`--cudagraph-metrics`): **1,974 exact-256 PIECEWISE dispatches +
  843 ragged padded-256 dispatches = 273,249 K-split executions per rank**
  (97 HC instances per dispatch) over the session; decode dispatched FULL at width ≤8
  (stock arm) 8,921×.
- **Replay stress**: 240 s bounded, **1,389 requests, 198 bursts of 4, zero bad
  outputs**, health 200 throughout.
- **Full battery**: shorts 8/8; needle **3/3 exact** at 17,878 / 23,795 / 31,652 prompt
  tokens (the exact battery that rejected PIECEWISE-256 before); vision **7/7**
  image-grounded incl. OCR "ZEBRA-42" byte-identical to the authority; degeneration
  scan 0 across boot+battery+stress+benchmark; `sudo xpu-smi` 4/4 normal.
- **Numerical sanity vs stock eager**: inputs bitwise-equal; first delta = the HC
  gate-mix draw (the expected BF16 reduction-order change); output rel_abs_sum 2.5e-3
  — inside the stock race's own envelope; 5/8 shorts and the OCR response
  byte-identical to the authority, forks are ordinary continuation choices.

Verdict PASS (`docs/HC_KSPLIT_SERVING_TEST.md`, evidence `runtime/hcsplit-c1/`);
promoted 2026-09-26 as the frozen Base authority (see the final runtime section).
**Base tuning is closed.**

## Final Base performance authority

C1, 3 unique-prompt runs per depth, temp 0.7 / top_p 0.8 / top_k 20 / seed 42 /
ignore_eos / completion 512; prompts byte-identical to the retained full-c1 authority
lane; ledger `runtime/hcsplit-c1/evidence/c1-lane/`.

**TTFT (median)**: p512 **0.1721 s** · p4096 **1.4884 s** · p8192 **1.9414 s**
(runs 0.1710/0.1721/0.1869 · 1.4884/1.4959/1.4838 · 1.9496/1.9414/1.9319).

**Prefill throughput** (prompt_tokens / TTFT): p512 ≈ **2,975 tok/s** (+88% vs
pre-K-split 1,586) · p4096 ≈ **2,752 tok/s** (+101% vs 1,372) · p8192 ≈ **4,220 tok/s**
(+84% vs 2,298).

**Decode (median)**: p512 **57.9245** · p4096 **57.8501** · p8192 **57.3798 tok/s**.

| Depth | decode vs v0.30 eager | vs pre-K-split FULL (`full-c1`) | vs v0.26 Flash-Next Base |
|---|---|---|---|
| p512  | 7.93× (7.3059) | −0.06% (57.9601) | +27.9% (45.3020) |
| p4096 | 8.00× (7.2319) | −0.16% (57.9404) | +27.4% (45.4162) |
| p8192 | 7.90× (7.2665) | −0.21% (57.5025) | +27.1% (45.1523) |

TTFT vs the pre-K-split authority: **−46.7% / −50.2% / −45.6%**. Do not conflate the
two effects: the K-split GEMM is ~+11% *slower* per HC down GEMM in isolation — but it
determinizes the width-256 forward, which is what makes the 256-token prefill graph
safe to capture; replayed 256-chunks are ~2× faster than the eager chunks the authority
had to run. Decode is neutral.

## Memory authority

- `--gpu-memory-utilization 0.85`, `--max-model-len 32768` (32K contract) — unchanged
  from first qualification; **do not change util** without a new campaign.
- Weights ≈ **18.7 GiB/rank** (18.04 pre-K-split): the contiguous K-halves cost
  ≈ **+0.65 GiB/rank** (97 modules × 2 halves), with the **original 10240-wide weights
  retained on device** for the ≤64 stock arm (1.5× for this weight class).
- **KV capacity 356,485 tokens** (was 414,982 pre-K-split: −14.1%; the halves dominate,
  plus the 256-graph pool ≈651 token-equivalents) = **~10.88×** the 32K context
  (was 12.66×). Available KV 4.78 GiB.
- PLE table: zero device residency (2 pinned slabs × 40,000,192 rows × 4 ranks, host).
- Vision's encoder activation reserve is included in the same 0.85 budget.

## Prefix caching status

Precise status: the built-in prefix cache is **ON by stock v0.30 default** in the
qualified Base (engine logs `enable_prefix_caching=True`; the launch contract never
touches it). **Real cache hits have occurred and behaved correctly** during testing —
needle depths sharing openings produced 36.9–51.6% hit-rate intervals under 48-chunking
with exact recall, and the PIECEWISE-256 prefix-hit re-run reproduced its state
deterministically. **APC has NOT received its own dedicated qualification campaign** —
no claim is made that APC is frozen or qualified beyond this evidence; the benchmark
lanes used run-unique prompts so no number above depends on it.

## Graph contract (final Base posture)

- `cudagraph_mode = FULL_AND_PIECEWISE`, capture sizes **`[1,2,4,8,256]`**,
  `--max-num-batched-tokens 256` (chunk = graph size), `max_num_seqs 4`.
- **torch.compile mode NONE** — the arch's stock breakable-graph design; no
  Dynamo/Inductor in the accelerated path.
- M1 (and ≤4-row uniform) decode dispatches **FULL**; M=256 prefill chunks (exact and
  padded) dispatch **PIECEWISE**; the **K-split makes wide HC execution deterministic**,
  which is what makes the 256 entry safe.
- PIECEWISE 5/5 + FULL 3/3 capture at boot; graph memory profiled inside the 0.85
  budget before KV sizing.
- The qualified contract includes `--cudagraph-metrics` (as tested; decode cost bounded
  ≤0.2%).

## Required functionality (hard requirements of this runtime)

- **Vision** (image modality; launch-contract posture above).
- **B70 residency shim** via `LD_PRELOAD` — without it the boot does not fit host memory.
- **PLE pinned-host table** via `PLE_TABLE_PATH` (2-slab UVA mechanism above).
- **TP4 + expert parallelism** (`--tensor-parallel-size 4 --enable-expert-parallel`).
- **W4A16** weights (stock XPU WnA16 kernels; int4 group-128 pack-quantized).
- **Dense-QSA fallback for this checkpoint** (the serve-dir config surgery; the
  checkpoint has no indexer weights).

**Concurrency remains undecided** (`max_num_seqs 4`, 10.88× 32K KV headroom) — not
removed, not tuned. **MTP is planned later, deliberately deferred until after Base
closure** (28 `mtp.*` tensors auto-dropped at load; the v0.26 wedge evidence is part of
why the rebase happened). Video is unqualified (`video:0`).

---

# Part II — the MTP campaign (after Base closure)

Everything from here runs **on top of the frozen Base authority** (image bytes
`8aaa7309…` unless a section says otherwise). The MTP campaign asked one question in
stages: can native v0.30 speculative decoding pay for itself on this stack — and if
not at depth 1, does depth 3 change the arithmetic, and what is the real overhead
made of? Stage categories stay labeled. The end state is a separate candidate image
(`mtp3-c1`); **Base remains the non-speculative authority and fallback throughout**.

## Native v0.30 MTP — feasibility first (foundation, flags-only)

**What I found** (`docs/MTP_RECON.md`, static audit of the pinned tree + checkpoint
safetensors headers, no boot): v0.30 ships a complete native Qwen4Exp MTP
implementation (`vllm/models/qwen4_exp/nvidia/mtp.py`, registry-resolved via
`Qwen4ExpMTP`), so the entire v0.26-phase0 bespoke donor port stays dead. The
checkpoint feeds it natively:

- **28 BF16 `mtp.*` tensors** (≈5.21 GiB total → ≈1.30 GiB/rank under TP4+EP4),
  interleaved across shards 00001–00016, covered by the quantization ignore regex
  `re:.*mtp\..*` → the drafter is unquantized BF16 by design.
- `mtp.layer_types = ["full_attention"]`, `mtp.hybrid = true` — **the drafter layer
  is full-attention only: no GDN, no PLE, no indexer in the draft path.**
- `mtp_num_hidden_layers = 1` (one reusable draft layer), shared embeddings/lm_head
  (`mtp_use_dedicated_embeddings = false`).
- Base's load-time `WeightsMapper("mtp." → None)` is a pure skip; nothing is damaged
  when speculative decoding is off.

**What that gave me.** The first candidate was **flags-only**: `+ --speculative-config
'{"method":"mtp","num_speculative_tokens":1}'` on the frozen image, eager posture
first. MTP1 eager (`docs/MTP1_EAGER_BRINGUP.md`, PASS): drafter loaded via a second
17-shard pass (7.5 s, page-cached) with zero missing/unexpected tensor warnings;
**shorts 8/8, needles 3/3 exact at 17,878/23,795/31,652, vision 7/7 including OCR
"ZEBRA-42"**; counters proved MTP active (drafts = draft tokens exactly at depth 1);
**APC stayed stock-ON with the Mamba cache auto-set to `'align'`** — the drafter-legal
mode — and the APC ALIGN / PLE / GDN speculative state machinery (save/migrate/
restore, commit/rollback on rejection) produced **zero spec-state errors** across the
session, including 39,104 real prefix-cache hits under repeat traffic. The MTP1
accelerated graph lane (`docs/MTP1_ACCELERATED_BRINGUP.md`, PASS) then re-entered
FULL_AND_PIECEWISE `[1,2,4,8,256]` with target FULL tiers `[2,4,8]`, draft-prefill
FULL `[2,4,8]`, and no draft-decode graphs (MTP1's `steps==1` early return), with the
full battery green and the 1-token invariant everywhere. The **v0.26 oneCCL
RingTransmit never-retired wedge did not reproduce** in any posture — eager, graph,
or burst — which is the direct payoff of the rebase decision.

## MTP1 — correct, then structurally unprofitable (correctness + decision)

MTP1 correctness was never in question (above). The performance receipt
(`docs/MTP1_PERFORMANCE.md`, C1 methodology byte-equal to the Base receipt) measured
the loss:

- Initial correctness-window acceptance ≈ **81%** (eager lane; greedy battery +
  burst window 2,134/2,633), healthy at every depth; benchmark-window acceptance
  88–93% under temp-0.7 sampling.
- Final performance: **47.47 / 48.73 / 47.16 tok/s median** at p512/p4096/p8192 vs
  Base 57.92 / 57.85 / 57.38 — **−15.8…−18.1% at every depth** with TTFT +4…+21%.
  Run distributions do not overlap Base's at any depth.
- Step cost ≈ **40 ms** = **2.32× a Base step** (17.3 ms) per (draft + 2-row verify)
  step. **Structural break-even failure**: break-even needs mean acceptance length
  ≥ 2.32, but depth 1 is hard-capped at 2.0 tokens/step (draft + bonus) — even at
  100% acceptance the lane sits ≈ −14%. This is not an acceptance-quality problem
  and no tuning can fix it at depth 1.
- Disposition: **MTP1 retained as functional** (the packaged dispatcher supports it;
  smoke-tested on every later image) **but not as the preferred performance lane.**

## MTP3 — legality and the first depth win (feasibility + performance)

The MTP1 receipt redirected the question to depth. A pinned-source legality audit
(`docs/MTP3_FEASIBILITY_PERFORMANCE.md` §1) proved **iterative reuse of the single
MTP layer is the designed native path**:

- The only depth gate in `config/speculative.py:1470-1488` is divisibility
  (`num_speculative_tokens % n_predict == 0` when deeper than the drafter); with
  `n_predict = 1` every depth passes. (The earlier recon's "depth >1 would be
  rejected" inference was wrong and is corrected there.)
- `MTPSpeculator.propose()` runs draft step 0 then loops the remaining steps through
  the *same* module — `layers[spec_step_idx % num_mtp_layers]` = layer 0 every step,
  feeding the multi-stream hidden back: classic EAGLE-style module reuse, native.
  **MTP3 = 3 draft iterations, target verify width 4** (3 drafts + 1 bonus row).
- Graph posture **derived natively** from the same capture list
  `[1,2,4,8,256]`: target FULL tiers round up to multiples of `decode_query_len=4`
  → **FULL [4,8]**, PIECEWISE keeps `[1,2,4,8,256]` including M=256; draft-prefill
  FULL `[4,8]`; draft-decode `FULL_DECODE_ONLY [1,2,4]` — the **first lane that
  captures draft-decode graphs** (MTP1 skipped them), via the Flash-Attention
  draft-decode metadata update at `dcp=1`.

Initial untuned receipt (same image, same C1 methodology, fresh prompts): medians
**57.16 / 64.10 / 78.63 tok/s** — −1.3% at p512 but **+10.8% at p4096 and +37.0% at
p8192**, the first speculative lane to beat Base at depth; initial step ≈ **43 ms**
(42.9–43.7, flat across depths). The economics flip because the two extra draft
forwards + two extra verify rows cost only +7–9% per step (2.47–2.52× Base step)
while the ceiling doubles to 4 tokens/step: break-even ≈ 2.5 meanlen sits *below*
the measured meanlen at p4096/p8192 (2.76/3.44), and acceptance now **grows** with
context depth (48→59→81% medians) — depth-3 drafting monetizes long-context
predictability. **Depth-3 economics proved viable**; not promoted at this point.

## Step-cost profile — the GDN metadata hotspot (profiling)

To find where the 43 ms went, a monkey-patch host/device profiler ran on disposable
overlay containers, image untouched (`docs/MTP_STEP_COST_PROFILE.md`, code in
`prof/`). Perturbation gate first: **XPU `Event.record` costs ~150 µs live** on this
driver (empty-queue record forces submission) — 8 events/step added +3.6%, so the
headline posture became walls-only (+0.74%). Findings:

- The MTP3 step is **host-bound**: attention-metadata build = **39.3 ms of the
  43.5 ms step**, of which GDN builders 37.9 ms — the per-layer speculative-decode
  branch of `GDNAttentionMetadataBuilder.build` (~0.6 ms × the layer count in tiny
  host-dispatched ops: masks, block-table gather, spec-state staging copies/fills),
  re-derived per layer for identical batch-level inputs. Base's non-spec branch of
  the same builder costs ~5× less per layer.
- Device floor ≈ **21.2 ms**/step (target verify FULL@4 17.5 + draft graphs 3.65),
  already overlapping the host section — eliminating the redundant rebuild would put
  the step at the device floor, ≈1.9× serving.

**The broad hotspot was real** — GDN speculative metadata preparation dominated the
step. Several of the profile's *attributions* were wrong, which is the next section.

## Independent Astra verification and correction (measurement audit)

An independent verification pass (`docs/MTP_OVERHEAD_ASTRA.md`, evidence
`runtime/astra/`) re-measured everything with monotonic timestamps, direct worker
cadence, and exact production harnesses, and **corrected the profile**:

- **Exactly 36 GDN builds per uniform target step, not 40–43.** There is no draft
  GDN — the MTP head is full attention. The cProfile hook had counted outside the
  filtered steady-step window (its 8,280 calls = 36 × 230 unfiltered steps).
- **The ~38 ms GDN host span was not 38 ms of exclusive host work.** The first
  builder costs ~19 ms and the remaining 35 total ~18.8 ms because two CPU-mask
  device-index operations perform blocking indexed H2D copies whose **first wait
  absorbs previously queued target/draft device work**. A standalone builder
  harness takes ~20 ms for all 36 builders.
- **The engine is asynchronous — a two-batch pipeline** (`step_with_batch_queue`,
  queue size 2, `AsyncScheduler`): the worker re-enters while previous device work
  is still queued (re-entry gap measured ~0.09 ms). The synchronous
  `EngineCore.step` mental model misses this; whole-request client walls cannot
  establish an exclusive engine/RPC gap.
- **GDN state indices and block tables differ across every group** (36 unique per
  step; example layer-0/1/2 state rows `[201,313,208,309]`, `[285,202,222,17]`,
  `[211,363,188,14]`) — so whole-metadata hoisting/aliasing across layers **would
  have been incorrect**. Common batch-level values can be shared only with proper
  per-group graph-buffer handling.
- Also corrected: the device-scalar `fill_` sync hypothesis was tested and rejected
  (the blocking waits live in `index → to → _to_copy → copy_`); the pinned runner's
  multi-group aligned-index hook is dormant on these builders; MTP1→MTP3's apparent
  GDN-width cost is mostly device waiting charged to the first builder.

This audit is what made the retained optimizations safe to keep small and
per-group-faithful instead of a blind metadata hoist.

## Retained optimization 1 — per-builder fused ALIGN gather + uniform staging (performance, in the MTP image)

**What changed** (`gdn_spec_metadata.py` + `gdn_attn.py`): the Mamba-ALIGN state
gather and the uniform speculative staging (masks, token indices, accepted counts,
query starts, state staging) are fused into **one kernel per builder**. Isolated
trace per builder: **11 kernels + 8 copies + 2 blocking indexed H2D waits → 1
kernel, 0 copies, 0 in-builder waits**. The **original persistent graph buffers are
preserved** — every destination the captured FULL graphs read is still written by
name, so no graph rewiring or buffer aliasing exists.

Eligibility is gated (real spec rows a prefix, uniform width, ALIGN mode, full-graph
capacity, the exact supported GDN builder) with exact/fallback correctness gates —
15 uniform cases, 16 strided/boundary/mixed/ragged cases, 12 per-builder graph
checks; Base/prefill/mixed/ragged/dtype/sleep configurations fall back to the stock
path unchanged.

**Result**: serving MTP3 step **43.519 → 22.562 ms** (p4096 worker cadence); MTP1
39.942 → 18.8 ms; Base inert (17.303 → 17.307 ms).

## Retained optimization 2 — all-compatible-group staging (performance, in the MTP image)

**What changed** (`gdn_spec_metadata_batched.py`): when every GDN group is eligible
in a step, **one grouped launch writes the individual existing destination
buffers** of all 36 builders via a constant pointer plan — still no graph-buffer
aliasing, still every used field and padded row rewritten each step. The plan is
builder-owned and **invalidated on changed pointers or layout** (a same-address
stride change defeats the old pointer-only key — regression-proven); non-int32
block tables and sleep-enabled configurations retain the per-builder route.

**Result**: the additional serving gain is small — **~0.064 ms** (matched-output
ABBA; isolated 36-builder staging ~4.12 → 0.36 ms, mostly hidden) — because by then
the runtime had become **device-bound**. Retained anyway: compatible, exact-gated,
and measured positive (packaged comparison independently confirms ~0.075 ms).

## Final bottleneck — the runtime is device/graph bound (measurement)

Cumulative image, p4096 steady state: **worker cadence 22.437 ms** against a
critical path of **target FULL@4 ≈ 17.399 ms + first draft ≈ 1.307 ms + fused
drafts 2+3 ≈ 2.303 ms = 21.009 ms** of graph replay, plus necessary eager
logits/sampling (~1.35 ms: logits GEMM 0.540, top-k/top-p 0.546). The final
metadata stage is **~0.49 ms host with zero individual GDN `build()` calls** on the
uniform path. Removing ~4 ms *more* metadata host work bought only ~0.06 ms of wall
— **host optimization is effectively exhausted; the runtime is device/graph bound**.
Further substantial wins require a separate target/draft graph/kernel effort (not
undertaken; no acceptance/TP/EP/precision/feature lever was used anywhere).

Measurement hygiene that matters for anyone re-reading the receipts: the full XPU
trace **perturbed later prefills (~13% slower despite matched cache hits)**, so the
performance authority is the fresh fully-uninstrumented boot below, never a traced
window.

## Final MTP3 performance receipt (fresh, no profiler)

Fresh server, `ASTRA_OVERLAY=0`, no observer, C1 streaming, the authority sampling
and timing convention (first-to-last content; TTFT separate), 512 completion
tokens, same v7/v8/v9 prompt bytes as the before-lane for acceptance control:

| Depth | tok/s (measured) | client interval | acceptance before → after |
|---|---:|---:|---|
| p512  | **111.26** | 22.114 ms | 48.33% → 48.33% |
| p4096 | **149.25** | 22.435 ms | 74.47% → 78.65% |
| p8192 | **153.74** | 22.789 ms | 78.00% → 83.79% |

- **p512 is the direct proof**: identical text, identical 209 drafts / 303 accepted
  (acceptance matched at 48.33%), interval halved 42.755 → 22.114 ms — approximately
  a **2× runtime throughput improvement attributable to the runtime alone**.
- Deep (p4096/p8192) acceptance **varies across boots and prompts even on stock
  with the same seed** — so not every observed deep gain is attributable to the
  runtime optimization. Estimated throughput at the *old* acceptance with the new
  measured step: **≈110.8 / 144.2 / 146.6 tok/s** (retained beside measured values
  in `runtime/astra/evidence/campaign-summary.json`). Even at fixed old acceptance
  the lane is far above Base (57.92/57.85/57.38) at depth.
- APC posture was matched between lanes (p512/p4096 zero hit tokens; each p8192
  request 2,496 hit tokens in both lanes).

## Upstream breadcrumbs (researched, not reproduced here)

Full research dump with raw diffs and API responses: `runtime/astra/research/`;
dispositions in `docs/MTP_OVERHEAD_ASTRA.md` "Upstream dragnet". The ones that
shaped decisions:

- **[#38020](https://github.com/vllm-project/vllm/pull/38020)** — the old ALIGN
  fusion PR (approved, stale, closed unmerged); its stride/bounds discussion seeded
  the ALIGN-gather fusion design.
- **[#58737](https://github.com/vllm-project/vllm/pull/58737)** — open upstream
  ALIGN fusion (seven gathers → one); kernel-level claims only. Adapted and tested
  here; its benefit is subsumed by the full staging fusion.
- **[#52297](https://github.com/vllm-project/vllm/pull/52297)** — GDN common
  precompute; its review threads (residual host-mask copies, per-group staging)
  independently corroborated where the real cost sits.
- **[#58762](https://github.com/vllm-project/vllm/pull/58762) / [#58763](https://github.com/vllm-project/vllm/pull/58763)** —
  MRV2 metadata reuse and uniform slicing; maintainers favor update_block_table
  reuse with consistently shared capture buffers. This runtime instead preserves
  every existing destination buffer.
- **[#53774](https://github.com/vllm-project/vllm/pull/53774)** — closed unmerged
  K3 revert documenting stale raw pointers from temporary capture allocations; the
  direct reason the grouped plan retains tensor owners and invalidates on
  pointer/layout change.
- **[#49451](https://github.com/vllm-project/vllm/pull/49451)** — merged revert of
  unconditional capture metadata after FlashAttention capture failures; capture
  policy preserved here.
- Intel/XPU history: [intel/llm-scaler #711](https://github.com/intel/llm-scaler/pull/711)
  and [vllm-xpu-kernels #600](https://github.com/vllm-project/vllm-xpu-kernels/pull/600)
  (speculative conv/null-state and ragged traversal — informed the eligibility
  gates; no binary kernel replacement needed), [intel/llm-scaler #728](https://github.com/intel/llm-scaler/issues/728)
  (reduced draft vocabulary would change draft support — rejected as
  outside exact-metadata semantics), plus #49736/#43565 (XPU GDN MTP support
  already represented in the pinned tree).

Blindly enabling upstream's new multi-group aligned-index hook would trip a
CUDA-only assertion on this pinned XPU tree; the fingerprints, blocking-copy trace,
graph-safe staging, and differential harness are upstreamable findings (nothing
posted externally).

## The MTP3 qualification campaign and promotion (freeze of the MTP lane)

One bounded campaign on the exact renamed bytes, no tuning, no profiling
overlay (`docs/MTP3_QUALIFICATION.md`, evidence `runtime/mtp3-c1/evidence/`,
promotion record `runtime/mtp3-c1/PROMOTION.json`):

- **Boot/posture**: health 200 at ~5:50; graph tiers exactly as derived
  (target FULL 2 + PW 5/5; draft-prefill FULL 2 + PW 5/5; draft-decode
  FULL 3); model 19.94 GiB/rank; KV 3.42–3.56 GiB/rank; PLE pinned-host 4/4;
  shim in 11 processes; mamba cache `align`; only the known-benign warnings.
- **Basic MTP gate**: exactly 3 draft tokens/proposal (9 drafts / 27 draft
  tokens sanity); FULL@4 dispatched ×17 and PIECEWISE-256 ×15 exact + 4 padded
  on a p4096 probe; counters advance and reconcile.
- **Correctness regression**: shorts 8/8; needles 3/3 **exact** at
  17,878/23,795/31,652 (completion heads byte-match the Base authority);
  vision 7/7 grounded incl. OCR "ZEBRA-42"; post-work text coherent;
  degeneration scan clean over all 18 responses; APC ALIGN active with
  **39,104 real hit tokens (49.8% of queried) and exact recall**; `sudo
  xpu-smi` 4/4 normal; zero error signatures.
- **Performance receipt** (C1, 3 unique runs/depth, fresh variants 13–15,
  authority sampling/timing, all asserts green):

| depth | runs tok/s | median | vs Base | TTFT med | acc med | meanlen | step |
|---|---|---:|---:|---:|---:|---:|---:|
| p512  | 148.4908/159.5740/142.6220 | **148.4908** | +156.3% | 0.1852 s | 76.99% | 3.31 | 22.29 ms |
| p4096 | 161.7311/152.5831/130.7389 | **152.5831** | +163.7% | 1.6009 s | 80.98% | 3.43 | 22.47 ms |
| p8192 | 142.5867/146.2887/122.8481 | **142.5867** | +148.4% | 2.4178 s | 74.47% | 3.23 | 22.68 ms |

  Worker cadence without profiling (meanlen ÷ tok/s) = 22.29–22.68 ms — the
  qualified optimized-image cadence, unchanged by the rename. Acceptance ran
  higher/lower than the astra receipt at different depths (77% vs 48% at
  p512; 74% vs 84% at p8192) — the documented cross-boot/prompt variance law;
  throughput tracks meanlen at the fixed step exactly.
- **Concurrency (inside the unchanged `max_num_seqs=4` contract)**:
  C1/C2/C4 × p512/p4096/p8192, 256-token requests, all complete, 3-token
  invariant everywhere, health 200. Aggregate tok/s: p512 118.6/141.3/122.6,
  p4096 97.9/88.2/71.4, p8192 59.3/66.3/58.6 (C1/C2/C4); acceptance
  55.9–80.1%; FULL@8 dispatched live under waves; peak KV 16.9% (= 4×8192+
  prompts of 200,326 exactly); no degeneration. Envelope qualification only.
- **Active stability (no idle soak)**: 724 s / 117 mixed cycles — greedy
  shorts, shared-prefix needle requests (17,878-token prompt, recall asserted
  **exact all 234 passes**), all 7 vision fixtures rotated, alternating
  C1/C2/C4 decode waves. **0 errors, 0 stalls, health 200 every cycle**,
  counters monotone (forward-progress guard never tripped); window acceptance
  63.5% / meanlen 2.906; APC under sustained shared-prefix traffic 4.13M hits;
  full-session invariant 76,713 draft tokens = 3 × 25,571 exactly; xpu-smi
  4/4 normal.
- **MTP1 regression smoke** (`SPEC_DEPTH=1`, second boot, same image): server
  up, **draft tokens == drafts (exactly 1/proposal)**, 3 shorts clean, the
  17,878-token needle exact, health 200. No MTP1 perf campaign (MTP3 is the
  preferred lane).

**Verdict PASS → `mtp3-c1` promoted to qualified MTP3 authority** (project-
internal). Frozen Base re-verified unchanged after both boots; nothing
published externally; b70ctl/model packs untouched. Qualification limits:
bounded campaign (no long-duration soak), single MTP3 boot, concurrency only
within the current contract, video unqualified.

## The MTP3 authority (neutral identity)

```text
local/qwen38-flashnext-v030:mtp3-c1
sha256:3bc0ee337a732b9341f1653264a0672b7661bcf37c926dc3cda404a1c02d5656
```

*(2026-09-26 supersession note: the `mtp3-c1` tag now aliases the unified
runtime authority `ad6d44d7…` — see the unified-runtime section below. This
digest `3bc0ee33…` records the pre-correction MTP3 authority bytes, which
remain retained by image ID.)*

Retag-only (bytes identical to the experimental build; manifest
`build/astra-cumulative/manifest.json` carries the retag record) of the cumulative
MTP image = Base authority + exactly **four overlaid vLLM files** (GDN entry point
`gdn_attn.py`, uniform staging `gdn_spec_metadata.py`, grouped staging
`gdn_spec_metadata_batched.py`, MRV2 attention dispatch `attn_utils.py`; complete
delta `build/astra-cumulative/source_delta.patch`, source-delta sha256
`a611f3fc…`). Launch: `scripts/launch-mtp3-c1.sh` (digest-gated; `SPEC_DEPTH`
selector, default 3; MTP1 smoke via `SPEC_DEPTH=1`).

Status: **qualified MTP3 authority** (all six gates PASS, 2026-09-26 — see the
qualification section above). Base is not replaced by this lane.

**Qualified serving posture since the capacity recovery (2026-09-26, same
bytes): util 0.91** via `scripts/launch-mtp3-util.sh` (`MTP_UTIL=0.91`) — KV
~307–308K tokens at the 32K contract (9.4× 32K, was 200,279 at util 0.85), or
the native-context posture `MAXLEN=262144` (KV 403,260 tokens, 1.54× 262K).
`scripts/launch-mtp3-c1.sh` (util 0.85) remains the frozen qualification
record and fallback. Campaign: the next section + `docs/MTP3_UTIL_CAPACITY.md`.

## The util-0.91 capacity recovery (KV/context reclaim, launch-only)

Capacity campaign, not tuning — one variable (`--gpu-memory-utilization`) on
the exact MTP3 authority bytes. Sweep 0.89/0.90/0.91, cheap runtime gate per
candidate (health, short+vision+p8192 needle, 3-drafts invariant, APC
shared-prefix, xpu-smi, error scan) — all three clean:

| util | KV tokens (32K mml) | device free steady |
|---|---:|---|
| 0.85 (old) | 200,279 (6.11×) | ~4.5 GiB |
| 0.89 | 271,765 (8.29×) | 3.6–4.1 GiB |
| 0.90 | 289,706 (8.84×) | 3.2–3.8 GiB |
| **0.91 (selected)** | 306,903 / 308,158 (9.37–9.40×) | 2.98–3.5 GiB |

0.92+ rejected on receipts: +0.01 util ≈ +18K tokens but −0.3 GiB
out-of-budget headroom (transient absorption beyond the 1.18 GiB profiled
activation); 0.91's rank-0 floor of ~3.0 GiB is comfortable, ~2.7 GiB is not.
Selected 0.91 then passed the full battery on a clean relaunch (shorts 8/8,
needles 3/3 exact at 17,878/23,795/31,652, vision 7/7 incl. OCR, APC 0.9308
shared-prefix hits, degeneration clean over 19 responses, 3× draft invariant,
4/4 devices, zero error signatures) and the perf sanity on fresh variant 16:
**step 21.90–22.73 ms and TTFT 0.186/1.607/2.421 s — unchanged vs the
retained authority (22.29–22.68 ms; 0.185/1.601/2.418 s)**; tok/s 167.9 /
155.9 / 153.2 rode higher acceptance (82–89% vs 74–81%, documented
cross-boot law), not a step-time change. Context proof (separate launch,
`MAXLEN=262144`, same bytes/util): engine KV **403,260 tokens, 1.54× native**;
deterministic needles exact at **65,408 / 130,980 / 262,027** prompt tokens
(99.95% of native ceiling, one request, greedy recall exact; peak engine KV
usage 35.5%; 4/4 devices normal). Observed receipt fact: engine KV token
accounting is max-model-len dependent (same 5.24 GiB pool reports ~307K
tokens at 32K vs 403K at 262K) — reported as-is. Memory unchanged elsewhere:
model 19.94 GiB/rank, graphs 0.53 GiB, activation 1.18 GiB. Full sweep +
ledgers: `docs/MTP3_UTIL_CAPACITY.md`, `runtime/mtp3-u89|u90|u91|u91ctx/`.

## The deep-prefill DEVICE_LOST campaign — root cause: GDN convolution 32-bit state offsets (correctness fix — in the unified image)

The investigation journal is `docs/DEEP_PREFILL_DEVICE_LOST.md` →
`DEEP_PREFILL_APC_DISCRIMINATOR.md` → `ALIGN_STATE_BLOCK_LIFETIME.md` →
`ALIGN_OBS_INSTRUMENTED_RUN.md` → `PW_ALLOC_LOCALIZATION_RUN.md` → the full
report `docs/GDN_CONV_INDEX64_FIX.md`; receipts under `runtime/pw-rootcause/`
and `runtime/mtp3-gdn-guard/`. The campaign is **closed**.

**Failure.** The pinned vllm-xpu-kernels 0.1.14.1 GDN causal convolution
(commit `6d92b1bfbf32767ecda8e819613eb151e70030ad`) multiplies an `int` state
ID by an `int` state stride before adding that element offset to the
convolution state tensor. Production convolution state is bf16 with block
stride **425,984 elements** (851,968 bytes): valid state ID **5041** still
fits the 32-bit product; valid ID **5042** produces 2,147,811,328 and exceeds
INT32_MAX. Seven expressions share the defect across the non-speculative,
speculative, chunked, tiled and state-update implementations —
`csrc/xpu/gdn_attn/causal_conv1d.hpp` ×3, `xe_2/chunk_causal_conv1d_xe2.hpp`
×2, `xe_2/chunk_causal_conv1d_tiled_xe2.hpp` ×2. No invalid state ID, stale
block or wrong ALIGN column is involved: the IDs were always valid, the
consumer's multiplication was too narrow.

**Fix.** The seven expressions promote the state ID via
`static_cast<int64_t>(states_id)` **before** multiplying by the state stride
(exact patch `build/gdn-index64/source_delta.patch`). Tensor layouts, state
allocation, cache policy, kernel selection, quantization and graph policy are
unchanged; the existing chunk-delta rule already used 64-bit state-offset
products and is untouched. To avoid replacing unrelated operators in the
installed package, the corrected GDN translation unit is packaged as a small
private Torch operator library — `libgdn_index64.so` (built with oneAPI
2026.0.0-947 against the parent image's torch 2.13.0+xpu headers; operator
source `build/gdn-index64/gdn_index64.cpp`) — selected by a single dispatch
change in `vllm/_xpu_ops.py`: `torch.ops._xpu_C.gdn_attention` →
`torch.ops._gdn_index64.gdn_attention` (`build/gdn-index64/vllm_delta.patch`).
Build manifest and artifact hashes: `build/gdn-index64/manifest.json`. Both
sibling images carry **byte-identical** correction artifacts
(`libgdn_index64.so` sha256 `0fc700d3…`, `_xpu_ops.py` sha256 `1b4195c7…`).

**Causal proof.** A source-level parent guard on the actual GDN input tensors
raises on historical request 13, layer
`language_model.model.layers.38.linear_attn`, non-speculative prefill, state
ID **5042**, stride `[425984,2560,1]`, identically on all four ranks — the
exact request where unguarded runs first lost MTP acceptance
(`runtime/mtp3-gdn-guard/offset-guard/`). Direct operation comparison
(`scripts/test-gdn-index64.py`; receipt
`build/gdn-index64/evidence/operator-test.log`): stock operator on compact
state at ID 0 vs corrected operator on identical state at the production
stride at IDs **5041, 5042, 5623, 6600** — all convolution outputs, gate
output and updated state **bitwise equal** across prefill widths 256 and 64,
decode width 1, speculative width 4, plus three graph cases at ID 5623 with
eight replays each: **19/19 PASS**. The corrected serving lanes then completed
**84 cold long prompts** (MTP3 56/56, Base 28/28, both including the original
14 payload identities in order) — exposure far beyond the historical failure
point. The retained-tensor arithmetic also reproduces every recorded fault
address exactly: `5623 × 425984 = 2,395,308,032`, whose signed-32 result
`-1,899,659,264` lands 720,896 bytes past the mapped weight region
(`runtime/pw-rootcause/retained-offset-arithmetic.json`, 8/8 locations).

**What this closes.**
- APC/ALIGN **exposes** the arithmetic boundary by accumulating valid
  checkpoint state IDs over repeated long serving; the ALIGN state/index
  bookkeeping itself was never invalid (the instrumented ALIGN run had
  already exonerated it).
- Base, MTP1 and MTP3 share the affected convolution — speculative decoding
  is not a necessary condition; the APC-off discriminator stayed clean only
  because its smaller active state-ID set never reached the product limit.
- The graph#40 / PW-workspace localization was the **asynchronous reporting
  point**, not the cause: returning from the eager GDN host call establishes
  submission, not device completion, so the fault surfaced at the following
  graph replay call.
- The "800-MiB workspace allocation" was mislabeled: it is the drafter's BF16
  routed-expert weight `draft.model.layers.0.mlp.experts.routed_experts.w13_weight`,
  shape `[128,2560,1280]` = exactly 838,860,800 bytes. The actual
  WorkspaceManager buffer is **1,310,720 bytes (~1.25 MiB)**; its 800-MiB
  neighbor is an allocator segment, not one workspace tensor. No workspace
  sizing, lifetime or resizing fix was required or made.
- The earlier **Astra speculative-metadata** optimization was audited and
  exonerated: its uniform speculative fast path is inactive during ordinary
  (non-speculative) prefill, it is absent from Base behavior, and it does not
  alter the convolution's native index arithmetic. It did not cause the
  failure.

## The unified runtime authority — one image, two launch postures (freeze, 2026-09-26)

A pre-promotion static audit (no boot) established that the corrected MTP3
image and the corrected Base sibling carry **byte-identical correction
artifacts**, and their only other difference is the four-file Astra
speculative-metadata overlay inherited from `mtp3-c1`. Docker RootFS layer
containment proves the tree: `gdn-index64-c1` = Base-authority chain + 8
Astra COPY layers + 4 correction layers, with image Env/Entrypoint/Cmd/Labels
identical throughout the lineage (file-only overlays). Every Astra change is
gated on speculative execution — `gdn_attn.py` requires `use_spec_decode`,
`gdn_spec_metadata.eligible()` returns None without it,
`gdn_spec_metadata_batched.prepare()` returns `{}`, and `attn_utils.py`'s
prebuilt dict stays empty when `num_decode_draft_tokens_cpu` is absent — so
with no `--speculative-config` all four files execute the stock paths.
Speculation itself is flags-only on this stack: the first MTP1 candidate was
the frozen Base image plus flags, and the drafter never loads without the
flag (`WeightsMapper("mtp." → None)` is a pure load-time skip). Retained
depth-0 evidence agrees: the overlay candidate served drafts=0 traffic at
Base cadence **17.303 vs 17.307 ms** and decode 57.92–57.96 tok/s, equal to
the stock control (`runtime/astra/evidence/base-regression.json`), and the
installed-source harness checked the packaged dispatcher at **depths 0/1/3**
with exact comparisons including Base and graph replay. The corrected GDN
dispatch sits at the shared entry point, so Base mode uses the corrected
operation identically.

Verdict: the Base sibling is a causal/regression artifact, not a second
production runtime. **One unified image serves both postures**; the Base and
MTP3 launch surfaces differ only by launch configuration
(`--speculative-config`, util/maxlen):

```text
Unified runtime authority (promoted 2026-09-26, retag-only — bytes NOT rebuilt):
  local/qwen38-flashnext-v030:runtime-authority
  = local/qwen38-flashnext-v030:gdn-index64-c1   (build/provenance alias)
  = local/qwen38-flashnext-v030:base-authority   (compatibility alias, moved)
  = local/qwen38-flashnext-v030:mtp3-c1          (compatibility alias, moved)
  sha256:ad6d44d7a8aee41a4d419befbcaf78c7d73e0acd7f7aaaf597b5ff283601b448

Superseded Base authority (bytes retained):
  sha256:8aaa7309de1624a98b5b4fae7b43bdda105679c8d8f3e4600ada0f16555699e7
  (still tagged hcsplit-c1; scripts/launch-hcsplit-c1.sh relaunches it verbatim)

Superseded MTP3 authority (bytes retained, untagged):
  sha256:3bc0ee337a732b9341f1653264a0672b7661bcf37c926dc3cda404a1c02d5656
  (present in the local store; recoverable by image ID)

Validation sibling (evidence only, NOT a production lane):
  local/qwen38-flashnext-v030:gdn-index64-base-c1
  sha256:33647e33688d6ccf07c1eccf122918fde0c2cc49a9ee135391c2b4c36cf8781c
```

Launch surfaces (both digest-gated to the unified digest):
- **Base posture** (speculation off): `scripts/launch-gdn-index64-base.sh` —
  the frozen Base contract verbatim (TP4+EP, 32K, btk 256, seqs 4, util 0.85,
  FULL_AND_PIECEWISE `[1,2,4,8,256]`, `--cudagraph-metrics`, vision image:2,
  PLE pinned-host, shim), with **no** `--speculative-config`.
- **MTP3 posture**: `scripts/launch-gdn-index64-mtp3.sh` (`MTP_BOOT_ACK=1
  MTP_UTIL=0.91`, `SPEC_DEPTH` default 3) — the qualified util-0.91 capacity
  posture. `scripts/launch-mtp3-c1.sh` (util 0.85) remains the frozen MTP3
  qualification contract and fallback; its digest gate moved to the unified
  digest (identity-only edit, no argument changes).

Qualification basis (retained; details in the previous section and
`docs/GDN_CONV_INDEX64_FIX.md`): MTP3 56/56 and Base 28/28 cold long prompts
with the original 14 payload identities exact and both historical short
follow-ups; shorts 8/8; needles 3/3 exact through 31,652 tokens; vision 7/7
incl. ZEBRA-42 OCR; repeated APC hits with exact recall (93.075% / 97.73%);
APC ON / ALIGN, FULL+PIECEWISE graph acceleration, TP4+EP, W4A16, pinned-host
PLE and residency shim all preserved; MTP3 three-draft invariant
7,263 draft tokens / 2,421 proposals; health 200; 4/4 devices normal; zero
runtime exceptions. Performance deltas negligible: MTP3 step +0.31% / +0.55%
/ −0.17% at p512/p4096/p8192; Base decode −0.07% / −0.11% / −0.12%;
warmup-matched Base p512 TTFT 173.533 vs 172.100 ms (+0.83%). Bounded-
qualification caveat: the Base serving battery ran on the Base sibling; the
unified image's Base posture rests on the byte-identical correction
artifacts, the source-level gates above, and the retained depth-0 overlay
evidence. Promotion record: `build/gdn-index64/PROMOTION.json`.

## Honest boundaries / deferred work

- **APC dedicated qualification still pending** at Base-freeze time (the MTP
  campaign since exercised APC heavily with exact outputs — see Part II).
- **Concurrency envelope not yet tuned or frozen** (the MTP3 qualification ran
  C1/C2/C4 inside the unchanged `max_num_seqs=4` contract).
- **Video unqualified**; image qualified up to `image:2`, 224–448 px fixtures only.
- **MTP executed after Base closure** (Part II): native and correct at depth 1 and
  3; MTP1 arithmetically unprofitable; MTP3 qualified as a separate authority
  (`mtp3-c1`, `3bc0ee33…`) and later superseded — both lanes now share the
  unified corrected authority (`ad6d44d7…`, see the unified-runtime section).
  The v0.26 oneCCL RingTransmit wedge never reproduced on v0.30 in
  any posture.
- **Base tuning is CLOSED** unless a real runtime issue appears.
- **Released as the B70 model pack `qwen38-flashnext-b70` 1.0.0 (2026-09-27)** —
  see the release section at the end of this recipe. Concurrency tuning and
  video remain deferred; upstream reports remain parked (below).
- Upstream-reportable items parked: the width-gated HC K-split (and the oneDNN skinny
  large-K race it works around), the PLE cross-stream capture fix, the 2^34-byte
  XPU pin ceiling, the WORKDIR checkout shadowing, and the PIECEWISE padded-width
  poisoning at 64 (superseded operationally by the K-split but still a stock defect).
- The HC race itself is **worked around, not fixed**: stock `F.linear` at
  `[N,10240]`-class shapes remains nondeterministic above width 64; any future width
  >64 path that bypasses the gated arm inherits it.
- Environment scope: 4× Arc Pro B70, host driver family matched to the pinned image
  stack; other hardware unqualified. `xpu-smi` must run under `sudo` on this host.

## The final runtime

```text
Unified/current runtime authority (promoted 2026-09-26, retag-only — bytes NOT rebuilt):
  local/qwen38-flashnext-v030:runtime-authority
  = local/qwen38-flashnext-v030:gdn-index64-c1   (build/provenance alias)
  = local/qwen38-flashnext-v030:base-authority   (compatibility alias, moved)
  = local/qwen38-flashnext-v030:mtp3-c1          (compatibility alias, moved)
  sha256:ad6d44d7a8aee41a4d419befbcaf78c7d73e0acd7f7aaaf597b5ff283601b448

  Serves BOTH postures from one image (see the unified-runtime section for
  the audit): Base = same contract with NO --speculative-config
  (scripts/launch-gdn-index64-base.sh); MTP3 = + --speculative-config
  '{"method":"mtp","num_speculative_tokens":3}' at util 0.91
  (scripts/launch-gdn-index64-mtp3.sh, MTP_BOOT_ACK=1 MTP_UTIL=0.91).

Superseded Base authority (2026-09-26 → 2026-09-26, bytes retained):
  local/qwen38-flashnext-v030:hcsplit-c1
  sha256:8aaa7309de1624a98b5b4fae7b43bdda105679c8d8f3e4600ada0f16555699e7
  (= the K-split freeze; superseded by the GDN index-64 correction;
   relaunchable verbatim via scripts/launch-hcsplit-c1.sh)

Superseded MTP3 authority (bytes retained by image ID):
  sha256:3bc0ee337a732b9341f1653264a0672b7661bcf37c926dc3cda404a1c02d5656
  (= Base authority + 4-file GDN speculative-metadata overlay; superseded by
   the same correction; qualification/util-capacity receipts remain its record)

Retained pre-K-split fallback:
  local/qwen38-flashnext-v030:full-c1
  sha256:7c8a35dfe8ba7fdc4e9e263038d3402d0c3b220530f540a50b48bf41022a4b6b
  (posture: FULL_AND_PIECEWISE [1,2,4,8], eager 256-chunk prefill; relaunchable
   verbatim via scripts/launch-full-c1.sh)

Validation sibling (evidence only, NOT a production lane):
  local/qwen38-flashnext-v030:gdn-index64-base-c1
  sha256:33647e33688d6ccf07c1eccf122918fde0c2cc49a9ee135391c2b4c36cf8781c

Parent chain (immutable):
  vllm/vllm-openai-xpu v0.30.0, amd64 sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9
  └ local/qwen38-flashnext-v030:base-c5 sha256:4e09fa32ac9c0b7dcb9571d6ee42ac0012be83e30c937c9d4a518e4438ac2372
    (official image + 2-file overlay [XPU gate, PLE pinned-host] + shim; also the
     PIECEWISE/eager fallback image)
    └ full-c1 sha256:7c8a35df… (1-file PLE capture fix)
      └ hcsplit-c1 sha256:8aaa7309… (2-file width-gated HC K-split — old Base authority)
        └ mtp3-c1 sha256:3bc0ee33… (4-file GDN speculative-metadata overlay — old MTP3 authority)
          └ runtime-authority sha256:ad6d44d7… (+ libgdn_index64.so and the
            _xpu_ops.py GDN dispatch correction — the unified authority)
```

The unified authority image = official 0.30 + exactly **four overlaid vLLM
files** (arch gate; PLE pinned-host; PLE FULL-capture fix; HC K-split +
load-time prep hook) + the residency shim at `/opt/b70-residency-shim/` + the
four-file GDN speculative-metadata overlay + the GDN index-64 correction
(`libgdn_index64.so` + `_xpu_ops.py` dispatch). Full provenance:
`build/manifest.json` → `build/full-c1/manifest.json` →
`build/hcsplit-c1/manifest.json` (+ `build/hcsplit-c1/PROMOTION.json`) →
`build/astra-cumulative/manifest.json` → `build/gdn-index64/manifest.json`
(+ `build/gdn-index64/PROMOTION.json`).

**Exact launch contract (Base posture)** = `scripts/launch-gdn-index64-base.sh`
(digest-gated to the unified digest; byte-equal serve args to the frozen
`scripts/launch-hcsplit-c1.sh` contract, minus nothing). Essential form:

```text
vllm serve /models/flashnext-w4a16-serve \
  --tensor-parallel-size 4 --enable-expert-parallel --dtype bfloat16 \
  --max-model-len 32768 --max-num-batched-tokens 256 --max-num-seqs 4 \
  --gpu-memory-utilization 0.85 --trust-remote-code \
  --served-model-name devan-carlin/Qwen3.8-Flash-Next-W4A16 \
  --limit-mm-per-prompt '{"image":2,"video":0}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,256]}' \
  --cudagraph-metrics

env (required): PLE_TABLE_PATH=/models/flashnext-w4a16-serve/ple_table_qwen4exp.pt
  LD_PRELOAD=/opt/b70-residency-shim/libl0_peer_residency_shim.so
  VLLM_XPU_ENABLE_XPU_GRAPH=1  CCL_SYCL_ALLREDUCE_TMP_BUF=1
  CCL_SYCL_ALLGATHERV_TMP_BUF=1  ZE_FLAT_DEVICE_HIERARCHY=FLAT
  ZE_AFFINITY_MASK=0,1,2,3  VLLM_WORKER_MULTIPROC_METHOD=spawn
  VLLM_TARGET_DEVICE=xpu  + HF offline trio
mounts: model snapshot (read-only) → /models/flashnext-snap;
  serve dir (patched config + symlinks) → /models/flashnext-w4a16-serve
```

- **Model/revision**: `devan-carlin/Qwen3.8-Flash-Next-W4A16` @
  `40b8f18df4d4a32cb6e687a51c78207e5e438522` (serve dir = `serve/flashnext-w4a16-serve/`
  with the dense-QSA config surgery).
- **Graph policy**: FULL_AND_PIECEWISE `[1,2,4,8,256]`, mode NONE, btk 256.
- **Performance authority**: TTFT 0.1721 / 1.4884 / 1.9414 s; prefill ≈2,975 / 2,752 /
  4,220 tok/s; decode 57.9245 / 57.8501 / 57.3798 tok/s (C1 medians; vs eager
  7.9–8.0×, vs v0.26 Base +27–28%, TTFT −46…−50% vs pre-K-split). Measured on
  the pre-correction Base authority `8aaa7309…`; the GDN index-64 correction
  moves Base decode ≤0.12% and matched-warmup p512 TTFT +0.83% (unified-runtime
  section) — these remain the reference numbers.
- **Memory authority**: util 0.85, 32K, weights ≈18.7 GiB/rank, KV 356,485 tokens
  (10.88×), PLE zero-device-resident, shim required.
- **Qualification summary**: 4-boot causal determinism gate (no relocated divergence);
  full battery (shorts 8/8, needles 3/3 exact to 31,652, vision 7/7 incl. OCR);
  1,389-request replay stress clean; 273,249 K-split executions/rank proven in
  dispatch; 0 degeneration/fault signatures; 4/4 devices normal. The corrected
  unified image re-passed the battery shape (shorts 8/8, needles 3/3, vision 7/7,
  28/28 + 56/56 cold long prompts, health 200, 4/4 devices; see the
  GDN-correction and unified-runtime sections).
- **Deferred/unqualified**: APC qualification, concurrency freeze, video,
  release/pack packaging, upstream reports (see honest boundaries). MTP1 uses
  the same corrected operation (depth-1 smoke retained) but received no
  dedicated MTP1 campaign on the unified image.

## The B70 model pack release (2026-09-27)

The unified authority above is the production runtime of the released pack
`qwen38-flashnext-b70` v1.0.0 ("Qwen3.8-Flash-Next B70 Pack (Base / MTP3)"):

- **Production image**: `ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0`, registry
  manifest digest `sha256:85512b52c09fa660a2e7fe441417129e7c47fac727fd85f66ccea6b65e0a9122`
  — the unified authority `ad6d44d7…` plus ONE deterministic packaging layer
  (baked dense-QSA serve config sha256 `91fa33ca…` + the serve-directory
  preparator ENTRYPOINT). Both modes use the same image bytes.
  Build/push records: `packaging/PROMOTION.json`.
- **Model authority**: `devan-carlin/Qwen3.8-Flash-Next-W4A16` @
  `40b8f18d…` (31 files, 179,841,735,233 B, PLE table included; ungated).
- **Profiles**: 8, TP4-only — `w4a16-{base,mtp3}-tp4-{32768,65536,131072,
  262144}`. Base = util 0.85 + speculation OFF; MTP3 = util 0.91 + MTP depth
  3 (`num_speculative_tokens: 3`). APC stock-ON / ALIGN, vision image:2 /
  video:0 (unqualified), pinned-host PLE, residency shim, btk 256, seqs 4,
  FULL_AND_PIECEWISE `[1,2,4,8,256]` compile-NONE — the qualified contract
  unchanged, expressed through pack.json layers.
- **Ladder law**: a successful 262144 boot of a mode qualifies every lower
  context tier of that mode on the same contract; the two max-context b70ctl
  smokes (Base, MTP3) are the pack's qualifying events. No lower-context
  boots exist or are required.
- **BetterBench v0.6.0 authorities** (`benchmarks/betterbench-v0.6.0/
  {base,mtp3}-final/`): Base weighted decode 58.34 tok/s, prefill 2,476
  tok/s @23.6K, C4 183.7; MTP3 weighted decode 119.86 tok/s, C2 195.0,
  prefill 2,290 @23.6K; 3-draft invariant exact. Concurrency numbers are
  recorded envelope data; concurrency tuning is deferred.
- **Deferred**: concurrency tuning, video, MTP1-as-pack-mode, upstream
  submissions (none planned). No runtime investigation remains open.

Pack contracts live in `B70-LLM-Controller-Model-Packs/Qwen3.8-Flash-Next/`
(dev) and `B70-Controller/model-packs/Qwen3.8-Flash-Next/` (published);
release tag `qwen3.8-flash-next-v1.0.0`.

## Tool calling — the 1.0.1 launch-contract correction (2026-09-27)

A chat frontend sent the released 1.0.0 server `tools` +
`"tool_choice": "auto"` and got HTTP 400 (`"auto" tool choice requires
--enable-auto-tool-choice and --tool-call-parser to be set` — the static
guard in `vllm/renderers/online_renderer.py`). The runtime was never at
fault; the launch contract omitted vLLM's tool flags.

**Static audit of the exact pinned image** (vLLM 0.30.0, transformers 5.16.1,
model revision `40b8f18d…`): `qwen3_xml` is registered
(`Qwen3EngineToolParser`, combined-engine adapter over `vllm/parser/qwen3.py`)
and imports clean; `qwen3` is registered as the reasoning parser
(`Qwen3ParserReasoningAdapter`); the standalone `chat_template.jinja` (which
takes precedence over the older embedded template) renders the complete Qwen3
XML tool loop — `<tools>` system block, `<tool_call><function=…><parameter=…>`
assistant calls, `<tool_response>` tool results, and continued generation —
in exactly the syntax the parser engine defines; Base and MTP3 share the one
tokenizer/template. No custom template and no other flags are required.

**Fix**: pack **1.0.1**, launch-contract only — three flags added once at the
shared runtime layer of `pack.json` (`--enable-auto-tool-choice`,
`--tool-call-parser qwen3_xml`, `--reasoning-parser qwen3`), inherited
identically by both modes and all 8 profiles. Runtime id, image tag, and
digest are **unchanged** (`85512b52…`); no rebuild, no GHCR push (the 1.0.4
vision-flag pack-only precedent). Validator gained hard assertions for the
parser pair and per-profile inheritance; render-diff proved the delta is
exactly those five command tokens.

**Live proof** (b70ctl path, 262144, `runtime/pack-smoke-1.0.1/`): Base
15/15 and MTP3 16/16 — the original 400 shape now returns 200 with a valid
OpenAI `tool_calls` structure (name + JSON arguments exact), tool-result
round trip coherent, streaming tool calls assemble correctly, `tool_choice:
"none"` stays plain text, reasoning/content separation works in stream and
non-stream form (vLLM 0.30 names the field `reasoning`), ZEBRA-42 vision and
APC intact, MTP3 3-draft invariant exact (408 = 3×136), clean logs, 4/4
devices. Full record: `docs/TOOL_CALLING_1.0.1.md`; release tag
`qwen3.8-flash-next-v1.0.1`.

## Evidence index (breadcrumbs)

| Stage | Document | Evidence root |
| v0.26 closure | `../qwen3.8-flash-next/v026-reference/retired/V026-CAMPAIGN-SUMMARY.md`, `INDEX.md` | `../qwen3.8-flash-next/v026-reference/` (read-only archive) |
| v0.30 recon / stack pin | `RECON.md` | `evidence/image-inspect-v030-amd64.json`, `image-source/vllm/` |
| Core Base (gate + PLE + shim) | `docs/BASE_BRINGUP.md` | `build/manifest.json`, `runtime/base-c5/evidence/` |
| Vision | `docs/VISION_BRINGUP.md` | `runtime/vision-b1/evidence/` |
| Retention + eager baseline | `docs/FULL_RUNTIME_RETENTION.md` | `runtime/vision-b1/evidence/{retention,c1-eager-baseline}/` |
| PIECEWISE acceleration | `docs/ACCELERATED_BRINGUP.md` | `runtime/accel-b2/evidence/`, `runtime/accel-b1/evidence/` (FULL failure) |
| FULL-graph fix | `docs/FULL_GRAPH_LANE.md` | `build/full-c1/manifest.json`, `runtime/full-c1/evidence/` |
| Prefill-256 rejection | `docs/PREFILL_256_TEST.md` | `runtime/prefill-c1/evidence/` |
| Prefill mechanism recon | `docs/PREFILL_GRAPH_RECON.md`, `docs/PREFILL_REPLAY_CORRUPTION_RECON.md` | `runtime/full-c1/evidence/`, source citations in docs |
| Width brackets | `docs/PREFILL_WIDTH64_TEST.md`, `…WIDTH32…`, `…WIDTH48…` | `runtime/width{64,32,48}-c1/evidence/` |
| btk-48 rejection | `docs/SAFE_WIDTH_PREFILL_TEST.md` | `runtime/btk48-c1/` |
| M256 localization | `docs/M256_REPLAY_ISOLATION.md`, `M256_SEGMENT_DIFFERENTIAL.md`, `M256_COLLECTIVE_DIFFERENTIAL.md`, `M256_HC_BREAK_TEST.md` | `m256-isolation/`, `m256-segdiff/`, `m256-colldiff/`, `m256-hcbreak/` + `runtime/{segdiff,colldiff,hcbreak, hciso, hcfix}-c1/` |
| HC race + K-split offline | `docs/HC_GATE_MIX_ISOLATION.md`, `docs/HC_DOWN_GEMM_FIX_RECON.md` | `m256-hciso/`, `runtime/hciso-c1/`, `runtime/hcfix-c1/` |
| Final qualification + promotion | `docs/HC_KSPLIT_SERVING_TEST.md` | `build/hcsplit-c1/{manifest.json,PROMOTION.json}`, `runtime/hcsplit-c1/` |
| MTP recon (native feasibility) | `docs/MTP_RECON.md` | checkpoint/safetensors audits, `image-source/vllm/` |
| MTP1 eager + accelerated bring-up | `docs/MTP1_EAGER_BRINGUP.md`, `docs/MTP1_ACCELERATED_BRINGUP.md` | `runtime/mtp1-{eager,graph}-c1/evidence/` |
| MTP1 performance LOSS receipt | `docs/MTP1_PERFORMANCE.md` | `runtime/mtp1-graph-c1/evidence/c1-lane/` |
| MTP3 legality + first receipt | `docs/MTP3_FEASIBILITY_PERFORMANCE.md` | `runtime/mtp3-graph-c1/evidence/` |
| Step-cost profile (GDN hotspot) | `docs/MTP_STEP_COST_PROFILE.md` | `prof/`, `runtime/prof-*/` |
| Astra correction + optimizations + final receipt | `docs/MTP_OVERHEAD_ASTRA.md` | `astra/`, `build/astra-cumulative/`, `runtime/astra*/` |
| MTP3 qualification (this campaign) | `docs/MTP3_QUALIFICATION.md` | `runtime/mtp3-c1/evidence/` |
| MTP3 util capacity recovery | `docs/MTP3_UTIL_CAPACITY.md` | `runtime/mtp3-u89|u90|u91|u91ctx/` |
| Deep-prefill DEVICE_LOST investigation | `docs/DEEP_PREFILL_DEVICE_LOST.md`, `DEEP_PREFILL_APC_DISCRIMINATOR.md`, `ALIGN_STATE_BLOCK_LIFETIME.md`, `ALIGN_OBS_INSTRUMENTED_RUN.md`, `PW_ALLOC_LOCALIZATION_RUN.md` | `runtime/pw-rootcause/`, `runtime/align-obs*/`, `runtime/pwobs*/` |
| GDN index-64 fix + sibling qualification | `docs/GDN_CONV_INDEX64_FIX.md` | `build/gdn-index64/` (+ `PROMOTION.json`), `runtime/mtp3-gdn-guard/`, `runtime/mtp3-gdn-index64-c1/`, `runtime/gdn-index64-base-c1/` |
| Unified runtime promotion | `RUNTIME_RECIPE.md` unified-runtime section | `build/gdn-index64/PROMOTION.json`, `build/gdn-index64/promotion-inspect-{pre,post}.txt` |
| Tool calling 1.0.1 (launch-contract fix) | `docs/TOOL_CALLING_1.0.1.md` | `runtime/pack-smoke-1.0.1/` |
| Launchers | `scripts/launch-*.sh` (unified Base posture: `launch-gdn-index64-base.sh`; unified MTP3: `launch-gdn-index64-mtp3.sh`; frozen fallbacks: `launch-hcsplit-c1.sh`, `launch-full-c1.sh`, `launch-mtp3-c1.sh`) | — |
---

# Pack addendum — `qwen38-flashnext-b70` 1.0.0 (2026-09-27)

Everything above is the campaign/runtime history of the unified authority
image. This section records how that authority became the released B70
model pack. Nothing above changed; the pack adds exactly one deterministic
packaging layer and a launch contract expressed through pack.json layers.

## Image identity

```text
parent (campaign-internal, never rebuilt for the pack):
  local/qwen38-flashnext-v030:runtime-authority
  sha256:ad6d44d7a8aee41a4d419befbcaf78c7d73e0acd7f7aaaf597b5ff283601b448

pack production image (GHCR, tag pushed once, immutable):
  ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0
  manifest digest sha256:85512b52c09fa660a2e7fe441417129e7c47fac727fd85f66ccea6b65e0a9122
  (= local/qwen38-flashnext-v030:pack-1.0.0, image Id sha256:85512b52…)
```

Digest-capture law satisfied at publication: push output, buildx imagetools
inspect, and the registry `Docker-Content-Digest` header all reported the
same manifest digest, and a digest-pull round trip reproduced the built
image Id. The pack pins tag AND digest AND registry ref; the registry
manifest digest is the authority. Parent provenance lives in the campaign
workspace (`packaging/PROMOTION.json`).

## The packaging layer

`FROM` the unified authority + one layer:

- `/opt/b70-flashnext/serve-config.json` — byte-copy of the qualified
  dense-QSA serving config (sha256 `91fa33ca…`), provably equal to the
  pinned-revision `config.json` (sha256 `b9ef7d7d…`) minus exactly the five
  `text_config.indexer_*` keys. `--hf-overrides` cannot express key removal
  on vLLM 0.30 (shallow `config.update()`), so config surgery on a served
  directory is the only correct mechanism, and this is its byte-exact form.
- `/opt/b70-flashnext/prepare-serve.sh` (0755, the ENTRYPOINT) — receives
  the Controller-resolved `vllm serve <readonly mount> [flags…]`, fail-closed
  checks for the PLE table and the weight index under the mount, rebuilds
  `/work/flashnext-serve` in the writable container filesystem (symlinks to
  every mounted file + the baked patched config), and execs
  `/opt/venv/bin/vllm serve /work/flashnext-serve <flags…>` verbatim. The
  Controller-mounted snapshot is never written (read-only bind; verified).

Served bytes, flags, and env are byte-equivalent to the qualified launchers
modulo Controller-owned port publishing, entrypoint mechanics, and caches
moved from the campaign's `/experiment-runtime` bind to image-internal
`/work/*` (established pack precedent). The PLE table path
(`PLE_TABLE_PATH=/models/devan-carlin__Qwen3.8-Flash-Next-W4A16/ple_table_qwen4exp.pt`)
resolves straight off the readonly Controller mount — the same pinned-host
2-slab load as every qualified boot.

## The launch contract (pack.json layering)

One model, one runtime, two modes, eight profiles (`runtime → model → mode →
profile` merge; env keys strictly partitioned across layers):

- runtime layer: the fixed device docker args, the 16 shared env vars
  (CCL tmp-buffer flags, offline trio, `VLLM_TARGET_DEVICE=xpu`,
  `VLLM_XPU_ENABLE_XPU_GRAPH=1`, `ZE_FLAT_DEVICE_HIERARCHY=FLAT`,
  `VLLM_WORKER_MULTIPROC_METHOD=spawn`, shim `LD_PRELOAD`, `PLE_TABLE_PATH`,
  `/work/*` caches), and the shared
  `vllm serve --host 0.0.0.0 --port 8000 --dtype bfloat16
  --enable-expert-parallel --trust-remote-code --max-num-batched-tokens 256
  --max-num-seqs 4 --limit-mm-per-prompt {"image":2,"video":0}
  --compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE",
  "cudagraph_capture_sizes":[1,2,4,8,256]} --cudagraph-metrics`.
- model layer: the mount positional + `--served-model-name`.
- mode `base` (declared first ⇒ b70ctl default): `--gpu-memory-utilization
  0.85`, **no** speculative config (drafter never loads).
- mode `mtp3`: `--gpu-memory-utilization 0.91` + `--speculative-config
  {"method":"mtp","num_speculative_tokens":3}`.
- each profile: `ZE_AFFINITY_MASK=0,1,2,3`, `--tensor-parallel-size 4`,
  `--max-model-len <32768|65536|131072|262144>`.

APC stays stock-ON (never flag-touched; engine logs `enable_prefix_caching=True`,
Mamba cache `'align'`). torch.compile mode NONE is the architecture's stock
breakable-graph design (no `mode` key, unlike compiled packs).

## Qualification basis for this pack

The two max-context boots through the normal b70ctl path — Base
(`w4a16-base-tp4-262144`) and MTP3 (`w4a16-mtp3-tp4-262144`) on the released
image digest — are the pack's qualifying events; every lower context tier
inherits from the successful 262144 boot of its mode (ladder law). Underlying
runtime qualification (shorts, needles to 31,652 tokens, vision incl.
ZEBRA-42 OCR, cold long-prompt batteries 56/56 MTP3 + 28/28 Base, APC exact
recall, 3-draft invariant, BetterBench v0.6.0 both lanes) is retained in the
campaign workspace evidence roots indexed above.

Deferred (explicitly not in this release): concurrency tuning
(`max_num_seqs 4` unchanged), video (unqualified, `video:0`), MTP1 as a pack
mode, any upstream submission.


---

# Pack addendum — `qwen38-flashnext-b70` 1.0.1 (2026-09-27)

A chat frontend sent the 1.0.0 server `tools` + `"tool_choice": "auto"` and
received HTTP 400 — vLLM's static guard in `renderers/online_renderer.py`
rejects auto tool choice when no tool parser is configured. The runtime was
healthy; the 1.0.0 launch contract had omitted vLLM's tool-calling flags.
Pack 1.0.1 is the launch-contract correction. **Nothing about the runtime
image changed**: same runtime id `qwen38-flashnext-runtime-1.0.0`, same image
tag `ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0`, same manifest digest
`sha256:85512b52…` — no rebuild, no GHCR push (the 1.0.4 vision-flag
pack-only precedent; no image tag proliferation).

## Static audit (the exact pinned stack, no inference from upstream)

- `qwen3_xml` registered in `vllm/tool_parsers/__init__.py` →
  `Qwen3EngineToolParser` (adapter over the combined Qwen3 parse engine
  `vllm/parser/qwen3.py`); registry import verified in-image.
- `qwen3` registered in `vllm/reasoning/__init__.py` →
  `Qwen3ParserReasoningAdapter`; import verified. The engine defines
  `<think>`/`</think>` and treats `<tool_call>` as an implicit reasoning end —
  tool and reasoning parsing are one designed combination.
- The pinned revision's standalone `chat_template.jinja` (9,993 B; takes
  precedence over the older embedded template under transformers 5.16.1) was
  rendered in-image with a full tool conversation: tool definitions
  (`<tools>` system block), assistant tool calls
  (`<tool_call><function=…><parameter=…></parameter></function></tool_call>`),
  `tool` role results (`<tool_response>` under a user turn), and continued
  generation after results — byte-for-byte the syntax the `qwen3_xml` parser
  engine defines. Base and MTP3 share the one tokenizer/template (single
  model + runtime entry in the pack).
- No custom chat template, no further flags. The three flags below all exist
  on the pinned CLI; the only cross-flag constraint is
  "auto requires --tool-call-parser" (satisfied).

## The delta

```diff
- "version": "1.0.0"
+ "version": "1.0.1"
  runtimes[0].launch.command:
+   "--enable-auto-tool-choice"
+   "--tool-call-parser", "qwen3_xml"
+   "--reasoning-parser", "qwen3"
```

Flags live once at the shared runtime layer, inherited identically by both
modes and all 8 profiles (the validator's duplicate-flag rejection makes
double-placement structurally impossible). `validate.py` additionally pins
version 1.0.1, repins `PACK_SHA256` (`12566bdc…` → `5cba8a7c…`), and asserts
the parser pair plus per-profile inheritance; render-diff 1.0.0 → 1.0.1
proved the complete delta is exactly those five command tokens across all 8
profiles with every other render field and the digest byte-identical.

## Qualification (b70ctl path, 262144, released digest)

Both max-context boots on the installed 1.0.1 pack (`runtime.BuildLaunch` →
`runtime.Start`; container args carry the three flags):

- **Base 15/15**: health 200 (~5.5 min); no-tools chat coherent; tools +
  `tool_choice:"auto"` → HTTP 200 with OpenAI `tool_calls` (name
  `calculator`, arguments `{"expression": "17*23+5"}` JSON-exact);
  tool-result round trip (role `tool` → "…is **396**."); streaming tool call
  assembles with `finish_reason="tool_calls"`; `tool_choice:"none"` plain
  text; reasoning separation in stream + non-stream (field `reasoning`, zero
  raw tag leakage); ZEBRA-42 OCR exact; APC hit delta 2,496/2,698 with exact
  recall; zero spec counters; clean logs; 4/4 devices.
- **MTP3 16/16**: depth-3 active; 3-draft invariant exact (408 = 3×136);
  the same tool surface green end-to-end; APC hit 1,664/2,698; health 200;
  zero runtime/spec/parser exceptions; 4/4 devices.
- Server log on both boots: `enable_auto_tool_choice: True`,
  `tool_call_parser: 'qwen3_xml'`, `reasoning_parser: 'qwen3'`,
  `"auto" tool choice has been enabled.`, `enable_prefix_caching=True`.

Evidence: campaign workspace `docs/TOOL_CALLING_1.0.1.md` +
`runtime/pack-smoke-1.0.1/`. Claim scope: the tested client-side calculator
schema at 262144 through the b70ctl pack path; lower contexts inherit by the
ladder law. Deferred as before: concurrency tuning, video, MTP1-as-pack-mode.

# Source patch publication — pack 1.0.2

The [published patch index](../patches/README.md) now ships inside the model-pack archive with the exact retained vLLM, vllm-xpu-kernels, serving-packaging and B70 residency-shim source needed to account for the promoted runtime. The provenance chain is the pinned vLLM 0.30.0 XPU image (`sha256:e4446310…`) → base-c5 → full-c1 → hcsplit-c1 → astra-cumulative/mtp3-c1 → gdn-index64-c1 → the unchanged production packaging layer. The native patch targets vllm-xpu-kernels 0.1.14.1 commit `6d92b1bfbf32767ecda8e819613eb151e70030ad`. Each patch README identifies its source base, affected files, stage and retained authority; `patches/SHA256SUMS` records distributed bytes.

Pack 1.0.2 changes only the pack version, documentation and source distribution. The image remains `ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0` (`sha256:85512b52c09fa660a2e7fe441417129e7c47fac727fd85f66ccea6b65e0a9122`). The model revision, modes, eight profile command lines, environment and mounts are identical to 1.0.1. The release archive includes `README.md`, `RUNTIME_RECIPE.md`, `pack.json` and `patches/**`.

# Tool-choice semantic fix — pack 1.0.3 (2026-10-07)

The shared runtime launch adds exactly one `--exclude-tools-when-tool-choice-none` argument for all eight Base/MTP3 profiles. The existing vLLM 0.30 renderer then excludes current tool definitions for `tool_choice:"none"`; historical assistant calls and tool results remain represented. Previously, the flag was disabled and none-mode prompts still advertised tools, allowing selected call XML to be generated before parser suppression.

Temporary qualification at context 262144 retained every existing mode setting: Base utilization 0.85 with speculation off; MTP3 utilization 0.91 with exactly three speculative tokens. Both modes passed 20 byte-identical retained calculator none requests, with correct 396 answers, null calls, no selected call XML, finite logprobs and healthy immediate controls. Both also passed non-streaming/streaming none, automatic calls and tool-result continuations; default-to-auto, explicit auto, named and required choices; disabled/enabled thinking and supported low reasoning effort; tool history; and the existing red-circle vision smoke with tools present under none.

Streaming selected-token audits use the existing `return_token_ids` response-metadata option where necessary to retain parser-buffered chunks. This adds no decoding constraint. Normal non-none prompts remain unchanged; none prompts match the corresponding tools-omitted forms, including reasoning prefixes and image placeholders.

The image digest, model pin, source patch bundle, environment, mounts and all other serving arguments remain unchanged. This is a pack launch fix requiring a new pack archive/catalog version and runtime recreation after installation, with no vLLM/model/template patch or image rebuild. Qualification evidence is retained in the campaign workspace `runtime/rr-exclusion-qualification-20261007`; no prefill scheduling work was performed.
