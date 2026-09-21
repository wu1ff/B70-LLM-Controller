# Qwen3.6-35B-A3B on Intel Arc Pro B70 — the runtime recipe

This is the technical history of the vLLM runtime that serves Qwen3.6-35B-A3B on Intel Arc Pro B70 GPUs: where it started, what was broken or missing at each step, what changed, why, and what each change produced — in the order the changes actually stack.

It is written for someone who wants to understand this runtime or investigate the same source areas, so it carries breadcrumbs (upstream projects, PRs, files, kernels). It is not an SHA-by-SHA reproduction guide and not a container-build manual. Detailed per-episode qualification records live in the private development repository (`qwen3.6-35b-a3b`, Forgejo); this document is the pack-facing summary of what is baked into the image, what is a launch policy, and what the final qualified state is.

Everything except the launch policies (graph capture, scheduler settings, memory utilization, affinity — all expressed in `pack.json`) is baked into the final image. The final image is **one image for every profile**: Base and DFlash, FP8 and INT4, TP1/TP2/TP4.

## Where this runtime came from

The runtime is a direct descendant of the qualified Qwen3.8-27B v26 platform (the same foundation stack: vLLM 0.26 on Torch 2.14 XPU with the pinned Ubuntu 24.04 + Intel OMIX user-mode base, the native XPU kernel closure, the Level Zero peer-residency shim, and the oneCCL temporary-buffer launch contract). Qwen3.6-35B-A3B is a routed-expert (MoE) hybrid GDN model, which is where the port work began.

| Component | Role in this runtime |
|---|---|
| vLLM | `v0.26.0` generation, routed-expert + GDN + speculative paths modified as below |
| vLLM XPU kernels | `v0.1.11.1` generation + the PR #459 backport (surface-height fix) |
| Grouped-GEMM native library | Xe2 W4A16 path with the intel/llm-scaler PR #442 pitch guard |
| DFlash drafter support | `qwen3_dflash.py` aux-dtype overlay (5 lines, loaded only under a speculative config) |

## The chain at a glance

1. Routed-expert FP8 rehome — flip Qwen3.6's routed experts onto the native grouped-GEMM MoE path and restore graph capture (first Qwen3.6 image delta; FP8 Base baseline)
2. INT4 AutoRound MoE enablement — property-gated XPU construction of the native W4A16 grouped-GEMM path for packed AutoRound routed experts, plus an availability probe fix so the unloadable ARK XPU library falls through to the working dense INT4 kernel (second image delta; INT4 Base authority)
3. vllm-xpu-kernels PR #459 backport — the 2^24 load-surface-height truncation fix (native library swap; closed a DFlash TP4 history-conditioned correctness corruption at its root)
4. DFlash concurrency correction — 64-bit convolution-state indexing in the native library plus the Graph64 capture policy (TP2 conv-state slot×stride intermediates overflowed signed 32-bit above ~5 active requests; C2+ batches dispatched eager)
5. Scale-prefetch pitch guard — intel/llm-scaler PR #442 (native grouped-GEMM library; closed the INT4 TP2 prefix-caching blocker that scheduler 2048 exposed)
6. DFlash prefix-caching correction — upstream #51113 align/resume + #51603 multimodal encoder-cap ordering + #48109 XPU high-bit pointer handling (two Python files)
7. Mamba sub-block-progress scheduler — Kimi K3 PR #50000 hunk (one Python file; makes TP2 prefills interleave with decodes under the speculative clamp)
8. Shared-GDN APC window-index correction — one Python file (`gdn_attn.py`): common APC window-index columns computed once in the shared GDN build context instead of per-group (30 groups) — the final image delta
9. Context ceilings at the model-native 262144 — launch policy only
10. The one-card DFlash lane (INT4 DFlash TP1) — launch policy only
11. Unified Base + DFlash authority — no image change: five Base lanes re-qualified on the exact DFlash-authority bytes

→ the final qualified runtime, published as `ghcr.io/wu1ff/qwen36-35b-a3b-b70:1.0.0`.

Each stage below is labeled correctness, feature, performance, or reliability, because those categories have different shelf lives.

## Routed-expert FP8 rehome (feature)

**What was wrong.** The frozen v26 foundation could not boot Qwen3.6 FP8 at all: the reference routed-expert (MoE) execution path defeated graph capture on this platform.

**What changed.** A one-file image delta re-homing the routed experts onto the native grouped-GEMM MoE path used by this stack.

**What that produced.** Graph capture restored; FP8 Base TP2 qualified cold ×3 with bounded accuracy and became the campaign baseline; TP4 followed as a launch-profile derivation (tensor parallel + affinity only).

## INT4 AutoRound routed MoE enablement (feature)

**What was wrong.** The AutoRound INT4 checkpoint's routed experts had no XPU construction path, and the ARK availability probe let an unloadable XPU library be selected.

**What changed.** Two one-file deltas: property-gated XPU construction of the existing native W4A16 grouped-GEMM MoE method, and an availability probe that requires the XPU library to actually load (falling through to the working dense INT4 kernel otherwise).

**What that produced.** Native W4A16 grouped-GEMM execution on every rank for the INT4 target; INT4 Base TP1/TP2/TP4 qualified with full 262144-context admission. INT4 serves its natural bfloat16 dtype — no `--dtype` override anywhere in the pack's INT4 profiles.

## The 2^24 surface-height truncation (correctness — the DFlash TP4 corruption)

**What was wrong.** With a speculative pool above 2^34 bytes per rank, certain DFlash boots returned stable wrong answers — history-conditioned, reproducible, and initially blamed on scheduling. The real cause: the Xe load-surface-height field is 24 bits; when a KV view's row-block geometry made the effective surface height hit 2^24, loads silently read the wrong rows. Asserts that would have caught it are compiled out of the release kernels.

**What changed.** The vllm-xpu-kernels PR #459 fix, backported as a native-library swap (plus dist-info and source mirror).

**What that produced.** The TP4 corruption class eliminated at the root; DFlash serving qualified at TP4. The diagnostic chain that isolated this (pool-size laws, byte-layout brackets, the exact 2^34-per-rank boundary, a float32-view model, and single-GPU kernel differentials) is the reason the pool geometry in this pack's lane contracts is treated as a qualified property, not a tunable.

## DFlash concurrency correction (correctness)

**What was wrong.** Two defects surfaced under concurrent DFlash serving: capture sizes `[1,8]` covered only the single-request batch shape (C2+ dispatched eager), and eight native convolution-state `slot × stride` intermediates were signed 32-bit — at TP2 the stride is 2,097,152 elements, so slot ≥1024 (about five active requests) overflowed.

**What changed.** A conv-index64 native rebuild plus the widened capture policy. The pack's FP8 DFlash lanes carry Graph64 (`[1,2,4,8,16,32,64]`); the INT4 DFlash TP2/TP4 lanes carry Graph128 (`[1..128]`) because 16 concurrent requests × 8 verifier rows at N=7 need the 128-row graph; INT4 DFlash TP1 carries Graph16 (2 requests × 8 rows). These are launch policies, qualified per lane.

**What that produced.** FP8 DFlash qualified through true active C8 (Running 8 / Waiting 0); INT4 DFlash TP2/TP4 through true resident C16.

## The scale-prefetch pitch guard (correctness — INT4 TP2 + prefix caching)

**What was wrong.** Turning the scheduler up to 2048 tokens (required by the align-mode prefix cache; see below) exposed a latent native-kernel defect on INT4: an unsupported Xe2 W4A16 grouped-GEMM 2D scale-prefetch hint on specific scale-row pitches, which had been invisible at the old 256-token scheduler shapes.

**What changed.** intel/llm-scaler PR #442: skip the prefetch hint on unsupported pitches — no precision change, no fallback, compiled execution stays enabled. Baked as a two-file COPY of the proven bytes (native library + source mirror), promoted retag-only.

**What that produced.** The INT4 TP2 prefix-caching blocker closed; APC on with scheduler 2048 across every lane; INT4 TP1/TP4 re-proven by same-bytes regression probes.

## DFlash prefix-caching correction (correctness)

**What was wrong.** Prefix caching plus DFlash hit two upstream defects: scheduler block-align/resume ordering poisoned reused state (#51113 class), multimodal encoder-cap ordering was wrong (#51603), and XPU pointer handling overflowed int64 wraps in mamba state math at high addresses (#48109 — a hard startup blocker).

**What changed.** The three upstream fixes (#51113, #51603, #48109) as a two-file Python overlay, promoted retag-only after live qualification.

**What that produced.** APC real hits at exact block quanta with DFlash active under restore; vision and chunked multimodal correct under APC; the #51113 poisoning class not reproducible (byte-identical references ×3).

## Mamba sub-block-progress scheduler (reliability)

**What was wrong.** On FP8 DFlash TP2, the effective scheduler clamp (2048 minus reserved slots per request) fell below the 2048-token align block, so fresh prefills could not make progress without stalling decode waves.

**What changed.** The Kimi K3 PR #50000 scheduler hunk (one Python file): fresh blocks progress privately 0→clamp, then a small re-align completes the 2048 boundary; only the aligned boundary is hash-published.

**What that produced.** FP8 DFlash TP2 qualified with APC on at util 0.91 (the calculated capacity: 2705 × 4 MiB blocks, true multi-boundary resident C8). The same branch is live on INT4 DFlash TP2 (align block 2048 > clamp 1952) and dormant at every geometry where the align block already fits the effective budget (TP4, all Base lanes).

## Shared-GDN APC window-index correction (performance — the final image delta)

**What was wrong.** Each of the model's ~30 GDN attention groups rebuilt its own APC window-index columns during shared-prefix metadata construction — identical integer work repeated 30 times, measurable on INT4 DFlash TP4 decode.

**What changed.** One Python file (`vllm/v1/attention/backends/gdn_attn.py`): common window-index columns computed once in the shared build context and reused by all groups; per-group physical block-table gathers unchanged.

**What that produced.** Same-contract INT4 DFlash TP4 combined decode +9.6% (287.1 → 314.6 tok/s); all four DFlash profiles re-qualified on these bytes, one boot each, each boot also carrying a full BetterBench 0.4.0 sweep; final weighted decode 269.3 / 323.3 / 278.7 / 314.6 tok/s (FP8 TP2 / FP8 TP4 / INT4 TP2 / INT4 TP4). This is the image published as `:1.0.0`.

## Context ceilings at the model-native 262144 (launch policy)

All four DFlash TP2/TP4 lanes were live-qualified at the model-native 262144 maximum — whole-context ~260.6K-token needle recovery 3/3 plus ~160K chunked-prefill sanity per lane, `--max-model-len` the only profile delta. A successful native-max qualification covers every lower tier on the identical contract, which is why each lane publishes 32K/64K/128K/256K without per-tier boots. The five Base lanes carry their retained 262144 authorities unchanged. Concurrency, performance, and vision receipts were earned at context 32768 on the DFlash lanes; the 262144 qualification covers admission and whole-context addressability.

## The one-card DFlash lane (launch policy)

INT4 DFlash TP1 is a real qualified lane, not a degraded default: one boot at context tier 65536 with the full battery — near-ceiling admission, APC A/B/B2 with byte-identical repeats, deterministic state including post-vision, single and ordered-two-image vision, chunked multimodal, image-grounded and ordinary tool calls, RestartCount 0. Its defining geometry: the drafter's 8 unsharded KV heads win the unified page size at 8 MiB; the qualified util-0.90 pool is 971 blocks. That pool is why the lane stops at 65536 — 131072 and 262144 are capacity exclusions (real admission demand 976/1664 blocks), not untested gaps. `max_num_seqs` is 2 (Running 2 / Waiting 0 directly observed; C4+ not supported), with Graph16 as the C2 envelope (2 requests × 8 verifier rows at N=7) and an effective scheduler clamp of 2036 (2048 − 6 reserved slots × 2 requests).

## Unified Base + DFlash authority (no image change)

The final stage built nothing. A full installed-runtime census (66,341 files per image, identical path sets, exactly 10 differing files) proved the DFlash authority a strict functional superset of the historical Base authority for the Base lanes: every Base-stage correction present byte-identically, deltas classified as shared corrections (dormant where geometry allows), one DFlash-only file (loaded only under a speculative config, which no Base profile passes), and source mirrors. All five Base profiles then re-qualified on those exact bytes — one boot each, image identity the only launch-contract delta, 5/5 PASS with KV pools exactly the retained values.

The historical Base authority (and every earlier tag) remains pullable by digest but nothing in this pack references them.

## The Base prefix-caching law

Every profile in this pack serves with automatic prefix caching on and a 2048-token scheduler. The cache auto-resolves to align mode (the Mamba/GDN hybrid block must equal the scheduler token budget); restore happens at exact block quanta, and repeated identical requests are byte-identical. INT4 Base TP1 runs util 0.84 rather than 0.82 because at 0.83 its pool landed at exactly the context ceiling with zero margin — the extra hundredth restores the margin class. These are qualified values.

## The final runtime

```text
image     ghcr.io/wu1ff/qwen36-35b-a3b-b70:1.0.0
digest    sha256:8360c9a7e3d8e1bfb890d1f57cd59533958bf5140331d144eb05beb9ac8bc790
pull ref  ghcr.io/wu1ff/qwen36-35b-a3b-b70@sha256:8360c9a7… (immutable, in pack.json)
```

One image, 38 profiles. Base never loads the DFlash support file; DFlash adds the V2 model runner environment and the speculative config, both as launch policy in `pack.json`, never as a second image.

## Qualified models and profiles

| Target | Revision | Modes / topologies |
|---|---|---|
| `Qwen/Qwen3.6-35B-A3B-FP8` | `95a723d08a9490559dae23d0cff1d9466213d989` | Base + DFlash at TP2/TP4, contexts to 262144 |
| `Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound` | `65f69c73f17488236c85c85211f6ba28d7106157` | Base at TP1/TP2/TP4 to 262144; DFlash at TP1 (≤64K) and TP2/TP4 (to 262144) |

DFlash assistant: `z-lab/Qwen3.6-35B-A3B-DFlash` at `f181eece646affea2c38b2765f1aaa01a9734ccd`, N=7, mounted only in DFlash mode.

Vision (XPU-default FLASH_ATTN encoder attention, 2-image limit), tools (`qwen3_coder` parser with auto tool choice), and prefix caching are qualified on every profile the pack exposes.

## Launch-contract factoring

`pack.json` factors each profile the way the qualified launchers were actually written: the runtime layer holds the shared docker shape (devices, by-path bind, IPC, shm, security opts), the 13 shared environment variables (platform device selection, offline HF, oneCCL temporary buffers, the residency shim preload, the pack cache root), and the common `vllm serve` framing including reasoning parsing, the chat template behavior, remote code trust, the multimodal limits and processor sizing, and the tool-call parser. The model layer holds the mount path, dtype (FP8 `float16`; INT4 natural bfloat16), and served name. The DFlash mode adds the V2 runner variable and the speculative config. Each profile adds only its tensor-parallel size, context tier, sequence cap, memory utilization, card affinity, and (DFlash) its qualified graph policy. Environment keys are partitioned across layers — a key never appears in two layers.

## Honest boundaries

- The FP8 DFlash TP2 lane runs util 0.91 — a calculated capacity (2705 × 4 MiB blocks), not headroom to borrow for other lanes.
- FP8 DFlash is qualified through true active C8; FP8 true C16 is not qualified and the profiles cap `max_num_seqs` at 8 accordingly.
- INT4 DFlash TP1 is qualified for C1–C2 only; its 128K/256K exclusion is capacity, not coverage.
- INT4 Base TP2 with prefix caching cannot host 16 simultaneous long (≥2048-token) prefills — a geometry constraint (Mamba alignment blocks), not a defect.
- On classic-graphdriver Docker daemons the image-ID verification convention (manifest digest as `.Id`) does not hold; this pack targets containerd image-store daemons, matching the established Qwen3.8 pack convention.
- The public GHCR package inherits some cosmetic labels from its base-image lineage (experiment-era descriptions); the qualified authority is the digest above and this document, not the label text.
