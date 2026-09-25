# Qwen3.8-27B on Intel Arc Pro B70 — the runtime recipe

This is the technical history of the vLLM runtime that serves Qwen3.8-27B on Intel Arc Pro B70 GPUs here: where it started, what was broken or missing at each step, what I changed, why, and what each change produced — in the order the changes actually stack.

It is written for someone who wants to understand this runtime or investigate the same source areas themselves, so it carries breadcrumbs (upstream projects, PRs, commits, functions, kernels, thresholds). It is not an SHA-by-SHA reproduction guide, not a dependency or container-build manual, and it does not attempt to rebuild Intel's entire software stack.

Each stage below is a correctness fix, a feature port, a performance tuning change, or a reliability fix — labeled as such, because those categories have different shelf lives. Everything except the graph-capture policy (a launch flag) and the final oneCCL temporary-buffer environment (two launch variables) is baked into the final image.

## Where this runtime came from

This work grew out of Intel's pre-release **llm-scaler** vLLM 0.26 integration for Arc Pro B60/B70 GPUs (github.com/intel/llm-scaler). The runnable Intel image actually in hand was the previous generation, `intel/llm-scaler-vllm:0.21.0-b1` (vLLM `0.21.1.dev0+gad7125a43`), whose working Qwen3.8 stack became the behavioral oracle for the ports described below. The pre-release 0.26 line itself never existed as a pullable image during this work — its published integration artifacts were 0.21-era, and Intel's first public 0.26 image tag appeared only after this runtime's core was already built and qualified. So the runtime that ultimately qualified was rebuilt around pinned public sources of the same 0.26 generation, then modified as described here:

| Component | Pinned identity |
|---|---|
| vLLM | `v0.26.0` — commit `568afb3a13806beb53bb2e6bd518269357b237c0` |
| vLLM XPU kernels | `v0.1.11.1` — commit `a6929869587fa6c40dbe393eedc96eeab076bfbb` |
| PyTorch | `2.14.0.dev20260713+xpu` (git `9bfc8327…`) |
| Triton XPU | `3.7.2+git5fcc14d9` |
| OS / user-mode base | Ubuntu 24.04 + Intel OMIX GPU stack (compute-runtime `26.14.37833.4`, Level Zero `1.28.2`, SYCL RT `2026.0.0`, oneCCL `2022.0.0`) |

## The chain at a glance

1. Rebuild on Torch 2.14 / SYCL 2026 (foundation for everything after)
2. XPU memory-info fallback (real free memory)
3. KV-cache admission vs. the cold-compile peak
4. Qwen3.8 FP8 → W8A16 rehome (performance)
5. TP collective segmentation — all-reduce first, then all-gather for MTP1
6. Native XPU kernel closure (the 11-object native layer)
7. dFlash2 port, then the compiled batch-1 walk (feature, then performance)
8. GDN v-head out-of-bounds guard (correctness — the repeated-"!" bug)
9. Draft context-K per-layer normalization (correctness, in the ported file)
10. FP32 residual + aux-tap overlays (precision)
11. TP4 FP16 provider overlays (performance)
12. Level Zero peer-residency shim (containment)
13. Graph-capture policy (launch tuning, not an image change)
14. Patch 009 — TP2 initialization reliability (in the final image)
15. Patch 010 — conservative TP2 concurrent-serving containment (historical; superseded)
16. oneCCL temporary-buffer launch contract — the final TP2 serving reliability fix (launch environment, no image change)
17. dFlash2 exact Gumbel-noise caching (performance — in the final image)
18. dFlash2 widened capture, PIECEWISE64 at TP2/TP4 (launch tuning, not an image change)
19. DFlash2 proposal-lifecycle scheduler fix (correctness — in the final image)
20. XPU Mamba pointer-overflow fix, upstream #48109 (correctness — in the final image; enables DFlash2 ALIGN APC)

→ the final qualified runtime.

---

## Rebuilding on Torch 2.14 / SYCL 2026

**What was wrong.** The pre-release 0.26 line declared a Torch 2.12 XPU build. On B70 that build carried a 20 MiB allocator segment floor (PyTorch `c10/core/AllocatorConfig.h`) that made every graph capture balloon to ~2.25 GiB per worker, and its speculative width-8 decode step sat at a 28.49 ms median. It also predated the 29-argument `gdn_attention` operator schema that vLLM 0.26's speculative decode expects from the XPU kernels.

**What I changed.** Rebuilt vLLM 0.26 and its XPU kernel package on the Torch `2.14.0.dev20260713+xpu` nightly with SYCL RT 2026.0.0 and the pinned Ubuntu 24.04 + Intel OMIX user-mode foundation. This is the base every later change applies to.

**What that gave me.** Graph-capture memory collapsed 2.25 → ~0.35 GiB per worker (−85%); the width-8 step median dropped to ~24.4 ms; steady decode on the reference validation lane rose 247.6 → 294.7 tok/s — with bit-exact canonical output, so a pure rebase, not a behavior change.

## XPU memory reporting returned zero

**What was wrong.** The pinned B70 userspace driver (compute-runtime `26.14.37833.4`) does not advertise the usable-memory extension. The kernel package's `getMemoryInfo` queried it unchecked and consumed an untouched zero-filled field, so every device reported `free = 0.00 GiB` — silently breaking KV-cache admission.

**What I changed.** The upstream compatibility fallback from vllm-xpu-kernels PR #499 (commit `6231eb3290…`). One subtlety worth knowing: the fallback is compiled into the kernel package's `_C.abi3.so`, and a from-scratch rebuild of the native closure *without* that patch regresses `getMemoryInfo` back to zero on this driver. The final runtime deliberately installs the fallback-carrying object.

**What that gave me.** Real free memory (~31.7–31.9 GiB per device at idle) and working KV admission; deterministic TP2/TP4 oneCCL correctness checks passed from then on.

## The KV-cache budget was paying for a dead compiler

**What was wrong.** On XPU, ahead-of-time compilation on a cold start transiently consumed ~2.3 GiB — freed long before serving. vLLM retained that transient high-water mark in the peak accounting used for KV-cache sizing, so the engine computed a negative KV budget and failed startup before any graph capture. The memory was gone, but the bill remained.

**What I changed.** The accounting semantics of upstream vLLM PR #54209 (commit `fd0fdade…`; still open at the donor revision): keep the legitimate pre-compile peak, reset the allocator high-water mark after compilation, measure the real compiled profile forward, then fold the earlier peak back in with `max()`. Adapted to activate only for the standard-allocator XPU AOT path (`vllm/compilation/monitor.py`, `vllm/utils/mem_utils.py`).

**What that gave me.** Graph-disabled and width-1 piecewise lanes both admit ~1 GB of KV and reach readiness; width-1 capture/replay proven with exact token parity.

## Qwen3.8 FP8: requantize once, serve W8A16 (performance)

**What was wrong.** Vanilla v0.26 executed the checkpoint's serialized `[128,128]` block-FP8 weights through the apply-time block-scaled route at every linear call. Numerically correct, materially slow on B70.

**What I changed.** At load time, dequantize each TP shard once, requantize to persistent per-tensor E4M3, and serve through the existing W8A16 kernel `_xpu_C.fp8_gemm_w8a16` (vLLM's `XPUW8A16FP8LinearKernel`) — zero apply-time requantization cost.

**What that gave me.** All 256 intended FP8 linears converted exactly once, deterministic token parity, and +65% decode throughput (65.4–65.8% across p512/p4096/p8192 decode lanes). The single biggest performance win in the chain.

## Large TP collectives wedge the device

**What was wrong.** Live TP prefills completed a contiguous BF16 `[256,5120]` (2.5 MiB) native all-reduce but died at `[512,5120]` (5 MiB) with `UR_RESULT_ERROR_DEVICE_LOST`. Native oneCCL submissions above ~2.5 MiB can wedge the B70 device engine — no retry, no fallback, just a lost device.

**What I changed.** In `XpuCommunicator.all_reduce`, split any contiguous result larger than 2,621,440 bytes into increasing-offset views of at most that size, reduced in order: one logical input, one logical result, no algorithm changes. MTP1 later exposed the same boundary on the gather side — its gathered `mtp.fc` output (`ColumnParallelLinear(..., gather_output=True)`) all-gathers past the limit during prefill — so `XpuCommunicator.all_gather` got the equivalent segmentation (bound constant `_XPU_ALL_GATHER_MAX_NATIVE_OUTPUT_BYTES = 2,621,440`; ordered dim-0 chunks, then concat). All-reduce was fixed first; all-gather needed the same treatment before MTP1 was usable.

**What that gave me.** TP2/TP4 lanes from p512 to p8192 plus health gates pass, MTP1 prefills pass on TP2/TP4, and segmentation ships in every later image.

## The native kernel closure

**What was missing.** The model's XPU execution path — GDN attention, grouped GEMM, the W8A16 GEMM, attention and logits kernels, the XPU memory allocator — lives in vllm-xpu-kernels' native objects, and they must be linked against the exact Torch/SYCL they run with. Eleven shared objects in total, including `_xpu_C.abi3.so` and the GDN attention companion `libgdn_attn_kernels_xe_2.so`.

**What I changed.** Built the closure fresh from the pinned `vllm-xpu-kernels` commit with the pinned oneAPI 2026 toolchain, CPU-only, gated on the two things that matter: `torch.ops._xpu_C.gdn_attention` exposing exactly the 29-argument speculative schema (`num_spec_decodes`, `spec_query_start_loc`, `num_accepted_tokens`), and correct SYCL linkage.

**What that gave me.** The native layer everything else depends on, with rebuilds proven deterministic (all eleven objects byte-identical across rebuilds). This is also the object the GDN correctness fix below was applied to — surgically, which is why the closure's structure matters.

## dFlash2 simply wasn't there

**What was missing.** dFlash2 — this model family's five-layer draft-model speculative decoding (K=7) — does not exist anywhere in the vLLM 0.26 tag. The 0.21-generation Intel runtime had a working implementation; that runtime was the behavioral oracle for what correct output looks like.

**What I changed.** Ported the dFlash2 V1 path: the draft-model definition, proposal generation, standard lossless rejection sampling, candidate-head GEMM repair, context-KV construction, and probabilistic sampling — roughly 19 files across `vllm/v1/spec_decode` and the model executor.

**What that gave me.** Both-topology qualification passed (semantic, widening, and canonical gates), and MTP1 coexistence stayed bit-exact — the port is inert when `method=mtp` is requested.

### The compiled batch-1 walk (performance)

The probabilistic walk (`select_dflash2_path_with_noise`) ran eagerly at ~1.47 ms per decode cycle — about 3.9% of the whole cycle in the batch-1 lane. Running the exact same walk through `torch.compile` (batch 1 only, four value-preserving clones to isolate compiler annotations; batch > 1 stays eager; values and RNG unchanged; greedy path untouched) cut it to ~0.22 ms. Measured real-workload probabilistic gains: +3–4% median on matched lanes.

## The repeated-"!" failure: GDN v-head out of bounds

**What was wrong.** MTP1 output collapsed into a repeated "!" token. This was not sampling — a kernel was corrupting weights. In the GDN companion, `chunk_prepare_kernel` computed `v_head_id = total_sg_id / chunk_range` unbounded. On B70 the Xe2 chunk-prefill launch geometry is 32 work-groups × 32 sub-groups = 1024 sub-groups; at TP4 the model has 12 local v-heads, and 1024 mod 12 = 4 — so sub-groups 1020–1023 computed `v_head_id == 12` on a 12-head tensor. They read past `A_log[12]`/`dt_bias[12]` (garbage → ±e38) and wrote a full head-row past the FP32 temporary (~512 B past the allocation on a 75-token prefill). In the MTP1 runtime layout the adjacent allocation was `layers.1.mlp.gate_up_proj.weight_scale` — corrupted on all four ranks → nonfinite logits → token 0 forever. Base and dFlash2 fired the same illegal write, but their allocator layouts left it silent.

**What I changed.** The upstream guard (vllm-xpu-kernels commit `18b78776a`, 2026-07-09, present in current upstream; the pinned `a692986` predates it): early-return when `v_head_id >= num_v_heads`. Applied as a surgical rebuild of only the companion object — `chunk_prepare_kernel` is compiled into the shared `libgdn_attn_kernels_xe_2.so`, which `_xpu_C` links, so replacing `_xpu_C` alone never executes the guard (learned the hard way). One compile edge plus one link edge; every other native object stayed byte-identical; exported ABI unchanged (538/538 symbols).

**What that gave me.** The one-token reproducer emits the correct token on 4/4 ranks with the weight scale intact; sustained MTP1 output is bit-exact against the Base oracle at every index; the first *valid* MTP1 medians became FP8 82.4 / INT4 110.1 tok/s (earlier MTP1 numbers were measured on corrupted weights and are worthless).

## Draft context-K: five normalizations, not one

**What was wrong.** Draft acceptance sat at 444/497 (89.3%) when it should have been essentially total. vLLM 0.26 groups the five draft layers' learned context-K normalization into one `[5,ctx,2,128]` XPU `rms_norm` call — but the XPU kernel indexes a single weight vector, so layers 1–4 silently used layer 0's K-norm weights.

**What I changed.** One existing RMSNorm call per draft layer, using that layer's own `[128]` weight vector. Five calls instead of one, no new kernel, execution and sampling untouched (`_normalize_context_k` in the draft model definition — a correction inside the file the dFlash2 port introduced).

**What that gave me.** Full 448/448 acceptance over 64 rounds restored, the canonical output hash reproduced, ~+11% throughput on the same validation lane.

## FP32 residual and aux-tap overlays

FP16 residual rounding inside the fused RMSNorm degraded deep-layer tap precision, and the draft auxiliary-tap geometry needed FP32 intermediates it wasn't getting. Two small file overlays fix it: `GemmaRMSNorm.forward_native` with FP32-residual semantics, and an FP32 aux-tap interface for the draft path — both precision corrections carried over from the 0.21 oracle's proven behavior. Tap precision restored; both overlays ship in every qualified image.

## TP4 FP16 provider overlays (performance tuning)

To be clear about category: the recent stages above this were correctness; this is tuning. The qualified TP4 FP8 Base profile wanted the faster frozen FP16-native M=1 linear and page-attention paths that v0.26 does not select by default. Eight provider files overlaid onto the vLLM tree: ESIMD W8A16/GEMV providers, GDN bundle helpers, and an FP16 Eagle page-attention policy. Result: +2.5–4.1% composite across p512/p4096/p8192 at TP4 — the retained TP4 FP8 Base profile. Correctness gates were unchanged and re-passed; no behavior was claimed fixed.

## Level Zero peer residency: ~101 GiB of host RAM for nothing

**What was wrong.** On multi-B70 serving, the upstream Level Zero runtime mirrors large non-oneCCL device allocations into host memory. A TP4 Base launch sat at ~101 GiB of driver-accounted host residency merely loaded and idle; unshimmed serving ran ~110.5 GiB. Host memory burned with zero serving benefit — the model and KV cache live on the devices.

**What I changed.** A narrow loader-interposition shim (`LD_PRELOAD`, binding `libze_loader.so.1`, exporting no dynamic symbols of its own) that tracks device allocations through `zeMemAllocDevice` / `zeDriverGetMemAllocProperties` and suppresses the duplicate host peer mappings on non-CCL-tracked allocations. To be precise about what it is not: it does not offload the model or the KV cache, does not replace or modify oneCCL, and does not fix anything inside Intel's driver. It contains the pathology at the loader boundary.

**What that gave me.** Loaded host memory pinned to 4.0–4.5 GiB across all serving lanes (~92 GiB of peer residency suppressed in the measured validation run), full release on shutdown, steady-decode cost bounded at ~1.4–3.8 tok/s. Bypassing the shim would buy only +1.7–3.0% decode — a bad trade for ~101 GiB of RAM.

## Graph-capture policy (launch tuning, not an image change)

Sixty-five PIECEWISE graph wrappers cost 1.30–1.39 ms of cumulative host dispatch per target forward in Base/MTP1 decode. Capturing the decoded shapes as one FULL graph — `cudagraph_mode=FULL_AND_PIECEWISE`, capture lists `[1]` (Base) and `[1,2,4]` (MTP1) — removes that dispatch. dFlash2 stays PIECEWISE (FULL measured neutral there, +0.17%); its capture list is topology-dependent since the 2026-09-12 widening — see stage 18 below. FP8 Base +2.66% and INT4 MTP1 +3.71% median on matched edits. This is a launch-flag change; the image bytes are untouched.

## Patch 009 — TP2 initialization reliability

**What was wrong.** TP2 at 32K/64K contexts could intermittently die during engine initialization — despite memory fitting comfortably. During warm-up, the segmented native collectives (from the segmentation stage above) submit very long uninterrupted chunk bursts: ~1,120 chunk collectives inside the single 4096-token memory-profiling forward, ~410 chunks inside one 4096-row logits all-gather. On B70 these intermittently wedge a device engine; the GuC heartbeat then kills the stalled job, the driver bans the context, and the next blocking call returns `UR_RESULT_ERROR_DEVICE_LOST`. The race was probabilistic; memory capacity was never the limiter.

**What I changed.** Warm-up path only. A `set_warmup_serialize()` flag in the XPU communicator that inserts a `torch.xpu.synchronize()` after every native chunk in the segmented collective loops while set; the worker sets it around memory profiling, compile warm-up, and kernel warm-up, and clears it before graph capture and before serving; plus five sampler-stage drains. The serving path is byte-identical to the parent at this stage, and the production `CCL_SYCL_ALLREDUCE_TMP_BUF=1` setting is retained.

**What that gave me.** TP2 32K ×3 and 64K ×3, plus further clean 64K inits, real probabilistic requests with healthy speculative counters, a ~47K-token real prompt, and the TP4/64K regression — zero device-loss events in every window. Cost: ~4 s of one-time serialized initialization.

## Patch 010 — TP2 concurrent-serving reliability (historical containment, superseded)

Patch 009 fixed initialization; concurrent serving then exposed a second, related collective-liveness problem one layer up.

**What was wrong.** At concurrency ≥ 2, TP2 dFlash2 serving died silently: generation throughput to zero, RPC timeouts, `EngineDeadError` — with no OOM, no `DEVICE_LOST`, and all GPUs reporting normal. Above the captured widths every forward runs eager and submits long back-to-back native collective bursts (~12–27 ops per decode step; ~90–130 all-reduces per mixed prefill+decode step). These wedge a device engine with the same kernel-internal circular-wait mechanism patch 009 documented — except silently: the GuC heartbeat is not tripped. Device-side ground truth: both ranks' submission streams were byte-identical (48,991 collectives), but device completion stopped mid-forward, inside a plain `(121,5120)` fp16 all-reduce. The ranks had done everything right; the device just stopped retiring their work.

**What I changed.** One file: a serving-path burst bound in `XpuCommunicator.all_reduce` / `.all_gather`. Every native collective (or segment chunk) whose contribution is ≥ 96 KiB (`_XPU_SERVE_DRAIN_MIN_BYTES`) is followed by `torch.xpu.synchronize()`. The floor sits above the largest graph-replayed collective — `(8,5120)` = 80 KiB — so graph replay and sub-floor eager ops are untouched; graph capture is explicitly guarded; patch-009's warm-up discipline is preserved. All-gather must be bounded too: all-reduce-only variants were refuted by controlled A/B (one hung at concurrency 4, another on the first request at concurrency 1 — the undrained logits all-gather is itself a wedge point).

**What that gave me.** The minimal 2-client reproducer went from hanging 3/3 to 5/5 clean episodes (111 requests, zero hangs). BetterBench concurrency sweep: **48/48 at every level — 1, 2, 4, 8, 16** — with throughput now actually scaling; single-stream 160/160; prefill 2K–64K pass (64K = a 47,097-token prompt); TP4 dFlash2 64K regression pass.

**The honest cost** (measured, accepted at the time, not tuned away): TP2 SQL 178.7 → 160.3 tok/s (−10.3%); single-stream decode mean 91.9 → 88.3 tok/s (−3.9%); prefill −5.4% (64K) to −10.5% (2K). As with patch 009, this bounds the trigger pattern of a driver-level stall; the stall mechanism itself is not claimed fixed.

This established that the serving/liveness problem was containable — but the serialization cost was real, and follow-up work (below) showed the drain was not the final answer. Patch 010 is retained as historical evidence and a known fallback containment; it is **not part of the final serving contract**.

## The oneCCL temporary-buffer launch contract — the final TP2 serving fix

The last step is not a code change at all: it is a launch-environment correction on the patch-009 runtime, and it retires patch 010 from the serving path.

**How the investigation got here.** Two cadence experiments (Candidates A and B) tried to narrow patch-010's ≥ 96 KiB drain so that only the actually-vulnerable op patterns paid the synchronization cost. Both recovered most of the lost performance during warm, qualified runs — and both failed a fresh-server promotion gate by reproducing the cold-server wedge, one of them even in the batch-size-1 posture. That result proved two things: the vulnerable state also exists during early B=1 serving (fresh eager collectives before the engine settles), and synchronization *narrowing* was not the real solution. The attention then moved from when collectives are submitted to *which buffers* they operate on.

**What was wrong.** With `CCL_SYCL_ALLGATHERV_TMP_BUF=0`, oneCCL SYCL gathers — including medium/large shape-variable gathers on the serving path — can operate directly on user buffers. Those buffers are fresh or recycled allocations whose shape changes with the batch, so every serving-time gather forces Level Zero IPC exchange/open behavior on memory the communicator has never seen. That dynamic, serving-time direct-IPC churn on changing user gather buffers is the strongly supported trigger of the cold TP2 liveness failures.

**What I changed.** The launch contract only:

```text
CCL_SYCL_ALLREDUCE_TMP_BUF=1    (already the production posture)
CCL_SYCL_ALLGATHERV_TMP_BUF=1   (the correction; returns to the v21 value)
```

With `CCL_SYCL_ALLGATHERV_TMP_BUF=1`, those gathers are staged through the persistent communicator-owned temporary-buffer pool allocated once at communicator initialization — removing serving-time direct IPC churn on the changing user gather buffers instead of serializing around it. No drain, no cadence code, no rebuild: the final runtime is the patch-009 image bytes, unchanged.

**Classification — say this precisely.** The root-cause mechanism is **strongly supported**, not absolutely source-level confirmed: no oneCCL/xe source defect was definitively located and repaired. What is proven is that the buffer-routing change eliminated the observed cold TP2 liveness failures across the whole qualified campaign.

**What that gave me.** On the exact patch-009 image (`68f1d8a8…`) with both variables set:

- 3/3 independent fully cold C1→C2 episodes PASS (fresh container, cold caches, every episode: hardware 4/4 normal).
- 3 independent fully cold full concurrency sweeps — every sweep 48/48 at C1, C2, C4, C8, and C16; campaign-wide 1008/1008 requests.
- Zero occurrences of any failure signature: no `shm_broadcast` timeout, no `sample_tokens` timeout, no `EngineDead`, no `DEVICE_LOST`, no GuC timeout, no xe reset, no device coredump; hardware 4/4 normal throughout.
- Performance with no containment cost: TP2 SQL 178.6 tok/s / 44.79 ms/round (100% draft acceptance); 64K prefill 2,566.9 tok/s on a 47,097-token prompt.

Representative fully cold concurrency sweeps (aggregate tok/s):

| Sweep | C1 | C2 | C4 | C8 | C16 |
|---|---|---|---|---|---|
| A (qualification) | 76.2 | 95.8 | 173.2 | 297.8 | 288.8 |
| B | 75.7 | 92.0 | 175.0 | 284.1 | 296.7 |
| C (daytime confirmation) | 77.7 | 98.8 | 175.1 | 287.8 | 294.9 |

(A later host-side audit showed a long-lived warm server can sit above these cold bands — C1 ~78, C2 ~104, C4 ~182, C8 ~305 — warmth, not host topology, is that gap; see the host note below.)

Evidence: `repro/v26-tp2-onexccl-ipc-cache/` (cold-discriminators: the 3-episode cold gate, full qualification sweeps, SQL gate, and the daytime confirmation).

### Host-side tuning was audited and nothing was retained

The same rig (AMD EPYC 7443, 24C/48T, NPS=4) went through a full host-side audit during this work: CPU frequency governor, CPU/thread affinity of the engine and worker threads, GPU IRQ affinity, automatic NUMA balancing, idle states, and the oneCCL worker-affinity controls. The kernel's default placement/frequency policy was best overall. Explicit pinning could buy ~5–6% at C1 but cost ~5–8% at C2 and ~3–4% at C4/C8 — a trade this runtime's serving profile cannot accept. No persistent host setting belongs in the runtime requirements; the machine stays at its default posture (evidence: `repro/v26-tp2-host-latency/`).

## dFlash2 exact Gumbel-noise caching (performance — in the final image)

The eager dFlash2 proposal walk draws its Gumbel noise deterministically from per-request seeds. The original implementation rebuilt that noise every round: it copied the GPU-resident seed tensor to the host (`seeds.detach().to("cpu")` — a GPU→CPU synchronization boundary on the serving path each round), re-seeded a private CPU generator per request, and re-drew the identical `[num_steps, top_k]` values — identical because each request's seed is immutable once assigned. Every round paid the same sync plus the same CPU construction for the same numbers.

**What I changed.** One file (`vllm/v1/spec_decode/dflash.py`): a `_cached_dflash2_gumbel_noise` method that caches the exact per-request noise keyed by request ID, prunes completed requests, and rebuilds only on geometry changes (batch composition or shape). The seed derivation (`_dflash2_sampling_seeds`) still runs every round, so the original GPU RNG draws and generator state advancement are preserved exactly; the proposal distribution and the rejection sampler are untouched. The image delta vs the patch-009 authority is exactly this one file (the parent's 48 layers are an exact prefix of the 49).

**What that gave me.** Exactness first: 44/44 transition cases on the packaged image (4 seeds × 11 add/remove/reorder patterns, seeded/unseeded mixtures) preserve the exact noise tensors, the exact GPU RNG states, and the exact subsequent draws; greedy temp-0 output is token-exact 15/15 vs the same-campaign control. Performance: ~2.9–3.2 ms/update recovered in the investigative pairs, and the full benchmark confirmed ≈2.9 ms (−6.0%) single-stream update p50 in **every** category, with additional warm-concurrency gains at every level. No claim is made that stochastic (seeded temp>0) output becomes deterministic — it was not deterministic before the change either.

This change alters submission timing on the serving path, so it received fresh cold qualification of its own: readiness-only cold initialization, 3/3 fully cold C1→C2 episodes, the full benchmark, the SQL lane (byte-identical canonical output, 100% acceptance), a ~47K-token prefill probe, and a full current-boot health campaign with zero new fault-class events — all on the exact promoted bytes.

## dFlash2 widened capture — PIECEWISE64 at TP2/TP4 (launch tuning, not an image change)

dFlash2 speculative decoding with N=7 means one request produces **8 verifier rows** per forward (7 drafted + 1 bonus). Under the original dFlash2 capture list `[1,2,4,8]`, a single request (8 rows) replayed inside a captured graph, but two concurrent requests (16 rows) exceeded the 8-row ceiling and fell back to eager execution — the C2/C4 concurrency cliff.

**What I changed.** The launch flag only: `cudagraph_mode=PIECEWISE`, `cudagraph_capture_sizes=[1,2,4,8,16,32,64]`, `max_cudagraph_capture_size=64` for dFlash2 at **TP2 and TP4**. TP1 keeps the originally qualified `[1,2,4,8]` ceiling (widened capture was qualified at TP2/TP4 only). Base and MTP1 keep their independently qualified FULL_AND_PIECEWISE policies.

**What that gave me.** The C2 cliff disappeared: short-lane C2 update −31% / C4 −23%; the authoritative benchmark measured C2 +37.8% and C4 +26.0% aggregate (before the noise cache), C1 unchanged, and greedy temp-0 token-exact 15/15. Qualified with 3/3 fully cold C1→C2 episodes, the full benchmark, the SQL lane (178.70 tok/s, byte-identical output), and clean health windows throughout. At TP4 the widened policy is part of the qualified contract as well (SQL 260.0 tok/s, C16 541.2 aggregate).

---

## DFlash2 proposal-lifecycle crash fix (2026-09-21, in the final image)

An external report surfaced a probabilistic-dFlash2 engine death inside long agentic conversations: once histories grew through the 62–65K-token region, a verification round arrived whose speculative slots had no proposal distribution `q` behind them, and the strict verifier (`GPUModelRunner._get_spec_decode_draft_probs()`) correctly killed the engine rather than sample against a wrong distribution. The proposal cache has a single-round lifetime; two scheduler-side lifecycle holes let slots outlive their `q`:

1. **Tail/drafter-fit boundary.** The drafter refuses to propose when the batch's max optimistic sequence length plus the 8 query tokens (7 speculative + 1 dFlash bonus) exceeds the effective drafter context (boundary 65528 at the 64K tier). The parent `AsyncScheduler` still assigned `[-1]*7` placeholder spec slots on those rounds — exactly the rounds the worker produces null `q`.
2. **Non-preemptive deferral.** Any path that leaves a RUNNING request with pending spec slots out of a step's schedule (token-budget exit, allocation failure, zero-token skip, async max-tokens skip, decode-eligibility skip, pause) drops it from the worker's persistent batch, so the next proposal snapshot cannot contain a `q` row for it.

**What I changed.** Two Python scheduler files only (`vllm/v1/core/sched/scheduler.py`, `async_scheduler.py`; retained as `patches/011-dflash2-proposal-lifecycle/`). The scheduler now mirrors the runner's drafter-fits projection exactly and assigns placeholder slots only when a `q` can exist; a single post-running-loop sweep invalidates the pending slots of requests not scheduled this step (the same defense preemption already had). A suppressed round degrades to plain 1-token decode and a fresh proposal follows when the drafter fits again. Nothing else moved: the strict verifier, the proposer, and the rejection sampler are byte-identical; no fallback distribution is ever substituted.

**What that gave me.** The deterministic reproducer flips from crash to clean suppression exactly between 65528 and 65529; 80/80 sequential agent steps through the original crash window (histories to 64,340 tokens, one live deferral handled cleanly); a 126,483-token long-context prompt 512/512; TP2 and TP4 both pass; 103,264 draft rows with 0 cache misses and 0 unmatched `q`; SQL continuation +14% (the documented warm-server upside ceiling). Inertness for Base/MTP1/non-probabilistic methods was proven deterministically against the exact bytes (42-scenario parent-vs-candidate differential, identical digest), not re-qualified by server campaigns. DFlash2 APC is OFF in every lane of this promotion — APC enablement is the next separate campaign.

---

## DFlash2 ALIGN APC + the XPU Mamba pointer fix (2026-09-22, in the final image)

The last blocker for prefix caching on this hybrid-GDN runtime was an address bug, not a scheduler bug. Qwen3.8's hybrid config resolves the Mamba cache mode to ALIGN whenever prefix caching is on; the ALIGN manager (`MambaSpecDecodeGPUContext.initialize_from_forward_context`) stores raw device pointers into `torch.int64` tensors. `data_ptr()` is an unsigned 64-bit virtual address and the Level Zero USM allocator routinely returns VAs ≥ 2^63, so the int64 store routes through a C `long long` and dies with `ValueError: Overflow when unpacking long long` (exactly `mamba_utils.py:649` `state_base_addrs` and `:732` `block_table_ptrs`). CUDA survives the same upstream code by pointer-range luck.

**What I changed.** One file: the retained upstream PR #48109 fix (`patches/012-xpu-mamba-pointer-overflow/`) — a `_reinterpret_u64_as_i64` helper wrapped around exactly those two stores. Subtracting 2^64 yields the same 64-bit pattern in two's-complement signed form; dtypes, call sites, and the Triton consumer ABI (int64 load → offset arithmetic → bit-cast to pointer) are unchanged. For pointers < 2^63 the helper is the identity, and for pointers ≥ 2^63 the parent crashed — there is no input where both complete with different bits, which is the Base/MTP1 inertness argument (their launch contracts are untouched).

**What that gave me.** DFlash2 production APC turned ON (`--enable-prefix-caching`, effective ALIGN via the normal config path), qualified 2026-09-22 with verdict PROMOTE: 12/12 delta lanes across INT4 TP1/TP2/TP4 and FP8 TP2/TP4, standard + uncensored artifacts each, with real FA+Mamba prefix-cache hits, tool calls, restart episodes, 951/951 strict proposal-q rows matched (0 misses) on the new lanes, and a 12/12 health campaign with zero fault-class events. Two utilization envelopes were corrected as pack-envelope fixes (independently required by the APC-off parent, not an ALIGN tax): INT4 DFlash2 TP1 32K → 0.91, INT4 DFlash2 TP2 256K → 0.86; all other DFlash2 profiles stay 0.82. Effective posture: APC ON, Mamba cache mode ALIGN, probabilistic N=7, standard rejection, AsyncScheduler. Text and tools only at promotion time — vision was resolved separately the same day without touching these bytes (next section).

---

## SD conv-state migration fix (2026-09-25, in the final image)

Two files, `patches/013-sd-conv-state-migration/`: both speculative-decode
conv-state copy sites now source the full conv-state block from the
per-token block-table column of the aligned accepted position
(`get_conv_copy_spec` SD branch → `block_ids[cur_block_idx +
num_accepted_tokens - 1]`; `_copy_mamba_state_block` SD conv branch →
`state[bt[src_col + token_bias]]`, the same source column the exact
temporal branch uses). The shipped code used a row-offset within the
running column, corrupting every acc≥2 ALIGN advance/checkpoint — the
root cause of the spontaneous `!`-degeneration regression (closed
2026-09-25; full chain in CURRENT.md §1.19 and
`issue-forensics/apc/align-address-fix/` + `bang-regression-20260924/`).
acc=1/bias=0 copies are byte-identical to the parent (offline-proven).
Qualification on the exact promoted bytes: b6 instrumented gate + §N+5
battery (TP4 FP8 unc 262144) + representative production lanes FP8 TP2
64K std and INT4 TP1 32K @0.91 — all PASS with real APC hits and normal
acceptance; the historical 12-lane matrix deliberately not rerun.
DFlash2 posture, launch contract, envelopes: unchanged.

## Vision: VISION-LAUNCH-FLAG-FIX (2026-09-22, launch contract only — no image change)

Vision input died on every 1.0.x lane with `RuntimeError: could not create a primitive` out of oneDNN, reached from the ViT encoder's scaled-dot-product-attention. The mechanism, proven from the frozen 1.0.3 bytes plus one diagnostic boot:

```text
v0.21-era tuning flag --mm-encoder-attn-backend TORCH_SDPA
  -> copied forward verbatim into every v0.26 recipe/launch contract
  -> on image input the head-dim-72 ViT SDPA path asks oneDNN 3.12 for
     a primitive it cannot create
  -> engine death on the FIRST image request (text-only traffic never
     reaches the encoder, which is why text/tools qualification never
     saw it)

remove the stale override
  -> XPU platform default selection: FLASH_ATTN
  -> existing vllm-xpu-kernels ViT flash path (kernels byte-identical
     to the vision-qualified Qwen3.6 r9-c1 runtime)
  -> vision works
```

The fix is exactly the removal of `--mm-encoder-attn-backend TORCH_SDPA`
from the launch contract (recipes, pack runtime command). No explicit
`FLASH_ATTN` override was added — the qualified posture is *no explicit
override*, letting the XPU default select FLASH_ATTN. No source patch, no
image rebuild was needed AT THAT TIME: the then-runtime bytes remained
`c0c9b8f3…` (tag 1.0.3). (Since 2026-09-25 the ACTIVE runtime is
`f3006020…` / tag 1.0.6 — the SD conv-migration promotion, §1.19 of
CURRENT.md — carrying the identical no-explicit-override posture.)

**External-report history corrected.** The external deployment capture
previously cited as a "working vision contrast" was not one: its first
image request reached patch embedding (the Triton `_bilinear_pos_embed_kernel`
frame in its traceback) and then died at the SAME TORCH_SDPA
primitive-creation path. The old inference that the presence of the Triton
kernel in the trace proved working vision upstream is wrong — it only
proved the request got as far as patch embedding before the encoder
attention backend failed. Both our lanes and the external capture fail at
the identical defect; removing the stale flag is the complete fix.

**What is live-proven.** On INT4 DFlash2 TP2 65536 @util 0.82, APC ON
(effective ALIGN), the diagnostic boot (2026-09-22, container
`qwen38-visionfix-c1`, image identity-gated `c0c9b8f3…`) proved: text
PASS; vision cold PASS (512×512, grounded answer, 1.9 s); vision warm
PASS (second image, 1.4 s); APC + vision PASS (11K-token text producer →
image-bearing consumer, engine prefix-cache hit 45.1%, SpecDecoding live);
vision-grounded tool call PASS (parsed structured `report_colors`
arguments); health 200; zero fault signatures; whole-battery memory
envelope ≤ ~0.5 GiB/device (256-patch fixtures — a floor, not a ceiling).

The backend fix itself is vision-path/model-level and carries by
launch/source equivalence to every other profile (the flag was
runtime-level and identical across Base/MTP1/DFlash2 recipes). It is NOT
separately live-qualified on every Base/MTP1/uncensored/topology
combination, and this document does not claim it was.

**TP1 vision boundary (documented, not re-measured).** INT4 TP1 32K
@util 0.91 has only ~0.9–1.1 GiB residual per device. The TP2 diagnostic
measured ≤ ~0.5 GiB/device for small images, but that is a floor and
larger images cost more. Therefore INT4 TP1 32K: text/tools qualified;
the vision backend fix applies by launch equivalence; the vision memory
envelope is NOT qualified; large-image/multimodal use is not recommended
on this marginal profile. No boot was spent resolving it.

---

## Short-prefix APC behavior (hybrid FA+Mamba geometry — expected, not a defect)

DFlash2 APC uses 1024-token hybrid cache blocks. Because speculative
decoding drops one EAGLE lookahead block and Mamba state is materialized
only at safe chunk boundaries (2048-token cadence under the production
ALIGN prefill), very short shared prefixes may not produce reusable
cache state. Measured live on the TP2 production geometry (2026-09-22
closeout matrix, 15 exact tokenizer-verified boundaries on the final
public-path boot; full table in the runtime journal §24 and
`issue-forensics/apc/short-prefix/SHORT-PREFIX-RESULT.md`):

```text
with B = floor(shared_tokens / 1024):
  hit = 1024 * (B - 1)        normally
  hit = 1024 * (B - 2)        when the divergence lands in the FINAL
                              token of a hash block
                              (shared_tokens mod 1024 == 1023)

shared < 2048        ->  hit 0
shared 2048          ->  hit 1024        shared 3071  ->  hit 0 (edge)
shared 3072          ->  hit 2048        shared 4096  ->  hit 3072
shared 5120          ->  hit 4096        shared 5119  ->  hit 2048 (edge)
shared 6144          ->  hit 5120        shared 7168  ->  hit 6144
```

The staircase restores roughly `shared − 1024` tokens (one block is
always dropped for the EAGLE lookahead), stepping 1024 tokens at a
time. Two adjacent shared lengths can differ by one block when the
divergence sits in the last token of a hash block. A zero hit below
~2K shared tokens (and the one-block dip at block edges) is APC
geometry working as designed — the fixed-point reconciliation restores
only a prefix every surface can vouch for simultaneously. It is not an
APC failure and needs no user action; prefixes ≥ 2048 restore
normally. (An earlier revision of this section stated a coarser
2048-step staircase with a ~3072 threshold — that reading was an
artifact of prior evidence landing only on odd block counts; the 1.0.5
documentation corrects it to the measured law above.)

## INT4 TP2 256K — maximum-capacity profile warning

INT4 DFlash2 (and Base/MTP1) TP2 at 262,144 context is a
maximum-capacity profile. Exact qualified numbers (2026-09-22 §20
campaign): KV capacity 264,248 tokens at util 0.86 — a capacity ratio of
~1.01× the configured context window; runtime peak free memory in the
232K test ~1.8 GiB/device. The runtime is stable; the limitation is KV
concurrency at maximum context. INT4 TP2 256K is intended for one
near-full-context request at a time — useful concurrency should not be
expected when requests approach 256K tokens. (INT4 TP4 256K and FP8
TP4 256K have materially larger capacity ratios and are not covered by
this warning.)

---

## The final runtime

What came out the other end of that chain:

```text
local/qwen38-v26-dflash2:convfix-c1
Image ID: sha256:f3006020add52dedfed08947b05234ef4c058b47158afe61c0e79f9ef60a6f4a
(publication: ghcr.io/wu1ff/qwen38-27b-b70:1.0.6
 @sha256:f3006020add52dedfed08947b05234ef4c058b47158afe61c0e79f9ef60a6f4a — promoted
 2026-09-25 retag-only, no rebuild; the previous authority
 c0c9b8f382298bdd90f78ae2f4700637241c7a8e2b6c7ab591dc933c76b73cbf
 remains published/pullable as tag 1.0.3 and by digest)
```

These are the patch-009 bytes (stage 14) plus exactly four installed deltas since — the dFlash2 exact Gumbel-noise cache (stage 17), the DFlash2 proposal-lifecycle scheduler fix (stage 19), the XPU Mamba pointer-overflow fix (stage 20, upstream #48109), and the SD conv-state migration source-column fix (stage 21, patches/013 — two files). The final serving corrections that are NOT in the image are the required launch environment:

```text
CCL_SYCL_ALLREDUCE_TMP_BUF=1
CCL_SYCL_ALLGATHERV_TMP_BUF=1
```

and the topology-dependent dFlash2 capture list (stage 18): PIECEWISE `[1,2,4,8,16,32,64]` max 64 at TP2/TP4, `[1,2,4,8]` max 8 at TP1; Base/MTP1 keep their FULL_AND_PIECEWISE policies. Patch-010 serving drains are **not required** and not present in this runtime; patch 009's warm-up initialization discipline is baked in. Everything else above is baked into the image except the graph-capture policy and the two variables.

Qualified performance on the promoted bytes (TP2 dFlash2, 64K, warm): concurrency C1/C2/C4/C8/C16 = 83.7 / 141.1 / 238.6 / 325.3 / 321.6 aggregate tok/s; single-stream update p50 ≈ 45.7–46.2 ms across all benchmark categories; 64K prefill ≈ 2565.5 tok/s on a 47,098-token prompt; SQL 178.90 tok/s / 44.72 ms per 8-token round, 100% draft acceptance, canonical output byte-identical. At TP4 (same bytes, widened capture): single-stream decode ≈ 136.9 tok/s with update p50 ≈ 31.8–32.3 ms; concurrency 126.0 / 189.6 / 337.1 / 471.0 / 541.2; 64K prefill ≈ 3501.7 tok/s; SQL 260.0 tok/s / 30.77 ms/round, output byte-identical; C16 runs a true resident batch of 16 at ~45.6% KV peak with TTFT p50 ≈ 320.7 ms. Earlier per-mode medians on the pre-noise-cache lineage (INT4 Base 77.9 / MTP1 110.1; FP8 Base 54.6 / MTP1 82.4 tok/s — modes untouched by the two new stages) remain the reference for those lanes. Reliability is qualified under the launch contract above: 3/3 fully cold episodes and 3/3 fully cold full sweeps 48/48 at every level on the 2026-09-11 authority, re-qualified cold 3/3 on these bytes (2026-09-12), plus the TP4 campaign — zero failure signatures in every window.

## Qualified models and profiles

Exact public model identities and revisions (a revision mismatch is a stop condition, not a refresh):

| Role | Repository | Revision |
|---|---|---|
| Target (standard FP8) | `Qwen/Qwen3.8-27B-FP8` | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` |
| Target (uncensored FP8) | `orcarouter/Qwen3.8-27B-Uncensored-FP8` | `9228df5c6c9c509e1019f83b4e085cf643118bac` |
| Target (standard INT4) | `RedHatAI/Qwen3.8-27B-INT4` | `2fb0debc365fb6c1683d7d3ad7722470919627a8` |
| Target (uncensored INT4) | `noon-at-cgn/Qwen3.8-27B-Uncensored-W4A16-AutoRound` | `0e10c9f6b5b8a97fba199e82c49690d272f776ce` |
| dFlash2 draft | `incoai/Qwen3.8-27B-DFlash2` | `dedf8df68adfb1afeaf7b7480c0a0243108177b4` |

Serving modes: **Base** (plain decode), **MTP1** (the model's own one-token multi-token-prediction head), and **dFlash2** (draft-model speculative decoding, K=7) — where qualified. The qualified profile matrix totals 96 profiles: 18 standard FP8 + 18 uncensored FP8 + 30 standard INT4 + 30 uncensored INT4 (the INT4 side spans Base/MTP1/dFlash2 across TP1/TP2/TP4 at 32K–256K contexts). Model weights are not redistributed; acquire the pinned revisions from their publishers.

## Honest boundaries

- Qualified on Intel Arc Pro B70 (1–4 cards) with the host user-mode driver family matched to the image's pinned stack (compute-runtime `26.14.37833.4` era). Other hardware and driver families are unqualified.
- Patches 009/010 and the collective segmentation are behavioral bounds on a driver-level stall pattern; the xe-internal defect itself is not claimed fixed. The final launch contract (`CCL_SYCL_ALLGATHERV_TMP_BUF=1`) removes the dynamic gather-buffer IPC exposure rather than serializing around it — its root-cause mechanism is strongly supported by the qualified campaign, not absolutely confirmed at oneCCL/xe source level. Patch 010 remains a proven fallback containment if the launch contract ever needs to be abandoned.
- The Level Zero shim contains a driver-accounting pathology; it does not repair the driver.
- The dFlash2 noise cache preserves exact noise tensors and GPU RNG state per request; it does not make seeded stochastic sampling deterministic (that was never deterministic), and no all-temperature output-parity claim is made beyond the validated greedy lane.
- The promoted image's qualification record retains one unexplained hardware event: a 2026-09-12 08:04Z cold-initialization engine fault (GPU1, during draft weight loading, before any serving) that preceded the campaign, resolved by a host reboot. Three subsequent fully cold initializations and the entire qualification campaign on the same bytes were clean, the fault was never attributed to the cache (the cached path had not executed at the point of failure), and the prior-boot fault history of the affected cards is part of the retained record.
- `orcarouter/Qwen3.8-27B-Uncensored-FP8` is access-controlled on Hugging Face; the other four repositories resolve publicly at the pinned revisions.
- DFlash2 Automatic Prefix Caching is ON in production since 2026-09-22 (`--enable-prefix-caching`; the Qwen3.8 hybrid config resolves the Mamba cache mode to ALIGN), qualified text + tools across INT4 TP1/TP2/TP4 and FP8 TP2/TP4. Vision was fixed the same day by the VISION-LAUNCH-FLAG-FIX (removal of the stale `--mm-encoder-attn-backend TORCH_SDPA` override → XPU default FLASH_ATTN ViT): live-qualified INT4 DFlash2 TP2 64K (cold/warm/APC-composed vision, vision-grounded tool call); other profiles carry the fix by launch equivalence, with the INT4 TP1 32K vision memory envelope explicitly unqualified (see the vision section above). Base and MTP1 keep their engine-default APC posture. The 2026-09-21 scheduler fix does not weaken the strict proposal-probability verifier: a suppressed round degrades to plain decode rather than substituting a fallback distribution.
