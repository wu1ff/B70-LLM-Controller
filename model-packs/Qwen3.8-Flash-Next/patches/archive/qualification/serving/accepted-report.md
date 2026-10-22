# Integrated serving efficiency — complete

**Healthy and idle on port 8000.** Local candidate `qwen38-flashnext-dflash2:integrated-serving-v1`, **`sha256:6e2ab4e686680a5208ff0119c4637341cba81942bd39b6576ea14dc8f547b83b`**. Accepted `long-prefill-v1`, its original launch/container, prior evidence and published MTP5 remain preserved. No publication, GHCR, Controller-pack or README changes.

**Cause and retained implementation.** A waiting long prompt that could not fit an aligned chunk stopped the queue scan even when a later short prompt fit the leftover budget. The scheduler now reuses its skipped-request queue, preserves the earlier request's position, and reserves a slot for its later admission. KV-holding waiters retain their existing priority. The other change removes redundant batch-size-one top-k/top-p specialization and performs bounded filter warmup on initialized standalone tensors at startup. No live request/KV/RNG state is warmed; sampling/proposal/rejection arithmetic is unchanged.

Only four runtime files differ from the accepted image, whose layers are an exact prefix. **1024 budget / 832 full chunks / original native kernels** remain. No adaptive chunking or exclusive cadence is shipped. [Patch](/home/wulf/Desktop/qwen3.8-flash-next-dflash2-v032/patches/integrated-serving.patch), [recipe](/home/wulf/Desktop/qwen3.8-flash-next-dflash2-v032/build/integrated-v2/Dockerfile), [manifest](/home/wulf/Desktop/Benchmarking/flashnext-integrated-serving-20261022/selected-manifest.json).

**Mixed serving, two paired repeats.** Bodies, sampling, caps, arrivals and APC conditions match. Active coding is followed by 23535-token and 6009-token prompts, then a short 144-token prompt. Baseline engine admission alone delayed the short request 5.33–5.34 s.

| Measure | Before | Candidate |
|---|---:|---:|
| Short TTFT | 5.709 s | **.533 s** |
| Short own completion | 8.899 s | 8.427 s |
| Active-code completion | 8.267 s | 8.355 s |
| Long-A completion | 7.296 s | 7.411 s |
| Long-B completion | 7.705 s | 7.821 s |
| Full arrival-to-drain | 9.449 s | **8.977 s** |
| Complete-run aggregate | 69.21 tok/s | 73.52 tok/s |

Actual outputs **654→660** are retained: coding 355→355, A 44→45, B 55→60, short 200→200. The other requests finish 88–116 ms later; this tradeoff is not hidden by the aggregate. Production repeats confirm drain **8.994/8.986 s** and short TTFT **.555/.536 s**. Active-stream maximum observed gaps remain ~220 ms: this fixes admission delay, not prefill execution pauses. Small-sample maxima are burst-update observations, not per-token latency or qualified tails.

Other paired drain means (s): idle **5.451→5.460**, original mixed **7.691→7.627**, staggered C4 **6.229→6.237**, tool-result continuation with APC **3.270→3.262**, APC-cold **4.500→4.524**. All 34 bodies/arrival schedules match. Standalone/staggered/continuation outputs match exactly; mixed/admission output differences and all finish reasons remain in the raw records. Staggered C4 retains 1536 output tokens and >2 s of four-stream overlap. Aggregate always includes arrival through final drain. [Paired detail](/home/wulf/Desktop/Benchmarking/flashnext-integrated-serving-20261022/paired-mixed-comparison.json).

**One complete unchanged stock BetterBench v0.6.0 run: 336/336 measured + 32 warmups pass.** Native HTML visibly includes eight categories, C1/C2/C4 and all four prefill depths. Fresh APC salts; zero prefix hits/preemptions. Measured finishes: 238 length / 98 stop. The authoritative before is `/home/wulf/Desktop/Benchmarking/flashnext-long-prefill-betterbench-full-20261021T234459/results.html`.

| Stock measure | Before | Candidate |
|---|---:|---:|
| Weighted decode, tok/s | 158.61 | 160.88 |
| TTFT p50, ms | 138.19 | 137.78 |
| Stream-update p99, ms | 27.37 | 26.14 |
| C1 aggregate, tok/s | 143.35 | 146.12 |
| C2 aggregate, tok/s | 215.15 | 208.14 |
| C4 aggregate, tok/s | 309.62 | 303.39 |

C2/C4 stock aggregates are **3.3%/2.0% lower**; this is not a uniform throughput win. Unseeded trajectories differ: C1/C2/C4 output totals 19579/19422/19640→18674/19225/19096. Fixed-output controls stayed close to baseline; they do not erase the lower stock observations or establish statistical attribution.

| Stock target depth | Actual median input | Before prefill tok/s | Candidate full run |
|---|---:|---:|---:|
| 2000 | 1556.5 | 4576.56 | 4328.64 |
| 8000 | 5934.5 | 5005.88 | 4926.78 |
| 16000 | 11842 | 4779.71 | 4705.18 |
| 32000 | 23589.5 | 4665.06 | 4569.18 |

Full-run prefill is lower by 5.4% at 2K and 1.6–2.1% elsewhere. An exact-body/configuration 2K follow-up (same two warmups/eight measurements, new salts) measured **4568.74 tok/s / 339.92 ms TTFT**, with zero reuse/preemption. The 2K difference did not repeat; the full report is unchanged. Effective prefill is actual prompt tokens/client TTFT, including first-token overhead. These are APC-cold measurements with warm model/OS/PLE caches, not OS-cold results.

Archived coding preserves the output hash, **3077 tokens/471 cycles**, and every acceptance count: warmed decode **270.04→271.47 tok/s**, TTFT **1.753→1.727 s**. Sampled control preserves the complete output, **5046 tokens/1171 cycles**, all acceptance-position counts and tool-call finish: warmed decode **156.17→156.26 tok/s**.

**First-use tradeoff.** Bounded live timing found separate ~8 s top-k/top-p JIT calls at prefill and verification; GC was negligible. The shared variant serves width 8 in <1 ms after width 1. **54 exact filter cases per rank** cover six batch sizes, three filtering modes, ties and masks. Startup warmup costs approximately **12.3 s**, retaining **995840 tensor bytes/rank** then. This work moves to startup; only the redundant second specialization is eliminated. First sampled candidate control is **150.81 tok/s**, exact accepted output/counts, largest observed update gap **1.114 s**. Retained uninstrumented accepted first-use rate: **122.12 tok/s**; diagnostic baseline gap: **7.918 s**. Profiling overhead is not credited as a throughput gain. Other sampler initialization and lazy target-graph creation remain; cold coding timings are retained, not presented as warm performance.

**Correctness/resources.** Two real-scheduler unit tests pass (spare-token admission, priority/progress, last-slot protection). **22/22** final interface checks pass: APC hit/divergence/continuation/retired-slot reuse, reasoning, tools, structured output, two-image vision and reordered C4 waves. All ranks pass C1–C4 graph/eager guards; zero logged fallback/capture/guard failures. Eager prefill/partial verification is intentional.

**261632 input + 512 reserved = 262144** native execution passes cold/cached with the accepted **302-token answer and exact code hash**. TTFT **111.696/1.229 s**; reuse **0/260416 tokens**. Final container has zero preemptions, OOM or serving errors. Primary DRM-client VRAM **29.765–29.774 GiB/rank**, matching retained workload snapshots. Privileged `sudo xpu-smi` reports all four devices normal; no host/driver/reset/rebind changes.

TP4+EP, four slots, **7785 KV blocks / 491165-token capacity**, context 262144, utilization .91, APC, original **5 MiB/rank / 1024-row Disk BF16 PLE stage**, residency shim, W4A16 group-128 weights/scales, BF16 execution and FP32 recurrent state remain intact. Pinned target `4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897` and BF16 draft `a22a1744e85c17364dcc54a6a420af7135d4c7f6` are unchanged. Four simultaneous full-native contexts remain outside the original requirement.

**Rejected/remaining.** Existing 512 cap worsened mixed drain 7.69→10.18 s. Extra decode-only steps retaining mixed prefill (different from the prior exclusive policy) increased long TTFT to ~6.6/7.3 s without a complete-run win. A fully integrated 2048-capacity/1664-chunk prototype passed 176 native and 20 live output/state checks and improved long TTFT ~5.1→4.53 s, but changed coding to 3182 tokens/516 cycles (255.46 versus 270.82 tok/s at the same ~24.1 ms cycle time) and sampled output to 8216 tokens. Its expanded native/PLE payload is excluded. A diagnostic cadence wrapper's duplicate-argument failure was corrected and preserved.

Residual ~200 ms gaps are mainly target work: ~169–170 ms full prefill, ~177–178 ms mixed. Scheduler work is <1 ms; host waits shift between metadata and PLE, with similar rank timing. Tested chunk/cadence policies did not earn more scheduler complexity. Further execution improvements remain unproven; no hardware ceiling is claimed.

**Exact service:** `flashnext-integrated-final`, `99bdddf9c1387d26b622c43e5c1c2400bb92dd9d9b0b86d259bd19ebff7fbc33`, started `2026-10-22T09:20:51.214095815Z` (02:20:51 Phoenix), restart count **0**, policy **no**, HTTP 200 and idle. Accepted command/environment/mount/device contract matches; development RPC 404, no campaign mounts/custom worker/scheduler. Accepted image/original stopped container and launch remain available.

**Final native HTML:** `/home/wulf/Desktop/Benchmarking/flashnext-integrated-serving-20261022/final-stock/results.html`.
[Machine-readable results](/home/wulf/Desktop/Benchmarking/flashnext-integrated-serving-20261022/results-summary.json). Raw evidence, failures, manifests, original launch and report screenshots: `/home/wulf/Desktop/Benchmarking/flashnext-integrated-serving-20261022`. Previous complete record: `/home/wulf/Desktop/Benchmarking/flashnext-integrated-serving-20261022/rr-before.md`.
