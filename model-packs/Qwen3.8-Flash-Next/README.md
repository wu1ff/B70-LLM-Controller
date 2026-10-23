# Qwen3.8-Flash-Next on four B70s

> Editorial draft · DFlash2 model pack staged locally. Public installation still
> selects the published 1.0.4 pack. The new runtime has not been published.

## Wulf's introduction

*[Wulf: your perspective, what you wanted to build, and what this system means in
practice. Leave the opening in your own voice.]*

*[Hero image and short project introduction to be selected by Wulf.]*

## What we built

DFlash2 is speculative decoding: a smaller assistant proposes tokens, and the
main model checks them before committing the accepted output. Here, the assistant
uses features from completed target layers to draft seven proposals per cycle.

| Component | This implementation |
| --- | --- |
| Hardware | Four Intel Arc Pro B70 GPUs, 32 GB each |
| Parallelism | Tensor parallelism across all four cards (TP4), with expert parallelism |
| Main model | Intel AutoRound Qwen3.8-Flash-Next, W4A16 (4-bit weights, BF16 execution) |
| Assistant | External Qwen3.8-Flash-Next-DFlash2, BF16, seven proposals |
| Context and serving | Native 262,144-token context; four request slots |
| Runtime | Custom vLLM 0.32.0 XPU implementation, `integrated-serving-v1` |

The B70 implementation combines custom native kernels, guarded decode graphs,
and verification/state handling with disk-backed BF16 PLE and a residency shim
for GPU memory placement. The [runtime recipe](RUNTIME_RECIPE.md) records the
exact configuration, model revisions and capacity limits.

*[Hardware photo and architecture illustration: target completed-layer features
→ DFlash2 selector/drafter → seven proposals → target verification and state commit.
Wulf: describe the machine, choices and practical experience.]*

## BetterBench performance

The full BetterBench v0.6.0 rerun on **22 October 2026** used the accepted
`integrated-serving-v1` runtime: **336/336 measured requests completed**, plus
**32 warmups**. Weighted combined decode was **160.11 tok/s**, median time to first
token (TTFT) **139.39 ms**, and stream-update p99 **25.82 ms**. The graphic rounds
these headline values for display.

[![BetterBench full release report: category decode, stream-update latency, concurrency, prefill and complete result tables](patches/assets/betterbench-integrated-serving-v1-full.png)](patches/assets/betterbench-integrated-serving-v1-full.png)

| Concurrent requests | Completed | Aggregate throughput | Median decode per request | Median TTFT |
| --- | --- | --- | --- | --- |
| 1 (C1) | 48/48 | 146.50 tok/s | 183.46 tok/s | 140.99 ms |
| 2 (C2) | 48/48 | 227.12 tok/s | 145.60 tok/s | 175.95 ms |
| 4 (C4) | 48/48 | 316.01 tok/s | 106.58 tok/s | 196.47 ms |

Aggregate throughput counts output across concurrent requests; it differs from
the per-request rate and the weighted single-stream score. Stream-update gaps
measure delivery of token batches, rather than individual token latency. This run
used fresh prefix-cache salts with warm model/OS/PLE caches. The report records
zero prefix-cache hits and zero server errors. Its small-sample tail estimates
should be read with the qualifications printed in the report.

[Open the complete BetterBench PNG at original resolution](patches/assets/betterbench-integrated-serving-v1-full.png).
The [retained qualification reports](patches/archive/qualification/README.md)
and the runtime recipe preserve earlier measurements; the figures above refer
to this latest rerun. One run does not establish a universal gain.

## Real-world aquarium test

**ABYSS: An Alien Aquarium** is the three.js build session captured by these
reports: one completed turn spanning **3h 02m 55.8s** on 22 October 2026,
Phoenix time.

*[Wulf: introduce the aquarium, what you asked it to build, and your experience
working with it.]*

### Session overview

[![ABYSS aquarium session overview: lifecycle, prefix caching, DFlash2 acceptance, client latency and three compactions](patches/assets/aquarium-telemetry-overview.png)](patches/assets/aquarium-telemetry-overview.png)

The report accounts for **520,445 generated tokens**: 500,651 from main requests
and 19,794 from compaction. It records **517 main requests**, **520 tool calls**
and **three compactions**. Pooled client decode was **119.10 tok/s**; the whole-turn
effective rate, including tools, waits and compaction, was **47.42 tok/s**.

These are session measurements with different counting windows from BetterBench.
Client first-delta timing includes harness overhead; server counters cover the
enclosing capture window. The telemetry graphics retain their contemporaneous
benchmark references (138 ms and 4,665 tok/s), while the BetterBench section above
uses the later rerun. No speedup baseline was run for the aquarium session.

### Generation through the session

The first two panels show the saved generation-rate samples and cumulative
output, with the three compaction events marked on the same time axis.

[![Aquarium trace panel 1: server generation rate over the full session, including compaction markers and legend](patches/assets/aquarium-trace-01-generation-rate.png)](patches/assets/aquarium-trace-01-generation-rate.png)

[![Aquarium trace panel 2: cumulative generated tokens, comparing the server counter with settled main-request output](patches/assets/aquarium-trace-02-generated-tokens.png)](patches/assets/aquarium-trace-02-generated-tokens.png)

### Context growth and prefix caching

Prompt occupancy peaked at **209,678 of 262,144 tokens (79.99%)**. These panels
follow context growth, the compaction resets, and recent-window cache and draft
acceptance ratios.

[![Aquarium trace panel 3: prompt occupancy and all three context compaction cycles](patches/assets/aquarium-trace-03-prompt-occupancy.png)](patches/assets/aquarium-trace-03-prompt-occupancy.png)

[![Aquarium trace panel 4: prefix-cache ratio, draft acceptance and acceptance of all seven drafts over time](patches/assets/aquarium-trace-04-cache-and-acceptance.png)](patches/assets/aquarium-trace-04-cache-and-acceptance.png)

### Draft acceptance and compaction details

The two detail panels are presented separately so the position labels and
compaction table remain readable. The reported **3.53 tokens per cycle** is an
acceptance yield, not a measured speedup factor.

[![Aquarium trace panel 5: draft acceptance at each of the seven proposal positions](patches/assets/aquarium-trace-05-acceptance-by-position.png)](patches/assets/aquarium-trace-05-acceptance-by-position.png)

[![Aquarium trace panel 6: timings, prompt sizes and next-request latency for the three compaction events](patches/assets/aquarium-trace-06-compactions.png)](patches/assets/aquarium-trace-06-compactions.png)

### Client activity and evidence

[![Aquarium trace panel 7: client wait, reasoning, text, tools and compaction timeline, followed by integrity counters and evidence notes](patches/assets/aquarium-trace-07-client-timeline.png)](patches/assets/aquarium-trace-07-client-timeline.png)

Each panel links to its original-resolution crop.
[Open the complete, unchanged full-session trace report](patches/assets/aquarium-telemetry-full.png)
or [the unchanged overview](patches/assets/aquarium-telemetry-overview.png).
The [visual asset record](patches/assets/README.md) retains report identities,
measurement notes and crop boundaries.

## Installation and operation

*[Public DFlash2 installation walkthrough follows publication and a separate clean
install test. Until then, use the isolated local workflow in the technical recipe.]*

The [runtime recipe](RUNTIME_RECIPE.md) covers dependencies, model acquisition,
local Controller import and launch configuration. Native context is a per-request
ceiling; four full-length sessions do not fit simultaneously.

## Source access, credits and limitations

[Exact production reconstruction](patches/production/README.md) ·
[complete campaign archive](patches/INDEX.md) ·
[credits and notices](patches/licenses/NOTICE.md) ·
[source access](patches/licenses/SOURCE_ACCESS.md) ·
[open distribution findings](patches/licenses/SOURCE_DISTRIBUTION.md).

*[Wulf: final acknowledgments and perspective. Retain upstream/model attributions.]*

The accepted runtime and independent reconstruction passed focused live checks;
cold startup, sampled-concurrency assignment and finite KV capacity have documented
limits. Source/distribution decisions and executable notice incorporation remain
open. Publication is blocked pending their resolution.
