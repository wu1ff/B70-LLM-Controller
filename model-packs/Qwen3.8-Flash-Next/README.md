# Qwen3.8-Flash-Next on four B70s

> Editorial draft · DFlash2 model pack staged locally. Public installation still
> selects the published 1.0.4 pack. The new runtime has not been published.

## Wulf's introduction

*[Wulf: your perspective, what you wanted to build, and what this system means in
practice. Leave the opening in your own voice.]*

*[Hero image and short project introduction to be selected by Wulf.]*

## Hardware and architecture

Four Intel Arc Pro B70 32 GB cards, TP4 with expert parallelism. Intel AutoRound
W4A16 target, BF16 external DFlash2 assistant, seven proposals, native 262144-token
context and four request slots. Disk-backed BF16 PLE and the residency shim are
part of the accepted runtime.

*[Hardware photo and architecture illustration: target completed-layer features
→ DFlash2 selector/drafter → seven proposals → target verification and state commit.
Wulf: describe the machine, choices and practical experience.]*

## BetterBench screenshots and comparisons

*[Reserve report screenshots, category comparisons and C1/C2/C4 graphs.]*

Verified reference: one unchanged stock BetterBench v0.6.0 run, **336 measured
requests plus 32 warmups**, completed. Weighted decode **160.88 tok/s**; median
TTFT **137.78 ms**. C1/C2/C4 aggregate rates **146.12 / 208.14 / 303.39 tok/s**.
These aggregates describe different concurrency workloads. Compared with the
preceding long-prefill runtime, C2/C4 were 3.3%/2.0% lower. Results used fresh APC
salts with warm model/OS/PLE caches; one run does not establish a universal gain.
See the [retained report and measurement summary](patches/archive/qualification/README.md).

## A real three.js project

*[Wulf: choose the finished application, demo, screenshots and story. Reserve a
reproducible input/output artifact and tool/session timeline before adding figures.]*

## DSH telemetry, context growth and compaction

*[Reserve DSH session telemetry and context-growth/compaction graphs. No finalized
DSH metrics are claimed. Wulf: choose the session and explain the experience.]*

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
