# Reviewed visual assets

These are Wulf's supplied finished reports, selected for local README integration.
The three complete PNGs are byte-for-byte copies from the Flash-Next workspace.
The seven detail images are lossless crops of the full aquarium telemetry report:
no resizing, resampling, redesign or changed pixels. Every crop retains its panel
heading, chart, legend and text. The original complete reports remain available.

| Complete report | Dimensions | Workspace source |
| --- | --- | --- |
| [BetterBench full release report](betterbench-integrated-serving-v1-full.png) | 3000 × 8434 | `results.png` |
| [Aquarium overview](aquarium-telemetry-overview.png) | 2360 × 2840 | `abyss-telemetry-figure.png` |
| [Aquarium full-session traces](aquarium-telemetry-full.png) | 2360 × 5374 | `abyss-telemetry-full.png` |

The source workspace is `qwen3.8-flash-next-dflash2-v032`. [manifest.json](manifest.json)
records SHA-256 identities, crop coordinates, captions, runtime/checkpoint evidence,
counting windows, units and authorization. Crop coordinates below are
`(left, top, right, bottom)` in original pixels, with right/bottom exclusive.
Cuts follow the blank gutters between existing panels. The two panels in the
side-by-side detail row are separated at the central gutter.

| Readable crop | Dimensions | Original crop coordinates |
| --- | --- | --- |
| [1 · Generation rate](aquarium-trace-01-generation-rate.png) | 2240 × 785 | `(60, 600, 2300, 1385)` |
| [2 · Generated tokens](aquarium-trace-02-generated-tokens.png) | 2240 × 670 | `(60, 1385, 2300, 2055)` |
| [3 · Prompt occupancy](aquarium-trace-03-prompt-occupancy.png) | 2240 × 660 | `(60, 2055, 2300, 2715)` |
| [4 · Cache and acceptance](aquarium-trace-04-cache-and-acceptance.png) | 2240 × 710 | `(60, 2715, 2300, 3425)` |
| [5 · Acceptance by position](aquarium-trace-05-acceptance-by-position.png) | 1108 × 940 | `(60, 3425, 1168, 4365)` |
| [6 · Compactions](aquarium-trace-06-compactions.png) | 1108 × 940 | `(1192, 3425, 2300, 4365)` |
| [7 · Client timeline and evidence](aquarium-trace-07-client-timeline.png) | 2240 × 1009 | `(60, 4365, 2300, 5374)` |

BetterBench matches the finished `results.png` in the
`flashnext-release-betterbench-full-20261022T193611` campaign. That run used
`qwen38-flashnext-dflash2:integrated-serving-v1`, accepted image
`sha256:6e2ab4e686680a5208ff0119c4637341cba81942bd39b6576ea14dc8f547b83b`.
The README's two-decimal figures come from the campaign summary; the PNG rounds
its headline cards. Combined decode is a weighted single-stream score, concurrency
throughput is aggregate, and stream-update latency measures token batches.
The [runtime recipe](../../RUNTIME_RECIPE.md) and
[qualification archive](../archive/qualification/README.md) retain earlier runs.

Aquarium capture `3f1b27fb-2569-4bdb-8865-e0c8a35f70c4`, session
`4d83c905-4ef3-4393-97ac-38c205f203de`, covers 22 October 2026, Phoenix time.
The reports identify the Qwen3.8-Flash-Next + DFlash2 / B70 / vLLM XPU family;
they do not embed an immutable image digest or checkpoint revision. Evidence and
attribution in the report footers are preserved. Server counters use the enclosing
capture window; client lifecycle timings include harness overhead. Pooled decode
and whole-turn effective throughput use different denominators. No aquarium
speedup baseline was run. `C1/C2/C3` in these traces label compaction events;
`C1/C2/C4` in BetterBench label concurrent requests.

The aquarium graphics' 138 ms benchmark reference and 4,665 tok/s prefill estimate
are contemporaneous values, preserved unchanged. They are distinct from the latest
BetterBench rerun shown above them in the README. The referenced archived frame
2662 is not embedded in the reports; Wulf waived a separate aquarium screenshot.

Report authorship/evidence attribution remains as supplied by Wulf. Local staging
is authorized by this task; no new license or publication permission is inferred.
Hardware/architecture imagery and Wulf's personal narrative remain editorial work.
