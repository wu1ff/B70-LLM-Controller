# Qwen3.6-35B-A3B

This is the qualified Qwen3.6-35B-A3B pack I use with B70 LLM Controller. It contains two served checkpoints, one DFlash assistant, a frozen runtime, and the exact B70 configurations that passed testing. Every profile — Base and DFlash, FP8 and INT4, one to four cards — runs on the same single runtime image.

## Models

| Variant | Repository | Revision |
| --- | --- | --- |
| Standard FP8 | `Qwen/Qwen3.6-35B-A3B-FP8` | `95a723d08a9490559dae23d0cff1d9466213d989` |
| Standard INT4 | `Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound` | `65f69c73f17488236c85c85211f6ba28d7106157` |

DFlash uses `z-lab/Qwen3.6-35B-A3B-DFlash` at revision `f181eece646affea2c38b2765f1aaa01a9734ccd` as a support artifact. It is not a checkpoint you serve directly.

Both target repositories and the DFlash artifact are ungated. `b70ctl` downloads the exact revisions above and checks them against the file inventories in `pack.json`. The INT4 repository's local marker names its `main` branch; the pack still pins the commit SHA above, which byte-verification confirmed against every file in the frozen inventory.

## Modes

- **Base** serves the selected checkpoint without speculative decoding, with automatic prefix caching on.
- **DFlash** attaches the assistant model with seven speculative tokens per step, the V2 model runner, and automatic prefix caching on.

Mode availability depends on the checkpoint, card count, and context. A mode only appears when that exact combination has a profile.

## Hardware and context support

| Checkpoint | Cards / TP | Qualified contexts and modes |
| --- | --- | --- |
| FP8 | 2 or 4 | 32K, 64K, 128K, 256K: Base, DFlash |
| INT4 | 1 | 32K, 64K, 128K, 256K: Base; 32K, 64K: DFlash |
| INT4 | 2 or 4 | 32K, 64K, 128K, 256K: Base, DFlash |

FP8 has no TP1 profile at all — the checkpoint's runtime footprint exceeds a practical one-B70 deployment, so no such lane was qualified or is exposed. INT4 DFlash TP1 tops out at 64K: the 128K and 256K tiers are capacity exclusions on the one-card pool (real admission demand exceeds the qualified util-0.90 KV budget), not untested gaps. Each lane was qualified at its maximum context on the identical contract, and that qualification covers every lower tier the pack publishes.

Only combinations that passed qualification are included. `b70ctl` reads the exact 38-profile matrix from `pack.json`, so unsupported combinations simply do not appear.

## Performance

These are retained qualified results from the final runtime, not new measurements. All numbers are DFlash measurements on the promoted runtime pinned below, taken at a context of 32,768 on the qualified test host — they are not Base guarantees, and Base throughput is a different quantity that should not be inferred from this table.

Test host: AMD EPYC 7443, 4x Intel Arc Pro B70 32 GB, PCIe Gen4 x16.

BetterBench 0.4.0 combined decode throughput (weighted across workload categories), temperature 0.7, single stream:

| Checkpoint | Cards | TP | Combined decode |
| --- | --- | --- | ---: |
| FP8 | 2x B70 | TP2 | 269.3 tok/s |
| FP8 | 4x B70 | TP4 | 323.3 tok/s |
| INT4 | 2x B70 | TP2 | 278.7 tok/s |
| INT4 | 4x B70 | TP4 | 314.6 tok/s |

These are representative measurements, not guaranteed minimums. Performance varies with prompt shape, sampling, context, concurrency, host configuration, and serving mode.

## Runtime

The pack uses this final qualified runtime — one image for all 38 profiles:

```text
ghcr.io/wu1ff/qwen36-35b-a3b-b70@sha256:8360c9a7e3d8e1bfb890d1f57cd59533958bf5140331d144eb05beb9ac8bc790
```

This is the 2026-09-20 unified runtime authority: the exact bytes that were already the qualified DFlash authority, re-qualified for all five Base lanes so that Base and DFlash share one image. No separate Base runtime exists or is referenced.

If the image is missing, `b70ctl` pulls that immutable reference and verifies that Docker reports the expected image ID before offering any profile that uses it. One caveat inherited from the established convention: on Docker daemons using the containerd image store, the image ID equals the registry manifest digest recorded in `pack.json`, and verification succeeds. A classic-graphdriver daemon reports the config digest as the image ID instead, which would not match; this pack targets the containerd image-store daemon class, same as the existing Qwen3.8 pack.

The runtime mounts the B70 devices (`/dev/dri`) plus one fixed read-only bind of `/dev/dri/by-path` so serving resolves cards by their stable device paths. That exact bind is the only host mount the pack is allowed to request; `b70ctl` rejects any other pack-controlled mount.

## Runtime recipe

For the runtime build history, compatibility work, qualified launch contract, and technical provenance, see [RUNTIME_RECIPE.md](RUNTIME_RECIPE.md).

## Using the pack

Install from the public catalog via **Model Packs → Browse Available Packs**, or import this pack directory with **Model Packs → Import Local Pack** while working from the source tree. Choose which checkpoints to prepare during installation; existing exact model revisions and the runtime are reused.

Then open **Run Model**, choose the checkpoint, and select from the card, context, and mode values shown. Choose Local or LAN access, set the port, and start it.

`pack.json` is intentionally readable if you want to inspect all 38 profiles, model file inventories, and launch data yourself.

## Licenses

The Controller's MIT license does not relicense the model weights, runtime components, or other third-party software. Check the upstream terms for each repository and component you use.
