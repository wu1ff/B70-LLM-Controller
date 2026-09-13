# Qwen3.8 27B

This is the qualified Qwen3.8-27B pack I use with B70 LLM Controller. It contains four served checkpoints, one dFlash2 assistant, a frozen runtime, and the exact B70 configurations that passed testing.

## Models

| Variant | Repository | Revision |
| --- | --- | --- |
| Standard FP8 | `Qwen/Qwen3.8-27B-FP8` | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` |
| Uncensored FP8 | `orcarouter/Qwen3.8-27B-Uncensored-FP8` | `0f3cdb83820a8190ffedaef5b29cf4a635e49b4d` |
| Standard INT4 | `RedHatAI/Qwen3.8-27B-INT4` | `2fb0debc365fb6c1683d7d3ad7722470919627a8` |
| Uncensored INT4 | `noon-at-cgn/Qwen3.8-27B-Uncensored-W4A16-AutoRound` | `0e10c9f6b5b8a97fba199e82c49690d272f776ce` |

dFlash2 uses `incoai/Qwen3.8-27B-DFlash2` at revision `dedf8df68adfb1afeaf7b7480c0a0243108177b4` as a support artifact. It is not a checkpoint you serve directly.

Four of the five repositories are ungated. `orcarouter/Qwen3.8-27B-Uncensored-FP8` is gated on Hugging Face (`auto`), so downloading it requires an authenticated account that has accepted its terms; `b70ctl` treats it as a gated model and reports access failures accordingly. `b70ctl` downloads the exact revisions above and checks them against the file inventories in `pack.json`. The 1.0.1 revision repin of the Uncensored FP8 model is provenance-only: every runtime-relevant file at the new revision is byte-identical to the qualified artifact (only `README.md` differs), so no requalification was needed.

## Modes

- **Base** serves the selected checkpoint without speculative decoding.
- **MTP1** uses the checkpoint's MTP path with one speculative token.
- **dFlash2** uses the assistant model with seven probabilistic proposals and standard rejection sampling.

Mode availability depends on the checkpoint, card count, and context. A mode only appears when that exact combination has a profile.

## Hardware and context support

| Checkpoint | Cards / TP | Qualified contexts and modes |
| --- | --- | --- |
| Standard FP8 | 2 | 32K, 64K: Base, MTP1, dFlash2 |
| Standard FP8 | 4 | 32K, 64K, 128K, 256K: Base, MTP1, dFlash2 |
| Uncensored FP8 | 2 | 32K, 64K: Base, MTP1, dFlash2 |
| Uncensored FP8 | 4 | 32K, 64K, 128K, 256K: Base, MTP1, dFlash2 |
| Standard INT4 | 1 | 32K: Base, MTP1, dFlash2; 64K: Base, MTP1; 128K: Base |
| Standard INT4 | 2 or 4 | 32K, 64K, 128K, 256K: Base, MTP1, dFlash2 |
| Uncensored INT4 | 1 | 32K: Base, MTP1, dFlash2; 64K: Base, MTP1; 128K: Base |
| Uncensored INT4 | 2 or 4 | 32K, 64K, 128K, 256K: Base, MTP1, dFlash2 |

FP8 starts at two cards; FP8 at TP2 is qualified at 32K and 64K. Each uncensored checkpoint is qualified lane-for-lane against its standard counterpart, so the two matrices match for the same quant. Standard and uncensored FP8 each carry 18 profiles; the INT4 variants carry 30 each.

Only combinations that passed qualification are included. `b70ctl` reads the exact matrix from `pack.json`, so unsupported combinations simply do not appear.

## Performance

These are retained qualified results, not new measurements. All tables are FP8 with dFlash2 on the promoted runtime and image pinned in the Runtime section below. TP2 uses two B70 cards and TP4 uses four. The SQL lane ran at a context of 32,768; BetterBench ran at 65,536.

The numbers are representative measurements from the qualified test host, not guaranteed minimums. Performance varies with prompt shape, sampling, context, concurrency, host configuration, and serving mode. INT4, Base, and MTP1 configurations are different quantities and should not be inferred from these tables.

Test host: AMD EPYC 7443, 4x Intel Arc Pro B70 32 GB, PCIe Gen4 x16, current qualified runtime.

### SQL continuation

| Cards | TP | Median decode | Round latency | Acceptance |
| --- | ---: | ---: | ---: | ---: |
| 2x B70 | TP2 | 178.9 tok/s | 44.72 ms | 100% |
| 4x B70 | TP4 | 260.0 tok/s | 30.77 ms | 100% |

LocalMaxxing SQL continuation lane: temperature 0, 256 output tokens, concurrency 1, 2 warmups plus 5 measured rounds against the canonical output, with 100% speculative acceptance. The five TP4 samples spanned 258.9 to 260.8 tok/s (mean 259.92); the median is the headline value.

### BetterBench single stream

Decode throughput per workload category:

| BetterBench workload | TP2 tok/s | TP4 tok/s |
| --- | ---: | ---: |
| Chat | 70.5 | 108.0 |
| Code | 91.6 | 142.6 |
| File edit | 111.4 | 159.6 |
| JSON | 125.3 | 177.5 |
| Math | 126.3 | 178.6 |
| Prose | 67.0 | 97.9 |
| Reasoning | 70.4 | 100.4 |
| Summarization | 118.5 | 167.4 |
| Combined | ~93.4 | ~136.9 |

Combined is a convenient aggregate of the categories above, not a claim that any particular prompt runs at that speed.

### BetterBench concurrency

Aggregate decode throughput across simultaneous requests:

| Concurrent requests | TP2 aggregate tok/s | TP4 aggregate tok/s |
| ---: | ---: | ---: |
| 1 | 83.7 | 126.0 |
| 2 | 141.1 | 189.6 |
| 4 | 238.6 | 337.1 |
| 8 | 325.3 | 471.0 |
| 16 | 321.6 | 541.2 |

BetterBench runs its concurrency phase after its decode and prefill phases, so these retained values are intentionally warm, not cold-start numbers.

### BetterBench prefill

Median prefill throughput by prompt depth:

| Prompt depth | TP2 tok/s | TP4 tok/s |
| ---: | ---: | ---: |
| ~2K | 2,792.0 | 3,608.4 |
| ~8K | 3,001.2 | 3,766.5 |
| ~16K | 2,932.5 | 3,747.7 |
| ~32K | 2,813.1 | 3,679.4 |
| ~64K | 2,565.5 | 3,501.7 |

The depth labels are readable roundings; BetterBench uses its actual corpus token counts, roughly 1,556 / 5,960 / 11,836 / 23,585 / 47,098 tokens for these five depths.

### Measurement method

BetterBench 0.4.0 with temperature 0.7, top_p 0.95, top_k 20, and unique_nonce enabled. Single-stream: 3 warmups plus 20 passes per category. Prefill: 2 warmups plus 8 measured runs per depth with max_tokens 16 and temperature 0. Concurrency: 48 requests at each of 1, 2, 4, 8, and 16.

RUNTIME_RECIPE.md remains the deep technical companion for this runtime's provenance and qualified launch contract.

## Runtime

The pack uses this final qualified runtime:

```text
ghcr.io/wu1ff/qwen38-27b-b70@sha256:78a3720f542f8d7974aa6cf38eff4bfd612fcecb1bc7c06a42ddeb4d6febe1aa
```

This is the 2026-09-12 promotion: the previous authority bytes plus one
installed change — the dFlash2 exact Gumbel-noise cache (per-request noise
is reused exactly instead of rebuilt every round; GPU RNG behavior
preserved; inert in Base and MTP1). The previous digest
`sha256:68f1d8a8…d00690b` remains pullable by digest as the retained parent.

Every launch requires `CCL_SYCL_ALLREDUCE_TMP_BUF=1` and
`CCL_SYCL_ALLGATHERV_TMP_BUF=1`. Patch 010's serving synchronization was
historical containment and is superseded for production. The 96-profile
matrix remains unchanged.

The dFlash2 graph policy is topology-qualified: PIECEWISE capture sizes
`[1,2,4,8,16,32,64]` with max capture size 64 at TP2 and TP4 (seven
proposals make one request verify 8 rows, so the old 8-row ceiling replayed
only single requests and left larger batches eager); the TP1 dFlash2
profiles keep the originally qualified `[1,2,4,8]` max 8. Base and MTP1
keep their own unchanged graph policies.

If the image is missing, `b70ctl` pulls that immutable reference and verifies that Docker reports the expected image ID before offering any profile that uses it.

The runtime mounts the B70 devices (`/dev/dri`) plus one fixed read-only bind of `/dev/dri/by-path` so serving resolves cards by their stable device paths. That exact bind is the only host mount the pack is allowed to request; `b70ctl` rejects any other pack-controlled mount.

## Runtime recipe

For the full runtime build history, compatibility work, qualified launch contract, and technical provenance, see [RUNTIME_RECIPE.md](RUNTIME_RECIPE.md).

## Using the pack

Import this pack directory with **Model Packs → Import Local Pack** while working from the source tree. Choose which checkpoints to prepare during installation; existing exact model revisions and the runtime are reused.

Then open **Run Model**, choose the checkpoint, and select from the card, context, and mode values shown. Choose Local or LAN access, set the port, and start it.

`pack.json` is intentionally readable if you want to inspect all 96 profiles, model file inventories, and launch data yourself.

## Licenses

The Controller's MIT license does not relicense the model weights, runtime components, or other third-party software. Check the upstream terms for each repository and component you use.
