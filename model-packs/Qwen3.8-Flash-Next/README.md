# Qwen3.8-Flash-Next

This is the qualified Qwen3.8-Flash-Next (Qwen4Exp) pack I use with B70 LLM Controller. It contains one served checkpoint — weights, tokenizer, and the 102.4 GB PLE n-gram table together in a single pinned revision — a frozen unified runtime, and the exact B70 configurations that passed testing. Both modes, Base and MTP3, run on the same single runtime image and differ only by launch configuration.

## Models

| Variant | Repository | Revision |
| --- | --- | --- |
| W4A16 | `devan-carlin/Qwen3.8-Flash-Next-W4A16` | `40b8f18df4d4a32cb6e687a51c78207e5e438522` |

The repository is ungated and public. The pinned revision is 31 files, 179,841,735,233 bytes total (≈ 180 GB) — `ple_table_qwen4exp.pt` alone is 102,400,493,290 bytes and **is part of the revision**, so the download covers everything the runtime needs. `b70ctl` downloads the exact revision above and checks it against the 31-file inventory in `pack.json`; plan for the full ≈ 180 GB on first install.

## Modes

- **Base** serves the checkpoint without speculative decoding (util 0.85). The MTP drafter never loads in this mode.
- **MTP3** enables multi-token-prediction speculative decoding at exactly depth 3 (`{"method":"mtp","num_speculative_tokens":3}`, util 0.91): each proposal drafts exactly 3 tokens and the target verifies 4 rows. The mode display is `MTP3` — the depth is part of the contract; MTP1 exists in the runtime as a capability but is not a pack mode.

Mode availability depends on card count and context; a mode only appears when that exact combination has a profile.

## Hardware and context support

| Checkpoint | Cards / TP | Qualified contexts and modes |
| --- | --- | --- |
| W4A16 | 4 | 32K, 64K, 128K, 256K: Base, MTP3 |

TP4 on 4× Arc Pro B70 is the only topology this pack publishes. No sub-TP4 lane was ever qualified for this checkpoint (the W4A16 GEMM K≥6144 race is TP1/TP2-reachable), so no such profile exists.

Context ladder law: each mode was qualified at its maximum context — 262144 — on the identical runtime contract, and that qualification covers every lower tier the pack publishes (32K/64K/128K are data-only context ceilings; they require no separate boot). The native `max_position_embeddings` of the checkpoint is 262,144.

Only combinations that passed qualification are included. `b70ctl` reads the exact 8-profile matrix from `pack.json`, so unsupported combinations simply do not appear.

## Hard runtime requirements (baked into the image)

These are properties of the pinned runtime image, not options: pinned-host PLE table (2-slab UVA load, zero device residency — `PLE_TABLE_PATH` resolves into the read-only model mount), the B70 Level-Zero peer-residency shim (`LD_PRELOAD`), TP4 + expert parallelism, W4A16 bf16 serving, breakable FULL_AND_PIECEWISE XPU graphs (torch.compile mode NONE) — Base: `[1,2,4,8,16,256,512,1024]` at `--max-num-batched-tokens 1024` / `--max-num-seqs 16`; MTP3: `[1,2,4,8,256]` at `--max-num-batched-tokens 256` / `--max-num-seqs 4` (see [Base serving profile](#base-serving-profile)), automatic prefix caching ON with the ALIGN Mamba cache mode, and the dense-QSA serve configuration.

Dense QSA is a correctness requirement for this checkpoint: the pinned config ships five `text_config.indexer_*` keys but the checkpoint contains no indexer weights, so the runtime image carries the patched serving config (the pinned config minus exactly those five keys, sha256 `91fa33ca…`) and its entrypoint rebuilds the serve directory inside the container at boot (`/work/flashnext-serve`) from the read-only Controller model mount. The mounted snapshot is never written.

Vision is qualified (image modality, `{"image":2,"video":0}`); **video is not qualified** and stays at 0.

## Tool calling (1.0.1)

OpenAI-compatible automatic tool calling is part of the qualified launch contract (pack 1.0.1; the 1.0.0 pack omitted the flags and rejected `tool_choice:"auto"` with HTTP 400). The shared runtime layer of `pack.json` adds `--enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3` — one set of flags inherited identically by Base and MTP3 and every profile; the runtime image is unchanged from 1.0.0 (same digest `85512b52…`, no rebuild, no new GHCR tag).

Both parsers ship inside the pinned image, and the model's own chat template emits exactly the Qwen3 XML tool syntax the parser reads — no custom template, no other flags. What was tested and qualified at 262144 through the normal b70ctl path, on both Base and MTP3: `tools` + `tool_choice:"auto"` returning a valid OpenAI `tool_calls` structure (correct function name, JSON-exact arguments), tool-result round trip with the `tool` role, streaming tool calls, `tool_choice:"none"` returning plain text, and reasoning/content separation (vLLM 0.30 exposes the thinking under the message's `reasoning` field). Ordinary chat, vision, and prefix caching were regression-checked in the same smokes. Claim scope is the tested surface — a client-side calculator function schema; no claim about arbitrary tool schemas or structured outputs.

## Performance

Retained qualified results from the promoted unified runtime (BetterBench v0.6.0, 32K contract, temperature 0.7, single stream unless noted), not new measurements. Test host: 4× Intel Arc Pro B70 32 GB.

| Quantity | Base | MTP3 |
| --- | ---: | ---: |
| Combined weighted decode | 58.34 tok/s | 119.86 tok/s |
| Prefill @ ≈23.6K-token prompts | 2476 tok/s | 2290 tok/s |
| TTFT p50 (shorts) | 107.5 ms | 123.0 ms |
| Concurrency C4 aggregate | 183.7 tok/s | 138.3 tok/s |

Where each mode wins: MTP3 wins every single-stream decode category (+69% to +155%) and C1/C2 aggregates (2.00×/1.82×); Base wins prefill throughput at every depth (+7.3–8.5%), TTFT at every depth/level, and C4 aggregate (+32.7%). The two modes are complementary rather than overlapping. The numbers in this table are from the original qualification at `max_num_seqs 4` and a 256-token prefill chunk; the Base profile now uses 16 slots and a 1024-token chunk (next section). MTP3 keeps the qualified 4 / 256.

## Base serving profile

Base mode launches with `--max-num-batched-tokens 1024`, `--max-num-seqs 16` and capture sizes `[1,2,4,8,16,256,512,1024]`. These three flags moved from the shared runtime layer to the mode layer, so MTP3 still renders exactly the qualified `256` / `4` / `[1,2,4,8,256]` (only the argument order changed). Utilisation stays 0.85.

Measured on 4× Arc Pro B70, TP4 + EP, this checkpoint (`devan-carlin` @ `40b8f18d`), vLLM v0.30.0 with patches 001–006. The prefill row ran in an image built from the official v0.30.0 XPU image with these patches plus one boot-memory patch; the slots row ran with the PLE table in INT8 (proposed separately) at utilisation 0.87–0.88:

| change | measured |
| --- | --- |
| prefill chunk 256 → 1024 | prefill 2,674 → 4,711 tok/s at an 18.7K-token prompt, 1,686 → 3,711 tok/s at 98K; decode, C4 aggregate and MMLU/TruthfulQA unchanged (paired test, p = 1.0) |
| 4 → 16 slots | chat mix (1–4K in, 0.5–1.5K out) aggregate decode 181 / 289 / 364 / 425 tok/s at 4 / 8 / 12 / 16 concurrent, 0 preemptions; an agent mix of 20–60K-token prompts saturates at 8 and preempts at 16 (KV pool full) |

**Keep every capture size a power of two.** The 4 → 16 measurement above used `[1,2,4,6,8,12,16,256,512,1024]`. With those non-power-of-two sizes some requests in the minutes after a boot produced a repeated single token from the first output token on (NaN logits; with `logprobs` the response is HTTP 400 "Out of range float values are not JSON compliant: nan"). Under the same load, 40 of 84 requests hit it in 7 minutes with that list and 0 of 115 in 18 minutes with `[1,2,4,8,256,512,1024]`. The 16-slot power-of-two list above has served on the same vLLM tree since. With this checkpoint it held 20 minutes of 16 concurrent ~40K-token uncached prompts: 104 requests, 0 errors. This is strong evidence, not a proof: we have not isolated which size triggers it, so we keep every size a power of two.

## Runtime

The pack uses this single qualified runtime — one image for all 8 profiles and both modes:

```text
ghcr.io/wu1ff/qwen38-flashnext-b70@sha256:85512b52c09fa660a2e7fe441417129e7c47fac727fd85f66ccea6b65e0a9122
```

This is the 2026-09-27 packaging layer (serve preparator + patched config + entrypoint) on top of the unified runtime authority that serves Base (speculation off) and MTP3 (depth 3) from one image — including the GDN convolution state-index 64-bit correctness correction that fixed the deep-prefill DEVICE_LOST failure mode. No separate Base or MTP3 runtime image exists or is referenced.

If the image is missing, `b70ctl` pulls that immutable reference and verifies that Docker reports the expected image ID before offering any profile that uses it. On Docker daemons using the containerd image store, the image ID equals the registry manifest digest recorded in `pack.json`, and verification succeeds; a classic-graphdriver daemon reports the config digest instead, which would not match. This pack targets the containerd image-store daemon class, same as the existing packs.

The runtime mounts the B70 devices (`/dev/dri`) plus one fixed read-only bind of `/dev/dri/by-path` so serving resolves cards by their stable device paths. That exact bind is the only host mount the pack is allowed to request; `b70ctl` rejects any other pack-controlled mount. All caches live inside the container (`/work/*`).

## Source patches (1.0.2)

The pack now distributes the [exact source patches and build recipes](patches/README.md) behind the published Flash-Next B70 runtime. This is a source and provenance release: the production image remains `ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0` at digest `sha256:85512b52c09fa660a2e7fe441417129e7c47fac727fd85f66ccea6b65e0a9122`, and all eight profile launch settings remain those of pack 1.0.1.

## Runtime recipe

For the runtime build history, compatibility work, qualified launch contract, and technical provenance, see [RUNTIME_RECIPE.md](RUNTIME_RECIPE.md).

## Using the pack

Install from the public catalog via **Model Packs → Browse Available Packs**, or import this pack directory with **Model Packs → Import Local Pack** while working from the source tree. There is one target checkpoint; preparing it is the ≈ 180 GB download. Existing exact model revisions and the runtime are reused on update.

Then open **Run Model**, choose the checkpoint, and select from the card, context, and mode values shown. Choose Local or LAN access, set the port, and start it. First boot takes roughly six minutes to readiness.

`pack.json` is intentionally readable if you want to inspect all 8 profiles, the model file inventory, and launch data yourself.

## Licenses

The Controller's MIT license does not relicense the model weights, runtime components, or other third-party software. Check the upstream terms for each repository and component you use.
