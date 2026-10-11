# Qwen3.8-Flash-Next — Disk

This pack serves Qwen3.8-Flash-Next W4A16 with B70 LLM Controller using one qualified Disk runtime image and separate Base and MTP3 modes. Version **1.0.3** promotes the qualified Disk runtime and cumulative source/build documentation. The image is available in the existing GHCR package under `:disk`; the [1.0.3 release](https://github.com/wu1ff/B70-LLM-Controller/releases/tag/qwen3.8-flash-next-v1.0.3) provides the complete model-pack archive and its SHA256 checksum.

## Supported configurations and features

| Checkpoint | Hardware | Context selections | Modes |
| --- | --- | --- | --- |
| `devan-carlin/Qwen3.8-Flash-Next-W4A16` | 4× Intel Arc Pro B70, TP4 + expert parallelism | 32768, 65536, 131072, 262144 | Base, MTP3 |

All eight existing profile identities are retained. The native 262144 ceiling includes **input plus output**. Final packaged-image qualification covered both modes at native context, including 262112 input + 32 output; lower contexts are rendered ceilings and have not each received an independent serving campaign. TP1, TP2, MTP1 and video are not pack configurations.

Image vision (at most two images per prompt), reasoning separation, OpenAI tools, automatic prefix caching and Mamba ALIGN remain supported. Automatic tool choice uses the in-image `qwen3_xml` parser with `qwen3` reasoning. `--exclude-tools-when-tool-choice-none` preserves the tool-choice fix: requests with `tool_choice:"none"` exclude tool definitions from the rendered prompt. Required and named tools, tool round trips, streaming, ordinary chat and reasoning were covered by retained qualification. Arbitrary schemas and model-generated code are not guaranteed correct.

## Checkpoint, storage and host memory

The ungated checkpoint is pinned to revision `40b8f18df4d4a32cb6e687a51c78207e5e438522`: 31 files, **179,841,735,233 bytes** total. This includes the original **102,400,493,290-byte BF16 PLE table**, integrated MTP tensors, tokenizer and vision configuration. Budget roughly 180 GB for the checkpoint plus image, caches and installation staging. No conversion or external assistant checkpoint is needed.

Disk PLE maps the original BF16 table from the read-only checkpoint and gathers only TP-owned rows into bounded pinned staging before device execution. At MBT1024 that stage is **5 MiB per rank / 20 MiB across TP4**. It removes Vanilla's whole-table pinned slabs without changing PLE precision. File-backed page cache is reclaimable but can grow substantially; Disk still uses host memory for the runtime, weights, caches and mapped pages. No minimum-RAM or storage-cold endurance guarantee has been established. SSD/NVMe latency and OS cache history affect performance.

The image's entrypoint builds `/work/flashnext-serve` using checkpoint symlinks and a baked dense-QSA config: the pinned config minus exactly five unsupported `text_config.indexer_*` keys. It leaves the mounted checkpoint intact. The B70 residency shim, XPU graphs, offline settings and caches under `/work` are part of the launch contract.

## Base and MTP3

| Setting | Base | MTP3 |
| --- | --- | --- |
| GPU memory utilization | 0.85 | 0.91 |
| Speculation | Off; drafter not loaded | Integrated MTP, exactly 3 tokens; local argmax reduction |
| Batch-token budget / slots | 1024 / 4 | 1024 / 4 |
| Prefill policy | Aggregate cap 256 when a request reaches decode | Fixed 1024; qualified draft-prefill gates |
| FULL_AND_PIECEWISE captures | 1,2,4,8,256,512,1024 | 1,2,4,8,64,128,256,512,832,1024 |

Base's conditional cap supports mixed prefill/decode scheduling. MTP3 retains the accepted eager/KV-only draft prefill, post-capture warmup, fused metadata, device split plan, native tiled reducer and 65536-byte CCL LL threshold. The exact environments and arguments are in `pack.json`; no performance tuning is required to select either mode.

## Startup and retained performance

Broad standalone qualification of the preserved Disk predecessor reached health in **274.46 seconds Base / 292.23 seconds MTP3**, once per mode. The corrected RC retains equivalent executable logic and native payload; its focused Controller qualification reached health in **265.3 seconds Base / 294.4 seconds MTP3** and passed completion, MTP3 streaming/stopping and native-library checks. Allow several minutes; compilation and cache history vary. Controller Start returns while the server is loading. Runtime shows Starting until `/health` succeeds; it provides no automatic startup timeout or rollback.

Retained thinking-enabled FixedStepClock coding requests (seed57, temperature0.7/top_p0.95/top_k20, 21 functional checks) give these descriptive single-stream rates:

| Prompt / cache | Base tok/s | MTP3 tok/s | Base / MTP3 TTFT seconds |
| --- | ---: | ---: | --- |
| 65845 / cold APC | 50.00 | 120.33 | 17.257 / 16.823 |
| 65845 / reused APC | 50.24 | 122.21 | 0.317 / 0.476 |
| 164149 / cold APC | 45.50 | 108.77 | 56.365 / 56.624 |
| 164149 / reused APC | 45.48 | 111.12 | 0.692 / 1.112 |

These are retained n=1 observations with differing output lengths/trajectories, not a universal speed claim or matched Base regression estimate. Cold APC means a fresh request cache namespace, with natural OS page-cache history. Four-slot mixed serving passed functional checks; throughput/fairness, sustained native-ceiling generation and storage-cold endurance remain unmeasured. Historical Vanilla BetterBench results are retained in the recipe and should not be treated as Disk measurements.

## Runtime identity and installation

The published GHCR tag is `ghcr.io/wu1ff/qwen38-flashnext-b70:disk`. The exact qualified RC Docker image ID is:

```text
sha256:b13fed6dd54051395f603409ada440c74afcbd5331810eee3acd62315d7787e3
```

`pack.json` pins that ID in `runtime.digest` and uses the verified immutable acquisition reference in `runtime.registry`:

```text
ghcr.io/wu1ff/qwen38-flashnext-b70@sha256:b13fed6dd54051395f603409ada440c74afcbd5331810eee3acd62315d7787e3
```

The remote linux/amd64 manifest, config and all 22 layer descriptors match the qualified RC. An anonymous clean pull through Controller's unchanged `runtime.AcquireOwned` returned this exact Docker `.Id`; the pulled installed inventory matched all 3,118 entries, including all 15 corrected Python notices. The registry manifest happens to equal `.Id` on the supported containerd image store. The distinct OCI config digest is `sha256:2399885f6fbc59cdb602fe869350ddd80e085f62da25d55859d683b5e3450baf`; classic image stores can expose that config ID and fail the existing Controller identity contract. Controller launches by the verified ID.

The existing GHCR package now exposes exactly one version with the sole tag `disk`. The superseded Vanilla `1.0.0` index, amd64 manifest and attestation were removed only after replacement verification. Original local Disk/Vanilla images and source history remain available for recovery; `qwen38-flashnext:disk` retains its predecessor identity and `qwen38-flashnext:disk-rc` retains the corrected RC.

Use the real TUI menus:

1. **Model Packs → Browse Available Packs**, select Flash-Next 1.0.3 and the W4A16 target, Continue and confirm installation. For a downloaded and extracted archive, **Import Local Pack** accepts its directory. Complete exact checkpoints and the qualified local image are reused; an absent runtime is acquired by the verified immutable registry reference and checked against its Docker ID. Import does not switch a running server.
2. **Run Model**, choose the target (and version label if several versions are installed), four cards, a context, Base or MTP3, Local or LAN access, and port 8000. Select Start.
3. **Runtime → Refresh / View Logs** to follow readiness. Use **Stop** before switching mode or context; the next Start replaces the stopped managed container. Exiting the TUI leaves the detached service running.

**Browse Available Packs** and Update are the normal distribution path. Verify the downloaded archive against the release’s `SHA256SUMS` when importing manually. There are no pack-management command-line subcommands in `b70ctl`.

## Limitations and manual rollback

Native context qualification used a 32-token output at the ceiling; sustained long generation there remains unmeasured. Retain the recipe's bounded numerical/reduction-order equivalence, first-use/cache effects, unresolved historical synthetic-startup-audit cause and model-output limitations. No broader topology, video, concurrency performance or minimum-RAM claim follows from these checks.

Retain the older Vanilla **1.0.2 source pack/archive** and image before updating. A successful Controller update can retire older packs and unshared owned images; KeepModels only preserves checkpoints. To roll back, stop Disk in Runtime, **Import Local Pack** from the retained Vanilla directory, then choose its version-labelled target and the desired Base/MTP3 profile. Start and wait for health. Catalog browsing does not offer downgrades, and the Controller does not automatically roll back a later serving failure.

Vanilla recovery uses the retained **local** image `sha256:85512b52c09fa660a2e7fe441417129e7c47fac727fd85f66ccea6b65e0a9122` and saved 1.0.2 source/archive. Its former GHCR reference is retained only in dated history and is no longer a download or rollback source. Keep these local assets before any Controller update can retire unshared owned images.

For a failed managed Disk integration, the retained qualified standalone `launch.py` and prepared Base/MTP3 receipts offer recovery after freeing port 8000. See the existing [runtime recipe](RUNTIME_RECIPE.md) for build identities, archive checksums and recovery commands.

## Sources, provenance and licenses

[patches/README.md](patches/README.md) describes the cumulative accepted Disk source, native builds, dependency locks and checksums. Historical Vanilla 001–007 deltas remain labelled history. The public source export excludes Intel compiler package binaries; reproducible native builds require the exact locked packages to be obtained separately.

No identifiable KaloyanNaumov/Lumnus executable source ships in Disk. Research and configuration influence remains acknowledged in the historical provenance records. Historical co-author trailers describe those records; they do not assign authorship to a new release. The residency shim is Wulf's own work.

The Controller's MIT license does not relicense model weights or third-party runtime components. Preserve the actual vLLM/kernel Apache-2.0 notices, SYCL-TLA BSD and embedded third-party notices, and vendor/Torch notices carried with the distributed sources and image.

The release preserves the original local qualified `:disk` image and uses the notice-corrected `:disk-rc` as GHCR `:disk`. Installed executable logic, native binaries and runtime configuration remain equivalent. The distributed licenses/notices and historical [source-delivery findings](patches/disk/licenses/README.md) are retained without changing third-party terms. The complete accepted Disk implementation and native build recipes ship inside the model-pack archive under `patches/disk/`; compiler packages and model weights are acquired separately. Earlier provisional 1.0.3 artifacts are historical evidence and are superseded by the published archive and release checksum. The historical [Vanilla 1.0.2 release](https://github.com/wu1ff/B70-LLM-Controller/releases/tag/qwen3.8-flash-next-v1.0.2) and source remain available, but its retired GHCR image cannot be pulled there.
