# Qwen3.8 Flash-Next Disk — W4A16 on four B70 GPUs

[Overview](#overview) · [Optimizations](#runtime-optimizations) · [Performance](#betterbench-performance) · [Benchmark system](#benchmark-system-and-methodology) · [Observed Resource Usage](#observed-resource-usage) · [Installation](#supported-configurations-and-installation) · [Limitations and licenses](#limitations-attribution-and-licenses)

## Overview

Flash-Next Disk is a released runtime for serving **Qwen3.8-Flash-Next W4A16 on four Intel Arc Pro B70 GPUs**, using tensor parallelism (TP4) and expert parallelism. It maps the original BF16 PLE embedding table from checkpoint storage and gathers the rows needed for execution, avoiding large pinned-memory slabs holding the complete table.

Model pack **1.0.3** provides one runtime image with **Base** and **MTP3** modes. MTP3 uses the checkpoint's integrated draft model for three-token speculative decoding; no external assistant checkpoint or PLE conversion is needed. Both modes support context ceilings up to **262144 tokens**, image vision, tool calling, reasoning and automatic prefix caching.

## Runtime optimizations

### Memory architecture

- **Disk-backed BF16 PLE mapping** preserves the original table's precision while letting the OS manage file-backed pages.
- **Selected-row gathering** transfers only each TP rank's required rows rather than uploading a complete PLE shard.
- **Bounded pinned staging** uses **5 MiB per rank / 20 MiB across TP4** at the 1024-token batch budget (MBT1024). This is the PLE staging allocation, not total host-memory use.
- The required **B70 peer-residency shim** contains host-memory mirroring of large peer allocations while preserving pool and collective residency.

### Prompt processing

- **Wider target prefill and graph captures:** MTP3 uses a 1024-token budget and expanded capture sizes for target prompt processing.
- **Intermediate draft-tail suppression:** wholly intermediate prefill keeps the first prompt-KV pass and skips its two unused later proposals.
- **KV-only draft prefill:** eligible intermediate batches populate the draft's prompt KV while skipping discarded attention output, MoE and proposal work. Final prefill, mixed batches and decode retain ordinary execution.
- **Eager true draft prefill:** avoids a known large draft-prefill graph registration failure while retaining target-prefill and target/draft decode graphs.
- **Post-capture synchronization and warmup:** clears startup capture state and fences the collective path before readiness.
- **Base conditional scheduling:** pure prefill can use 1024 tokens; once a running request reaches decode, the aggregate scheduled-token budget becomes 256, including decode work. MTP3 retains a fixed 1024 budget.

### Generation

- **Fused speculative attention metadata** generates eligible live integer metadata on the device, reducing host-side staging.
- **Device-generated split plans** use current sequence lengths on every replay rather than lengths fixed during graph capture.
- **Native tiled attention reduction** handles qualified MTP3 attention shapes, with native fallback for other shapes.
- **Selected collective configuration** enables oneCCL temporary buffers and uses a 65536-byte small-message all-reduce threshold in MTP3.

Required correctness protections include the hyperconnection (HC) wide-GEMM K-split workaround, GDN 64-bit state indexing and partial-verification/padding guards, and bounds-checked MoE expert remapping. These changes have bounded numerical/reduction-order equivalence; they do not guarantee arbitrary model outputs. See the [runtime recipe](RUNTIME_RECIPE.md) for implementation, construction and qualification details.

## BetterBench performance

**114.0 tok/s weighted single-stream generation**, measured with the released Disk image in MTP3 mode on 2026-10-11 using BetterBench v0.6.0.

This is BetterBench's default weighted mean of six category decode medians: **code 30%, reasoning 20%, prose 15%, JSON 15%, file editing 10%, summarization 10%**. Chat and math are measured with zero headline weight. It is not the arithmetic mean of all eight medians. The weighting and timing definitions are documented in the [methodology](../../betterbench-results/2026-10-11-flash-next-disk-mtp3/METHODOLOGY.md).

### Single-stream generation

| Category | Median decode (tok/s) | TTFT p50 (ms) |
| --- | ---: | ---: |
| Chat | 99.0 | 104.8 |
| Code | 107.4 | 99.0 |
| File editing | 124.2 | 109.8 |
| JSON | 143.3 | 106.7 |
| Math | 139.9 | 104.6 |
| Prose | 100.9 | 104.0 |
| Reasoning | 97.1 | 105.4 |
| Summarization | 132.9 | 115.2 |

Each category completed **20/20 measured requests**. Decode throughput excludes time to first token (TTFT); TTFT ends at the first text-bearing reasoning or answer update.

### Prefill

| Target prompt tokens | Actual prompt tokens, median | Median TTFT (s) | Median prompt processing (tok/s) |
| ---: | ---: | ---: | ---: |
| 2,000 | 1,556.5 | 0.630 | 2,490 |
| 8,000 | 5,934.5 | 1.926 | 3,099 |
| 16,000 | 11,842 | 3.356 | 3,525 |
| 32,000 | 23,589.5 | 6.125 | 3,849 |

Each depth completed **8/8 measured requests**, with a 16-token generation cap. Targets are approximate; prompt-processing throughput is each request's actual prompt tokens divided by TTFT, including network and first-token generation time. The largest measured prompt was **23,689 tokens**, well below the runtime's 262144 ceiling.

### Concurrency

| Concurrent requests | Aggregate output (tok/s) | Per-request median decode (tok/s) | TTFT p50 (ms) | Successful requests |
| --- | ---: | ---: | ---: | ---: |
| C1 | 111.0 | 122.1 | 103.8 | 48/48 |
| C2 | 184.0 | 108.6 | 172.8 | 48/48 |
| C4 | 249.0 | 73.0 | 200.1 | 48/48 |

Aggregate throughput is successful output tokens divided by the level's elapsed wall time, including TTFT and scheduling. It is not the sum of per-request medians. Total output rises with load while individual streams slow; C4 does not scale linearly from C1.

All **336 measured requests** succeeded. Full distributions, stream-update metrics and percentile warnings are in the [standalone HTML report](../../betterbench-results/2026-10-11-flash-next-disk-mtp3/results.html), with [sanitized results JSON](../../betterbench-results/2026-10-11-flash-next-disk-mtp3/results.json) for inspection. Download the HTML file to view its interactive charts in a browser. **All performance numbers here describe MTP3; Base was not included in this campaign.**

## Benchmark system and methodology

| Component | Measured system |
| --- | --- |
| CPU | AMD EPYC 7443, one socket, 24 cores / 48 threads |
| Motherboard | HUANANZHI H12D-8D V2.0; four NUMA nodes |
| Memory | 256 GiB DDR4, eight 32 GiB DIMMs; frequency not independently verified |
| GPUs | Four Intel Arc Pro B70 GPUs, 32 GiB each |
| GPU links | Verified host-facing PCIe 4.0 ×16 on all four cards |
| OS / kernel | Ubuntu 26.04 LTS / Linux 7.2.0-070200-generic |
| Checkpoint storage | Samsung MZVLB1T0HBLR-000L2 NVMe, 1.024 TB, ext4 |
| Released runtime | Pack 1.0.3, `ghcr.io/wu1ff/qwen38-flashnext-b70:disk`, identity `b13fed6dd540…` |
| Serving stack | vLLM 0.30.0+xpu; PyTorch 2.13.0+xpu; Triton XPU 3.7.2; oneCCL 2022.0.0 |
| Serving profile | TP4 + expert parallelism, BF16 activations, MTP3, 262144 context, four slots, MBT1024, GPU memory utilization 0.91 |
| Benchmark | BetterBench v0.6.0, corpus 1.0; source commit `d00ad5ec8098c06584a88ec3468bacd37d5ed098` |

The [hardware record](../../betterbench-results/2026-10-11-flash-next-disk-mtp3/HARDWARE.md) retains the exact image identity and driver/runtime versions. DDR4 firmware speed fields disagree, so no operating frequency is claimed.

The campaign used all eight categories (29 corpus prompts): three discarded warmups and 20 measured passes per category; two warmups and eight samples at each 2K–32K prefill target; and 48 requests at each of C1/C2/C4. It reused one healthy MTP3 server with the **262144-context profile**, without exercising prompts near that ceiling.

Fresh nonce prefixes and regenerated prefill bodies prevented prior prompt-prefix/KV reuse. APC remained enabled; the OS page cache was retained. **Fresh-prefix measurements are not storage-cold measurements.**

Interpretation limits:

- Category and concurrency sampling used temperature 0.7, top_p 0.95 and top_k 20, without a request seed override. Prefill used temperature 0.
- Corpus output caps were retained: **126/160 category requests (78.8%)** reached their limit. Generated tokens include reasoning and potentially incomplete answers; these are performance measurements, not task-quality scores.
- Four first-use Triton JIT warnings appeared during concurrency. Associated latency spikes remain in the results; later runs may avoid those compilation costs.
- TTFT p99 has only 20 category or 48 concurrency observations; prefill has eight samples per depth. Tail estimates are under-sampled and should be treated as roughly the slowest observed, not reliable population percentiles.

### Observed Resource Usage

**2026-10-11, post-BetterBench, loaded and idle observations:** six read-only snapshots five seconds apart, plus three privileged GPU checks; zero running/waiting requests. This is **one known-working system, not established minimum hardware requirements**. GiB/MiB are binary units.

| Resource | Observed usage | Capacity / interpretation |
| --- | --- | --- |
| Host RAM, idle | **32.65–32.70 GiB used; 214.78–214.84 GiB available** | 256 GiB installed; 247.48 GiB OS-visible. MemTotal − MemAvailable estimates host-wide RAM not readily available for new work, rather than all physically occupied RAM. |
| Linux cache, idle | **131.59–131.63 GiB** active + inactive file pages; 1.78 GiB reclaimable kernel cache | Nonfree RAM was 164.03–164.12 GiB including cache. |
| Runtime cgroup, idle | **27.65 GiB**: 20.37 GiB anonymous, 6.72 GiB file pages | Docker displays 25.65 GiB after subtracting inactive file cache. |
| Swap, idle | **13.22 MiB host; zero runtime usage** | 50.00 GiB host swap. |
| GPU 0 VRAM, idle | **29.63 GiB used; 2.26 GiB available** | 31.89 GiB usable; 32 GiB advertised. |
| GPU 1 VRAM, idle | **29.24 GiB used; 2.65 GiB available** | Same capacity. |
| GPU 2 VRAM, idle | **29.24 GiB used; 2.65 GiB available** | Same capacity. |
| GPU 3 VRAM, idle | **29.24 GiB used; 2.65 GiB available** | Same capacity. |
| PLE, idle / reported staging | **18.59 GiB file-cache residency; 20 MiB pinned stage TP4** | Original BF16 file 95.37 GiB / 102.4 GB; staging is only part of host RAM. |
| CPU, idle | Runtime **6.57–7.53% of one logical CPU** (0.137–0.157% of all 48); host 0.51–1.33% | Load averages 0.17–0.26 / 0.12–0.13 / 0.11; host includes other work/inspection. |
| Checkpoint / filesystem | **167.49 GiB / 179.84 GB** checkpoint; **180.72 GiB available** | Actual ext4 filesystem capacity 887.62 GiB. |
| Docker / caches | Image-list size **15.87 GiB**; writable layer **157.09 MiB** | Shared image bytes included; unique cost unavailable. Writable layer includes Triton 16.83 MiB and XDG 8.44 MiB caches. |

VRAM uses privileged xe accounting, corroborated by **sudo xpu-smi**; GPU IDs follow xpu-smi indices. Whole-device usage includes driver/other clients. vLLM's **reported allocations** are separate: 19.95 GiB/rank model-loading delta, 5.24–5.25 GiB/rank KV, 1.00 GiB/rank graph-capture delta and 403,444-token KV capacity. Host, cgroup, worker and PLE-cache views overlap; **do not add them**. Clean mapped PLE pages remain reclaimable and shared cache may be charged outside the runtime cgroup.

**Under load**, retained BetterBench samples show API-process RSS 2.97–2.99 GiB and peak KV occupancy 10.51%. Full-system/worker RAM and CPU, and per-device used VRAM under load, were not captured. Peak system memory and per-device VRAM under C4 were not measured. KV occupancy is not VRAM utilization. No additional inference was run.

## Supported configurations and installation

Both modes use four B70 GPUs, TP4 + expert parallelism, a 1024-token batch budget and four sequence slots.

| Mode | Speculative decoding | GPU memory utilization | Prefill scheduling | BetterBench campaign |
| --- | --- | ---: | --- | --- |
| Base | Disabled; no drafter loaded | 0.85 | Conditional aggregate cap 256 during mixed prefill/decode | Not included |
| MTP3 | Integrated MTP, exactly three tokens, local argmax reduction | 0.91 | Fixed 1024 budget with draft-prefill optimizations | Measured above |

Each mode offers **32768, 65536, 131072 or 262144 tokens**, counting input plus output. Supported features include:

- **Image vision:** up to two images per prompt.
- **OpenAI-compatible tool calling:** automatic (`qwen3_xml`), required and named tools, including tool round trips. `tool_choice:"none"` excludes tool definitions from the rendered prompt.
- **Reasoning:** `qwen3` parsing separates reasoning and answer channels; ordinary chat and streaming are supported.
- **Automatic prefix caching (APC)** and Mamba ALIGN state handling.

Install [B70 LLM Controller](../../README.md#installation), with working Intel XPU host drivers and Docker, then launch `b70ctl`:

1. **Model Packs → Browse Available Packs:** select Flash-Next **1.0.3**, choose the W4A16 target, Continue and confirm installation. Controller prepares the checkpoint and acquires/verifies the pinned runtime as needed. To install a downloaded [release archive](https://github.com/wu1ff/B70-LLM-Controller/releases/tag/qwen3.8-flash-next-v1.0.3), verify its release `SHA256SUMS`, extract it and select **Import Local Pack** with that directory.
2. **Run Model:** select the target/version, four cards, context, Base or MTP3, Local or LAN access, and a host port (for example 8000). Select **Start**.
3. **Runtime → Refresh / View Logs:** allow several minutes for loading and graph/JIT setup. Wait for **Running** before sending requests. Use **Stop** before changing mode or context; exiting the TUI leaves the service running.

With Local access and port 8000, point an OpenAI-compatible client at `http://127.0.0.1:8000/v1`, using model name `devan-carlin/Qwen3.8-Flash-Next-W4A16`. Pack management uses the TUI; `b70ctl` has no pack-install CLI subcommands.

The exact supported contract is in [pack.json](pack.json); construction and standalone serving are in [RUNTIME_RECIPE.md](RUNTIME_RECIPE.md), with the [complete runtime sources](patches/disk/README.md).

## Limitations, attribution and licenses

- **Hardware and modes:** only four-card B70 TP4 is packaged. TP1, TP2, MTP1 and video are unsupported.
- **Context:** native-ceiling qualification covered 262112 input + 32 output tokens. Sustained generation at the ceiling remains unmeasured; lower context selections were rendered/checked but did not each receive an independent serving campaign.
- **Concurrency:** four slots are supported. BetterBench used short corpus prompts at C1/C2/C4; it does not establish long-prompt throughput, fairness or sustained high-load behavior.
- **Storage and RAM:** see [Observed Resource Usage](#observed-resource-usage) for measured checkpoint, cache, RAM and VRAM usage. File-backed page cache can grow as different PLE rows are touched; minimum hardware requirements and storage-cold endurance remain unestablished.

The runtime retains bounded numerical equivalence and first-use/cache effects; arbitrary generated code and tool schemas can fail. The recipe records remaining qualification limits, including an unresolved historical synthetic-startup nonfinite cause.

Wulf's Disk implementation and residency shim retain upstream attribution. KaloyanNaumov/Lumnus research and configuration influence is acknowledged; the completed audit found no identifiable contributor executable source in final Disk. See the [provenance ledger](patches/disk/provenance/PATCH_PROVENANCE_LEDGER.md) and [contribution closeout](patches/disk/provenance/contribution-closeout.md).

The Controller's **MIT** license does not relicense model weights or runtime dependencies. The checkpoint/derived configuration uses the [Qwen Community License 1.0](patches/disk/licenses/external-model.txt); vLLM and XPU kernels retain **Apache-2.0**, and SYCL-TLA retains **BSD** and embedded third-party notices. Preserve the [distributed modification notices](patches/disk/NOTICE.md), vendor/Torch notices and each component's terms. The [dated license/source-delivery findings](patches/disk/licenses/README.md) retain unresolved corresponding-source and Intel recipient-term items, including the NumPy-bundled libgfortran source-identity gap; release availability does not resolve those findings.
