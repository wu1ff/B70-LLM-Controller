# Qwen3.8 Flash-Next Disk — Intel AutoRound on four B70 GPUs

Pack **1.0.4** serves Intel’s W4A16 AutoRound checkpoint with BF16 activations, TP4 + expert parallelism and **262144-token context**. One qualified profile provides four request slots and dynamic MTP: five verification tokens for one scheduled request, three for scheduled batches of two through four. The proposer can still produce five drafts.

Image `ghcr.io/wu1ff/qwen38-flashnext-b70@sha256:dc25c8cd205a828ad4df3ae9aa3e9505f0a51550010ac9a115336473ca268086` (single-platform linux/amd64; tag `disk`). Checkpoint `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`, revision `4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897`.

## Optimizations and supported behavior

- Intel-only preparation creates a dense serving view while preserving all 2340 AutoRound exclusions. The 128 contiguous BF16 PLE shards map directly from Intel’s safetensor after layout validation; no conversion, duplicate table or devan-carlin checkpoint is required.
- Selected-row PLE gathering uses a protected private mapping and **5 MiB pinned staging per rank / 20 MiB TP4**. Mapped file cache and other runtime allocations are additional host memory. The B70 residency shim remains required.
- MBT1024, expanded target captures, intermediate draft-tail suppression, KV-only/eager draft prefill and post-capture synchronization retain the qualified prefill path. Dynamic decode preserves single-query draft descriptors.
- Device-generated attention metadata and split plans, split-major scheduling for wider queries, native q2/q4/q6/q8 tiled reduction and the selected oneCCL buffers/65536-byte threshold accelerate generation.
- Uniform GDN metadata uses the actual verification length while retaining every reserved state slot; mixed/nonuniform/prefill batches retain the fallback.
- HC wide-GEMM, GDN 64-bit/partial-verification guards and bounds-checked MoE remapping remain intact. APC uses Mamba ALIGN with 832-token pages.

Image vision supports up to two images per prompt. Tools support automatic, required, named, disabled and continuation requests through `qwen3_xml`; `qwen3` separates reasoning and answer channels. The packaged qualified Qwen frontend preserves tokenizer, system/developer and high-effort reasoning behavior. Video, other TP sizes, fixed Base/MTP3 modes and other configured context ceilings are not advertised.

## BetterBench performance

**120.38 tok/s weighted single-stream generation**, measured on 2026-10-20 with the concurrency-corrected packaged image, BetterBench v0.6.0 / corpus 1.0. Weights: code 30%, reasoning 20%, prose 15%, JSON 15%, file editing 10%, summarization 10%; chat and math have zero headline weight.

| Category | Median decode (tok/s) | TTFT p50 (ms) | Success/attempts |
| --- | ---: | ---: | ---: |
| chat | 92.4 | 107.9 | 20/20 |
| code | 115.4 | 105.8 | 20/20 |
| file_edit | 137.9 | 110.2 | 20/20 |
| json | 162.8 | 107.1 | 20/20 |
| math | 164.6 | 104.8 | 20/20 |
| prose | 95.5 | 103.8 | 20/20 |
| reasoning | 91.9 | 107.1 | 20/20 |
| summarization | 148.5 | 117.5 | 20/20 |

| Prefill target | Actual prompt median | TTFT median (s) | Prompt processing (tok/s) | Success/attempts |
| ---: | ---: | ---: | ---: | ---: |
| 2,000 | 1,556.5 | 0.392 | 3,996 | 8/8 |
| 8,000 | 5,934.5 | 1.194 | 4,966 | 8/8 |
| 16,000 | 11,842.0 | 2.410 | 4,918 | 8/8 |
| 32,000 | 23,589.5 | 4.941 | 4,778 | 8/8 |

| Load | Aggregate output (tok/s) | Per-request decode median (tok/s) | TTFT p50 (ms) | Success/attempts |
| --- | ---: | ---: | ---: | ---: |
| C1 | 116.77 | 137.97 | 105.4 | 48/48 |
| C2 | 180.36 | 101.13 | 173.8 | 48/48 |
| C4 | 242.62 | 73.60 | 201.2 | 48/48 |

The preserved pre-fix Intel RC measured 118.91 weighted tok/s and C2/C4 aggregate 101.53/153.62 tok/s. The correction changes these by +1.2% and +77.6%/+57.9%. Against the 2026-10-11 devan-carlin/fixed-MTP3 release, final weighted throughput changes +5.6% and C2/C4 aggregate -2.0%/-2.5%. The historical release used another checkpoint and NVMe/filesystem; the matched Intel probes isolate the metadata correction. Five-token drafting and the original serving contract remain intact.

336/336 measured requests succeeded; 368 total calls including warmups, 0 failures, 0 preemptions and 0 prefix hits. Each category used three discarded warmups + 20 measurements; each prefill depth used two warmups + eight measurements; C1/C2/C4 each used 48 requests. Original sampling (temperature 0.7, top_p 0.95, top_k 20), corpus limits, nonce policy and timing were retained; prefill used temperature 0 and a 16-token cap.

Decode is `(completion_tokens−1)/(last text update−first text update)`; TTFT includes the first reasoning/answer update. Prefill is actual prompt tokens / TTFT; aggregate output includes TTFT and scheduling. OS file cache remains warm. The weighted headline is a mean of category medians. Stream updates may contain several tokens. 109/160 category responses reached their corpus limit; this is throughput measurement, not a quality score. Tail percentiles remain under-sampled.

## System and resource observations

AMD EPYC 7443 (24 cores / 48 threads), HUANANZHI H12D-8D, four NUMA nodes, 256 GiB DDR4 and four 32 GiB Intel Arc Pro B70s with host-facing PCIe 4.0 ×16. Ubuntu 26.04 / Linux 7.2.0, Docker 29.7.2 with containerd image store; vLLM 0.30.0 XPU, Torch 2.13.0+xpu, Triton XPU 3.7.2 and oneCCL 2022.0.0. DIMM firmware fields conflict, so no operating memory frequency is claimed. Checkpoint storage is a Solidigm SSDPFKNU020TZ 2 TB NVMe on XFS (`noatime`). No host/GPU tuning occurred.

The resource table below preserves loaded-idle observations from the pre-fix qualified Intel RC. The correction changes no memory limit, PLE allocation or residency setting; these GPU/PSS readings were not remeasured for the corrected image.

| Resource | Pre-fix RC observation |
| --- | --- |
| Host RAM after campaign, loaded idle | 33.30–33.32 GiB used (MemTotal−MemAvailable); 214.16–214.19 GiB available |
| Active + inactive file cache | 195.66–195.66 GiB; overlaps other memory views |
| Runtime cgroup / summed process PSS | 28.30–28.30 / 49.48–49.48 GiB; not additive |
| VRAM, GPU 0 / 1 / 2 / 3 | 29.74 / 29.35 / 29.35 / 29.35 GiB used, privileged whole-device snapshots |
| KV | 364,492 reported token capacity after restart; 16.31% peak occupancy in 15-second BetterBench samples |
| Storage | 181.24 GB Intel snapshot; 95.37 GiB PLE payload within it; 4.34 GB compressed image layers |

Those six idle snapshots followed the preserved pre-fix campaign. Host, cgroup, PSS and file-cache accounting overlap. VRAM includes driver/other-client use; KV occupancy is not VRAM utilization. These observations do not establish minimum RAM, a host-memory peak or storage-cold performance. SSD/NVMe and writable `/work` space for preparation/caches are required.

## Installation

Use Docker’s containerd image store with working Intel XPU drivers/devices. In `b70ctl`, choose **Model Packs**, select the Flash-Next pack and the Intel target, then install. Controller acquires the pinned image and Intel snapshot. In **Run Model**, select four cards, dynamic MTP5, 262144 context and port 8000; wait for **Running**. Exiting the TUI leaves the service running.

For local import, download `qwen38-flashnext-b70-1.0.4.tar.gz` and `SHA256SUMS` from the [1.0.4 release](https://github.com/wu1ff/B70-LLM-Controller/releases/tag/qwen3.8-flash-next-v1.0.4), verify the checksum, extract into a new directory and choose **Model Packs → Import Local Pack**. The archive includes the complete model-pack source.

Use OpenAI-compatible endpoint `http://127.0.0.1:8000/v1`, model `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`. The exact contract is in [pack.json](pack.json); rebuilding and standalone serving are in [RUNTIME_RECIPE.md](RUNTIME_RECIPE.md).

## Qualification and licenses

The corrected artifact passed 150 exact metadata cases, concurrent finite-logprob checks, cold/APC 262112-input + 32-output checks, ALIGN boundaries, short-after-long fingerprints, finite sampled logprobs, vision, tools, reasoning, concurrent coding/tool requests, direct C1–C4 schedule proof and restart. All four native binaries match the prior four-GPU/256-case reducer qualification. Sustained native-ceiling generation, video, more than four slots and arbitrary quality/fairness are outside this qualification.

Wulf’s Disk implementation/shim retains upstream attribution; Astra’s selected Intel/dynamic-MTP5 work and KaloyanNaumov/Lumnus research/configuration influence are recorded in [notices](patches/disk/NOTICE.md) and provenance. The Controller’s MIT license does not relicense the runtime or model. Qwen frontend/configuration and weights retain the [Qwen Community License 1.0](patches/disk/licenses/external-model.txt); vLLM/XPU kernels retain Apache-2.0 and SYCL-TLA its BSD/third-party notices. [Inherited source-delivery findings](patches/disk/licenses/README.md), including the NumPy-bundled libgfortran identity gap and Intel recipient terms, remain outstanding; publication does not assert their completion.
