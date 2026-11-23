# Qwen3.8 Flash-Next DFlash2 — vLLM 0.33

This runtime serves Qwen3.8 Flash-Next on four Intel Arc Pro B70 GPUs through an
OpenAI-compatible API. It combines an Intel AutoRound W4A16 target with a BF16 DFlash2
draft, custom XPU kernels and graphs, and disk-backed BF16 PLE embeddings.

## Benchmark system

| Component | Configuration |
| --- | --- |
| GPUs | 4× Intel Arc Pro B70, 32 GiB each |
| CPU | AMD EPYC 7443, 24 cores / 48 threads |
| RAM | 256 GiB installed RAM |
| Storage | NVMe-backed storage |
| Runtime | Customized vLLM 0.33.0 XPU |

## BetterBench performance

Two independent stock BetterBench 0.6.1 runs used the accepted v6 image and unchanged serving profile.
Decode and concurrency used the manufacturer's recommended thinking configuration: **temperature 1.0,
top-p 0.95, top-k 20**. Each run completed 344 measured samples, with 34 warmups discarded.

| Metric | Run 1 | Run 2 |
| --- | ---: | ---: |
| Weighted decode | 178.57 tok/s | 183.85 tok/s |
| C1 aggregate | 164.59 tok/s | 159.76 tok/s |
| C2 aggregate | 233.55 tok/s | 238.36 tok/s |
| C4 aggregate | 333.78 tok/s | 340.10 tok/s |
| Weighted TTFT (category p50) | 131.18 ms | 129.19 ms |

Weighted decode is the weighted arithmetic mean of category median single-stream decode rates,
excluding TTFT. C1/C2/C4 aggregate throughput measures total output tokens over the concurrency
sweep's wall time, including scheduling and TTFT.

Stock prefill uses greedy sampling. Each depth has eight measurements per run; the reported rate
is the median of prompt tokens divided by TTFT. Nominal targets are approximate filler sizes;
actual token counts are shown separately.

| Nominal prefill target | Actual prompt tokens, median R1 / R2 | Prefill, median R1 | Prefill, median R2 |
| --- | ---: | ---: | ---: |
| 16K (16,000) | 11,871 / 11,863 | 4,827.04 tok/s | 4,801.71 tok/s |
| 32K (32,000) | 23,576 / 23,625 | 4,695.76 tok/s | 4,677.96 tok/s |
| 64K (64,000) | 47,082.5 / 47,169.5 | 4,327.08 tok/s | 4,309.97 tok/s |

## What this runtime is

The W4A16 target, `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`, uses 4-bit weights and BF16 execution
and supplies predictions and multimodal features. The smaller, text-only BF16 draft,
`incoai/Qwen3.8-Flash-Next-DFlash2`, uses target features to propose seven tokens per cycle;
STANDARD target verification decides which proposals to accept. The target uses TP4 / EP4,
and the draft uses TP4 with probabilistic sampling.

The custom work is runtime engineering: kernels, speculative execution, state handling,
memory placement and scheduling. It does not fine-tune the target or draft checkpoint weights.
A supplied calibration tensor adjusts draft proposal scoring within that runtime.

## Engineering and optimizations

### Speculative decoding and recurrent-state correctness

The DFlash2 integration carries target features into the draft and handles accepted prefixes,
partial verification and padded request batches. Shared Gated DeltaNet (GDN) metadata staging
uses actual query lengths and aligned state indices, with guarded fallbacks for other layouts.
This keeps recurrent checkpoints and request slots consistent as acceptance and concurrency vary.

Proposal calibration weights unary and residual scores separately at each proposal position.
It acts on draft scores; target verification retains STANDARD accept/reject handling. The retained
thinking EOS guard suppresses premature end-of-sequence during the initial open reasoning segment
and recognizes tool handoffs.

### Native GPU kernels and BF16 convolution

Custom SYCL kernels accelerate GDN projections, hyperconnection (HC) reductions and mixture-of-experts
(MoE) execution. Dispatch selects kernels by shape and alignment; ordered reductions preserve the
intended arithmetic where required. Fused MoE work combines routing, shared-expert projections,
activation and gating, with per-block temporary-state cleanup to prevent reuse across batches.

The draft's grouped convolution fuses coefficient preparation, products and accumulation while
retaining the original BF16 rounding boundaries. These paths reduce launches and intermediate
memory traffic.

### Target/query graphs and input staging

Guarded target-verification and draft-query graphs replay supported C1–C4 decode shapes. Graph
ownership and output/state checks protect request-specific buffers; other shapes retain fallbacks.
The profile's generic eager setting coexists with these explicitly managed XPU graph paths.

Graph-input staging compiles the fixed destination-tree traversal into direct host operations,
retaining schema checks and copy ordering. A fused byte-copy kernel batches eligible contiguous
copies into one launch. Together these changes reduce repeated Python work and GPU idle time
between draft execution and target verification.

### Disk-backed PLE and preparation

The original BF16 PLE embedding table is mapped read-only from the target checkpoint.
Only tensor-parallel-owned rows enter bounded pinned staging before transfer to the GPUs, avoiding
a permanently pinned whole-table host allocation. The residency shim manages GPU memory placement;
the accepted profile uses eight CPU gather threads over the same mapped table.

For eligible verification batches, deferred PLE preparation lets graph-input copies queue before
the blocking row preparation. Device operations remain on the same stream, with flushing before
eager execution. Automatic prefix caching (APC) reuses eligible prompt prefixes and recurrent state.
NVMe latency and the reclaimable OS page cache still affect preparation and startup costs.

## System capabilities

The native limit is 262,144 input-plus-output tokens. The accepted profile reported **460,944 shared
KV-cache tokens** across four request slots. Capacity depends on active requests and cached prefixes;
one simultaneous native-limit request was qualified.

The runtime supports APC, streaming, separated Qwen3 reasoning, automatic `qwen3_xml` tool calls and
tool-result continuation, and schema-constrained structured output. Vision accepts up to two images
per prompt through the multimodal target; the speculative draft is text-only. Video is disabled.

## Installation and operation

Operation requires a compatible Linux Intel Xe/Level Zero stack, a container runtime and access to
the GPUs. B70 LLM Controller manages model acquisition, profile selection and container lifecycle.
Keep the complete model-pack source archive: Controller import retains the manifest and README,
not the full source tree. Use [RUNTIME_RECIPE.md](RUNTIME_RECIPE.md) for exact checkpoint revisions,
dependencies, setup, runtime configuration and launch procedures.

See the [tuning guide](docs/TUNING_GUIDE.md) for engineering decisions and measured results.

## Sources and notices

[Production sources and reconstruction](patches/production/README.md),
[build provenance and boundaries](patches/production/BUNDLE.md),
[retained source-access record](patches/licenses/SOURCE_ACCESS.md),
[credits and notices](patches/licenses/NOTICE.md) and
[source-distribution findings](patches/licenses/SOURCE_DISTRIBUTION.md).
