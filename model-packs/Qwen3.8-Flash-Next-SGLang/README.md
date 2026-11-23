# DFlash2 — SGLang

Qwen3.8 Flash Next W4A16 with the trained BF16 DFlash2 draft. Pack `dflash2-sglang` 1.0.0 requires **B70 LLM Controller 1.0.2 or newer**.

## Tested system and BetterBench results

Four Intel Arc Pro B70 GPUs (32,656 MiB each), an AMD EPYC 7443, 256 GiB of RAM, and NVMe checkpoint storage. The host used Linux `7.2.0-070200-generic` with `xe` and Docker 29.7.2 with the containerd image store. The Ubuntu 24.04.5 runtime uses Torch 2.14.0+xpu, SGLang based on `ecd3d75a2a790cdb3acba0ecdb4786ff4d4ab25d`, and sglang-kernel-xpu 0.6.0.

Two stock BetterBench 0.6.1 runs per runtime used corpus 1.0 and the same workload configuration, with 344 measured requests per run.

| Metric | SGLang | vLLM 0.33 DFlash2 v6 |
| --- | ---: | ---: |
| Weighted single-stream decode, tok/s | 199.47 / 199.94 | 178.57 / 183.85 |
| Weighted category TTFT p50, ms | 95.88 / 95.99 | 131.18 / 129.19 |
| C1 aggregate output, tok/s | 186.04 / 174.36 | 164.59 / 159.76 |
| C2 aggregate output, tok/s | 282.58 / 263.68 | 233.55 / 238.36 |
| C4 aggregate output, tok/s | 389.76 / 368.96 | 333.78 / 340.10 |

In the two SGLang runs, 79 and 86 single-stream requests reached the stock length cap during reasoning.

## Qualified capabilities

The `sglang-dflash2-tp4-262144` profile uses TP4/EP4, four active request slots, up to 262,144 input-plus-output text tokens per request, automatic prefix caching, and exact STANDARD DFlash2 sampling with seven proposals. Tested API paths include Chat Completions, streaming, function tool calls, constrained output, foreground Responses, and bounded image input.

Full-length multimodal requests, concurrent near-limit text, other GPU configurations, and extended soak remain unqualified. Stored or background Responses and hosted tools are outside this profile.

## Requirements

- Controller 1.0.2+, four B70s, a working `xe` driver and firmware, and Docker access to render devices `/dev/dri/renderD128` through `/dev/dri/renderD131`.
- About 170 GiB of NVMe storage for the managed checkpoints, plus space for the runtime image and JIT cache.
- Docker must use the containerd image store. Controller acquires runtime `sglang-a0042` by immutable digest. [Runtime notices](https://github.com/wu1ff/B70-LLM-Controller/releases/download/v1.0.2/dflash2-sglang-a0042-NOTICES.txt)

```text
Image  ghcr.io/wu1ff/qwen38-flashnext-b70@sha256:32d45cbaee2e127c79646045cb04dc9235cf3a991d6e137f2cb4cb07f27775e4
ID     sha256:32d45cbaee2e127c79646045cb04dc9235cf3a991d6e137f2cb4cb07f27775e4
```

| Checkpoint | Hugging Face repository | Revision |
| --- | --- | --- |
| W4A16 target | `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` | `4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897` |
| BF16 draft | `incoai/Qwen3.8-Flash-Next-DFlash2` | `a22a1744e85c17364dcc54a6a420af7135d4c7f6` |

## Installation and operation

1. Extract the pack. In `b70ctl`, open **Model Packs → Import Local Pack** and select the extracted directory.
2. Use Controller to prepare the pinned target and draft. Keep the Controller-managed directories and `.b70-model.json` markers. Existing vLLM v6 checkpoints at the same revisions can be reused.
3. In **Run Model**, select **Qwen3.8 Flash Next W4A16 — SGLang** and start the profile. Local binds `127.0.0.1` and LAN binds all host interfaces, on host port 8000 by default. Controller maps that port to 18093 and checks `/ready`. Use API model `flash-next-a0003` under `/v1`.
4. Use **Runtime** to check status, view logs, or stop serving. The first request at a new shape may need compilation. Controller keeps the Triton and TorchInductor caches between starts.

This runtime builds on SGLang, sgl-kernel-xpu, Intel SYCL-TLA, and the [B70 DFlash2 v6 runtime and adaptations](https://github.com/wu1ff/B70-LLM-Controller/releases/tag/qwen3.8-flash-next-v1.0.6). The target is Intel AutoRound's quantization of Qwen's checkpoint. Inco AI supplies the trained draft. The checkpoints and runtime are obtained separately under their respective terms.

**A detailed tuning guide and source patch set are planned for a future update.**
