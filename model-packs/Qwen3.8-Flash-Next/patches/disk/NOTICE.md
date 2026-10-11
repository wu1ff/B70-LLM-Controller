# Disk distributed-source notices

The cumulative overlays and native sources retain upstream vLLM and vllm-xpu-kernels Apache-2.0 licensing. The native base is vllm-project/vllm-xpu-kernels 0.1.14.1, commit 6d92b1bfbf32767ecda8e819613eb151e70030ad. SYCL-TLA is Intel commit 87f6850680a580654b9ea2c80dbc01aeb36ad231; its BSD license and embedded third-party notices are retained. licenses/ preserves upstream and inherited vendor/Torch notices. No model weights are distributed here; their terms remain with the pinned checkpoint.

Wulf changed the following upstream files for B70 Flash-Next Disk. Each distributed file now carries a prominent modification comment; all original upstream notices and code are retained. Paths below are relative to this directory. Added standalone extensions are recorded in the existing source manifests and diffs.

| Distributed modified files | Changes |
| --- | --- |
| `vllm/_xpu_ops.py`, `vllm/model_executor/layers/linear.py` | Native dispatch and XPU loading corrections |
| `vllm/models/qwen4_exp/__init__.py`, `vllm/models/qwen4_exp/nvidia/hyperconnection.py`, `vllm/models/qwen4_exp/nvidia/model_state.py`, `vllm/models/qwen4_exp/nvidia/mtp.py`, `vllm/models/qwen4_exp/nvidia/ngram_embedding.py` | BF16 mapped PLE, hyperconnection and MTP integration |
| `vllm/v1/attention/backends/gdn_attn.py` | Corrected GDN and metadata dispatch |
| `vllm/v1/core/sched/scheduler.py` | Conditional aggregate prefill budget |
| `vllm/v1/worker/gpu/attn_utils.py`, `vllm/v1/worker/gpu/model_states/mamba_hybrid.py`, `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py`, `vllm/v1/worker/xpu_model_runner.py` | APC/ALIGN, metadata and speculative staging |
| `vllm_xpu_kernels/flash_attn_interface.py`, `vllm_xpu_kernels/fused_moe_interface.py` | Attention reducer and guarded MoE dispatch |
| `native/source/csrc/moe/remap_hidden_states.cpp` | Global/local expert bounds and excluded-row pointer guard |
| `native/source/csrc/xpu/gdn_attn/causal_conv1d.hpp`, `native/source/csrc/xpu/gdn_attn/xe_2/chunk_causal_conv1d_xe2.hpp`, `native/source/csrc/xpu/gdn_attn/xe_2/chunk_causal_conv1d_tiled_xe2.hpp` | 64-bit indexing |
| `native/source/csrc/xpu/attn/xe_2/kernel/paged_decode_kernel.hpp`, `native/source/csrc/xpu/attn/xe_2/paged_decode.hpp` | Tiled attention reduction |

The `vllm/` and `vllm_xpu_kernels/` paths in this table are under `overlay/opt/venv/lib/python3.12/site-packages/`. Native paths are relative to this directory. The existing cumulative patches give the exact changes. Apache-2.0 requires notices on changed files and retention of its license/applicable upstream attribution; it does not impose a corresponding-source offer. BSD/MIT components likewise require their applicable notices, rather than a blanket source offer.

The derived serving configuration is accompanied by the actual Qwen Community License 1.0 in `licenses/external-model.txt`. No checkpoint weights or BF16 PLE table are included. Downstream commercial-use conditions remain those of that license.

**Registry status, 2026-10-11:** the corrected RC was published under the separately authorized GHCR replacement and verified as the sole `:disk` version. Public model-pack release remains unapproved. The local `qwen38-flashnext:disk-rc` candidate preserves all 15 Python modification comments in one added layer; the original qualified `:disk` image remains unchanged. See [licenses/README.md](licenses/README.md) for corresponding-source delivery and Intel recipient-license requirements. This NOTICE is neither a written source offer nor permission to publish the image.

Intel compiler packages are intentionally absent. Their exact identities and official acquisition location are locked in manifests/compiler-lock.json. The full compiler packages are build tools, and their presence in a local reproduction archive does not establish redistribution rights. Obtain them separately under their applicable terms. They are used only in the native build stage and do not enter the final image.

The B70 residency shim is Wulf's own work; no external shim permission is required. Its retained source has no separate license header; do not infer an external author's license or invent an additional permission requirement.

The completed contributor source audit found no identifiable KaloyanNaumov/Lumnus executable source incorporated in final Disk. Research and configuration influence remains acknowledged in provenance/. Historical commit messages and co-author trailers are preserved as historical records only; do not carry those trailers onto a new release commit as inferred co-authorship. PR #4 configuration direction informed four-slot conditional Base prefill; direct-pinned BF16 remained Speed research. INT8/NVMe, CPU-KV, API-default policy and alternate checkpoint implementations are excluded. Wulf's independent Disk mmap, scheduler, GDN, metadata, native and serving changes remain separately evidenced.

Issue #3 and PRs #4–#8 have completed historical closeouts. Issue #9, https://github.com/wu1ff/B70-LLM-Controller/issues/9, remains for final release reconciliation; this task changes no GitHub state.
