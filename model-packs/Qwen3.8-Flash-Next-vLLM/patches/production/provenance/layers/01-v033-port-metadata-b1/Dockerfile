FROM local/dflash2-v033:vendor-exact
COPY overlay/ /
ENV CCL_SYCL_ALLREDUCE_TMP_BUF="1"
ENV HF_HUB_OFFLINE="1"
ENV LD_PRELOAD="/opt/b70-residency-shim/libl0_peer_residency_shim.so"
ENV TMPDIR="/work/tmp"
ENV TRANSFORMERS_OFFLINE="1"
ENV XDG_CACHE_HOME="/work/xdg-cache"
ENV CCL_SYCL_ALLGATHERV_TMP_BUF="1"
ENV CCL_SYCL_ALLREDUCE_LL_THRESHOLD="65536"
ENV VLLM_CACHE_ROOT="/work/vllm-cache"
ENV VLLM_TARGET_DEVICE="xpu"
ENV HF_DATASETS_OFFLINE="1"
ENV TRITON_CACHE_DIR="/work/triton-cache"
ENV VLLM_WORKER_MULTIPROC_METHOD="spawn"
ENV VLLM_XPU_DFLASH_FIXED_SPLITS="32"
ENV ZE_FLAT_DEVICE_HIERARCHY="FLAT"
ENV B70_PLE_DISK_BF16="1"
ENV HF_HOME="/work/hf-cache"
ENV PLE_TABLE_PATH="/models/model-00016-of-00017.safetensors"
ENV VLLM_USE_V2_MODEL_RUNNER="1"
ENV ZE_AFFINITY_MASK="0,1,2,3"
ENV VLLM_XPU_INC_WNA16_BACKEND="w4a16"
ENV B70_DFLASH_LIVE_GRAPH_CHECKS="1"
ENV B70_DFLASH_FUSED_GDN_METADATA="1"
WORKDIR /opt/venv
ENTRYPOINT ["/opt/b70-flashnext/prepare-serve.sh"]
CMD ["vllm", "serve", "/models", "--host", "0.0.0.0", "--port", "8000", "--dtype", "bfloat16", "--enable-expert-parallel", "--max-num-seqs", "4", "--max-num-batched-tokens", "1024", "--gpu-memory-utilization", "0.91", "--limit-mm-per-prompt", "{\"image\":2,\"video\":0}", "--trust-remote-code", "--enable-prefix-caching", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_xml", "--reasoning-parser", "qwen3", "--exclude-tools-when-tool-choice-none", "--enforce-eager", "--served-model-name", "Intel/Qwen3.8-Flash-Next-W4A16-AutoRound", "--speculative-config", "{\"method\": \"dflash\", \"model\": \"/draft\", \"num_speculative_tokens\": 7, \"draft_tensor_parallel_size\": 4, \"draft_sample_method\": \"probabilistic\", \"rejection_sample_method\": \"standard\", \"attention_backend\": \"TRITON_ATTN\"}", "--tensor-parallel-size", "4", "--max-model-len", "262144"]
