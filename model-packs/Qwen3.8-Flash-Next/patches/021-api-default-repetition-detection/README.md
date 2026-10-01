# Server default for vLLM's repetition stop

- **Purpose:** Server default for vLLM's repetition stop.
- **Category:** api.
- **Pinned upstream base:** vLLM 0.30.0 XPU image `vllm/vllm-openai-xpu@sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9` (`gced6857af`; extracted `image-source/vllm`) plus patches 001–005 and the vLLM half of 006 (the 1.0.2 tree).
- **Files affected:** `vllm/entrypoints/openai/chat_completion/protocol.py`.
- **Applied runtime stage:** Not in a published image. Proposed stage: `b70-api`, on top of `gdn-index64-c1`.
- **Currently shipping:** No. Not in `ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0`; proposed. It runs on our 4× Arc Pro B70 host in a tree built from 001–006 plus this series (github.com/Lumnus/b70-flash-next, release `0.30.0-b70.1`).
- **Technical explanation:** Opt-in. `B70_DEFAULT_REPETITION_DETECTION=max=1,min=1,count=128` sets a server default for vLLM's built-in n-gram repetition stop (finish reason `repetition`) for clients that never send the field. A request's own `repetition_detection` wins. Chat endpoint only. We use it as a backstop against single-token NaN loops.
- **Reproduction/application:** Apply `021-api-default-repetition-detection.patch` after 020 from the vLLM tree root (`patch -p1 --fuzz=0`).
- **Retained provenance:** github.com/Lumnus/vllm, branch `b70/v0.30.0`, commit `21546bab3a` (`0010-b70-default-repetition-detection`); exported as `patches/vllm/0010-b70-default-repetition-detection.patch` in github.com/Lumnus/b70-flash-next. This file is the same change rewritten as a plain unified diff against the 1.0.2 tree; applying the proposed patches to that tree gives files byte-identical to ours.
