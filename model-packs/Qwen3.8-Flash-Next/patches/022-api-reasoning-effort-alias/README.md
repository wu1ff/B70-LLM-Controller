# OpenRouter-style reasoning.effort alias

- **Purpose:** OpenRouter-style reasoning.effort alias.
- **Category:** api.
- **Pinned upstream base:** vLLM 0.30.0 XPU image `vllm/vllm-openai-xpu@sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9` (`gced6857af`; extracted `image-source/vllm`) plus patches 001–005 and the vLLM half of 006 (the 1.0.2 tree).
- **Files affected:** `vllm/entrypoints/openai/chat_completion/protocol.py`.
- **Applied runtime stage:** Not in a published image. Proposed stage: `b70-api`, on top of `gdn-index64-c1`.
- **Currently shipping:** No. Not in `ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0`; proposed. It runs on our 4× Arc Pro B70 host in a tree built from 001–006 plus this series (github.com/Lumnus/b70-flash-next, release `0.30.0-b70.1`).
- **Technical explanation:** Opt-in with `B70_REASONING_EFFORT_ALIAS=1`: maps `{"reasoning": {"effort": X}}` onto `reasoning_effort` when the request did not set it (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, `ultra`; anything else is ignored), so 020's budgets and the template effort apply to clients that send the OpenRouter shape (Hermes does).
- **Reproduction/application:** Apply `022-api-reasoning-effort-alias.patch` after 021 from the vLLM tree root (`patch -p1 --fuzz=0`).
- **Retained provenance:** github.com/Lumnus/vllm, branch `b70/v0.30.0`, commit `547ce5aca0` (`0012-b70-reasoning-effort-alias`); exported as `patches/vllm/0012-b70-reasoning-effort-alias.patch` in github.com/Lumnus/b70-flash-next. This file is the same change rewritten as a plain unified diff against the 1.0.2 tree; applying the proposed patches to that tree gives files byte-identical to ours.
