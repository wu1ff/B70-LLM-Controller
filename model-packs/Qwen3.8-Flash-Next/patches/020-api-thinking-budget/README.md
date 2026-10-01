# Per-effort thinking budget; server default presence penalty

- **Purpose:** Per-effort thinking budget; server default presence penalty.
- **Category:** api.
- **Pinned upstream base:** vLLM 0.30.0 XPU image `vllm/vllm-openai-xpu@sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9` (`gced6857af`; extracted `image-source/vllm`) plus patches 001–005 and the vLLM half of 006 (the 1.0.2 tree).
- **Files affected:** `vllm/entrypoints/openai/chat_completion/protocol.py`.
- **Applied runtime stage:** Not in a published image. Proposed stage: `b70-api`, on top of `gdn-index64-c1`.
- **Currently shipping:** No. Not in `ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0`; proposed. It runs on our 4× Arc Pro B70 host in a tree built from 001–006 plus this series (github.com/Lumnus/b70-flash-next, release `0.30.0-b70.1`).
- **Technical explanation:** Opt-in. `B70_THINKING_BUDGET=minimal=512,low=512,medium=2048,high=4096,xhigh=8192,max=12288,ultra=12288,default=8192` caps reasoning tokens by the effort the client asked for (`default` when none was sent); nothing applies when thinking is off, and a request's own `thinking_token_budget` wins. `B70_DEFAULT_PRESENCE_PENALTY` sets a server default; a request's own value wins. Ungated: `reasoning_effort` also accepts `"ultra"`. Pairs well with `--reasoning-config` and a closing `reasoning_end_str`, so the forced close reads as a transition rather than a cut mid-sentence.
- **Reproduction/application:** Apply `020-api-thinking-budget.patch` after 006 (vLLM half) from the vLLM tree root (`patch -p1 --fuzz=0`).
- **Retained provenance:** github.com/Lumnus/vllm, branch `b70/v0.30.0`, commit `051c3ffa53` (`0009-b70-thinking-budget-per-requested-effort`); exported as `patches/vllm/0009-b70-thinking-budget-per-requested-effort.patch` in github.com/Lumnus/b70-flash-next. This file is the same change rewritten as a plain unified diff against the 1.0.2 tree; applying the proposed patches to that tree gives files byte-identical to ours.
