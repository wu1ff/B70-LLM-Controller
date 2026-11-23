

# --- B70 expanded campaign 2026-11-16: exact fused MoE glue (env-gated) -----
# B70_DFLASH_MOE_GLUE=1 binds per-block fast paths (router gate + shared expert)
# that reproduce the original arithmetic bit-for-bit; see b70_moe_glue.py.
if __import__("os").getenv("B70_DFLASH_MOE_GLUE", "0") == "1":
    from .b70_moe_glue import wrap_block_forward as _b70_wrap_moe_block

    Qwen4ExpSparseMoeBlock.forward = _b70_wrap_moe_block(Qwen4ExpSparseMoeBlock.forward)
