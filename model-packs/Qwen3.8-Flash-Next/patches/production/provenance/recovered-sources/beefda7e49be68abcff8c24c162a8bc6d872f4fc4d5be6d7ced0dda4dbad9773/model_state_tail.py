

# --- B70 expanded campaign 2026-11-16: deferred Disk-PLE prepare (env-gated) --
# B70_DFLASH_DEFER_PLE=1: C1-verify batches run the PLE prepare after the target
# graph's input update instead of before it (host ordering only); see b70_defer_ple.py.
if __import__("os").getenv("B70_DFLASH_DEFER_PLE", "0") == "1":
    from .b70_defer_ple import install as _b70_install_defer_ple

    _b70_install_defer_ple(Qwen4ExpModelState)
