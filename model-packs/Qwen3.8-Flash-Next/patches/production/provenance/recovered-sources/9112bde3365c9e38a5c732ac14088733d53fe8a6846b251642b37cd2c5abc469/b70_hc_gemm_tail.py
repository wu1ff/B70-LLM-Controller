

# --- B70 expanded campaign 2026-11-16 (exact; env-gated; default off) -------
# B70_DFLASH_EXACT_BA=1: GDN in_proj_ba at 9..32 rows (K=2560, N=24) uses a
#   native kernel computing the same single chained FP32 DPAS accumulation over
#   K as _ordered_hc_gemm (bitwise verified on random and live serving inputs).
# B70_DFLASH_HC_SPLIT32=10|20|40: native ordered HC down at exactly 32 rows uses
#   that split; every split computes identical per-tile partials and the same
#   sequential FP32 tile sum, so the result is bit-identical.
_b70_exact_ba = os.getenv("B70_DFLASH_EXACT_BA", "0") == "1"
_b70_hc_split32 = int(os.getenv("B70_DFLASH_HC_SPLIT32", "0"))
if _b70_hc_split32 not in (0, 10, 20, 40):
    raise ValueError("Unsupported exact-order HC split for 32 rows")
if _b70_exact_ba or _b70_hc_split32:
    if _b70_exact_ba:
        import vllm_xpu_kernels as _b70_vxk

        torch.ops.load_library(str(Path(_b70_vxk.__file__).with_name("libb70_exact_v1.so")))
    _b70_original_ordered_hc_gemm = ordered_hc_gemm
    _b70_original_parallel_hc_gemm = parallel_hc_gemm

    def ordered_hc_gemm(x, weight, valid_columns):
        if (
            _b70_exact_ba
            and x.dim() == 2
            and 9 <= x.shape[0] <= 32
            and x.shape[1] == 2560
            and tuple(weight.shape) == (24, 2560)
            and valid_columns == 24
            and x.dtype == weight.dtype == torch.bfloat16
            and x.device.type == "xpu"
            and x.is_contiguous()
            and weight.is_contiguous()
        ):
            return torch.ops.b70_exact_v1.gemm(x, weight, 1, 3)
        return _b70_original_ordered_hc_gemm(x, weight, valid_columns)

    def parallel_hc_gemm(x, weight, valid_columns):
        if (
            _b70_hc_split32
            and _ordered_native_hc
            and x.shape[0] == 32
            and x.shape[1] == 10240
            and weight.shape[0] in (320, 336)
            and x.is_contiguous()
            and weight.is_contiguous()
        ):
            return torch.ops.b70_ordered_hc_v2.forward(x, weight, valid_columns, _b70_hc_split32)
        return _b70_original_parallel_hc_gemm(x, weight, valid_columns)
