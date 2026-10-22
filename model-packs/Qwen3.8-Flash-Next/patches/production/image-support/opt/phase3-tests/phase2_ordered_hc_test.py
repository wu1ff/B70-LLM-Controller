"""Independent float64 oracle and repeatability for all relevant HC row widths."""

import json
import torch
from vllm.models.qwen4_exp.nvidia.b70_hc_gemm import ordered_hc_gemm

torch.manual_seed(7)
weights = (torch.randn(384, 10240) * 0.01).bfloat16()
weights[324:] = float("nan")
w = weights.to("xpu")
records = []
for rows in (1, 4, 13, 26, 64, 65, 256):
    x = (torch.randn(rows, 10240) * 0.2).bfloat16()
    expected = (x.double() @ weights[:324].double().T).bfloat16()
    gpu = x.to("xpu")
    first = ordered_hc_gemm(gpu, w, 324)
    for _ in range(8):
        result = ordered_hc_gemm(gpu.clone(), w, 324)
        assert torch.equal(first, result)
    actual = first[:, :324].cpu()
    relative = float(
        (actual.float() - expected.float()).norm() / expected.float().norm()
    )
    torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.001)
    assert first[:, 324:].count_nonzero() == 0
    records.append(
        {"rows": rows, "relative_l2": relative, "repeat_bitwise_equal": True}
    )
print(json.dumps(records, indent=2))
