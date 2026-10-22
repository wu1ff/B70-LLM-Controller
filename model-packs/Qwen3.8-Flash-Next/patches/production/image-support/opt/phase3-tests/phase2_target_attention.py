"""Native target GQA6/HD256/page832 decode repeatability and masking oracle."""

import json
import torch
from vllm._xpu_ops import xpu_ops

torch.manual_seed(61)
q = torch.randn(1, 6, 256, device="xpu", dtype=torch.bfloat16)
cache = torch.randn(3, 1, 832, 512, device="xpu", dtype=torch.bfloat16)
k, v = cache.transpose(1, 2).split(256, dim=-1)
length = 14
reference = torch.einsum(
    "qhd,khd->hqk", q.float(), k[1, :length].float().expand(-1, 6, -1)
) * (256**-0.5)
reference = torch.einsum(
    "hqk,khd->qhd", reference.softmax(-1), v[1, :length].float().expand(-1, 6, -1)
)
outputs = []
for i in range(8):
    cache[1, :, length:].fill_(float(i) * 1000)
    out = torch.full_like(q, float(i) * 7)
    xpu_ops.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=torch.tensor([0, 1], device="xpu", dtype=torch.int32),
        max_seqlen_q=1,
        max_seqlen_k=length,
        softmax_scale=256**-0.5,
        causal=True,
        out=out,
        block_table=torch.tensor([[1]], device="xpu", dtype=torch.int32),
        seqused_k=torch.tensor([length], device="xpu", dtype=torch.int32),
    )
    outputs.append(out.cpu())
errors = [float((x.float() - reference.cpu()).abs().max()) for x in outputs]
equal = [torch.equal(x, outputs[0]) for x in outputs]
print(json.dumps({"max_abs_errors": errors, "repeated_bitwise_equal": equal}))
assert all(equal)
torch.testing.assert_close(outputs[0].float(), reference.cpu(), rtol=0.02, atol=0.01)
