"""BF16 HC glue parity against the Intel XPU float-accumulation contract."""
import unittest
import torch
from vllm.models.qwen4_exp.nvidia.ops.hc import hc_combine, hc_combine_norm, hc_gate_mix


class HCMath(unittest.TestCase):
    def test_bf16_completed_hc_and_normalized_mix(self):
        torch.manual_seed(19)
        for rows in (1, 4, 65, 256):
            x = torch.randn(rows, 10240, dtype=torch.bfloat16, device='xpu')
            block = torch.randn(rows, 2560, dtype=torch.bfloat16, device='xpu')
            injection = torch.randn(rows, 4, dtype=torch.bfloat16, device='xpu')
            weight = torch.randn(10240, dtype=torch.bfloat16, device='xpu') / 4
            ref32 = x.float().view(rows,4,2560) + block.float()[:,None,:] * (2 * torch.sigmoid(injection.float()/4))[:,:,None]
            reference = ref32.flatten(1).to(torch.bfloat16)
            completed = hc_combine(x, block, injection, 4)
            torch.testing.assert_close(completed.cpu(), reference.cpu(), rtol=0.008, atol=0.03125)
            actual_x, normalized = hc_combine_norm(x, block, injection, weight, 1e-6, 4)
            torch.testing.assert_close(actual_x.cpu(), completed.cpu(), rtol=0, atol=0)
            groups = reference.float().view(rows,4,2560)
            norm = (groups * torch.rsqrt(groups.square().mean(-1,keepdim=True) + 1e-6)).flatten(1)
            norm = (norm * (1+weight.float())).to(torch.bfloat16)
            torch.testing.assert_close(normalized.cpu(), norm.cpu(), rtol=0.016, atol=0.03125)
            gate = torch.randn_like(x)
            mixed = hc_gate_mix(x, gate, 4)
            refmix = (x.float().view(rows,4,2560) * torch.sigmoid(gate.float().view(rows,4,2560))).mean(1).to(torch.bfloat16)
            torch.testing.assert_close(mixed.cpu(), refmix.cpu(), rtol=0.008, atol=0.015625)


if __name__ == '__main__':
    unittest.main(verbosity=2)
