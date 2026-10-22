"""Install reversible candidate implementations in the campaign workers."""

from contextlib import contextmanager
from types import MethodType
import torch
import torch.distributed as dist


def run(worker, command):
    from vllm.models.qwen4_exp.nvidia import c1_verify

    if command.get("grouped"):
        from vllm.distributed import get_tp_group

        @contextmanager
        def canonical(width):
            comm = get_tp_group().device_communicator
            original = comm.all_reduce

            def reduce(x):
                if x.dtype == torch.bfloat16 and x.ndim == 2 and x.shape[0] == width:
                    output = x.clone()
                    with dist._coalescing_manager(
                        group=comm.device_group, device=x.device
                    ):
                        for row in output.split(1):
                            dist.all_reduce(row, group=comm.device_group)
                    return output
                return original(x)

            comm.all_reduce = reduce
            try:
                yield
            finally:
                comm.all_reduce = original

        c1_verify.canonical_reductions = canonical
    if command.get("hc"):
        from phase4_hc_probe import project

        def down(module, x):
            linear = (
                module.input_mix_weight_down_block_inject
                if module.use_combine
                else module.input_mix_weight_down
            )
            valid = module.lora_rank + (module.hc_count if module.use_combine else 0)
            if x.shape[0] <= 4:
                return project(x, linear.weight, valid, rm=1, bn=16)
            from vllm.models.qwen4_exp.nvidia.b70_hc_gemm import ordered_hc_gemm

            return ordered_hc_gemm(x, linear.weight, valid)

        for module in worker.model_runner.model.modules():
            if hasattr(module, "_down_gemm"):
                module._down_gemm = MethodType(down, module)
    return {"rank": worker.rank, "applied": command}
