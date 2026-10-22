"""Probe native reduction order, then use v0.32's fixed-rank BF16 reduction."""

import json
import time
from pathlib import Path

import torch
from vllm.distributed import get_tp_group


def run(worker):
    comm = get_tp_group().device_communicator
    original = comm.all_reduce
    torch.manual_seed(94 + worker.rank)
    data = torch.randn(2560, device="xpu", dtype=torch.bfloat16)
    records = {}
    for name, operation in (("native", original), ("fixed_rank", comm._fixed_rank_sum)):
        outputs = []
        for trial in range(16):
            if trial % 4 == worker.rank:
                time.sleep(0.003)
            outputs.append(operation(data).cpu())
        records[name] = {
            "bitwise_equal": [torch.equal(x, outputs[0]) for x in outputs],
            "max_abs_from_first": [
                float((x.float() - outputs[0].float()).abs().max()) for x in outputs
            ],
        }
    comm.all_reduce = comm._fixed_rank_sum
    Path(f"/work/phase2-audit/collective-probe-rank-{worker.rank}.json").write_text(
        json.dumps(records, indent=2)
    )
