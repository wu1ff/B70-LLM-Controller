"""Retain native one-row reductions while removing copies and host calls."""

import json
from pathlib import Path
import time
import torch
import torch.distributed as dist
from vllm.distributed import get_tp_group


def run(worker, command):
    comm = get_tp_group().device_communicator
    torch.manual_seed(864 + worker.rank)
    x = torch.randn(4, 2560, device=worker.device, dtype=torch.bfloat16)
    native = comm.all_reduce

    def baseline():
        return torch.cat([native(row) for row in x.split(1)])

    def inplace():
        out = x.clone()
        for row in out.split(1):
            dist.all_reduce(row, group=comm.device_group)
        return out

    def asynchronous():
        out = x.clone()
        works = [
            dist.all_reduce(row, group=comm.device_group, async_op=True)
            for row in out.split(1)
        ]
        for work in works:
            work.wait()
        return out

    def grouped():
        out = x.clone()
        with dist._coalescing_manager(group=comm.device_group, device=worker.device):
            for row in out.split(1):
                dist.all_reduce(row, group=comm.device_group)
        return out

    ref = baseline()
    records = []
    for name, fn in [
        ("baseline", baseline),
        ("clone_once", inplace),
        ("async", asynchronous),
        ("coalesced", grouped),
    ]:
        try:
            y = fn()
            torch.xpu.synchronize()
            equal = torch.equal(y, ref)
            a = torch.xpu.Event(enable_timing=True)
            b = torch.xpu.Event(enable_timing=True)
            t = time.perf_counter()
            a.record()
            for _ in range(40):
                fn()
            b.record()
            b.synchronize()
            records.append(
                dict(
                    name=name,
                    equal=equal,
                    queue_ms=a.elapsed_time(b) / 40,
                    wall_ms=(time.perf_counter() - t) * 1000 / 40,
                )
            )
        except Exception as e:
            records.append(dict(name=name, error=repr(e)))
    Path(f"/work/phase4-audit/reduce-variants-rank-{worker.rank}.json").write_text(
        json.dumps(records, indent=2)
    )
    return records
