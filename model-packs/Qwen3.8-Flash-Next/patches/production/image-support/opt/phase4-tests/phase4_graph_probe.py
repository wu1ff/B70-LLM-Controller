"""Isolated XCCL capture/replay qualification; never uses serving groups."""

import json
import os
from pathlib import Path
import traceback
import torch
import torch.distributed as dist

rank = int(os.environ["LOCAL_RANK"])
torch.xpu.set_device(rank)
dist.init_process_group("xccl")
device = torch.device("xpu", rank)
report = {"rank": rank, "passed": False}
try:
    x = torch.empty(4, 2560, device=device, dtype=torch.bfloat16)

    def reduce():
        y = x.clone()
        with dist._coalescing_manager(device=device):
            for row in y.split(1):
                dist.all_reduce(row)
        return y

    x.fill_(rank + 1)
    for _ in range(3):
        reduce()
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        y = reduce()
    errors = []
    for step in range(32):
        x.fill_(rank + 1 + step)
        graph.replay()
        if not torch.equal(y.cpu(), torch.full_like(y.cpu(), 10 + 4 * step)):
            errors.append({"step": step, "graph": y.cpu().flatten()[:10].tolist()})
        z = reduce()
        if not torch.equal(z.cpu(), torch.full_like(z.cpu(), 10 + 4 * step)):
            errors.append({"step": step, "eager": z.cpu().flatten()[:10].tolist()})
    report.update(passed=not errors, errors=errors)
except Exception:
    report["error"] = traceback.format_exc()
finally:
    Path(f"/work/phase4-audit/xccl-graph-rank-{rank}.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report), flush=True)
    dist.destroy_process_group()
