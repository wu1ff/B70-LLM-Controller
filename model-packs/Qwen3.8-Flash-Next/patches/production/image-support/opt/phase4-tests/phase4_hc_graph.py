"""Stateless HC-only graph experiment; no communication or cache operations."""

import json
from pathlib import Path
import torch


def run(worker, command):
    from vllm.models.qwen4_exp.nvidia import c1_verify
    from vllm.distributed import get_tp_group

    runner = worker.model_runner
    graph_records = []
    active = [False]
    pool = torch.xpu.graph_pool_handle()
    methods = []

    def install(module, name):
        original = getattr(module, name)
        entry = {}
        methods.append((module, name, original))

        def call(*args):
            if not active[0] or args[0].shape[0] != 4:
                return original(*args)
            if not entry:
                inputs = [x.clone() if x is not None else None for x in args]
                original(*inputs)
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph, pool=pool):
                    output = original(*inputs)
                entry.update(graph=graph, inputs=inputs, output=output, count=0)
                graph_records.append(entry)
            else:
                for dest, source in zip(entry["inputs"], args):
                    if dest is not None:
                        dest.copy_(source)
            entry["graph"].replay()
            entry["count"] += 1
            return entry["output"]

        setattr(module, name, call)

    for module in runner.model.modules():
        if hasattr(module, "_down_gemm"):
            install(module, "mix")
            install(module, "combine_and_mix")
    original_verify = c1_verify.verify

    def verify(runner, batch, metadata, inputs):
        active[0] = True
        try:
            return original_verify(runner, batch, metadata, inputs)
        finally:
            active[0] = False
            Path(f"/work/phase4-audit/hc-graph-rank-{worker.rank}.json").write_text(
                json.dumps(
                    dict(
                        graphs=len(graph_records),
                        replays=sum(x["count"] for x in graph_records),
                        allocated=torch.xpu.memory_allocated(),
                        reserved=torch.xpu.memory_reserved(),
                    ),
                    indent=2,
                )
            )

    c1_verify.verify = verify
    worker._phase4_hc_graph_methods = methods
    worker._phase4_hc_graph_entries = graph_records
    return {"rank": worker.rank, "installed": "stateless HC graphs"}
