"""Report real HC graph ownership and cache/allocator capacity."""

import json
from pathlib import Path
import torch


def capture(worker):
    runner = worker.model_runner
    entries = [
        entry
        for module in runner.model.modules()
        for entry in getattr(module, "_c1_hc_graphs", {}).values()
    ]
    addresses = set()
    owned = 0

    def visit(value):
        nonlocal owned
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            if storage.data_ptr() not in addresses:
                addresses.add(storage.data_ptr())
                owned += storage.nbytes()
        elif isinstance(value, (tuple, list)):
            for x in value:
                visit(x)

    for entry in entries:
        visit(entry["inputs"])
        visit(entry["output"])
    stats = torch.xpu.memory_stats()
    segments = torch.xpu.memory_snapshot()
    graph_segments = [
        s for s in segments if tuple(s.get("segment_pool_id", (0, 0))) != (0, 0)
    ]
    report = dict(
        rank=worker.rank,
        hc_graphs=len(entries),
        hc_graph_replays=sum(e["replays"] for e in entries),
        graph_retained_inputs_outputs_bytes=owned,
        allocated=torch.xpu.memory_allocated(),
        reserved=torch.xpu.memory_reserved(),
        graph_pool_reserved_bytes=sum(s["total_size"] for s in graph_segments),
        graph_pool_segments=graph_segments,
        num_blocks=runner.kv_cache_config.num_blocks,
        allocator_stats=stats,
        config=repr(runner.kv_cache_config),
    )
    Path(f"/work/phase4-audit/resources-rank-{worker.rank}.json").write_text(
        json.dumps(report, indent=2)
    )
