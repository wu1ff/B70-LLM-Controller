"""Separate local projection width behavior from TP reduction width behavior."""

import json
from pathlib import Path
import torch
from vllm.distributed import get_tp_group


def run(worker):
    path = Path("/work/phase3-audit")
    captures = torch.load(
        path / f"phase3-vector-first-v3-probe-rank-{worker.rank}.pt", weights_only=False
    )
    x = (
        captures["layers.0.linear_attn.norm"][0][0]
        .to(worker.model_runner.device)
        .flatten(-2)
    )
    layer = worker.model_runner.model.language_model.model.layers[
        0
    ].linear_attn.out_proj
    apply = layer.quant_method.apply
    one = torch.cat([apply(layer, row[None], None) for row in x])
    vector = apply(layer, x, None)
    comm = get_tp_group().device_communicator
    native_one = torch.cat([comm.all_reduce(row[None]) for row in one])
    native_vector = comm.all_reduce(vector)
    fixed = comm._fixed_rank_sum(vector)

    def diff(a, b):
        return dict(
            equal=torch.equal(a, b), max_abs=float((a.float() - b.float()).abs().max())
        )

    report = dict(
        quant_method=type(layer.quant_method).__name__,
        local=diff(one, vector),
        native=diff(native_one, native_vector),
        native_one_vs_fixed=diff(native_one, fixed),
        native_vector_vs_fixed=diff(native_vector, fixed),
    )
    (path / f"linear-probe-rank-{worker.rank}.json").write_text(
        json.dumps(report, indent=2)
    )
    import itertools
    import torch.distributed as dist

    gathered = torch.empty((4, *one.shape), dtype=one.dtype, device=one.device)
    dist.all_gather_single(gathered.view(-1), one.view(-1), group=comm.device_group)
    matches = []
    for perm in itertools.permutations(range(4)):
        a, b, c, d = [gathered[i] for i in perm]
        options = [
            ((a + b) + c) + d,
            (a + b) + (c + d),
            (a + (b + c)) + d,
            a + ((b + c) + d),
            a + (b + (c + d)),
            (((a.float() + b.float()) + c.float()) + d.float()).to(a.dtype),
            ((a.float() + b.float()) + (c.float() + d.float())).to(a.dtype),
        ]
        for i, tensor in enumerate(options):
            if torch.equal(tensor, native_one):
                matches.append([perm, i])
    report["native_one_tree_matches"] = matches
    (path / f"linear-probe-rank-{worker.rank}.json").write_text(
        json.dumps(report, indent=2)
    )
