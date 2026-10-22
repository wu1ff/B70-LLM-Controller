"""Repeat stateless real TP4 MoE operations on identical input tensors."""

import json
from pathlib import Path
import torch


def probe(worker, run):
    records = []
    handles = []
    for index, layer in enumerate(
        worker.model_runner.model.language_model.model.layers
    ):
        saved = {}

        def before(module, args, kwargs, saved=saved):
            saved["args"] = tuple(
                t.clone() if isinstance(t, torch.Tensor) else t for t in args
            )
            saved["kwargs"] = kwargs

        def after(module, args, kwargs, output, saved=saved, index=index):
            original = output.clone()
            repeat = module.forward(*saved["args"], **saved["kwargs"])
            difference = original.float() - repeat.float()
            records.append(
                {
                    "layer": index,
                    "max_abs": float(difference.abs().max()),
                    "relative_l2": float(difference.norm() / original.float().norm()),
                    "bitwise_equal": torch.equal(original, repeat),
                }
            )
            return original

        handles.append(layer.mlp.register_forward_pre_hook(before, with_kwargs=True))
        handles.append(layer.mlp.register_forward_hook(after, with_kwargs=True))
    try:
        return run()
    finally:
        for handle in handles:
            handle.remove()
        Path(f"/work/phase2-audit/moe-probe-rank-{worker.rank}.json").write_text(
            json.dumps(records, indent=2)
        )
