"""Locate the first serial/vector difference without changing target execution."""

import json
from pathlib import Path
import torch


def install(worker):
    from vllm.models.qwen4_exp.nvidia import c1_verify

    if getattr(worker, "_phase3_probe_installed", False):
        return
    worker._phase3_probe_installed = True
    original = c1_verify.verify
    saved = set()

    def verify(runner, batch, metadata, inputs):
        ctrl = Path("/work/phase2-audit/control.json")
        control = json.loads(ctrl.read_text()) if ctrl.exists() else {}
        tag = control.get("tag")
        if not control.get("phase3_probe") or tag in saved:
            return original(runner, batch, metadata, inputs)
        captures = {}
        handles = []
        for name, module in runner.model.language_model.model.named_modules():
            if not name.startswith(("layers.0", "layers.1.")) and not (
                name.startswith("layers.") and name.count(".") == 1
            ):
                continue

            def hook(module, args, kwargs, out, name=name):
                values = (
                    [out]
                    if isinstance(out, torch.Tensor)
                    else list(out)
                    if isinstance(out, (tuple, list))
                    else []
                )
                values = [
                    x.detach().clone() for x in values if isinstance(x, torch.Tensor)
                ]
                if values:
                    captures.setdefault(name, []).append(values)

            handles.append(module.register_forward_hook(hook, with_kwargs=True))
        try:
            result = original(runner, batch, metadata, inputs)
        finally:
            for handle in handles:
                handle.remove()
        captures = {
            name: [[x.cpu() for x in values] for values in rows]
            for name, rows in captures.items()
        }
        torch.save(
            captures, Path("/work/phase3-audit") / f"{tag}-probe-rank-{worker.rank}.pt"
        )
        saved.add(tag)
        return result

    c1_verify.verify = verify
