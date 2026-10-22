"""Repeat real HC projections and complete mixers on identical BF16 inputs."""

import json
from pathlib import Path
import torch


def probe(worker, run):
    records = []
    originals = []
    for name, module in worker.model_runner.model.named_modules():
        if not hasattr(module, "_down_gemm"):
            continue
        for method in ("_down_gemm", "combine_and_mix"):
            original = getattr(module, method)
            valid = module.lora_rank + (module.hc_count if module.use_combine else 0)

            def wrapped(
                *args,
                original=original,
                name=name,
                method=method,
                valid=valid,
                **kwargs,
            ):
                copied = tuple(
                    t.clone() if isinstance(t, torch.Tensor) else t for t in args
                )
                first = original(*args, **kwargs)
                tensors = (first,) if isinstance(first, torch.Tensor) else first
                saved = tuple(
                    t.clone() if isinstance(t, torch.Tensor) else t for t in tensors
                )
                second = original(*copied, **kwargs)
                other = (second,) if isinstance(second, torch.Tensor) else second
                diffs = [
                    float(
                        (
                            (a[..., :valid] if method == "_down_gemm" else a).float()
                            - (b[..., :valid] if method == "_down_gemm" else b).float()
                        )
                        .abs()
                        .max()
                    )
                    for a, b in zip(saved, other)
                    if isinstance(a, torch.Tensor)
                ]
                records.append(
                    {
                        "name": name,
                        "method": method,
                        "max_abs": diffs,
                        "equal": all(x == 0 for x in diffs),
                    }
                )
                return saved[0] if isinstance(first, torch.Tensor) else saved

            originals.append((module, method, original))
            setattr(module, method, wrapped)
    try:
        return run()
    finally:
        for module, method, original in originals:
            setattr(module, method, original)
        Path(f"/work/phase2-audit/hc-probe-rank-{worker.rank}.json").write_text(
            json.dumps(records, indent=2)
        )
