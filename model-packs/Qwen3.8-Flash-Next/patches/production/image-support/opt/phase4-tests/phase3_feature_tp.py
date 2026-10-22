"""The completed five-layer interface must remain replicated across TP4."""

import json
from pathlib import Path
import torch

p = Path("/work/phase3-audit")
records = []
for branch in ["all", "first", "middle"]:
    payloads = [
        torch.load(
            p / f"phase3-vector-{branch}-v5-logits-rank-{rank}.pt", weights_only=False
        )
        for rank in range(4)
    ]
    equal = all(
        torch.equal(a, b)
        for payload in payloads[1:]
        for a, b in zip(payloads[0]["features"], payload["features"])
    )
    assert len(payloads[0]["features"]) == 5 and equal
    records.append(
        dict(
            branch=branch,
            tp4_bitwise_equal=equal,
            feature_shapes=[list(t.shape) for t in payloads[0]["features"]],
        )
    )
(p / "features-tp4.json").write_text(json.dumps(records, indent=2))
print(json.dumps(records))
