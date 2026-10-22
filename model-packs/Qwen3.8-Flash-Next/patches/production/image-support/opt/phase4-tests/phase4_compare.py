"""Apply the retained bitwise prefix/recovery gates to the Phase 4 candidate."""

import json
from pathlib import Path
import sys
import torch

version = sys.argv[1]
audit = Path("/work/phase2-audit")
p3 = Path("/work/phase3-audit")
out = Path("/work/phase4-audit")
records = []
for branch, count in [("all", 4), ("first", 1), ("middle", 2)]:
    for rank in range(4):
        tag = f"phase3-vector-{branch}-{version}"
        comparisons = {}
        for follow in ("", "-followup"):
            a = torch.load(audit / f"{tag}{follow}-rank-{rank}.pt", weights_only=False)
            refs = {
                "serial": audit / f"phase3-serial-{branch}-v5{follow}-rank-{rank}.pt",
                "independent": audit
                / (
                    f"{branch}-followup-sequential-phase3-v5-rank-{rank}.pt"
                    if follow
                    else f"{branch}-sequential-phase3-v5-rank-{rank}.pt"
                ),
            }
            assert a["accepted_inputs"] == count
            assert a["length"] == 13 + count * (2 if follow else 1)
            for label, path in refs.items():
                b = torch.load(path, weights_only=False)
                for key in ("tokens", "hidden"):
                    comparisons[f"{follow}/{label}/{key}"] = torch.equal(a[key], b[key])
                for group in ("states", "attention", "draft_context"):
                    assert a[group].keys() == b[group].keys()
                    for name, tensor in a[group].items():
                        comparisons[f"{follow}/{label}/{group}/{name}"] = torch.equal(
                            tensor, b[group][name]
                        )
        vl = torch.load(p3 / f"{tag}-logits-rank-{rank}.pt", weights_only=False)
        sl = torch.load(
            p3 / f"phase3-serial-{branch}-v5-logits-rank-{rank}.pt", weights_only=False
        )
        for key in ("logits", "hidden", "input_ids", "positions"):
            comparisons["all_rows/" + key] = torch.equal(vl[key], sl[key])
        for i, (a, b) in enumerate(zip(vl["features"], sl["features"])):
            comparisons[f"all_rows/feature-{i}"] = torch.equal(a, b)
        records.append(
            dict(
                branch=branch,
                rank=rank,
                comparisons=len(comparisons),
                passed=all(comparisons.values()),
                failures=[k for k, v in comparisons.items() if not v],
            )
        )
result = dict(
    version=version,
    records=records,
    comparisons=sum(r["comparisons"] for r in records),
    passed=all(r["passed"] for r in records),
)
(out / f"correctness-{version}.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
assert result["passed"]
