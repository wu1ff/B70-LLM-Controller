"""Check all committed caches after processing the recovery or bonus input."""

import json
from pathlib import Path
import sys
import torch

p = Path("/work/phase2-audit")
out = Path("/work/phase3-audit")
version = sys.argv[1]
records = []
for branch, count in [("all", 4), ("first", 1), ("middle", 2)]:
    for rank in range(4):
        a = torch.load(
            p / f"phase3-vector-{branch}-{version}-followup-rank-{rank}.pt",
            weights_only=False,
        )
        s = torch.load(
            p / f"phase3-serial-{branch}-{version}-followup-rank-{rank}.pt",
            weights_only=False,
        )
        r = torch.load(
            p / f"{branch}-followup-sequential-phase3-{version}-rank-{rank}.pt",
            weights_only=False,
        )
        assert a["length"] == s["length"] == r["length"] == 13 + count * 2
        assert torch.equal(a["tokens"], s["tokens"]) and torch.equal(
            a["tokens"], r["tokens"]
        )
        comparisons = {}
        for name, ref in [("serial", s), ("independent", r)]:
            for group in ["states", "attention", "draft_context"]:
                for key, tensor in a[group].items():
                    comparisons[name + "/" + group + "/" + key] = torch.equal(
                        tensor, ref[group][key]
                    )
            comparisons[name + "/hidden"] = torch.equal(a["hidden"], ref["hidden"])
        records.append(
            dict(
                branch=branch,
                rank=rank,
                length=a["length"],
                comparisons=comparisons,
                all_bitwise_equal=all(comparisons.values()),
            )
        )
result = dict(
    results=records, all_bitwise_equal=all(r["all_bitwise_equal"] for r in records)
)
(out / f"followup-comparison-{version}.json").write_text(json.dumps(result, indent=2))
print(
    json.dumps(
        [{k: v for k, v in x.items() if k != "comparisons"} for x in records], indent=2
    )
)
assert result["all_bitwise_equal"]
