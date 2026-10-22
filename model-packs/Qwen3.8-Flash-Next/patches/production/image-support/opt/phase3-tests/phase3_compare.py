"""Extend retained Phase 2 committed-prefix and all-row target-logit comparisons."""

import json
from pathlib import Path
import sys
import torch

audit = Path("/work/phase2-audit")
p3 = Path("/work/phase3-audit")
version = sys.argv[1] if len(sys.argv) > 1 else "v1"
results = []


def diff(a, b):
    d = a.float() - b.float()
    return dict(
        bitwise_equal=torch.equal(a, b),
        relative_l2=float(d.norm() / b.float().norm().clamp_min(1e-10)),
        max_abs=float(d.abs().max()),
        shape=list(a.shape),
    )


for branch, count in [("all", 4), ("first", 1), ("middle", 2)]:
    for rank in range(4):
        tag = f"phase3-vector-{branch}-{version}"
        a = torch.load(audit / f"{tag}-rank-{rank}.pt", weights_only=False)
        ref = torch.load(
            audit / f"phase3-serial-{branch}-{version}-rank-{rank}.pt",
            weights_only=False,
        )
        replay = torch.load(
            audit / f"{branch}-sequential-phase3-{version}-rank-{rank}.pt",
            weights_only=False,
        )
        assert a["accepted_inputs"] == ref["accepted_inputs"] == count
        assert a["rejected"] == ref["rejected"] == 4 - count
        assert torch.equal(a["tokens"], ref["tokens"]) and torch.equal(
            a["tokens"], replay["tokens"]
        )
        comparisons = {}
        for group in ["states", "attention", "draft_context"]:
            assert a[group].keys() == ref[group].keys() == replay[group].keys()
            for name, tensor in a[group].items():
                comparisons["serial/" + group + "/" + name] = diff(
                    tensor, ref[group][name]
                )
                comparisons["independent/" + group + "/" + name] = diff(
                    tensor, replay[group][name]
                )
        comparisons["serial/hidden"] = diff(a["hidden"], ref["hidden"])
        comparisons["independent/hidden"] = diff(a["hidden"], replay["hidden"])
        vl = torch.load(p3 / f"{tag}-logits-rank-{rank}.pt", weights_only=False)
        sl = torch.load(
            p3 / f"phase3-serial-{branch}-{version}-logits-rank-{rank}.pt",
            weights_only=False,
        )
        assert torch.equal(vl["input_ids"], sl["input_ids"])
        comparisons["all_rows/logits"] = diff(vl["logits"], sl["logits"])
        comparisons["all_rows/hidden"] = diff(vl["hidden"], sl["hidden"])
        for i, (av, bv) in enumerate(zip(vl["features"], sl["features"])):
            comparisons[f"all_rows/feature-{i}"] = diff(av, bv)
        results.append(
            dict(
                branch=branch,
                rank=rank,
                accepted_inputs=count,
                length=a["length"],
                comparisons=comparisons,
                all_bitwise_equal=all(v["bitwise_equal"] for v in comparisons.values()),
                worst_relative_l2=max(v["relative_l2"] for v in comparisons.values()),
            )
        )
output = dict(
    results=results, all_bitwise_equal=all(x["all_bitwise_equal"] for x in results)
)
(p3 / f"comparison-{version}.json").write_text(json.dumps(output, indent=2))
print(
    json.dumps(
        [{k: v for k, v in r.items() if k != "comparisons"} for r in results], indent=2
    )
)
for r in results:
    bad = {k: v for k, v in r["comparisons"].items() if not v["bitwise_equal"]}
    if bad:
        print(
            r["branch"],
            r["rank"],
            json.dumps(
                dict(
                    sorted(
                        bad.items(), key=lambda x: x[1]["relative_l2"], reverse=True
                    )[:8]
                )
            ),
        )
assert output["all_bitwise_equal"], "Mismatch: inspect differences before qualification"
