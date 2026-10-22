"""Compare every committed state with an independent fresh target prefill."""

import json
from pathlib import Path
import sys

import torch

directory = Path("/work/phase2-audit")
results = []
for mode, accepted in (("all", 4), ("first", 1), ("middle", 2)):
    for rank in range(4):
        actual = torch.load(directory / f"{mode}-v4-rank-{rank}.pt", weights_only=False)
        reference_path = (
            directory / f"{mode}-sequential-v4-rank-{rank}.pt"
            if "--incremental" in sys.argv
            else directory / f"{mode}-v2-replay-rank-{rank}.pt"
        )
        replay = torch.load(reference_path, weights_only=False)
        assert actual["accepted_inputs"] == accepted, (mode, actual["accepted_inputs"])
        assert actual["rejected"] == 4 - accepted, (mode, actual["rejected"])
        torch.testing.assert_close(actual["tokens"], replay["tokens"], rtol=0, atol=0)
        comparisons = {}
        for group in ("states", "attention", "draft_context"):
            assert actual[group].keys() == replay[group].keys()
            for name, tensor in actual[group].items():
                reference = replay[group][name].float()
                difference = tensor.float() - reference
                relative_l2 = float(
                    torch.linalg.vector_norm(difference)
                    / torch.linalg.vector_norm(reference).clamp_min(1e-10)
                )
                comparisons[group + "/" + name] = {
                    "relative_l2": relative_l2,
                    "max_abs": float(difference.abs().max()),
                    "bitwise_equal": torch.equal(tensor, replay[group][name]),
                    "shape": list(tensor.shape),
                }
        reference = replay["hidden"].float()
        difference = actual["hidden"].float() - reference
        comparisons["hidden"] = {
            "relative_l2": float(difference.norm() / reference.norm()),
            "max_abs": float(difference.abs().max()),
            "bitwise_equal": torch.equal(actual["hidden"], replay["hidden"]),
        }
        results.append(
            {
                "mode": mode,
                "rank": rank,
                "length": actual["length"],
                "comparisons": comparisons,
                "worst_relative_l2": max(
                    x["relative_l2"] for x in comparisons.values()
                ),
            }
        )
output = {
    "relative_l2_tolerance": 0.03,
    "results": results,
    "passed": all(x["worst_relative_l2"] < 0.03 for x in results),
}
output["target_bitwise_equal"] = all(
    value["bitwise_equal"]
    for result in results
    for name, value in result["comparisons"].items()
    if not name.startswith("draft_context/")
)
output["all_bitwise_equal"] = all(
    value["bitwise_equal"]
    for result in results
    for value in result["comparisons"].values()
)
if "--incremental" in sys.argv:
    output["passed"] = output["passed"] and output["target_bitwise_equal"]
name = (
    "state-comparison-incremental.json"
    if "--incremental" in sys.argv
    else "state-comparison.json"
)
(directory / name).write_text(json.dumps(output, indent=2))
print(
    json.dumps(
        [{k: v for k, v in x.items() if k != "comparisons"} for x in results], indent=2
    )
)
assert output["passed"], "A committed cache differs materially from fresh target replay"
