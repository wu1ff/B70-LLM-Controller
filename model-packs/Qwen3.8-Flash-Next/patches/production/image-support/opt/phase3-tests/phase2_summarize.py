"""Verify retained live tensors and summarize the completed C1 qualification."""

import json
from pathlib import Path
import torch

directory = Path("/work/phase2-audit")
state = json.loads((directory / "state-comparison-incremental.json").read_text())
assert state["passed"] and state["all_bitwise_equal"] and len(state["results"]) == 12
features = []
for capture in (0, 1):
    reference = torch.load(
        directory / f"features-rank-0-{capture}.pt", weights_only=False
    )
    for rank in range(4):
        payload = torch.load(
            directory / f"features-rank-{rank}-{capture}.pt", weights_only=False
        )
        assert len(payload["features"]) == 5
        assert all(
            torch.equal(a, b)
            for a, b in zip(reference["features"], payload["features"])
        )
        assert torch.equal(reference["positions"], payload["positions"])
    features.append(
        {
            "capture": capture,
            "packed_shape": list(torch.cat(reference["features"], -1).shape),
            "dtype": str(reference["features"][0].dtype),
            "tp4_bitwise_equal": True,
        }
    )
selectors = []
for rank in range(4):
    load = json.loads((directory / f"load-rank-{rank}.json").read_text())
    assert len(load["checkpoint_tensors_bitwise_verified"]) == 81
    assert (
        load["embedding_shared"]
        and load["lm_head_shared"]
        and load["hc_ordered_modules"] == 97
    )
    for path in sorted(directory.glob(f"selector-rank-{rank}-*.pt")):
        payload = torch.load(path, weights_only=False)
        for row in range(payload["candidates"].shape[0]):
            previous = 0
            for step in range(3):
                token = payload["proposals"][row, step]
                index = int(
                    (payload["candidates"][row, step] == token).nonzero().item()
                )
                torch.testing.assert_close(
                    payload["realized_scores"][row, step],
                    payload["scores"][row, step, previous].float(),
                    rtol=0,
                    atol=0,
                )
                previous = index
        selectors.append(path.name)
    assert (directory / f"canonical-rank-{rank}.jsonl").stat().st_size > 0
    finish = json.loads((directory / f"audit-finished-rank-{rank}.json").read_text())
    assert (
        finish["propose"] == "DFlashSpeculator.propose"
        and finish["sampler"] == "CandidateSampler.sample"
    )
assert (
    not (directory / "control.json").exists()
    and (directory / "audit-complete").exists()
)
output = {
    "passed": True,
    "features": features,
    "selector_payloads_verified": selectors,
    "committed_prefix_cases": 12,
    "state_tensors_bitwise_equal": sum(len(r["comparisons"]) for r in state["results"]),
    "maximum_state_relative_l2": max(r["worst_relative_l2"] for r in state["results"]),
    "all_bitwise_equal": True,
    "draft_loaded_all_ranks": True,
    "hc_ordered_modules_per_rank": 97,
    "forced_controls_removed": True,
    "audit_hooks_disabled": True,
}
(directory / "qualification-summary.json").write_text(json.dumps(output, indent=2))
print(json.dumps(output, indent=2))
