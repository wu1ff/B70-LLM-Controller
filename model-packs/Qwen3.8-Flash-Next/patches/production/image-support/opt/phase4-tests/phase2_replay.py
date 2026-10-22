"""Submit each recorded committed prefix as a fresh target prefill."""

import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, "/tests")
from phase2_live_helpers import completion, control

directory = Path("/work/phase2-audit")
for mode in ("all", "first", "middle"):
    record = torch.load(directory / f"{mode}-v2-rank-0.pt", weights_only=False)
    length = record["length"]
    control({"mode": "replay", "tag": mode + "-v2-replay", "length": length})
    response = completion(
        {
            "prompt": record["tokens"].tolist(),
            "temperature": 0,
            "max_tokens": 1,
            "return_token_ids": True,
            "cache_salt": "phase2-replay-" + mode,
        }
    )
    (directory / f"{mode}-replay-response.json").write_text(
        json.dumps(response, indent=2)
    )
    (directory / "control.json").unlink(missing_ok=True)
    print(mode, length, flush=True)
