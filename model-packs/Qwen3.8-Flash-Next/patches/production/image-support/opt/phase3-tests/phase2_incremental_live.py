"""Run request-scoped one-token replay fixtures without restarting the model."""

import json
from pathlib import Path
import time
import urllib.request
import sys

root = Path(__file__).resolve().parents[1]
directory = root / "runtime/work/phase2-audit"
baseline = json.loads((root / "evidence/phase2/control-completion.json").read_text())
prompt = baseline["response"]["choices"][0]["prompt_token_ids"]
expected = baseline["response"]["choices"][0]["token_ids"]
for mode, count in [("all", 4), ("first", 1), ("middle", 2)]:
    tag = mode + "-sequential" + (sys.argv[1] if len(sys.argv) > 1 else "-v4")
    request_id = "phase2-" + tag
    control = {
        "mode": "replay",
        "tag": tag,
        "length": 13,
        "request_id": request_id,
        "incremental_prefix": prompt + expected[:count],
    }
    (directory / "control.json").write_text(json.dumps(control))
    body = {
        "model": baseline["request"]["model"],
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": 1,
        "cache_salt": request_id,
        "request_id": request_id,
    }
    response = json.load(
        urllib.request.urlopen(
            urllib.request.Request(
                "http://127.0.0.1:8000/v1/completions",
                json.dumps(body).encode(),
                {"Content-Type": "application/json"},
            ),
            timeout=120,
        )
    )
    deadline = time.monotonic() + 90
    while not all((directory / f"{tag}-rank-{rank}.pt").exists() for rank in range(4)):
        errors = list(directory.glob(tag + "*error.txt"))
        if errors:
            raise RuntimeError("\n".join(p.read_text() for p in errors))
        if time.monotonic() > deadline:
            raise TimeoutError(tag)
        time.sleep(0.1)
    (directory / "control.json").unlink(missing_ok=True)
    print(mode, 13 + count, flush=True)
