"""C1 real inference, controlled acceptance branches and fresh target replay."""

import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "runtime/work/phase2-audit"
EVIDENCE = ROOT / "evidence/phase2"
MODEL = "Intel/Qwen3.8-Flash-Next-W4A16-AutoRound"


def completion(body):
    request = urllib.request.Request(
        "http://127.0.0.1:8000/v1/completions",
        json.dumps({"model": MODEL, **body}).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def control(value):
    temporary = AUDIT / "control.tmp"
    temporary.write_text(json.dumps(value))
    temporary.replace(AUDIT / "control.json")


baseline = json.loads((EVIDENCE / "control-completion.json").read_text())
prompt = baseline["response"]["choices"][0]["prompt_token_ids"]
expected = baseline["response"]["choices"][0]["token_ids"]
results = []
try:
    for mode in ("all", "first", "middle"):
        fixture_id = "phase2-" + mode + "-v4"
        control(
            {
                "mode": mode,
                "tag": mode + "-v4",
                "request_id": fixture_id,
                "target_tokens": prompt + expected,
            }
        )
        response = completion(
            {
                "prompt": prompt,
                "request_id": fixture_id,
                "temperature": 0,
                "max_tokens": 12,
                "ignore_eos": True,
                "return_token_ids": True,
                "logprobs": 5,
                "cache_salt": "phase2-" + mode,
            }
        )
        (EVIDENCE / f"controlled-{mode}.json").write_text(
            json.dumps(response, indent=2)
        )
        assert response["choices"][0]["token_ids"] == expected[:12], mode
        # Snapshot JSON is written separately by the reader below; torch payloads
        # remain raw evidence in the shared runtime directory.
        results.append({"case": mode, "target_control_tokens_equal": True})
    (AUDIT / "control.json").unlink(missing_ok=True)
    response = completion(
        {
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": 16,
            "ignore_eos": True,
            "return_token_ids": True,
        }
    )
    (EVIDENCE / "natural-greedy.json").write_text(json.dumps(response, indent=2))
    assert response["choices"][0]["token_ids"] == expected[:16]
    results.append({"case": "natural-greedy", "target_control_tokens_equal": True})
    for seed in (17, 82):
        response = completion(
            {
                "prompt": "A short story about a blue fox:\n",
                "temperature": 0.8,
                "top_p": 0.95,
                "top_k": 20,
                "max_tokens": 24,
                "seed": seed,
                "return_token_ids": True,
            }
        )
        (EVIDENCE / f"sampled-{seed}.json").write_text(json.dumps(response, indent=2))
        assert response["usage"]["completion_tokens"] > 1
        results.append({"case": f"sampled-{seed}", "completed": True})
finally:
    (AUDIT / "control.json").unlink(missing_ok=True)
(EVIDENCE / "live-results.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
