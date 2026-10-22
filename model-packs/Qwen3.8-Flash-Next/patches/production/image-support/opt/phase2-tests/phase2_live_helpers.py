import json
from pathlib import Path
import urllib.request


def control(value):
    directory = Path("/work/phase2-audit")
    tmp = directory / "control.tmp"
    tmp.write_text(json.dumps(value))
    tmp.replace(directory / "control.json")


def completion(body):
    request = urllib.request.Request(
        "http://127.0.0.1:8000/v1/completions",
        json.dumps(
            {"model": "Intel/Qwen3.8-Flash-Next-W4A16-AutoRound", **body}
        ).encode(),
        {"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(request, timeout=180))
