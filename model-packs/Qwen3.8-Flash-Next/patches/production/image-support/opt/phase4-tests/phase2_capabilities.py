"""Focused functional API checks; no throughput benchmark or native-length claim."""

import base64
import json
from pathlib import Path
import struct
import time
import urllib.request
import zlib

root = Path(__file__).resolve().parents[1]
model = "Intel/Qwen3.8-Flash-Next-W4A16-AutoRound"
results = []


def get(path):
    with urllib.request.urlopen("http://127.0.0.1:8000" + path, timeout=10) as r:
        return json.load(r)


def chat(label, messages, **extra):
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
        **extra,
    }
    request = urllib.request.Request(
        "http://127.0.0.1:8000/v1/chat/completions",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=180) as r:
        response = json.load(r)
    row = {
        "case": label,
        "request": data,
        "response": response,
        "elapsed_seconds": time.monotonic() - start,
    }
    results.append(row)
    (root / "evidence/phase2/capability-smoke.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "case": label,
                "message": response["choices"][0]["message"],
                "usage": response.get("usage"),
            }
        ),
        flush=True,
    )
    assert response["choices"][0]["finish_reason"] != "error"
    return response


def png(color):
    def chunk(t, b):
        return struct.pack("!I", len(b)) + t + b + struct.pack("!I", zlib.crc32(t + b))

    raw = (b"\0" + bytes(color) * 224) * 224
    data = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack("!IIBBBBB", 224, 224, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(data).decode()


models = get("/v1/models")
(root / "evidence/phase2/served-models.json").write_text(
    json.dumps(models, indent=2) + "\n"
)
assert models["data"][0]["max_model_len"] == 262144
r = chat(
    "short-target",
    [{"role": "user", "content": "What is 2 + 2? Answer with just the number."}],
    max_tokens=8,
)
assert "4" in r["choices"][0]["message"]["content"]
r = chat(
    "continuation",
    [
        {"role": "user", "content": "Remember the code ORCHID."},
        {"role": "assistant", "content": "The code is ORCHID."},
        {"role": "user", "content": "What was the code? Reply with the code only."},
    ],
    max_tokens=16,
)
assert "ORCHID" in r["choices"][0]["message"]["content"].upper()
r = chat(
    "two-image",
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What are the colors of the first and second images? Answer in order, briefly.",
                },
                {"type": "image_url", "image_url": {"url": png((255, 0, 0))}},
                {"type": "image_url", "image_url": {"url": png((0, 0, 255))}},
            ],
        }
    ],
    max_tokens=40,
)
assert all(c in r["choices"][0]["message"]["content"].lower() for c in ("red", "blue"))
tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "description": "Look up weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
r = chat(
    "tool-call",
    [{"role": "user", "content": "Call lookup_weather for Phoenix."}],
    tools=tools,
    tool_choice="auto",
    max_tokens=96,
)
calls = r["choices"][0]["message"].get("tool_calls")
assert calls and calls[0]["function"]["name"] == "lookup_weather"
assert "Phoenix" in json.loads(calls[0]["function"]["arguments"])["city"]
r = chat(
    "reasoning",
    [
        {
            "role": "user",
            "content": "Compute 17 times 3. Give the result after thinking briefly.",
        }
    ],
    chat_template_kwargs={"enable_thinking": True},
    max_tokens=160,
)
m = r["choices"][0]["message"]
assert m.get("reasoning") or m.get("reasoning_content")
assert "51" in (m.get("content") or "")

print("Sequential capability checks passed.", flush=True)
