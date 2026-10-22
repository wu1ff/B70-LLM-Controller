"""Reject drift in the pinned local source/build inputs before rebuilding."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
lock = json.loads((root / "build/inputs-lock.json").read_text())
for name, expected in lock["files"].items():
    path = root / name
    if not path.is_file():
        raise SystemExit("Build input missing: " + name)
    with path.open("rb") as stream:
        actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_hash != expected:
        raise SystemExit("Build input mismatch: " + name)
for name, expected in lock["repositories"].items():
    actual = subprocess.check_output(
        ["git", "-C", str(root / "sources" / name), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        raise SystemExit("Source revision mismatch: " + name)
print(
    "Verified",
    len(lock["files"]),
    "source/build/test inputs and",
    len(lock["repositories"]),
    "pinned revisions.",
)

if "--native" in sys.argv:
    actual = hashlib.sha256(
        (root / "build/native-out/libgdn_index64.so").read_bytes()
    ).hexdigest()
    if actual != lock["expected_native_sha256"]:
        raise SystemExit("Rebuilt native GDN payload differs from the selected binary")
    print("Verified selected rebuilt GDN binary identity.")
