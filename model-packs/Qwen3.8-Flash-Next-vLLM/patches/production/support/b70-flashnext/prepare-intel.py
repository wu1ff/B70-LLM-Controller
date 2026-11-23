# Copyright Wulf. Qwen model data remains under the Qwen Community License 1.0.
"""Prepare the original Intel dense serving view without importing vLLM.

Adapted from Wulf's frozen Disk prepare-intel.py. This is source preparation,
not a launcher. The destination must be new; no existing directory is removed.
"""

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path


ASSETS = Path(__file__).resolve().parent / "assets"
LAYOUT = Path('/opt/venv/lib/python3.12/site-packages/vllm/models/qwen4_exp/intel_checkpoint.py')


def read_locked_json(path: Path, expected: str) -> dict:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Checkpoint metadata identity changed: {path.name}")
    return json.loads(data)


def prepare(snapshot: Path, target: Path) -> None:
    snapshot = snapshot.resolve(strict=True)
    target = target.absolute()
    if target.exists() or target.is_symlink():
        raise ValueError("Serving destination must be a new directory")
    lock = json.loads((ASSETS / "checkpoint-lock.json").read_text())
    spec = importlib.util.spec_from_file_location("intel_checkpoint", LAYOUT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Missing standalone Intel checkpoint layout helper")
    layout = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layout)
    layout.ple_layout(snapshot / layout.PLE_FILE)
    config = read_locked_json(snapshot / "config.json", lock["source_config_sha256"])
    qualified = read_locked_json(
        ASSETS / "serve-config.json", lock["dense_config_sha256"]
    )
    removed = {
        key: config["text_config"].pop(key)
        for key in list(config["text_config"])
        if key.startswith("indexer_")
    }
    if removed != lock["removed_text_config"] or len(removed) != 5:
        raise ValueError("Intel checkpoint indexer configuration changed")
    if config != qualified:
        raise ValueError("Intel config differs from the qualified dense config")
    index = read_locked_json(
        snapshot / "model.safetensors.index.json", lock["source_index_sha256"]
    )
    excluded = {
        f"{layout.PLE_PREFIX}{i}.weight" for i in range(layout.PLE_SHARDS)
    }
    actual = {key for key in index["weight_map"] if ".ngram_embedding.shard_" in key}
    if actual != excluded or any(
        index["weight_map"][key] != layout.PLE_FILE for key in excluded
    ):
        raise ValueError("Intel PLE weight index differs from the qualified layout")
    index["weight_map"] = {
        key: value for key, value in index["weight_map"].items() if key not in excluded
    }
    index["metadata"]["total_size"] = lock["served_total_size"]
    shards = set(index["weight_map"].values())
    if layout.PLE_FILE in shards:
        raise ValueError("PLE-only shard must not enter the ordinary model loader")
    sizes = {item["path"]: item["size"] for item in lock["source_payload_identities"]}
    for name in shards:
        if (
            Path(name).name != name
            or not (snapshot / name).is_file()
            or (snapshot / name).stat().st_size != sizes[name]
        ):
            raise ValueError(f"Unsafe, missing or changed checkpoint shard: {name}")
    frontend = json.loads((ASSETS / "frontend.json").read_text())
    for name, expected in frontend["files"].items():
        actual = hashlib.sha256((ASSETS / "frontend" / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Qualified frontend changed: {name}")

    # All validation precedes writes. Preserve every remaining tensor name,
    # shard assignment, quantization exclusion and persistent hash buffer.
    target.mkdir(parents=True)
    for name in sorted(shards):
        (target / name).symlink_to(snapshot / name)
    (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (target / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2) + "\n"
    )
    for name in frontend["files"]:
        shutil.copyfile(ASSETS / "frontend" / name, target / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    prepare(args.snapshot, args.target)


if __name__ == "__main__":
    main()
