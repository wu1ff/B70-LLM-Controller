#!/usr/bin/env python3
"""Render the supported v6 production launch; execute only on request."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent
PRODUCTION = ROOT / "patches/production"


def make_command(target, draft, *, name="dflash2-v033", port=18083,
                 bind_host="127.0.0.1"):
    pack = json.loads((ROOT / "pack.json").read_text())
    frozen = json.loads((PRODUCTION / "runtime/profile.json").read_text())
    runtime = pack["runtimes"][0]
    layers = [runtime["launch"], pack["models"][0]["launch"],
              pack["modes"][0]["launch"], pack["profiles"][0]["launch"]]
    env, docker_args = {}, []
    command = list(runtime["launch"]["command"])
    for layer in layers:
        if env.keys() & layer["environment"].keys():
            raise ValueError("Duplicate environment key")
        env.update(layer["environment"])
        docker_args.extend(layer["docker_args"])
        command.extend(layer.get("command_args", []))
    command.remove("/models")
    command.insert(2, "/models")
    # The frozen development profile is retained verbatim as provenance.
    # These are the only deliberate production differences: omit the inert
    # diagnostic RPC mixin, its import path, and development HTTP endpoints.
    expected_command = list(frozen["command"])
    index = expected_command.index("--worker-extension-cls")
    if expected_command[index:index + 2] != ["--worker-extension-cls", "parity_worker.ParityWorker"]:
        raise ValueError("Unexpected frozen diagnostic extension")
    del expected_command[index:index + 2]
    expected_env = {key: value for key, value in frozen["env"].items()
                    if key not in {"PYTHONPATH", "VLLM_SERVER_DEV_MODE"}}
    if command != expected_command or env != expected_env:
        raise ValueError("Pack differs from the supported v6 production launch")
    if not 1024 <= port <= 65535 or not name or name.startswith("-"):
        raise ValueError("Invalid port or container name")
    cmd = ["docker", "run", "-d", "--name", name,
           "--publish", f"{bind_host}:{port}:8000", *docker_args]
    for source, destination in [(target, "/models"), (draft, "/draft")]:
        source = Path(source).resolve(strict=True)
        if not source.is_dir() or "," in str(source):
            raise ValueError("Model mount must be a directory without commas")
        cmd += ["--mount", f"type=bind,source={source},target={destination},readonly"]
    for key, value in sorted(env.items()):
        cmd += ["--env", f"{key}={value}"]
    return [*cmd, runtime["digest"], *command]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--name", default="dflash2-v033")
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--bind-host", choices=["127.0.0.1", "0.0.0.0"], default="127.0.0.1")
    parser.add_argument("--execute", action="store_true", help="Start the server; default only prints")
    args = parser.parse_args()
    cmd = make_command(args.target, args.draft, name=args.name, port=args.port,
                       bind_host=args.bind_host)
    if args.execute:
        pack = json.loads((ROOT / "pack.json").read_text())
        for model, source in zip(pack["models"], [args.target, args.draft]):
            for item in model["files"]:
                path = source / item["path"]
                if not path.is_file() or path.stat().st_size != item["size"]:
                    raise ValueError("Incomplete pinned model: " + str(path))
        digest = pack["runtimes"][0]["digest"]
        actual = json.loads(subprocess.check_output(["docker", "image", "inspect", digest]))[0]["Id"]
        if actual != digest:
            raise ValueError("Accepted image identity differs")
        for cid in subprocess.check_output(["docker", "ps", "-q"], text=True).split():
            info = json.loads(subprocess.check_output(["docker", "inspect", cid]))[0]
            if info["HostConfig"].get("Devices") or info["HostConfig"].get("DeviceRequests"):
                raise RuntimeError("A GPU container is already running: " + info["Name"])
        subprocess.run(cmd, check=True)
    else:
        print(shlex.join(cmd))


if __name__ == "__main__":
    main()
