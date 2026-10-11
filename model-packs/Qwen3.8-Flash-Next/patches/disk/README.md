# Cumulative accepted Disk source

This source-only export comes from the independently reproduced and qualified Disk package. It contains the final vendor-to-Disk overlays, complete native kernel sources/headers/dispatchers, Wulf's shim, exact native build recipes, SYCL-TLA archive, dependency locks, accepted 3,118-file installed manifest, licenses and historical provenance. No checkpoint, accepted native binary/object or Intel compiler package binary is distributed here.

The original portable source archive also contained all 15 compiler packages for local offline reproduction. This public export deliberately excludes those binaries. Obtain the 15 **exact** filenames and SHA256s in `manifests/compiler-lock.json` from its recorded official Intel apt pool or an authorized local cache. Review the packages' applicable terms separately. Providing just an arbitrary installed compiler is insufficient.

With the pinned official vendor dependency already present, run from this directory:

```bash
sha256sum -c SHA256SUMS
python3 build.py --compiler-dir /absolute/authorized/compiler-cache --tag qwen38-flashnext:disk-fresh-test --evidence /absolute/build-evidence
python3 verify.py qwen38-flashnext:disk-fresh-test --output /absolute/build-evidence/recheck.json
```

The wrapper checks every distributed source hash and all compiler hashes, then creates a disposable build context containing the compiler packages. The export carries Wulf modification comments on the 21 changed upstream source files. It validates qualified locks after removing only the notice for comparison and preserves all 15 Python notices in the runtime overlay. Only the six native compiler inputs are restored to qualified bytes inside the disposable compiler context, preserving native binary reproduction; their distributed sources retain the comments. `verify.py` requires the exact notice-bearing runtime sources and unchanged native/pristine/config payload. Historical installed/source locks remain qualification anchors.

For the existing qualified local image, the smaller release correction needs no compiler or native rebuild:

```bash
python3 build-notice-image.py --tag qwen38-flashnext:disk-rc --evidence /absolute/notice-build-evidence
python3 verify.py qwen38-flashnext:disk-rc --output /absolute/notice-build-evidence/payload.json
```

This recipe checks the exact qualified parent, compares each of the 15 files against the source export, and adds their notice-bearing bytes in one layer. Original file/directory ownership, permissions and timestamps are preserved. A fixed creation timestamp supports independent builds without cache. The parent `:disk` tag is never overwritten. The release candidate is `sha256:b13fed6dd54051395f603409ada440c74afcbd5331810eee3acd62315d7787e3`; focused Base/MTP3 serving qualification and the authorized GHCR replacement are complete. It is published as `ghcr.io/wu1ff/qwen38-flashnext-b70:disk`; the original local parent remains preserved.

Docker and native inputs then match the accepted package; compiler extraction remains `/toolchain/oneapi`, source remains `/build`. It runs `DOCKER_BUILDKIT=0 docker build --pull=false --no-cache --network=none`. Both stages use official vendor image `sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9`; see `manifests/vendor.json`. No accepted, experimental or Vanilla image is a build input. Only rebuilt native outputs and final source/config overlays enter the final runtime stage.

Linux amd64, Python3.11+, Docker's available legacy builder, adequate compiler memory/storage and the exact vendor image are prerequisites. No GPU, model, host compiler, pip resolution or build-time network is required. Native tools remain oneAPI2026.0.0-947, vendor Python3.12/Torch2.13.0+xpu, GCC13.3 and locked SYCL-TLA. Do not reapply `patches/` historical diffs to these already-patched source files. The MoE guard is present in `native/source/csrc/moe/remap_hidden_states.cpp`; `native/moe-inputs.sha256` checks its six direct build inputs before compilation.

The historical independent native build reproduced all 3,118 accepted installed hashes. Release preparation built only the notice layer. All 209 source-build inputs retain their qualified bytes except the 15 preserved Python comments; all native/compiler/dependency inputs match. The notice candidate has the exact 3,118-entry payload expected by `verify.py`. Complete filesystem, bytecode, ELF and configuration evidence is retained in the runtime release-preparation directory; no native rebuild or full serving requalification was run.

`profiles/base.json` and `profiles/mtp3.json` preserve the exact qualified standalone launch authorities. They are receipts, not Controller schema. Use the root `pack.json` through the real b70ctl TUI for managed serving. `launch.py` renders a standalone recovery command; append `--execute` only after stopping the managed server, verifying all four GPUs and freeing the endpoint. It does not stop another server.

[NOTICE.md](NOTICE.md) and [licenses/README.md](licenses/README.md) distinguish the candidate's corrected source notices from remaining source-delivery and Intel recipient-license requirements. Historical co-author messages are records only. The original local qualified `:disk` image remains unchanged. The notice-corrected RC is now the single verified GHCR `:disk` version; its manifest/config/layers, clean pulled Docker ID and Controller acquisition passed. The 1.0.3 public model-pack archive includes this complete accepted source/build export. The release checksum and catalog identify the final archive; previous provisional assets remain preserved as history.
