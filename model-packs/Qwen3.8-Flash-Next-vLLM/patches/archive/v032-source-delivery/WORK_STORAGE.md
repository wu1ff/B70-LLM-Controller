# DFlash2 storage compatibility with Controller 1.0.1

22 October 2026, America/Phoenix. The external `/work` mount is unnecessary.
Model pack **1.0.5** uses the accepted image unchanged with `/usr/bin/b70ctl`
**1.0.1**. No Controller source or installed-binary changes are required.

## What the accepted container used

The accepted service bound the development workspace's `runtime/work` to `/work`.
The proposed pack inherited a named-volume requirement from that arrangement.
Docker inspection shows `ReadonlyRootfs=false`, no image `VOLUME`, no tmpfs at
`/work`, user root (empty image `User`) and working directory `/opt/venv`.
A fresh container has no `/work`; its root filesystem is writable overlayfs on
the host's XFS NVMe Docker filesystem. `/tmp` already exists with mode 1777.

| Path | Writer and contents | Persistence needed for serving |
| --- | --- | --- |
| `/work/flashnext-serve` | `prepare-serve.sh` invokes `prepare-intel.py`; ten regular configuration/index/frontend files, 46,959,218 bytes, plus sixteen weight symlinks into `/models`. | No. The entrypoint deletes and recreates this directory on each boot using `mkdir(parents=True)`. |
| `/work/triton-cache` | Triton JIT compiler; generated SPIR-V, IR, metadata and small extension libraries. | No. Recompiled on demand; cold startup is slower. |
| `/work/vllm-cache` | vLLM runtime/cache code; small generated cache records. | No. Recreated on demand. |
| `/work/hf-cache`, `/work/xdg-cache` | Configured Hugging Face/XDG cache roots; absent in the accepted work inventory and short mount-free check. | No model acquisition occurs here: serving is offline against `/models` and `/draft`. |
| `/work/tmp` in the development service | Python/vLLM temporary distributed rendezvous, IPC and compiler files. | No. The pack sets `TMPDIR=/tmp` to use an existing writable temporary directory. |
| `astra500`, `phase2-audit`, `phase3-audit`, `phase4-audit`, `feature-audit`, `deep-c1`, `deep-c1-proposals`, `fullwidth`, `mtp7-gap`, `live-feature-results.json` under the old work bind | Explicit development probes, custom workers and comparison scripts wrote tensor captures, JSON reports and controls. These account for almost all of the old 23 GiB. None changed during the accepted service's last run before handover. | Historical evidence only. Default serving does not invoke those probes or custom workers. Original host files are retained. |

## Disk PLE stays on model storage

`PLE_TABLE_PATH=/models/model-00016-of-00017.safetensors` is unchanged. The entrypoint
also derives that exact path from its `/models` snapshot argument. `intel_checkpoint.py`
validates the original 102,400,512,256-byte shard and maps it privately; `b70_disk.py`
protects the mapping read-only. All four live worker process maps point to that
same read-only model bind. The serving weight index excludes the PLE-only shard.

The 1,024-row DMA stage is a 5,242,880-byte pinned CPU tensor per rank, with an
XPU prefetch buffer. It is memory, not a disk staging file. No PLE table or model
checkpoint is copied into the container layer. Target and assistant use the
existing Controller model and assistant mechanisms at `/models` and `/draft`,
both read-only. Model revisions, checkpoint layout, inference flags, seven
proposals, context 262144 and 7785 KV blocks are unchanged.

## Mount-free solution and verification

Only the `/work` mount pair was removed from runtime Docker arguments;
`TMPDIR` changed from `/work/tmp` to `/tmp`. Other environment values are identical
to the accepted container. No host-mount alias, anonymous volume, tmpfs, extra
Docker option, alternate Controller binary or image rebuild is involved.
`/work` and the serving directory are created root-owned with mode 0755 by the
existing image entrypoint; cache directories are created by their libraries.

A fresh container without `/work` successfully prepared the serving view twice,
with identical configuration/index hashes and no regular weight files. The
installed `/usr/bin/b70ctl` 1.0.1 then imported the source pack, reused the complete
target, acquired the pinned assistant through its resumable download flow and
reused the accepted image digest. All six assistant files match the pinned LFS
SHA256/Git blob identities. Its TUI rendered the four-card DFlash2 profile as
available and launched it using supported arguments and bridge port publishing.
The only mounts were read-only `/models`, `/draft` and `/dev/dri/by-path`.

Cold startup reached HTTP 200 health in 235.7 seconds. Model listing and two chat
requests passed; the short request returned `ready`. The longer request exercised
DFlash: 37 draft rounds, 259 proposed tokens and 60 accepted tokens. `/work` then
held about 45 MiB of serving metadata, 14 MiB of Triton cache and 8 KiB of vLLM
cache. Its largest regular file was the 23,778,698-byte weight index. Process
maps and all four PLE startup records confirmed unchanged placement and stage size.
No BetterBench, full-context or broad qualification campaign was run.

The preserved original container was restarted after the check. Its identity,
image, command, environment, Docker options and mounts are preserved. The
previously stopped `b70ctl-runtime` container was preserved by temporary renaming;
the qualified test container is retained stopped as `dflash2-mount-free-qualified`.
The temporary Controller state and model acquisition were isolated from the
user's normal Controller configuration.

The feature branch's Controller additions and tests are removed. All tracked
files outside this model-pack directory match branch base
`94eeda630f757c46dd6e02c4cc49387119efc811`, including `cmd`, `internal`, `VERSION`
and the root README. Version remains 1.0.1. The public 1.0.4 catalog is unchanged.

The restored validator tests (`go test ./internal/modelpack -run '^TestLoad'`)
and runtime unit tests pass. The full repository suite has three unchanged
pack-fixture failures: `TestFlashNextSourceArchiveLoad` and
`TestTrackedCatalogMatchesPublishedPack` still require the published 1.0.4
catalog/archive identity, and `TestFlashNextDiskQualifiedProfiles` still requires
the old MTP5 profile. Those assertions were restored with the original Controller
source; neither they nor the public catalog were rewritten for an unpublished
1.0.5 pack. Installed-binary import, acquisition and serving are the compatibility
checks for this change. The full repository suite is not claimed green.

Detailed local receipts are in the DFlash2 workspace's
`release-prep/work-mount-removal/`: original/image inspection, work inventory,
fresh-path check, installed-binary PTY transcripts, strong assistant hashes,
contract comparison, mount-free inspection/logs/metrics and service restoration.
The final archive is also imported using the installed binary. This establishes
local-image storage compatibility; existing distribution, registry-delivery and
public clean-install gates remain. No GitHub, GHCR or public release is updated.
