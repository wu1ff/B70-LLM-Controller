# DFlash2 runtime reference

## Runtime and requirements

One local runtime: `qwen38-flashnext-dflash2:integrated-serving-v1`, image ID
`sha256:6e2ab4e686680a5208ff0119c4637341cba81942bd39b6576ea14dc8f547b83b`.
This replaces the previous MTP5 pack contract. Pack version `2.0.0-dev.1` is local
staging; **2.0.0 is the proposed release** because runtime, assistant and profile
identities change. Maintainer ratification remains required. The intended public
tag is `ghcr.io/wu1ff/qwen38-flashnext-b70:dflash2`; no registry digest is assigned
and no DFlash2 image is publicly available. `pack.json` intentionally omits `registry`.
The public catalog continues to advertise the unchanged 1.0.4 archive.

The qualified host has four Intel Arc Pro B70 32 GB GPUs and 256 GB system RAM,
a Linux Intel Xe/Level Zero driver stack, Docker with BuildKit for source builds,
readable `/dev/dri` and `/dev/dri/by-path`, and fast local NVMe storage. Four GPUs
are required; 256 GB describes the tested host, not a proven minimum. Observed
primary GPU usage was 29.765–29.774 GiB/rank. Reserve **181240433939 bytes** for
the target, **1206437758 bytes** for all six draft-repository files, plus image,
build space and writable caches. Disk PLE maps the 102400512256-byte target shard;
allow host page-cache capacity and cold-storage latency. Do not place `/work` on RAM tmpfs.

Pinned dependencies: vLLM 0.32.0+xpu, Torch 2.14.0+xpu, Transformers 5.19.0,
Triton XPU 3.8.0, XPU kernels 0.1.15.4 and Intel runtimes/compiler 2026.1.1.
Immutable vendor/compiler image references and source locks are under
`patches/production/manifests/`. There are 14 inherited `pip check` findings shared
by vendor/accepted/rebuilt images; see the Phase 3 receipt. Full stock-wheel
source attestation remains unresolved.

## Target and draft

| Role | Repository | Revision | Read-only mount |
| --- | --- | --- | --- |
| W4A16 target | Intel/Qwen3.8-Flash-Next-W4A16-AutoRound | `4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897` | `/models` |
| BF16 assistant | incoai/Qwen3.8-Flash-Next-DFlash2 | `a22a1744e85c17364dcc54a6a420af7135d4c7f6` | `/draft` |

The target has 29 declared files, including its original BF16 PLE shard and
nonconsecutive shard inventory. The assistant has all six pinned repository files;
its 1206140008-byte weights hash to
`d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f`.
The engine loads config/weights; repository README, figure and SGLang patch are
acquisition contents, not executable runtime inputs. Stronger identities live in
`draft-pinned-api.json` and `target-pinned-inventory.json`; schema v1 checks paths
and sizes, not per-file hashes. Frontend assets and dense serving configuration are
baked in the image. The entrypoint creates `/work/flashnext-serve` without changing
checkpoints. No separate target conversion or PLE download is needed.

## Build and source reconstruction

Run from the unpacked model-pack directory:

```sh
python3 patches/production/tools/verify.py \
  --staging /tmp/dflash2-clean --report /tmp/dflash2-source.json
python3 patches/production/tools/build_native.py --plan
```

Use a new staging path. Verification is offline and requires only Python 3.12+,
Git and Bash. It replays 16 patches over clean v0.32 sources and compares every
accepted Python/data/support file; archive variants never enter production.
`patches/production/README.md` gives the native and portable image assembly path.
The original Phase 2 bundle SHA256 is
`b2c2d0ae12c229651c287d397b4e8f110b48399aa764228a6065c02bbe02445a`.

The retained Phase 3 independent no-cache/offline build matched all 12 custom
native libraries and all non-bytecode runtime files. Its image ID is
`sha256:ef0ece167b6bd7da3e24c794ba8e7d0a9834ef4209110f53ab8422f555bbf4c1`;
layer history/cache differences prevent OCI identity equality. Phase 4 verified
live equivalence, including APC, tools, vision and C2/C4 overlap. This pack does
not require another image rebuild. Notice-only source exports have separate hashes
and are not relabeled accepted source or qualified executable images.

## Runtime configuration

The sole profile is `w4a16-dflash2-tp4-262144`: TP4+EP, BF16 execution, native
262144 context, four sequences, 1024-token aggregate budget and aligned 832-token
full prefill chunks, utilization 0.91 and 7785 KV blocks. DFlash uses `/draft`,
seven proposals, draft TP4, probabilistic draft sampling, standard rejection and
`TRITON_ATTN`. The v2 runner and fixed split count 32 are required. `--enforce-eager`
is intentional: guarded target graphs explicitly cover supported C1–C4 decode;
prefill and partial-verification paths retain their eager guards.

APC is enabled. Disk BF16 PLE uses `PLE_TABLE_PATH=/models/model-00016-of-00017.safetensors`,
a 5 MiB/rank 1024-row stage and `B70_PLE_DISK_BF16=1`. The required residency shim
is preloaded from `/opt/b70-residency-shim/libl0_peer_residency_shim.so`.
Keep the baked test/helper paths in `PYTHONPATH`; they are accepted image contents,
not external campaign mounts. All flags/environment are in `pack.json`; the frozen
standalone contract is `patches/production/manifests/serving-contract.json`.

Controller binds target/draft read-only and supplies a writable persistent Docker
volume `b70ctl-qwen38-flashnext-dflash2-work` at `/work`; it does not reuse the live
campaign's work directory. It passes all four DRI devices, uses host IPC and the
recorded security options, and publishes port 8000 at the chosen Local/LAN address.
The campaign used host networking; Controller uses its existing bridge/port mapping.
No Controller-launched boot is claimed during this phase.

Build this branch's Controller from the repository root with
`/usr/lib/go-1.25/bin/go build -buildvcs=false -o /tmp/b70ctl-dflash2 ./cmd/b70ctl`.
The Go path follows this repository's Makefile; `-buildvcs=false` accommodates Go's
VCS-root detection in this linked worktree.
Installed `/usr/bin/b70ctl` 1.0.1 rejects the explicit work-volume declaration and
needs the validator change in this branch. In the branch-built TUI choose
**Model Packs → Import Local Pack**, enter this unpacked directory, then prepare
both pinned checkpoints. Local image acquisition reuses the exact accepted ID;
if absent, it reports no immutable registry reference. Do not substitute the
published MTP image. Import stores only manifest and README: retain the original
complete archive for recipes, notices and sources. Runtime resolution additionally
requires both complete models and four cards. Readiness checks do not reserve GPUs;
starting a second instance while the accepted service owns them is unsupported.

## Interfaces, performance and limitations

OpenAI-compatible chat/completions, streaming, Qwen3 reasoning, `qwen3_xml` automatic
tools and tool-result continuation, structured output and up to two images are
retained; video is disabled. Served model name is
`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`.

Accepted stock BetterBench v0.6.0: 336 measurements + 32 warmups; weighted decode
160.88 tok/s, median TTFT 137.78 ms, C1/C2/C4 aggregate 146.12/208.14/303.39 tok/s.
These are one workload run with warm OS/model/PLE caches and no APC hits, not a
quality score or a universal speed guarantee. C2/C4 aggregates were lower than the
preceding runtime. Archived custom coding 271.47 tok/s is a different fixture;
do not present it as the stock median. Source reports are in `patches/archive/qualification/`.

Native 261632 input + 512 reserved execution passed; cold/cached TTFT was
111.696/1.229 s with 0/260416 reused tokens. Observed aggregate KV capacity is
roughly 491K tokens and cannot hold four full-native contexts. Startup includes about 12.3 s
of bounded filter warmup; lazy graph/JIT work and roughly 200 ms mixed-prefill
stream gaps remain. Concurrent identical sampled requests may exchange client-to-output
assignments while exact output multisets and aggregate state agree. Finite checks
do not establish an unlimited soak or every workload. DSH/three.js telemetry is
reserved for editorial work, with no invented figures.

Publication remains blocked by the [distribution findings](patches/licenses/SOURCE_DISTRIBUTION.md),
notice incorporation, version ratification and an actual registry artifact. No
public clean-install test has been performed.
