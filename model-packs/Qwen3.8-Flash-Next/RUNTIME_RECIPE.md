# DFlash2 runtime reference

## Runtime and requirements

Accepted runtime: `qwen38-flashnext-dflash2:integrated-serving-v1`, image ID
`sha256:6e2ab4e686680a5208ff0119c4637341cba81942bd39b6576ea14dc8f547b83b`.
The model pack is **1.0.5**;
B70 LLM Controller / `b70ctl` remains **1.0.1**. The pack uses the existing
read-only target/assistant mounts and the image's ordinary writable filesystem.
Public runtime: `ghcr.io/wu1ff/qwen38-flashnext-b70:dflash2`, immutable GHCR index
`ghcr.io/wu1ff/qwen38-flashnext-b70@sha256:6e2ab4e686680a5208ff0119c4637341cba81942bd39b6576ea14dc8f547b83b`.
The verified linux/amd64 manifest is
`sha256:e33132c1e04d669929ed7909fe61478d27e88fa7816723c9bdf86b5e9ffc12d7`.
The published index preserves the accepted local image identity; no rebuild was used.

The qualified host has four Intel Arc Pro B70 32 GB GPUs and 256 GB system RAM,
a Linux Intel Xe/Level Zero driver stack, Docker with BuildKit for source builds,
readable `/dev/dri` and `/dev/dri/by-path`, and fast local NVMe storage. Four GPUs
are required; 256 GB describes the tested host, not a proven minimum. Observed
primary GPU usage was 29.765–29.774 GiB/rank. Reserve **181240433939 bytes** for
the target, **1206437758 bytes** for all six draft-repository files, plus image,
build space and writable caches. Disk PLE maps the 102400512256-byte target shard;
allow host page-cache capacity and cold-storage latency.

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
The retained independent no-cache/offline build matched all 12 custom native
libraries and all non-bytecode runtime files. Live equivalence covered APC,
tools, vision and C2/C4 overlap; OCI identity differs because of layer history
and caches. Build and qualification receipts remain in the archive. The published
runtime uses the accepted image unchanged. Notice-only source exports have
separate hashes and are not qualified executable images.

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
not external campaign mounts. All flags/environment are in `pack.json`; the qualified
mount-free contract is `patches/production/manifests/qualified-launch.json`.
`serving-contract.json` retains the original standalone capture.

The pack binds target and draft read-only through Controller's existing model and
assistant mechanisms. Its only explicit mount is the existing read-only DRI
by-path bind. No external bind, named volume or tmpfs is mounted at `/work`.

The accepted image runs as root in `/opt/venv` with a writable root filesystem.
Its existing `prepare-intel.py` creates `/work/flashnext-serve` with parents and
recreates that directory on every boot. It writes about 45 MiB of qualified
configuration, weight index and frontend files; weight files are symlinks into
`/models`. Triton/vLLM caches create their own directories. `TMPDIR=/tmp` uses
the image's existing mode-1777 temporary directory. Losing these files on
container recreation is safe; cold JIT caches can lengthen startup.

Disk PLE stays on NVMe in `/models/model-00016-of-00017.safetensors`, mapped
directly from the original 102,400,512,256-byte shard. The 1,024-row, 5 MiB/rank
DMA stage is pinned CPU memory. No model weight or PLE staging file is copied
into the container layer. See [the storage investigation](WORK_STORAGE.md).

Controller publishes port 8000 at the chosen Local/LAN address; the preserved
standalone service uses host networking. Image acquisition pulls the
immutable registry reference or reuses the accepted ID. Import stores only
manifest and README: retain the complete archive for recipes, notices and sources.
Runtime resolution requires both complete models and four cards; readiness checks
do not reserve GPUs. Installed b70ctl 1.0.1 compatibility includes import, acquisition, profile rendering
and a healthy mount-free launch; retained live qualification covers performance
and full context. Public artifact checks reuse that qualification without
interrupting the existing service.

## Interfaces, performance and limitations

OpenAI-compatible chat/completions, streaming, Qwen3 reasoning, `qwen3_xml` automatic
tools and tool-result continuation, structured output and up to two images are
retained; video is disabled. Served model name is
`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`.

Accepted stock BetterBench v0.6.0 rerun: 336 measurements + 32 warmups; weighted
combined decode 160.11 tok/s, median TTFT 139.39 ms, stream-update p99 25.82 ms,
C1/C2/C4 aggregate 146.50/227.12/316.01 tok/s. This is one workload run with
warm OS/model/PLE caches and no APC hits, not a quality score or universal speed
guarantee. The [README](README.md) retains the report and aquarium-session graphics;
earlier, distinct measurements remain in `patches/archive/qualification/`.

Native 261632 input + 512 reserved execution passed; cold/cached TTFT was
111.696/1.229 s with 0/260416 reused tokens. Observed aggregate KV capacity is
roughly 491K tokens and cannot hold four full-native contexts. Startup includes about 12.3 s
of bounded filter warmup; lazy graph/JIT work and roughly 200 ms mixed-prefill
stream gaps remain. Concurrent identical sampled requests may exchange client-to-output
assignments while exact output multisets and aggregate state agree. Finite checks
do not establish an unlimited soak or every workload.

Existing [source-distribution findings](patches/licenses/SOURCE_DISTRIBUTION.md),
source-access instructions and required notices are retained with the complete
production and experimental sources.
