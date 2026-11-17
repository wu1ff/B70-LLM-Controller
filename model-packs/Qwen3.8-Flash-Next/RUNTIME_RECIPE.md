# DFlash2 vLLM 0.33 runtime recipe

## Release identity and public delivery

Model pack **1.0.6**, release tag `qwen3.8-flash-next-v1.0.6`,
archive `qwen38-flashnext-b70-1.0.6.tar.gz`.
It follows the actual published 1.0.5 archive, whose SHA256 is
`b1bef2c8e382cc5ae69033221b9822a46686198c54c95d4028ec91e749acec9c`.
The [approved README](README.md) is included byte-for-byte.

**Installed Controller compatibility passed:** unchanged `/usr/bin/b70ctl`
1.0.1 validated, imported and launched this production configuration using
isolated XDG state and existing complete target/draft snapshots. The exact v6
image reached HTTP readiness in 240.6 seconds against the original checkpoint
and served reasoning, a required
tool call and continuation, and two four-request slot-reuse waves. No model
download, Controller change, image rebuild or benchmark rerun was needed.

The original development/qualification launcher included an inert diagnostic
worker extension. Production deliberately omits that extension, its two external
module binds, its import path and development HTTP endpoints. The frozen profile
and complete source provenance are preserved unchanged. See the explicit
[production differences and check receipt](CONTROLLER_COMPATIBILITY.json).
That receipt records the local compatibility exercise before publication. The
accepted image has now been published unchanged and independently inspected
anonymously at GHCR as recorded below.

Accepted image, used unchanged:

- Local reference: `local/dflash2-v033:expanded2-exact-v6`.
- OCI index / local Docker ID:
  `sha256:2fa91d1ac8c17ad866d9ae7eaea83ac1b1666ffd0ccceb7685564f55528c47af`.
- linux/amd64 runtime manifest:
  `sha256:0d9a3cfe4bec7a28400f9ad425b13c9853734f19086a11dbaf884f7faebb8d60`.

Published GHCR tag: `ghcr.io/wu1ff/qwen38-flashnext-b70:dflash2-v033`.
Use this complete immutable pull reference:

```sh
docker pull ghcr.io/wu1ff/qwen38-flashnext-b70@sha256:2fa91d1ac8c17ad866d9ae7eaea83ac1b1666ffd0ccceb7685564f55528c47af
```

The registry index, linux/amd64 manifest, config
`sha256:8467812da11b910466d0814e9361b6771ad2c66de678a57c3297a1b9e8507064`,
all 30 filesystem layer descriptors and the BuildKit/SLSA attestation are
identical to the accepted local OCI objects. Anonymous manifest/blob access was
verified. The index retains the attestation manifest
`sha256:2441d0483350fde5d5f8e88da30e2186a5dc7df716bdc11cc9deae1e03b318c1`.
The local Docker ID is the OCI index identity on the qualified Docker host;
the runtime manifest and config have their own identities. `pack.json` records
the verified immutable registry reference and accepted digest. Existing
`:dflash2` (v0.32) and `:disk` tags are preserved.

## Hardware, storage and installed dependencies

The accepted benchmark host has 4× Intel Arc Pro B70 (32 GiB each), AMD EPYC
7443 (24 cores / 48 threads), 256 GiB installed RAM and NVMe-backed storage.
Use compatible Linux Intel Xe/Level Zero drivers, Docker, readable `/dev/dri`
and all four GPUs. This describes the tested host, not a proven RAM minimum.

Reserve 181,240,433,939 bytes for the 29 declared target files and
1,206,437,758 bytes for the six draft-repository files, plus the image and
writable caches. The original 102,400,512,256-byte PLE shard remains on model
storage, mapped read-only; NVMe latency and reclaimable OS page cache affect
startup and preparation. There is no separate PLE conversion/download.

The accepted image contains Python 3.12.3, vLLM **0.33.0+xpu** at upstream
`4ba787ec9959e566d3b587a796dff07673324da5` plus the packaged patches, Torch
2.14.0+xpu, Transformers 5.20.2, Triton 3.8.0+xpu / triton-xpu 3.8.0,
vllm-xpu-kernels 0.1.15.4 and Intel/SYCL runtime 2026.1.1.
The [installed package census](patches/production/manifests/installed-metadata.json)
and [dpkg census](patches/production/manifests/installed-dpkg-status.txt) record
the full inherited stack. Old v0.32 dependency-check results do not qualify this
stack. No dependency upgrades or installations are part of this release staging.

## Exact target and draft

| Role | Repository | Revision | Read-only mount |
| --- | --- | --- | --- |
| W4A16 target, BF16 execution | Intel/Qwen3.8-Flash-Next-W4A16-AutoRound | `4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897` | `/models` |
| BF16 DFlash2 draft | incoai/Qwen3.8-Flash-Next-DFlash2 | `a22a1744e85c17364dcc54a6a420af7135d4c7f6` | `/draft` |

The target's nonconsecutive 17-shard naming and extra tensor file are intentional;
use the exact 29-file inventory in [pack.json](pack.json), not a generated sequence.
The assistant includes all six pinned repository files. Its recorded weight
SHA256 is `d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f`.
[Target identities](patches/production/manifests/target-pinned-inventory.json),
[draft identities](patches/production/manifests/draft-pinned-api.json) and
[model-reference audit](patches/production/manifests/model-reference-audit.json)
retain stronger provenance. This staging validates paths, sizes and small metadata;
it does not rehash checkpoint weights. Model weights are external inputs and are
not in the pack. Obtain these exact revisions under their own terms.

## Accepted serving configuration

The sole profile remains `w4a16-dflash2-tp4-262144`: target TP4 / EP4, BF16
execution, native 262,144 input-plus-output context, four sequences, 1,024-token
aggregate batch budget and utilization 0.91. KV blocks are **automatic**;
do not restore v0.32's historical 7,785-block override. The accepted profile
reported 460,944 shared KV-cache tokens; four full-native requests do not fit.
One simultaneous native-limit request was qualified previously.

DFlash2 uses `/draft`, seven proposals, draft TP4, probabilistic draft sampling,
STANDARD target rejection/verification and `TRITON_ATTN`. APC, Qwen3 reasoning,
automatic `qwen3_xml` tools, tool continuation, structured output and up to two
images are retained; video is disabled. Served name:
`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`.

The [frozen v6 profile](patches/production/runtime/profile.json), SHA256
`ed6e18ce9c7bf052757da399d94857f99ebff0def85c94bb725d83301b7a7a85`,
is preserved unchanged as development/qualification provenance. Production
[pack.json](pack.json) preserves all 40 serving environment entries and all
engine arguments after the diagnostic omissions below. In particular retain:

- `VLLM_USE_V2_MODEL_RUNNER=1`, `VLLM_XPU_DFLASH_FIXED_SPLITS=32`,
  `VLLM_XPU_INC_WNA16_BACKEND=w4a16`, four-rank affinity and the oneCCL settings.
- Live graph checks, fused/shared GDN metadata, query graphs, native HC 20,
  `ASTRA_DFLASH_RAW_REDUCE=0`, query block 8, support 16 and score scale 1.0.
- Thinking EOS guard, proposal calibration and exact BF16 draft convolution;
  compiled graph update and fused graph copy; eight PLE gather threads;
  exact BA, HC split32 10, MoE glue and deferred PLE, all enabled as captured.
- `B70_PLE_DISK_BF16=1`, original `/models/model-00016-of-00017.safetensors`,
  and `LD_PRELOAD=/opt/b70-residency-shim/libl0_peer_residency_shim.so`.
- `TMPDIR=/work/tmp` and the unchanged offline cache environment.

Generic `--enforce-eager` coexists with explicitly guarded target/query graph
paths for supported C1–C4 shapes. There is no new tuning. The entrypoint
[prepare-serve.sh](patches/production/support/b70-flashnext/prepare-serve.sh)
creates `/work/tmp` and cache directories and a fresh serving view under
`/work/flashnext-serve.XXXXXXXX/view`; it validates checkpoint/frontend metadata,
links weight shards and leaves checkpoints read-only. The writable container
filesystem supplies disposable work/cache storage; no external `/work` mount is used.
See [storage and portability](WORK_STORAGE.md).

## Supported production launch and diagnostics omitted

Use unchanged b70ctl 1.0.1's **Model Packs → Available Packs** workflow to
acquire 1.0.6 from the public catalog. Alternatively, download the archive and
`SHA256SUMS` from the [1.0.6 release](https://github.com/wu1ff/B70-LLM-Controller/releases/tag/qwen3.8-flash-next-v1.0.6),
run `sha256sum -c SHA256SUMS`, unpack into an empty directory and use
**Model Packs → Import Local Pack** on that directory. Select the target, review the plan, then open
**Run Model** and choose the four-card, 256K DFlash2 profile. On the test host,
both exact checkpoints and the exact image were reused; the Controller showed
**PROFILE AVAILABLE**, launched the image, and later showed **Running**.
The test used local access at port 18084 and was stopped through the Controller.
Production Controller configuration was not changed.

| Difference from frozen development launcher | Reason |
| --- | --- |
| Omit `--worker-extension-cls parity_worker.ParityWorker` | Adds 14 explicit diagnostic RPC methods; no constructor, startup/destructor hook, or serving-method override. |
| Omit `PYTHONPATH=/opt/dflash-parity` | Used only to import the removed diagnostic extension. |
| Omit `VLLM_SERVER_DEV_MODE=1` | Enables developer cache/RPC/info/sleep endpoints; ordinary serving does not use them. |
| Omit the two `/opt/dflash-parity/*.py` binds | Their extension is not selected. Both files remain in the frozen provenance subtree. |
| Use writable container `/work`; no external work bind | The accepted image entrypoint creates temporary/cache directories and a fresh serving view. |

Omitted diagnostics include parity snapshots, shared-GDN A/B controls, synthetic
metadata probes, explicitly armed timing wrappers, HC/cyclic-expert probes,
tuning mutations and `/work` probe scripts. No diagnostic RPC is invoked during
startup or ordinary requests. Some require additional campaign helpers that are
also unnecessary for serving. Core DFlash2, GDN/recurrent state, PLE, graph
execution, STANDARD verification and the EOS guard are implemented inside the
accepted image and remain enabled by their original settings.

All four ranks logged target and query graph capture for C1–C4, with passing
changed-input checks and no graph capture/equality/fallback failures. Generic
`--enforce-eager` still permits these explicit guarded graphs. The final
original-checkpoint exercise recorded 654 draft rounds,
4,568 effective draft tokens and 1,953 accepted tokens.
Seven-token speculation was configured and full-width target/query graphs ran
on all ranks. Counters deduct grammar-invalid drafts for required tool calls,
so mixed tool traffic need not total exactly seven counted tokens per round.
The earlier concurrency-only repeat counted 648 rounds / 4,536 proposals, exactly
seven per round. Both checks observed four simultaneous requests and
zero running/waiting afterward. The first short check had one response reach
its 512-token budget during reasoning; its HTTP response was successful with
`finish_reason=length`. Only the concurrency portion was repeated with a 1,024
response-token budget; all eight completed correctly. Pack settings were unchanged.
The first run used an existing pinned Controller checkpoint copy. The final
check used an existing discoverable inventory whose 29 target files are the
original qualification checkpoint inodes; all four workers mapped the original
PLE inode read-only. No links, model files or replacement mounts were created.
This is a targeted compatibility check, not a fresh quality/performance campaign.

Keep the complete unpacked archive for source, notices and recipes. Controller
import normally retains only manifest, README and local provenance; no external
worker assets are required by this production configuration.

The [portable launcher](launch.py) renders the same production configuration
from any directory. Replace the two example paths with complete pinned snapshots:

```sh
python3 /path/to/unpacked-pack/launch.py \
  --target /path/to/target-snapshot --draft /path/to/draft-snapshot \
  --name dflash2-v033 --port 18083
```

For the portable launcher, first pull the immutable GHCR reference above.
The default bind is `127.0.0.1`; `--bind-host 0.0.0.0` selects LAN. The launcher
checks the production differences against the frozen profile and preserves
private IPC, 32 GiB shared memory, unconfined seccomp, DRI access and the accepted
image digest with `--pull never`. It checks pinned file sizes and image identity
before execution. Adding `--execute` starts the server; stop that standalone
container with `docker stop dflash2-v033`. It supplies only the target/draft binds.

## Source package and build provenance

Accepted input: `dflash2-v033-v6-source.tar.gz`, SHA256
`cf0e839191f32991228668b92f8ebd1ee901a1b0facbb9a419c3cf67cc59b20e`.
Its complete `patches/production` subtree is integrated unchanged, including
its 851-file input lock. Run from the unpacked pack with a new temporary path:

```sh
python3 patches/production/tools/verify.py \
  --staging /tmp/dflash033-source-clean --report /tmp/dflash033-source.json
python3 patches/production/tools/build_native.py --plan
```

Python 3.12+, Git and Bash suffice for offline verification. Ten ordered `-p1`
patches reconstruct 3,679 files; 3,021 Python files, the calibration tensor,
17 native families and build-script syntax are checked. No GPU, Docker, network,
compiler or weights are needed. The exact calibration tensor SHA256 is
`f6a104dd32e6a41754a89a2bb1cd3a3f35ec04ef6083c68ffd17e73a75375624`.
Use the [production instructions](patches/production/README.md),
[provenance boundaries](patches/production/BUNDLE.md),
[patch series](patches/production/manifests/patch-series.json) and
[input lock](patches/production/manifests/input-lock.json).

The binary vendor foundation is
`vllm/vllm-openai-xpu@sha256:010bdc441c72ee222f459d3b688bb41317525f1bd7223afe6218490b0c526832`,
original config `sha256:7b3eb7c792a5d3a049effcb0d792ce95d4a482dd47a965d8a05a244a47682813`.
The actual native compiler is oneAPI **2026.0.0.20260331**, using 15 locked
2026.0.0-947 packages, with GCC 13.3.0-6ubuntu2~24.04.1. This compiler differs
from the inherited runtime 2026.1.1 and the older release's compiler.
SYCL-TLA is pinned to `87f6850680a580654b9ea2c80dbc01aeb36ad231`, archive
SHA256 `5938a31e3940d1c90e88d9126788990a25a5e0ac968e53345461459b5ea21218`.
Compiler packages, vendor image and model weights remain external inputs;
[immutable inputs](patches/production/manifests/immutable-inputs.json),
[compiler lock](patches/production/manifests/compiler-lock.json) and
[native inventory](patches/production/manifests/native-builds.json) specify them.

The final historical build is BuildKit `p06lpoihbb5s1lyvy9nno1sg0`, from
`sha256:848f48ff831ff244154bab6ef2938093bb0a588b2e2bb643a51c8a57b920789b`.
[Build references](patches/production/manifests/build-record-audit.json) retain
all ten layers/material references and [image lineage](patches/production/manifests/image-lineage.json)
retains the layer chain. Thirty OCI/SLSA attachment payloads retain their original
digests. The installed source differs from its Git HEAD and from the source mirror;
use installed hashes and replay results as authority. A full independent native
or image rebuild is not claimed and was not attempted. Historical source build
commands in the source-only package are documentation, not staging instructions.

## Accepted BetterBench 0.6.1 evidence

The existing two temperature-1.0 runs used top-p 0.95, top-k 20, 344 measured
samples and 34 discarded warmups each. Decode is the weighted arithmetic mean
of category median rates excluding TTFT; concurrency throughput includes TTFT
and scheduling. These are performance measurements, not quality scores.

| Metric | Run 1 | Run 2 |
| --- | ---: | ---: |
| Weighted decode | 178.57 tok/s | 183.85 tok/s |
| C1 / C2 / C4 aggregate | 164.59 / 233.55 / 333.78 tok/s | 159.76 / 238.36 / 340.10 tok/s |
| Weighted category-p50 TTFT | 131.18 ms | 129.19 ms |

Stock greedy prefill retains eight measurements per depth/run. At nominal
16K / 32K / 64K depths, actual prompt-token medians are 11,871 / 23,576 / 47,082.5
for run 1 and 11,863 / 23,625 / 47,169.5 for run 2. Median prompt/TTFT rates are
4,827.04 / 4,695.76 / 4,327.08 tok/s and 4,801.71 / 4,677.96 / 4,309.97 tok/s.
The [README tables](README.md#betterbench-performance) retain full formatting.
[Run 1 raw results](patches/benchmarks/betterbench-0.6.1/run-1/results.json),
[run 2 raw results](patches/benchmarks/betterbench-0.6.1/run-2/results.json) and
[benchmark provenance](patches/benchmarks/betterbench-0.6.1/benchmark-provenance.json)
are included. Raw SHA256 values are respectively
`a5ff42caa70f928eacedbc7c6daab742f0799186de39a6935fd850e5b464875c` and
`6c9e00051b2d0321576a3aa1d6b90cec66bab420e81c0b3015c2d69c8ba62204`.
No benchmark or qualification rerun is part of staging.

## Retained notices and historical scope

[Credits/notices](patches/licenses/NOTICE.md),
[source access](patches/licenses/SOURCE_ACCESS.md) and
[source-distribution findings](patches/licenses/SOURCE_DISTRIBUTION.md) retain
published attribution and open findings. Their v0.32 evidence does not attest
the v0.33 inherited stack. The old annotated delivery is preserved separately
under [v0.32 source delivery](patches/archive/v032-source-delivery/README.md).
Historical experiments/assets remain historical and are not replay inputs.
No new licensing infrastructure, clearance or source-delivery promise is added.
