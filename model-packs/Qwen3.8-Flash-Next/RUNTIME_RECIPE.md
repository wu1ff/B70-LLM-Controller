# Build and serve the Flash-Next Disk runtime

This recipe constructs the released **`qwen38-flashnext-b70` 1.0.3** runtime from
its official vendor dependency and cumulative [Disk source package](patches/disk/README.md),
then verifies its installed payload and launches Base or MTP3. The execution
order follows [build.py](patches/disk/build.py), the [Dockerfile](patches/disk/Dockerfile)
and [native/build.sh](patches/disk/native/build.sh).

The published image is `ghcr.io/wu1ff/qwen38-flashnext-b70:disk`. Its immutable
acquisition reference is:

```text
ghcr.io/wu1ff/qwen38-flashnext-b70@sha256:b13fed6dd54051395f603409ada440c74afcbd5331810eee3acd62315d7787e3
```

## 1. Build target and prerequisites

The serving target is **four Intel Arc Pro B70 cards**, TP4 with expert
parallelism, PP1/DP1, and the integrated `Qwen4ExpForConditionalGeneration`
checkpoint `devan-carlin/Qwen3.8-Flash-Next-W4A16` at revision
`40b8f18df4d4a32cb6e687a51c78207e5e438522`. Base leaves speculation disabled;
MTP3 uses the checkpoint's integrated draft model with exactly three speculative
tokens. No external assistant checkpoint is needed.

W4A16 describes the quantized model weights and 16-bit activation contract.
The external PLE table remains **BF16**, without conversion: the original
`ple_table_qwen4exp.pt` is 102,400,493,290 bytes. The complete pinned checkpoint
contains 31 files totaling 179,841,735,233 bytes; its inventory and mount path
are in [pack.json](pack.json) and [external-model.json](patches/disk/manifests/external-model.json).
Allow roughly 180 GB for the checkpoint, plus the image, caches and staging.
Disk uses bounded pinned PLE staging, but mapped pages and runtime allocations
still consume host memory. No minimum-RAM or storage-cold endurance guarantee
is established; SSD/NVMe latency and OS page-cache history affect serving.

Source compilation requires Linux amd64, host Python **3.11 or newer**, Docker
with its legacy builder available, sufficient compiler RAM/storage, the exact
vendor image, and the separately acquired compiler packages below. Compilation
needs no GPU, checkpoint, host compiler, pip resolution or build-time network.
Serving additionally requires working Intel XPU drivers, access to all four
cards through `/dev/dri` and `/dev/dri/by-path`, the complete checkpoint, and a
writable container `/work` for prepared configuration and caches.

## 2. Acquire and verify the inputs

Use the complete pack source from the [1.0.3 release](https://github.com/wu1ff/B70-LLM-Controller/releases/tag/qwen3.8-flash-next-v1.0.3)
or its tagged public checkout. For example:

```bash
git clone --branch qwen3.8-flash-next-v1.0.3 --single-branch \
  https://github.com/wu1ff/B70-LLM-Controller.git B70-LLM-Controller
```

For an extracted release archive, verify it against the release's `SHA256SUMS`
before using it. Define these paths once; replace the absolute placeholders
with your checkout or extracted pack, compiler cache, and new evidence directory:

```bash
PACK_DIR=/absolute/B70-LLM-Controller/model-packs/Qwen3.8-Flash-Next
DISK_DIR="$PACK_DIR/patches/disk"
COMPILER_DIR=/absolute/authorized/compiler-cache
EVIDENCE_DIR=/absolute/new-disk-build-evidence
REBUILD_TAG=qwen38-flashnext:disk-fresh-test
cd "$DISK_DIR"
sha256sum -c SHA256SUMS
```

All listed source-package hashes must pass. The wrapper also rejects missing or
unlisted files, so keep build evidence and compiler binaries outside this
source directory. The sources are **already cumulatively patched**. Historical
Vanilla 001–007 and `disk/patches/` diffs explain lineage; applying them again is
not a construction step.

Both Docker stages inherit the official **vLLM 0.30.0 XPU** linux/amd64 image.
Acquire its pinned multi-platform index, selecting amd64:

```bash
VENDOR_REF=vllm/vllm-openai-xpu@sha256:fc0e112afb64e3a06fe8daff34652435822a629412f38efce8f0f67a46636b8d
docker pull --platform linux/amd64 "$VENDOR_REF"
docker image inspect --format '{{.Id}} {{.Os}}/{{.Architecture}}' "$VENDOR_REF"
```

Expected local identity:

```text
sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9 linux/amd64
```

A trusted saved vendor image can supply the same identity. `build.py` checks
this local ID, architecture and OS before building; it does not perform a live
registry provenance lookup. Both `FROM` instructions use that ID. If Docker
reports a different ID, resolve the image-store mismatch described in section 5
before building.

The distributed and builder-supplied inputs are:

| Input | Authority | Supplied by |
| --- | --- | --- |
| Final Python/configuration overlay and native source | [overlay.json](patches/disk/manifests/overlay.json), [native-source.json](patches/disk/manifests/native-source.json) | Pack |
| Kernel source base `0.1.14.1` | Commit `6d92b1bfbf32767ecda8e819613eb151e70030ad`, recorded in [runtime-lock.json](patches/disk/manifests/runtime-lock.json) | Pack's cumulative source |
| SYCL-TLA source archive | Commit `87f6850680a580654b9ea2c80dbc01aeb36ad231`; [archive lock](patches/disk/manifests/sycl-tla.json) | Pack |
| Vendor filesystem and installed dependencies | [vendor inventory](patches/disk/manifests/vendor.json), [Python versions](patches/disk/manifests/vendor-python.json), [dpkg inventory](patches/disk/manifests/vendor-dpkg.tsv) | Builder acquires pinned image |
| 15 exact Intel compiler packages | oneAPI `2026.0.0-947`; [compiler-lock.json](patches/disk/manifests/compiler-lock.json) | Builder acquires separately |
| Model weights, BF16 PLE table and tokenizer/configuration | Pinned checkpoint inventory | Controller or builder downloads separately |

Obtain the compiler `.deb` files under their applicable terms from the lock's
official Intel apt `pool_base`, or an authorized cache. Each download URL is
`pool_base` plus its exact locked filename. Put all 15 files directly in
`COMPILER_DIR`; the wrapper verifies every SHA256. An arbitrary installed
oneAPI compiler does not satisfy these locks. The public package contains no
compiler binaries, model weights, PLE table, or accepted prebuilt native objects
or libraries.

## 3. Prepare the context and compile the native components

`build.py` first checks the vendor identity, refuses an occupied output tag,
verifies the complete distributed package and all compiler hashes, then creates
a disposable copy. It copies compiler packages into `dependencies/compiler/`
only in that copy and records the executed build inputs.

The distributed sources carry Wulf modification notices on 15 upstream Python
files and six upstream native files. The wrapper checks the historical source
locks after removing only the added notice for comparison. It **keeps all 15
Python notices** in the installed overlay and restores only the six native
compiler inputs to their qualified bytes in the disposable context, preserving
native binary reproduction. Distributed native sources retain their notices.
No historical implementation patch is applied by this reconstruction.

The native Docker stage proceeds in this order:

1. Extract the 15 `.deb` files with `dpkg-deb -x` into `/toolchain/oneapi`.
   This stages the toolchain without installing packages into the vendor runtime.
2. Copy `native/` to `/build`, copy the locked SYCL-TLA archive there, and extract
   it as `/build/sycl-tla-87f6850680a580654b9ea2c80dbc01aeb36ad231`.
3. Run `bash /build/build.sh`, which performs the following builds in order.

| Build step | Output and purpose |
| --- | --- |
| Check `moe-inputs.sha256`; run [gdn-build.sh](patches/disk/native/gdn-build.sh) | `libgdn_index64.so`: corrected convolution/state indexing and partial speculative verification |
| Copy `source/csrc` to `/build/csrc`, recheck the six direct MoE inputs; run [remap-build.sh](patches/disk/native/remap-build.sh) | `libmoe_remap_guard.so`: bounds-checked expert remapping |
| Run [attention-build.sh](patches/disk/native/attention-build.sh) | `libb70_attn_reduce.so`: accepted device-planned, tiled attention reduction |
| Compile `l0_peer_residency_shim.c` with vendor GCC | `libl0_peer_residency_shim.so`: B70 peer-residency containment |

The SYCL builds use the extracted compiler at
`/toolchain/oneapi/opt/intel/oneapi/compiler/2026.0`, vendor Python 3.12 and
Torch `2.13.0+xpu` headers/libraries, C++20, O3/NDEBUG and `spir64`. The scripts
preserve qualified absolute source/include paths, dispatcher definitions, link
libraries and relative rpaths. GCC 13.3 builds the C shim with C11/O2, PIC and
warnings as errors, linking `ze_loader`, `dl` and pthreads. Full flags remain
in the scripts.

The vendor vLLM/Torch/XPU/SYCL/Level Zero/XCCL/UCX stack is inherited. The build
adds three native extension libraries; it does not rebuild that whole stack or
run the kernel source tree's wheel/CMake build. GDN rebuilds the interface
translation unit with corrected inline kernels and links the existing
`libgdn_attn_kernels_xe_2.so`.

The retained native corrections have distinct roles. GDN uses 64-bit deep-state
offsets, packed actual query lengths for partial speculative verification, and
no state access for graph-padded requests. Compatible shared Mamba metadata
keeps APC/ALIGN state staging consistent. The MoE remapper validates global IDs
before `expert_map` reads, validates local IDs before count/offset access, and
avoids negative pointers for excluded rows. This prevents router-padding reads
such as `expert_map[-1]` while retaining valid-row arithmetic. Its six-input hash
check protects the final guarded source from replacement by a pristine snapshot.
The tiled reducer preserves ordered per-coordinate accumulation and short-case
normalization; its MTP3 selection conditions are described below.

`build.sh` stages the three extensions under
`/out/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/` and the shim under
`/out/opt/b70-residency-shim/`. Their accepted hashes are in `runtime-lock.json`.

## 4. Assemble the serving image

The final stage starts again from the same official vendor image. It copies
`overlay/` to `/`, then copies `/out/` from the native stage to `/`. Thus it adds
the cumulative Python/configuration sources and rebuilt native outputs without
copying `/build`, `/toolchain`, the compiler packages or native object files.
The inherited vendor filesystem, including its existing tools and licenses,
remains present.

The overlay installs XPU Qwen4Exp model selection, native dispatch, disk PLE,
HC/MTP logic, scheduler and GDN/Mamba/attention metadata sources into
`/opt/venv/lib/python3.12/site-packages/`. XPU selection uses the existing backend
with XPU support. The retained HC width-gated K-split workaround prepares
contiguous K=5120 weight halves at load time and, for widths greater than 64,
sums two BF16 down-GEMMs; smaller widths use the original GEMM. This avoids the
affected wide-GEMM execution path with bounded BF16 reduction-order differences;
it does not fix the underlying vendor operation. Graph-input
preparation refreshes committed ngram context and PLE buffers before FULL replay;
the inherited PLE capture path also uses the active capture stream.

Disk PLE maps the original `[320001536,160]` BF16 table with CPU `torch.load`
`mmap=True`, validates and protects its file mapping, and gathers only TP-owned
rows. Non-owned/padded contributions are zero. Each worker uploads the gathered
rows through one serialized pinned stage before forward or graph replay;
completion events protect host-stage reuse and ownership changes drain the old
stream. At MBT1024, `1024 × 16 × 160 × 2` bytes is **5 MiB per rank / 20 MiB
across TP4** of pinned host staging, with a corresponding device producer buffer.
This figure excludes other buffers and reclaimable file-backed page cache. The
selected Disk path allocates no whole-table pinned slabs or device PLE weight
shard.

The shim is installed at
`/opt/b70-residency-shim/libl0_peer_residency_shim.so` and activated by the launch
profile's `LD_PRELOAD`. It tracks device-allocation ownership and redirects
classified large peer residency to the owner, containing B70 host-memory
mirroring while preserving pool and oneCCL residency. It remains a required
serving component.

The final Docker instructions set `WORKDIR /opt/venv` so the vendor's
`/workspace/vllm` checkout does not shadow installed sources, mark
`/opt/b70-flashnext/prepare-serve.sh` executable, create the `/work` cache
directories, and compile Python bytecode. The entrypoint is that preparator.
Launch commands must begin `vllm serve <mounted-checkpoint> [flags...]`.

The [preparator](patches/disk/overlay/opt/b70-flashnext/prepare-serve.sh) checks for
the PLE table and weight index, symlinks the mounted checkpoint's visible entries
except `config.json` into `/work/flashnext-serve`, and copies the baked
[serve-config.json](patches/disk/overlay/opt/b70-flashnext/serve-config.json) there.
That config is the pinned model config minus exactly five unsupported
`text_config.indexer_*` keys, selecting dense attention while preserving vision and
quantization settings. The original checkpoint stays read-only. The preparator
executes `/opt/venv/bin/vllm serve /work/flashnext-serve` with the remaining flags;
`PLE_TABLE_PATH` continues to reference the original mounted table.

Upstream licenses/notices remain in the inherited image and distributed source.
All 15 Python modification notices reach the final image. The published RC was
produced by [build-notice-image.py](patches/disk/build-notice-image.py), which adds
those notices over the already qualified local predecessor without native
compilation, preserving its effective configuration and original layers. That
release correction explains the published image identity; a fresh source build
uses the two-stage recipe above and verifies the same installed payload.

## 5. Build and verify

With the inputs and variables from section 2 ready, run:

```bash
cd "$DISK_DIR"
python3 build.py --compiler-dir "$COMPILER_DIR" \
  --tag "$REBUILD_TAG" --evidence "$EVIDENCE_DIR"
python3 verify.py "$REBUILD_TAG" --output "$EVIDENCE_DIR/recheck.json"
```

The wrapper uses `DOCKER_BUILDKIT=0 docker build --pull=false --no-cache
--network=none`, records vendor inspection, build-input manifest, command, log,
result and image inspection in the evidence directory, then invokes `verify.py`
automatically. The second command is an optional independent payload recheck.
Use a fresh isolated tag; omit `--assign-disk` to preserve existing local aliases.

[verify.py](patches/disk/verify.py) starts a temporary network-disabled container
with the Python entrypoint, no GPU/device mounts and no model engine. It compares
all regular files under installed `vllm`, `vllm_xpu_kernels`, `/opt/b70-flashnext`
and `/opt/b70-residency-shim`, excluding bytecode, with the accepted inventory.
The historical hashes are adjusted only for the 15 required Python notices.
Success exits zero and records:

```text
accepted_entries = rebuilt_entries = exact_matches = 3118
required_notice_files = 15
missing = extra = []
nonmatching = {}
exact = true
```

The original independent native build and archive import matched all 3,118
historical installed entries; notice-corrected RC and clean-pull receipts match
all 3,118 notice-bearing entries. These checks establish the covered installed
payload, including native binaries and serving configuration. They exclude
bytecode, unrelated vendor filesystem files, layer metadata and build timestamps.
**An independently rebuilt image can have a different Docker ID and OCI image
bytes.** It is not the published immutable artifact merely because verification
passes; the unchanged Controller pack pins the published image's ID.

The supported containerd image store reports the released Docker `.Id` as the
platform manifest digest `b13fed6d…`. Its distinct OCI config digest is
`sha256:2399885f6fbc59cdb602fe869350ddd80e085f62da25d55859d683b5e3450baf`.
A classic image store can expose the config ID and fail the Controller's exact-ID
check; the same issue affects the vendor `FROM` identity. Check with
`docker image inspect` and `docker info --format '{{json .DriverStatus}}'`
(containerd reports `io.containerd.snapshotter.v1`). Use a containerd-enabled
Docker installation or a separate compatible daemon if the IDs differ; retagging
cannot change an image ID. Do not replace the pack's pinned digest to bypass it.

## 6. Launch Base or MTP3

For standalone serving, continue with the image just reconstructed and verified
in section 5. Set `MODEL_DIR` to the complete pinned checkpoint and use the same
`REBUILD_TAG` defined in section 2. [launch.py](patches/disk/launch.py) renders the
complete profile with the explicit image argument:

```bash
MODEL_DIR=/absolute/pinned-checkpoint
SERVE_IMAGE="$REBUILD_TAG"
docker image inspect --format '{{.Id}}' "$SERVE_IMAGE"
python3 "$DISK_DIR/launch.py" mtp3 --model-dir "$MODEL_DIR" \
  --image "$SERVE_IMAGE" --bind 127.0.0.1 --port 18000
```

The last command only prints the Docker launch. Use `base` for Base. After
stopping any managed service using those GPUs and freeing the chosen endpoint,
launch the reconstructed image with the same arguments and `--execute`:

```bash
python3 "$DISK_DIR/launch.py" mtp3 --model-dir "$MODEL_DIR" \
  --image "$SERVE_IMAGE" --bind 127.0.0.1 --port 18000 --execute
```

The script defaults to the historical local `qwen38-flashnext:disk` predecessor
alias, so pass `--image` explicitly. Its profiles fix native context at 262144;
it has no context-selection option. Use Controller profiles for the four
supported context selections with the published image.

Allow approximately **4½–5 minutes** for startup; cache/JIT history varies.
After readiness, the example endpoint is `http://127.0.0.1:18000/v1` and its
health check is:

```bash
curl --fail http://127.0.0.1:18000/health
```

### Published image alternative

To serve the published immutable runtime instead of the reconstruction, acquire
it separately and render the same standalone profile:

```bash
SERVE_IMAGE=ghcr.io/wu1ff/qwen38-flashnext-b70@sha256:b13fed6dd54051395f603409ada440c74afcbd5331810eee3acd62315d7787e3
docker pull --platform linux/amd64 "$SERVE_IMAGE"
docker image inspect --format '{{.Id}}' "$SERVE_IMAGE"
python3 "$DISK_DIR/launch.py" mtp3 --model-dir "$MODEL_DIR" \
  --image "$SERVE_IMAGE" --bind 127.0.0.1 --port 18000
```

The inspected ID must match the published ID above. Append `--execute` to the
render command to launch this alternative once the GPUs and endpoint are free.

For managed serving of the published image, use the **b70ctl TUI**:

1. **Model Packs → Browse Available Packs**: install Flash-Next 1.0.3 and its
   W4A16 target. **Import Local Pack** accepts an extracted pack directory.
   Controller reuses a complete checkpoint and exact image or acquires the
   immutable registry reference and verifies its Docker ID.
2. **Run Model**: select the target/version, four cards, context, Base or MTP3,
   Local or LAN access, and host port. Context belongs here: the profile supplies
   `--max-model-len` as **32768, 65536, 131072 or 262144**, counting input plus output.
3. Start, then use **Runtime → Refresh / View Logs** until `/health` succeeds.
   Start returns during loading. Use **Stop** before switching mode/context;
   exiting the TUI leaves the detached service running. There is no automatic
   startup timeout or rollback.

### Shared configuration and modes

The shared contract uses TP4+EP, four slots, MBT1024, BF16 dtype, XPU graphs,
spawn workers and FLAT hierarchy with affinity `0,1,2,3`. It supplies `/dev/dri`,
read-only `/dev/dri/by-path` and checkpoint mounts, host IPC, 32 GiB shm allowance
and the qualified security options. `B70_PLE_DISK_BF16=1`, the original table
path and shim preload are required. CCL temporary buffers are enabled; HF,
datasets and Transformers use offline mode, and temporary/HF/Triton/vLLM/XDG
caches live under `/work`. Vision allows two images and zero videos per prompt.
Automatic tool choice uses `qwen3_xml` with `qwen3` reasoning and
`--exclude-tools-when-tool-choice-none`. APC and Mamba ALIGN with 832-token pages
remain enabled through inherited defaults.

The exact standalone authorities are [base.json](patches/disk/profiles/base.json)
and [mtp3.json](patches/disk/profiles/mtp3.json). `pack.json` expresses them as
shared runtime, model, mode and context-profile settings; all eight profiles
retain the same mode settings apart from the selected context ceiling.

| Setting | Base | MTP3 |
| --- | --- | --- |
| GPU memory utilization | 0.85 | 0.91 |
| Speculation | Disabled; no drafter | Integrated MTP, 3 tokens, `use_local_argmax_reduction=true` |
| Batch-token budget / sequence slots | 1024 / 4 | 1024 / 4 |
| Scheduled-token policy | Aggregate cap 256 once any running request reaches decode | Fixed 1024; no conditional-cap gate |
| `FULL_AND_PIECEWISE` capture sizes | 1,2,4,8,256,512,1024 | 1,2,4,8,64,128,256,512,832,1024 |

Base selects `B70_CONDITIONAL_PREFILL_TOKEN_CAP=256`: pure prefill can use 1024,
while mixed decode/prefill limits the **aggregate scheduled-token budget** to
256, including decode work. This is not a fixed MBT256 profile.

MTP3 enables the following accepted behavior; these gates are absent in Base:

- `B70_PLE_MTP_PREFILL_1024=1` permits the wider prefill budget/captures.
  `B70_MTP_SKIP_PREFILL_TAIL=1` keeps the first prompt-KV pass and omits its two
  unconsumed later proposals on wholly intermediate prefill.
- `B70_MTP_PREFILL_KV_ONLY=1` writes the one-layer draft's prompt KV while
  retaining embeddings, hidden projection, HC input mixing, QKV normalization
  and RoPE. It skips discarded attention output, MoE, final mixing and proposal
  work. Final prefill, mixed batches, decode and unsupported cases use ordinary
  execution.
- `B70_MTP_DRAFT_PREFILL_EAGER=1` avoids the retained large draft-prefill graph
  registration failure. Target prefill and target/draft FULL decode graphs remain.
  `B70_XPU_POST_CAPTURE_WARMUP=1` performs a startup-only 2560-element BF16
  all-reduce after final capture, checks queue state and fences before readiness;
  it avoids the installed XCCL first-long-request capture-state failure.
- `B70_XPU_SPEC_META_FUSED=1` generates live integer attention metadata on device
  for the eligible uniform small-query speculative branch.
  `B70_XPU_SPEC_SPLIT_PLAN=1` also builds a fresh compact work list on each replay:
  1 split through causal length 1984, 32 through 16384, and 64 above it. Planning
  uses current device lengths rather than capture-time values.
- `B70_XPU_TILED_REDUCER=1` selects the compiled four-way reducer for the eligible
  single-logical-sequence, q4/h6/kv1/d256, BF16, 832-page shape and supported
  strides/options. Other shapes retain native fallback. It covers target
  verification and the target-shaped first draft pass.
  `CCL_SYCL_ALLREDUCE_LL_THRESHOLD=65536` is the selected small-message collective
  setting, retained together with that reducer.

## 7. Qualification, limitations and references

Broad normal-entrypoint qualification on the independently reconstructed
predecessor covered both modes at native context: BF16 mapped PLE/shim, vision,
reasoning, required/named/none tools, APC/ALIGN boundaries, mixed four-slot
serving without preemption, finite sampled logprobs, stopping, disconnect/reuse,
short-after-long behavior and **262112 input + 32 output** cold/APC recall.
MTP3 also demonstrated depth-three acceptance and native reducer mappings.
The notice-only RC preserves executable logic, native binaries and baked config;
focused Controller checks additionally verified both startups, deterministic
completion, MTP3 streaming/stopping and loaded libraries. All eight released
profiles passed Controller rendering and clean installation checks; lower
context ceilings did not each receive an independent serving campaign.

Retained final reconstructed-image FixedStepClock coding requests used thinking,
seed57, temperature0.7/top_p0.95/top_k20 and passed 21 functional checks. Across
65,845/164,149-token prompts with cold/reused APC, observed single-stream rates
were **45.48–50.24 tok/s Base / 108.77–122.21 tok/s MTP3**, n=1 per condition.
These differing outputs and trajectories are descriptive historical observations.
“Cold APC” uses a fresh request cache namespace with natural OS page-cache
history, not cold storage.

[README](README.md#betterbench-performance) now reports the released Disk image's
MTP3 BetterBench v0.6.0 campaign. It measured short-corpus C1/C2/C4 aggregate
output throughput of **111/184/249 tok/s**, with **48/48 successful requests per
level**. See the retained [methodology](../../betterbench-results/2026-10-11-flash-next-disk-mtp3/METHODOLOGY.md)
for workload and timing definitions.

Numerical/reduction-order equivalence is bounded by retained differential
checks; arbitrary generated code and schemas can fail. The historical synthetic
startup audit's nonfinite cause remains unresolved despite subsequent clean
qualification. First-use/cache effects, minimum RAM, storage-cold endurance,
long-prompt concurrency, fairness, sustained high-load throughput and sustained
native-ceiling generation remain unmeasured. TP1/TP2, MTP1 and video are not
supported pack configurations.

Useful authorities and history:

- [Cumulative source and build package](patches/disk/README.md), [patch index](patches/README.md),
  [installed inventory](patches/disk/manifests/accepted.json) and
  [runtime/native lock](patches/disk/manifests/runtime-lock.json).
- [Distributed notices](patches/disk/NOTICE.md) and
  [dated license/source-delivery findings](patches/disk/licenses/README.md).
  Retain vLLM/kernel Apache-2.0, SYCL-TLA BSD and embedded notices, vendor/Torch
  notices and the actual Qwen Community License. Controller MIT does not
  relicense those components. The dated investigation retains unresolved
  source-delivery/recipient-term findings; release completion does not establish
  new clearance or resolve the NumPy-bundled libgfortran source-identity gap.
- [Contributor ledger](patches/disk/provenance/PATCH_PROVENANCE_LEDGER.md) and
  [completed contribution closeout](patches/disk/provenance/contribution-closeout.md).
  No identifiable KaloyanNaumov/Lumnus executable source ships in final Disk.
  Research/configuration influence, including PR #4's prefill direction, remains
  acknowledged. Historical `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
  trailers describe original contributor records, not authorship of new releases.
  Wulf's residency shim is his own work.
- [Disk 1.0.3 release](https://github.com/wu1ff/B70-LLM-Controller/releases/tag/qwen3.8-flash-next-v1.0.3)
  and [completed issue #9 release closeout](https://github.com/wu1ff/B70-LLM-Controller/issues/9#issuecomment-6112407789).
  Pending-release strings in frozen source and preparation records describe their
  preparation dates; `pack.json` and the completed release define current state.
- [Archived prior recipe and complete Vanilla campaign](history/RUNTIME_RECIPE-2026-10-11.md)
  and [Vanilla 1.0.2 source release](https://github.com/wu1ff/B70-LLM-Controller/releases/tag/qwen3.8-flash-next-v1.0.2).
  The former Vanilla GHCR image is retired. Manual rollback requires a saved
  1.0.2 pack and local image; retain both before Controller updates can retire
  older packs/unshared owned images, then Stop, Import Local Pack and select its
  version-labelled profile. There is no catalog downgrade or automatic rollback.

Raw qualification, build, tuning and publication receipts referenced by the
local development documents `DISK_MTP3_MODEL_PACK_INTEGRATION.md` and
`DISK_TUNING_STATUS.md` are retained in the development workspace and are not
bundled public evidence. The summaries above and in README state their scope;
the distributed sources/manifests and public release/issue links are the public
verification references. Historical Vanilla BetterBench and campaign numbers
remain in the archive and do not describe Disk performance.
