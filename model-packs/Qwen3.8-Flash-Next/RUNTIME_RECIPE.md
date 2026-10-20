# Build and serve Flash-Next Disk 1.0.4

The Intel AutoRound image is **`ghcr.io/wu1ff/qwen38-flashnext-b70@sha256:dc25c8cd205a828ad4df3ae9aa3e9505f0a51550010ac9a115336473ca268086`**, a single linux/amd64 OCI manifest published under tag `disk`. The corrected source was rebuilt offline and uncached; all 3130 installed payload hashes matched its accepted inventory. Only two Python files differ from the preserved RC; all four rebuilt native binaries match that RC’s earlier independent builds. Anonymous remote manifest/config/layer checks and a clean Controller pull verified the exact Docker identity and all installed payloads. GHCR contains one version with the sole tag `disk`.

## Inputs and build

Use the complete [Disk source package](patches/disk/README.md), Linux amd64, Python 3.11+, Docker BuildKit with the containerd image store, and the exact vendor dependency:

```bash
docker pull --platform linux/amd64 vllm/vllm-openai-xpu@sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9
```

Obtain the 15 exact oneAPI 2026.0.0-947 packages from the official Intel apt pool or an authorized cache, using filenames/SHA256s in `manifests/compiler-lock.json`. They are build tools, supplied separately; an arbitrary installed compiler does not satisfy the locks. The package includes kernel sources based on commit `6d92b1bfbf32767ecda8e819613eb151e70030ad` and SYCL-TLA `87f6850680a580654b9ea2c80dbc01aeb36ad231`. No accepted/prebuilt native binary, compiler package or checkpoint is included.

From `patches/disk`:

```bash
sha256sum -c SHA256SUMS
python3 build.py --compiler-dir /absolute/compiler-cache --tag qwen38-flashnext:independent-test --evidence /absolute/build-evidence
python3 verify.py qwen38-flashnext:independent-test --output /absolute/build-evidence/recheck.json
```

Keep evidence and compiler binaries outside the source tree; choose an unused tag. The wrapper checks every shipped file, native/compiler lock and compiler package before a network-disabled, uncached build. Both stages inherit the pinned vendor image. Compilation needs no GPU, model, host compiler, pip resolution or network.

The native stage extracts oneAPI to `/toolchain/oneapi`, uses vendor Python 3.12/Torch 2.13.0+xpu and GCC 13.3, checks the MoE inputs, then rebuilds GDN index64, guarded MoE remapping, tiled attention reduction and the residency shim. Attention retains the canonical SYCL-TLA include path. All distributed native/Python modification notices remain; only the native notice prefixes are removed inside the disposable compiler context against explicit compiler-input locks. Historical diffs are already applied.

The runtime stage receives only final overlays and newly compiled native outputs. Checked-hash bytecode, fixed `SOURCE_DATE_EPOCH=1792512000`, timestamp-normalized single-platform export and `unpack=false` make the final image deterministic; Docker unpacks on first use. The verifier checks the complete final inventory, including all four native binaries and 19 notice-bearing Python files. [Accepted payload](patches/disk/manifests/accepted.json), [source locks](patches/disk/manifests/native-source.json), [compiler inputs](patches/disk/manifests/native-compiler.json) and [runtime lock](patches/disk/manifests/runtime-lock.json) are authoritative. Historical 1.0.3 locks are retained in provenance.

## Intel-only preparation

Checkpoint `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` is pinned to `4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897`: 29 actual files / 181240433939 bytes. The inventory follows the upstream index, including its nonconsecutive shard filenames and `model_extra_tensors.safetensors`; never infer a numeric shard glob. Controller’s normal downloader is sufficient.

At each boot, the entrypoint validates the original config against the qualified dense configuration, removes exactly five indexer configuration keys, preserves all 2340 AutoRound exclusions and filters exactly 128 PLE entries from the serving weight index. It links the other weight shards into writable `/work/flashnext-serve`, retaining vision, MTP and persistent hash buffers; selected loaders skip unused indexer weights.

PLE maps directly from `model-00016-of-00017.safetensors`. Validation checks the exact header/file sizes, all 128 numeric shard names, BF16 dtype, `[2500012,160]` shapes and contiguous byte offsets. The mapping begins at byte 20736 and exposes `[320001536,160]`, 102400491520 payload bytes. Its owner remains alive and the existing protection check makes the private mapping read-only before row gathering. Each TP rank retains bounded 5 MiB pinned staging; no second 102 GB table, `.pt` conversion, old checkpoint or extra mount is required.

The eight packaged qualified Qwen frontend files supply the tokenizer/chat template, generation and image processor behavior. Their hashes and retained Qwen license are recorded in [frontend.json](patches/disk/manifests/frontend.json). The raw Intel snapshot remains read-only; only preparation/caches use `/work`.

## Serving contract

The sole profile is `w4a16-mtp5-dynamic-tp4-262144`: four B70s, TP4+EP, BF16, four slots, MBT1024, memory utilization 0.91, 262144 total context, APC/ALIGN with 832-token pages, two images/no video, `qwen3_xml` tools and `qwen3` reasoning. The integrated proposer permits five drafts; verification uses five tokens for one scheduled request and three for batches two through four. Client connection count alone does not determine the schedule. Uniform GDN metadata derives its token extent from the actual verification length while gathering all state slots reserved for the maximum depth. Mixed/nonuniform/prefill batches retain the fallback.

Captures are `[1,2,4,6,8,64,128,256,512,832,1024]` under `FULL_AND_PIECEWISE`. The exact environments, device/security/IPC settings, shim, prefill gates, collective buffers and speculative JSON are in [the profile receipt](patches/disk/profiles/mtp5-dynamic.json). No research or runtime override mounts are part of the contract.

Use Controller installation described in [README.md](README.md). For standalone review/recovery, first free the GPUs and endpoint, then render or execute from `patches/disk`:

```bash
python3 launch.py mtp5-dynamic --model-dir /absolute/intel-snapshot --image ghcr.io/wu1ff/qwen38-flashnext-b70@sha256:dc25c8cd205a828ad4df3ae9aa3e9505f0a51550010ac9a115336473ca268086 --port 8000
# Add --execute to start the rendered command.
```

The installed identity must equal the manifest above on Docker’s supported containerd store. A classic image store can expose a config identity instead and fail Controller’s immutable identity check; do not repin around that mismatch.

## Qualification and measurement boundaries

The corrected image passed 150 metadata cases across three distinct cache groups for both staging paths, with every field equal to the installed fallback; guard tests cover 30 uniform cases and eight rejected layouts. Serving qualification repeated native 262112+32 cold/APC, six ALIGN boundary lengths, short-after-long fingerprints, two 164149-token coding/tool runs, concurrent coding/tools, vision/two images, reasoning/system/developer behavior, C1/C2/C3/C4 schedule counters and graceful restart with unchanged configuration. Varying concurrent output lengths also passed finite generated/top-5 logprob checks. Native reducer/mapping qualification is inherited through identical native/PLE payload hashes from the preserved RC; the publication check used a clean pack/image store and reused the exact previously qualified Intel snapshot.

The [README](README.md) reports BetterBench v0.6.0 / corpus 1.0 results: **120.38 weighted tok/s**, C1/C2/C4 aggregate **116.77/180.36/242.62 tok/s**. Resource observations are explicitly labeled as historical. Source commit `d00ad5ec8098c06584a88ec3468bacd37d5ed098`, all 53 tool/corpus files, workload definitions, sampling, timing, warmups and request counts match the historical campaign. Only the verified metadata correction distinguishes the serving implementation; no substitute coding fixtures enter those results. Performance does not establish sustained generation at the native ceiling, arbitrary output quality or untested configurations.

Retain [NOTICE.md](patches/disk/NOTICE.md), model/vendor/kernel/SYCL-TLA licenses and the existing [source-delivery findings](patches/disk/licenses/README.md). Publication does not resolve the inherited corresponding-source delivery, bundled libgfortran identity or Intel recipient-term findings.
