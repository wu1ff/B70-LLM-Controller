# Flash-Next Disk source package

Pack 1.0.4 builds the Intel AutoRound dynamic MTP5 runtime from the pinned official vendor image and canonical sources. The only qualified serving contract is BF16 TP4+EP, four slots, 262144 context, with five verification tokens for one scheduled request and three for batches of two through four. The root pack README and recipe record final qualification and measurements.

The source export contains cumulative Python overlays, complete native sources and guarded GDN/MoE dispatchers, Wulf's residency shim, build recipes, SYCL-TLA source, exact input/payload locks and licenses. It contains no checkpoint, experimental/prebuilt native outputs or compiler packages. Historical diffs under `patches/` must not be reapplied.

Obtain the 15 exact compiler packages in `manifests/compiler-lock.json` from its official Intel pool or an authorized cache. With the pinned vendor image available locally:

```bash
sha256sum -c SHA256SUMS
python3 build.py --compiler-dir /absolute/compiler-cache --tag qwen38-flashnext:independent-build --evidence /absolute/build-evidence
python3 verify.py qwen38-flashnext:independent-build --output /absolute/build-evidence/recheck.json
```

Builds use the locked vendor image in both stages, oneAPI 2026.0.0-947, vendor Python 3.12/Torch 2.13.0+xpu, GCC 13.3 and locked SYCL-TLA. The wrapper verifies every shipped file and compiler package before an offline, uncached BuildKit build. The fixed source-date epoch, timestamp-normalized single-platform export and explicit deferred unpack make the image reproducible. Docker must use its containerd image store. It verifies the exact final installed inventory afterward. Only native notice prefixes are stripped inside the disposable compiler context to preserve the established native compiler inputs; all distributed sources and runtime Python notices remain intact. Native compiler bytes have a separate explicit lock. Compiler packages stay outside the final stage.

`manifests/accepted.json` is the final installed-file authority; `overlay.json`, `native-source.json` and `native-compiler.json` lock final inputs. Runtime/checkpoint and verified registry identities are in `runtime-lock.json` and `ghcr-release.json`. Earlier publication and implementation provenance remains historical evidence.

The entrypoint prepares a dense serving view under writable `/work` from Intel revision `4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897`, preserving all 2340 AutoRound exclusions and persistent hash buffers. It filters only the 128 PLE weight-index entries. The validated BF16 payload maps directly from Intel's PLE safetensor; no conversion, second table or old checkpoint is required. The eight packaged, qualified Qwen frontend files preserve the established tokenizer, developer/system, reasoning, tool and image behavior.

Use root `pack.json` with Controller for managed installation and serving. `profiles/mtp5-dynamic.json` is the exact standalone contract; `launch.py mtp5-dynamic --model-dir /absolute/intel-snapshot --image <final-image> --port 8000` renders recovery argv. Add `--execute` to start it after freeing the GPUs and port.

See [NOTICE.md](NOTICE.md) and [licenses/README.md](licenses/README.md). Retained inherited corresponding-source delivery, bundled libgfortran identity and Intel recipient-term findings remain outstanding; publication does not assert their completion.
