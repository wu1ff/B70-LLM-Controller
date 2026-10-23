# Production source reconstruction

Entry point: `python3 tools/verify.py --staging /tmp/dflash2-clean --report /tmp/dflash2-source.json`.
Use a new staging directory. Python 3.12+, Git and Bash suffice; no network or GPU.
The verifier replays the exact 16 original patches and compares all accepted source,
frontend, helper, dependency and native input hashes. `accepted-overlay/` is comparison
material, never a replay input. `manifests/patch-series.json` fixes order and strip
levels. `archive/` is never applied. Keep both native fallback libraries.

This directory starts from the verified Phase 2 portable archive. Its original
`BUNDLE.md`, native inventory and input lock describe that phase's scope; their
then-unproved build statements are superseded by `../archive/qualification/phase3/`
and the Phase 4 live receipts. All twelve custom native outputs matched independently;
OCI identity differs. The accepted runtime remains frozen.

`tools/build_native.py --plan` lists native recipes. For later source builds it can
compile/export the twelve libraries using the two pinned OCI inputs. A portable
`tools/prepare_image.py` stages replayed sources and independently built libraries
for `build/Dockerfile.runtime`, preserving frontend, shim, baked tests, modes and
accepted PYTHONPATH. It does not run Docker. Existing Phase 3/4 evidence is used
for this pack validation; no inference image is rebuilt here.

`manifests/controller-staging.json` records the accepted local image, release version,
verified immutable public registry reference and Controller transport/storage
differences. `manifests/qualified-launch.json` records the mount-free launch
verified by installed b70ctl 1.0.1; `serving-contract.json` retains the original
standalone capture. The accepted runtime is published unchanged.

The reviewed 57-file notice overlay is incorporated in `distribution-source.tar.gz`.
`manifests/source-export.json` pins this separately hashed source artifact;
`tools/verify_notice_export.py --source /tmp/dflash2-clean/reconstructed` verifies
every exported byte against accepted inputs and the recorded comment insertions.
`tools/notice_export.py` can regenerate it. This does not alter accepted
reconstruction or qualify a new executable image. Keep the full enclosing pack
for original recipes, patches, exact accepted input locks and archival evidence.

For later source builds, the complete assembly sequence from the model-pack root
is shown below. Native/image compilation is unnecessary for Phase 5 validation;
the retained independent build/live receipts already supply that evidence.

```sh
python3 patches/production/tools/build_native.py --output /tmp/dflash2-native
python3 patches/production/tools/prepare_image.py \
  --source /tmp/dflash2-clean/reconstructed --native /tmp/dflash2-native \
  --output /tmp/dflash2-image-context
docker build --no-cache --platform=linux/amd64 --pull=false --network=none \
  --provenance=false --tag qwen38-flashnext-dflash2:source-rebuild \
  /tmp/dflash2-image-context
```

The native output and image context directories must be new. The assembler checks
all 3454 payload files against the retained Phase 3 hashes before writing a context,
and preserves installed modes. `manifests/image-payload.json` is the exact inventory;
the Dockerfile deliberately regenerates runtime bytecode rather than importing
compiled campaign artifacts. A rebuilt image has its own identity.
