# Selected v6 source reconstruction

Run the offline verifier from any directory, using a new staging directory:

```sh
python3 /path/to/v033-v6-source/patches/production/tools/verify.py \
  --staging /tmp/dflash033-source-clean \
  --report /tmp/dflash033-source-verification.json
```

Python 3.12+, Git and Bash suffice. No Docker, network, compiler, GPU, model
checkpoint or original campaign path is needed. The verifier checks the package
lock, extracts pinned upstream vLLM and the immutable vendor kernel Python
baseline, supplies vendor-generated version metadata, checks and applies all ten
patches in order, and compares the complete resulting 3,679-file source set.
It checks the exact calibration input, all Python syntax, the 17 native source
families and recipe syntax, and the selected STANDARD/EOS/profile contract.

`manifests/patch-series.json` is the application order; every patch uses `-p1`.
These are consolidated installed-source diffs derived from the immutable image
generations. Original builders, manifests, Dockerfiles, build records and exact
SLSA/OCI attachments remain under `provenance/`. The historical v0.32 16-patch
series is reference evidence and is not applied to v0.33. `accepted-overlay/`
contains the final 44 changed source files for comparison; the verifier never
uses that directory as a patch replay input.

For later native reconstruction, inspect the plan first:

```sh
python3 tools/build_native.py --plan
```

Use the exact vendor image and either the verified retained toolchain or all 15
separately obtained compiler packages from `manifests/compiler-lock.json`:

```sh
python3 tools/build_native.py --output /tmp/dflash033-native \
  --compiler-debs /absolute/path/to/locked-compiler-debs
# Alternatively: --toolchain /absolute/path/to/verified/extracted/toolchain
```

The wrapper verifies the toolchain tree, extracts the packaged pinned TLA,
uses the corrected GDN header tree for GDN and the common tree otherwise, and
runs the original v0.33 recipes in isolated CPU containers over the exact vendor
image. No devices or network are attached. Each result must match the installed
library SHA256. A differing build fails and is retained for inspection. The
PLE gather recipe is a documented reconstruction whose output was independently
byte-matched in this task; the original GCC invocation was unavailable.

Stage a new runtime context only after source and native verification:

```sh
python3 tools/prepare_image.py \
  --source /tmp/dflash033-source-clean/reconstructed \
  --native /tmp/dflash033-native --output /tmp/dflash033-image-context
```

This stages final installed deltas, all 17 native outputs, exact preparation and
frontend assets, source mirror deltas, and frozen baked image configuration. It
does not run Docker. The emitted Dockerfile starts from the exact official vendor
binary image. A future image build uses a new tag and has its own identity:

```sh
docker build --pull=false --network=none --tag local/dflash033-source:review \
  /tmp/dflash033-image-context
```

No such image build was performed here. This flattened reassembly is a source
build procedure over the vendor image; it does not recreate the historical
layer/attestation/config identity or demonstrate a bit-identical image.

The final serving profile remains a separate, unchanged file:
`runtime/profile.json`. Its diagnostic worker modules, exact launcher and full
Docker contract are included. The profile contains original workspace paths;
moving those bind paths is a future packaging change. The selected image alone
does not activate every v6 feature: preserve the profile environment and command.
No server should be started as part of source verification.
