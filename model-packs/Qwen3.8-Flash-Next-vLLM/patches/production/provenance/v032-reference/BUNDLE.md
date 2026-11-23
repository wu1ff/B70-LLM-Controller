# DFlash2 portable source reconstruction

Accepted runtime: `qwen38-flashnext-dflash2:integrated-serving-v1`, local image ID
`sha256:6e2ab4e686680a5208ff0119c4637341cba81942bd39b6576ea14dc8f547b83b`.
This is a source reconstruction candidate. Independent native binary and full-image
reproducibility have not been proved. Performance tuning is closed.

Everything to carry into the future model-pack archive lives beneath `patches/`,
because that archive includes regular `patches/**` files. Preserve the entire tree:

- `production/`: clean source inputs, exact 16-patch chain, accepted overlay for
  comparison, selected native sources, two shared header trees, one pinned TLA
  archive, frontend, shim, immutable references, recipes, manifests and verifier.
- `archive/`: original campaign sources and patches, rejected/superseded variants,
  historical locks and reconstruction evidence. Never apply archive patches to the
  production result. `patch-index.json` indexes all 97 original patch/diff paths;
  `python-variants.json` indexes historical Python variants by content hash.
- `licenses/`: original notices, accepted-version vendor notices, compiler-input
  notices, source-delivery evidence and explicit unresolved findings.

From any directory, run the bundled verifier with Python 3.12+, Git and Bash:

```sh
python3 /path/to/bundle/patches/production/tools/verify.py \
  --staging /tmp/dflash2-source-clean \
  --report /tmp/dflash2-source-verification.json
```

The staging path must not exist. This works offline without Docker, model weights,
the original workspace, previously patched files, a compiler or GPUs. The verifier
checks the complete bundle inventory, extracts clean pinned package sources,
applies each original incremental patch, supplies explicit vendor-generated version
metadata, compares every installed Python source hash and its file set, and checks
all selected native/build inputs, both header trees, TLA and supporting files.
`accepted-overlay/` is a comparison target, never an input to patch replay.

Only `astra500-dominant.patch` uses `git apply -p3`; the other fifteen use `-p1`.
The original 16 patches are immutable; `manifests/patch-series.json` defines their
order. Do not reapply the chain over the final overlay.

The native environment is reconstructed from the two image digests in
`manifests/immutable-inputs.json`. The compiler image supplies oneAPI 2026.1.1;
the vendor image supplies Torch 2.14.0+xpu headers/libraries, Python 3.12 headers,
GCC/binutils, Level Zero headers/loader and the unchanged GDN delta-rule library.
No apt or pip resolution and no local campaign image tag is needed. Each native
family builds separately at the original `/build` path using its unmodified shell
recipe. The GDN builder uses its own corrected int64 conv headers; the other ten
SYCL families share the common tree/TLA. The residency shim uses vendor GCC.

Inspect the exact native order, translation units, flags and output renames:

```sh
python3 /path/to/bundle/patches/production/tools/build_native.py --plan
```

For a later authorized native compilation, preload both pinned OCI images and
provide Docker BuildKit. This command builds no serving image and starts no server:

```sh
python3 /path/to/bundle/patches/production/tools/build_native.py \
  --output /tmp/dflash2-independent-native
```

The native recipe's order is GDN → residency shim → MoE → HC v3 → direct fallback
→ ownership groups → batch → beyond fallback → dominant layout → short prefill
→ long HC → long prefill v3. These libraries are independent of one another;
all rely on the pinned vendor environment. `Dockerfile.native` first materializes
the compiler/common includes, then builds the twelve isolated stages and exports
the installed names. It preserves both the HC v3-to-HC rename and all fallbacks.
The build wrapper compares every result against the accepted library hash and
returns failure for any difference. A matching source inventory alone does not
predict that outcome. Compiler nondeterminism and toolchain environment differences
remain possible until this test runs.

For subsequent image assembly, install the verified source delta and all twelve
native outputs over the pinned vendor image, add `support/b70-flashnext/` at
`/opt/b70-flashnext/`, and install the shim under `/opt/b70-residency-shim/`.
Keep `/opt/b70-flashnext/prepare-serve.sh` as entrypoint. The complete accepted
test/helper sources are retained under `image-support/` at their original absolute
path suffixes; its 110 files are also hash-checked. Preserve the entrypoint's
executable mode when assembling an image from an archive.
The complete accepted
argv, environment, mounts/devices and image configuration are frozen in
`manifests/serving-contract.json` and `accepted-image.json`. Paths `/models`,
`/draft` and writable `/work` are part of that contract. Model revisions and
inventories are immutable references, not bundled checkpoints.

A full-image recipe is a separate next phase: reconcile the inherited test/PYTHONPATH
layers and configuration rather than silently omitting them or claiming the same
OCI digest. The smallest useful next action is the twelve-family native build
and binary comparison above, followed by an independently tagged image assembly.

The legacy workspace input lock and Phase 4 rebuild launcher are archived unchanged.
`archive/historical-build/lock-reconciliation.json` explains all ten stale hashes;
the active lock tracks the portable integrated bundle, not mutable old worktrees.
Distribution blockers are in `../licenses/FINDINGS.md` and `UNRESOLVED.json`.
No publication, new release identity, inference qualification or legal clearance is
implied by this bundle.
