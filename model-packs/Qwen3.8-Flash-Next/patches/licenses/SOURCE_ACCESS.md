# Source access and companion delivery directions

The complete model-pack source archive contains `patches/**`. Controller import
stores only `pack.json` and README, so retain the original archive separately.
No DFlash2 public archive URL or registry digest exists yet. Do not use historical
1.0.4 sources to reproduce DFlash2. Its published identity is retained only in
`../archive/published-1.0.4.json`.

Included: clean vLLM v0.32 package source/license, exact 16-patch replay, final
comparison overlay, all twelve custom native families and their recipes, the two
header/reference roots, complete pinned SYCL-TLA source archive, baked frontend,
shim, helpers/tests, complete campaign originals and retained notices. Start at
`../production/README.md`; source verification is offline. The original 97 patch
paths and source variants remain retrievable through the indices in `../INDEX.md`.

For a changed-file-notice source export, first verify/reconstruct production, then:

```sh
python3 patches/production/tools/notice_export.py \
  --source /tmp/dflash2-clean/reconstructed --output /tmp/dflash2-notice-source
```

The new directory includes source/dependencies/support/notices and a complete
`NOTICE_EXPORT.json` hash inventory. It is explicitly a source-only derivative;
accepted hashes and all historical patches remain unchanged. Publication requires
a final review of notices and eventual executable incorporation.

External immutable inputs are vendor/compiler OCI digests in
`../production/manifests/immutable-inputs.json` and target/draft revisions in
`pack.json`. Obtain these under their own terms. Checkpoint/compiler executables
are not redistributed in this source archive.

Companion-source records are in `source-delivery/ubuntu-reconciliation.json` and
`wheel-reconciliation.json`: exact source versions, source archive URLs, hashes,
distribution patches and build evidence. For each referenced asset, retrieve the
recorded URL, verify its SHA256, retain distribution patches and build rules, and
publish maintained matching source-access directions adjacent to any later image.
`source-access-index.json` collects the preserved references; an upstream URL is
not a maintainer-hosted source offer or proof of complete corresponding source.
Large companion archives are retained in the local campaign supporting-source
collection, not embedded here. No workstation path is presented as public access.

NumPy's bundled libgfortran source/build identity, stock kernel-wheel attestation,
and Intel designation/recipient terms remain unresolved. Do not substitute a
nearby source version. [Current findings](SOURCE_DISTRIBUTION.md) explain the
remaining delivery/distribution decisions. There is no written source offer here.
