# Source access and companion delivery directions

Retain the complete model-pack archive: Controller import keeps only pack.json
and README (plus local source metadata), not patches/**. No public DFlash2 source
URL or registry digest has been assigned. The published 1.0.4 archive is historical
and cannot reproduce DFlash2.

## Included accepted and annotated sources

From the unpacked model-pack directory:

```sh
python3 patches/production/tools/verify.py \
  --staging /tmp/dflash2-accepted --report /tmp/dflash2-accepted.json
python3 patches/production/tools/verify_notice_export.py \
  --source /tmp/dflash2-accepted/reconstructed --report /tmp/dflash2-notices.json
```

The first command replays the exact 16 production patches, checks all accepted
Python/package/support hashes, twelve native families, two header roots and the
complete 5,703-file SYCL-TLA source tree, and verifies archival indices. It uses
immutable original source identities. Archive experiments are never replay inputs.

`../production/distribution-source.tar.gz` is the materialized annotated source
export. Verify its identity in `../production/manifests/source-export.json`, then
extract into a separate empty directory. Its NOTICE_EXPORT.json inventories every
file and maps all 57 accepted source hashes to separately hashed comment-only
copies. Original copyright, SPDX, attribution and earlier modification text remain
byte for byte after removing exactly the recorded insertion. All other export
bytes match the accepted/retained inputs. This is source delivery, not a new runtime.
Use the enclosing production/ recipes and manifests for accepted reconstruction;
the annotated tree is not a replacement for the accepted input-lock or image.

To reproduce the annotated export, first run accepted verification above, then:

```sh
python3 patches/production/tools/notice_export.py \
  --source /tmp/dflash2-accepted/reconstructed \
  --output /tmp/dflash2-annotated \
  --archive /tmp/dflash2-distribution-source.tar.gz
```

All original 97 patch paths, 113 native/build originals and 120 Python-variant paths
remain in the enclosing archive with their exact identities and dispositions.
Historical originals retain their provenance purpose; their changed-file notice
treatment is explicitly pending legal/release review, rather than silently deemed
compliant because production source notices are present.

## Companion-source candidate and delivery decision

A separate local source-only candidate, `dflash2-companion-source-candidate.tar.gz`, is
staged with SHA256 `abf44d9258366534cd3187ae012880b91885893a9e9ae902035c70ed95b92335`
(300960581 bytes). It contains 34 source assets: 30 Ubuntu files
from ten exact matched source families, NumPy/SoXR/PyZMQ sdists and libzmq 4.3.5.
`source-delivery/companion-delivery.json` lists every path, URL and SHA256. It is not
embedded in this Controller archive, and has no public URL or maintenance owner.

For a retrieved companion candidate, verify its SHA256 and internal SHA256SUMS.
For each Ubuntu family retain its .dsc and every checksummed member together and
run `dpkg-source -x <family>.dsc`; Debian/Ubuntu build rules and patches are inside
the native or Debian source archive. SoXR 1.1.0 sdist includes libsoxr
0.1.3-14-ga66f3ee; PyZMQ 27.1.0 build records identify the separately included
libzmq 4.3.5 source. Preserve their license files and build directions.

Before distributing an image, assign maintained adjacent source access, a named
owner and image/digest-to-source mapping. Verify retrieval without workstation
paths, credentials or paid source access. GPLv3 section 6(d) permits equivalent
source access at the same place or another server with clear adjacent directions;
do not select or promise that route until the complete scope and hosting exist.
LGPL-2.1 section 6 obligations can include source and replacement/relinking rights;
MPL-2.0 section 3.2 requires recipients to be told how to obtain covered source.
Legal must choose applicable component-specific delivery routes and exception
interpretations. No written source offer is made here.

`source-access-index.json` distinguishes source assets, binary provenance records
and upstream references. URLs/hashes are retrieval evidence, not a maintainer-hosted
offer. Exact NumPy libgfortran GCC source/build identity and stock kernel-wheel
build attestation remain open. Ten Ubuntu families are a limited reconciliation,
not a complete corresponding-source audit of every inherited executable.

Vendor/compiler OCI digests remain in `../production/manifests/immutable-inputs.json`;
checkpoint repositories/revisions remain in the enclosing pack.json. Obtain them
under their own terms. Compiler executables, native libraries and weights are not
redistributed in these source archives. [Distribution decisions](SOURCE_DISTRIBUTION.md)
include the precise Intel designation/recipient limits.

References: [GPLv3 section 6](https://www.gnu.org/licenses/gpl-3.0.html),
[LGPL-2.1 section 6](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html),
[MPL-2.0 section 3](https://www.mozilla.org/en-US/MPL/2.0/).
