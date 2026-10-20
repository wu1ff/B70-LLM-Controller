# Historical release-preparation findings — 2026-10-11

The investigation below is preserved as recorded preparation evidence. Current runtime identity and model-pack distribution directions are in the root README and RUNTIME_RECIPE. This release retains the prepared third-party licenses and notices without reopening this investigation or asserting new licensing clearance.

# Distributed components and source delivery

This source export and the frozen Docker image have different contents. The export includes changed vLLM/kernel source, native source and build scripts, the pinned SYCL-TLA source archive, a derived Qwen serving configuration and retained notices. It contains no checkpoint, BF16 PLE table, accepted native object/library or Intel compiler `.deb`. The image includes the official vendor filesystem and rebuilt native outputs; its inherited executables and shared libraries are redistributed when the image is published.

## Requirements satisfied in this source export

- Apache-2.0: the license copies, original notices and modification history are retained. The 15 changed upstream Python files and six changed native upstream files carry an explicit Wulf modification comment. New standalone files are distinguished from modified upstream files using the existing vendor/source manifests. An external inventory alone would not satisfy the changed-file notice requirement in [section 4(b)](https://www.apache.org/licenses/LICENSE-2.0.html).
- SYCL-TLA BSD-3-Clause and its embedded third-party terms: the pinned archive and original copyright, conditions and disclaimers are retained, including `SYCL-TLA.txt`. It is not relicensed as a whole under Apache.
- Torch/vendor notices: retain the existing notices for the source/build components actually used. The image also retains installed package copyright/license files at `/usr/share/doc/*/copyright`, `/usr/share/common-licenses/` and Python `.dist-info` license directories. A GPL reference in a permissive package's license bundle can describe unused build machinery or an optional alternative; that reference alone does not impose GPL on the package.
- Qwen configuration: retain `external-model.txt`, the actual Qwen Community License 1.0, for the derived `serve-config.json`. Downloaded weights/table remain separately acquired. The license's commercial-service conditions apply if their stated conditions are met; they do not create a general corresponding-source requirement.

No third-party permission or new release co-author trailer is needed for Wulf's residency shim. The verified provenance remains: no identifiable PR #4–#8 implementation incorporated. Original contributor messages and research/configuration acknowledgments remain in `provenance/`.

## Corrected image candidate

The original qualified image `sha256:fad9d4cfae29a4fc3abd70e7339e4f2784c3589c988f7ca3f6751ffb46806316` remains unchanged. The local `qwen38-flashnext:disk-rc` candidate is `sha256:b13fed6dd54051395f603409ada440c74afcbd5331810eee3acd62315d7787e3`. One added layer preserves the 15 required Python modification comments. Removing only each exact comment restores the qualified file bytes. Complete installed comparison, native hashes and image configuration prove equivalent executable logic and native behavior; Python source line numbers advance by one.

The historical notice-only recipe reproduced this correction without compilation. `build.py` preserves runtime Python notices while reconstructing only native compiler inputs inside its temporary compiler context; the distributed native source retains all six notices. `verify.py` checks the exact notice-bearing payload against historical qualified locks. No source, installed-payload or profile qualification lock is repinned.

## Identified Ubuntu corresponding sources are available

The inherited apt configuration has moving `deb` entries for Noble, updates, backports and security; it is not a pinned corresponding-source delivery. Exact source publications were retrieved through Ubuntu's primary archive, and all 30 `.dsc`/source files for these ten identified families were downloaded locally. Each `.dsc` version matches the installed source identity, its SHA256 member list matches all downloaded files, and the source archives retain Ubuntu patches and Debian build/install rules.

| Installed binary/source family | Exact source version |
| --- | --- |
| Bash | `5.2.21-2ubuntu4` |
| coreutils | `9.4-3ubuntu6.3` |
| Git | `1:2.43.0-1ubuntu7.3` |
| apt | `2.8.3` |
| dpkg | `1.22.6ubuntu6.6` |
| GCC 13 compiler tools | `13.3.0-6ubuntu2~24.04.1` |
| GCC 14 runtime libraries, including libgcc/libstdc++ | `14.2.0-4ubuntu2~24.04.1` |
| binutils | `2.42-4ubuntu2.10` |
| FFmpeg | `7:6.1.1-3ubuntu5` |
| GNU C Library, libraries and utilities | `2.39-0ubuntu8.9` |

Use the matching source package, including its distribution patches/rules, rather than a current upstream tag. Immutable-version source records are at `https://launchpad.net/ubuntu/+source/NAME/VERSION`; exact source-file URLs and checksums are recorded with the local copies. The GPL/LGPL copyright texts retained inside the image establish the applicable component licenses, including GCC runtime exceptions and FFmpeg's GPL-enabled Ubuntu build.

For [GPL-3 section 6(d)](https://www.gnu.org/licenses/gpl-3.0.html), free equivalent source access may use a third-party server with clear adjacent directions, but the image distributor remains responsible for availability. [GPL-2 section 3](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html) requires its own corresponding-source route; the online source-access alternative specifies the same designated place. Prepare companion source assets at the release's designated download location instead of relying on a bare link to a moving Ubuntu repository. [LGPL-2.1 sections 4/6](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html) require the library's source and, where relevant, a suitable replacement/relinking mechanism for a work using it. Retain replaceable shared libraries and avoid applying Intel proprietary restrictions to LGPL/GPL components.

No new written offer is issued. A universal three-year retention promise is unnecessary for a chosen online equivalent-access route; that minimum belongs to applicable written-offer routes. Maintain matching sources while the corresponding binaries are offered, and preserve a fallback copy if third-party source hosting disappears. The ten-family check resolves those previously identified source-availability examples; it is not a new certification of every installed package.

## Identified wheel-bundled libraries

The exact upstream PyPI wheels were retrieved and their identified native-library bytes matched the image. Python-SoXR `1.1.0` bundles LGPL libsoxr `0.1.3-14-ga66f3ee`: its matching source archive includes all 138 libsoxr entries, the wrapper and CMake build scripts, and that exact version string. PyZMQ `27.1.0` bundles MPL-2.0 libzmq `4.3.5`: the matching binding source archive and libzmq release source were downloaded, including the wheel's build recipe. Preserve their license files and provide the actual corresponding-source directions beside the image; [MPL-2.0 section 3.2](https://www.mozilla.org/en-US/MPL/2.0/) governs executable-form source availability. No duplicate source download is automatically needed for source-only Python packages already delivered readably inside the image, such as html2text.

One exact source identity remains unresolved: NumPy `2.2.6` ships `numpy.libs/libgfortran-040039e1-0352e75f.so.5.0.0`, byte-matched to its official cp312 manylinux wheel. Its own GPL-3 source obligation remains despite the GCC Runtime Library Exception; the exception avoids imposing GPL on eligible independently compiled applications. NumPy's matching source archive contains no GCC/libgfortran source, and the library lacks a `.comment` section identifying its originating GCC source release. Obtain the exact matching GCC/libgfortran source, distribution patches and build material from the NumPy/OpenBLAS wheel build provenance or a verified upstream source delivery. Do not substitute the image's different Ubuntu GCC 14 source or infer a compiler version from SONAME 5.0.0. No dependency replacement or recompilation is required to resolve this documentation/source-delivery item.

## Intel runtime grant and remaining recipient terms

The five installed `2026.0.0` wheels containing the identified runtimes are `intel-cmplr-lib-rt`, `intel-cmplr-lib-ur`, `intel-opencl-rt`, `intel-openmp` and `intel-sycl-rt`. All 52 recorded native files are explicitly listed in Intel's 2026.0.0 `credist.txt`, extracted from the locked official `intel-oneapi-dpcpp-cpp-2026.0-2026.0.0-947_amd64.deb` (SHA256 `ffa5fc9f0e7f73c3f44254d030296fdc28b589f515923542f7395d9471857528`). Every installed runtime file also matches that compiler material byte-for-byte. Thus the prior missing-designation concern is resolved. The designation header references April 2023; the installed wheels supply the August 2024 EULA. Retain the supplied August 2024 terms, supported by the same release's explicit file list and exact binary match.

The [installed Intel EULA](https://cdrdv2-public.intel.com/777700/intel-end-user-license-agreement-for-developer-tools-version-august-2024.pdf), sections 2.1(D), 2.2, 3.1(viii) and 7.3, permits designated redistributables as part of this product. It requires an executable-code recipient agreement prohibiting reverse engineering/decompilation/disassembly, retention of notices, the specified liability limitations binding upon recipients, and the stated distributor responsibilities. Existing license copies are retained; a stored license file alone does not prove a binding recipient agreement. Before publication, establish that recipient-license arrangement for the Intel proprietary files, with separately licensed third-party code governed by its own terms. No separate permission request, compiler-binary publication or runtime recompilation is indicated by the verified designation.

## Remaining publication work

Prepare maintained companion source access and exact directions for the identified components; complete the NumPy-bundled libgfortran source identity; establish Intel recipient terms; then run the minimum focused serving checks on the corrected candidate during a later authorized maintenance window. Retain all existing upstream licenses/notices. No dependency upgrade/removal/replacement, additional contributor permission or full-stack requalification is implied. The source/compiler evidence and downloaded artifacts are together in the runtime release-preparation directory. No image promotion, registry login, push or public release occurred.

## Current 1.0.4 source candidate

The source candidate contains 19 notice-bearing modified upstream Python files, including model.py and cudagraph_utils.py, plus the retained six native notices. The qualified Qwen tokenizer, chat template, generation and processor assets accompany the Intel-derived serving configuration under external-model.txt. Runtime copies of that model license and the kernel Apache license are included with the frontend. Final source and installed-payload locks supersede the historical acceptance hashes; historical identity receipts remain in provenance. This update does not resolve the inherited source-delivery, bundled libgfortran or Intel recipient-term findings above.
