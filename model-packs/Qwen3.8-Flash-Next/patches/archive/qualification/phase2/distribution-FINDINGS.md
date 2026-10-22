# Accepted-version source and distribution findings

Reviewed for the accepted 0.32 vendor stack on 2026-10-22. This is a source
reconstruction candidate. Original source bytes, SPDX/copyright text, Intel/BSD
native headers, frontend/model notices and previous attribution records are
preserved. The Controller's MIT license does not relicense these components.

`historical-1.0.4/` is explicitly historical, including its Torch 2.13 and Intel
2026.0 records. `accepted-vendor/` contains notices extracted from the accepted
image's installed distributions. `compiler-input/` contains only readable notices
from the pinned OMIX image, including its current 2026.1.1 `credist.txt`.
Compiler executables, native binaries and checkpoints are excluded.

The current stack is Torch 2.14.0+xpu, vLLM 0.32.0+xpu, XPU kernels 0.1.15.4,
Triton XPU 3.8.0 and Intel runtime wheels 2026.1.1. The full installed distribution
list and dpkg records accompany production manifests. The stock XPU wheel's source
commit remains unattested. The retained Intel reference sources and two exact
header trees support the custom native build; they are not an attestation of the
stock wheel's complete corresponding source.

`source-delivery/ubuntu-reconciliation.json` compares the accepted dpkg source
versions with the previously retrieved ten source families. All ten match,
including their retained source-file SHA256 checks. The exact URLs, versions,
distribution patches and build-rule archive references are retained. Large Ubuntu
source assets remain external companion delivery material and are not necessary
for reconstructing this custom overlay atop the immutable vendor image. A future
image distributor still needs maintained matching source access and clear adjacent
directions. This limited check does not clear every inherited executable.

The accepted NumPy 2.2.6 libgfortran, SoXR 1.1.0 extension and PyZMQ 27.1.0 libzmq
match the historical wheel-evidence hashes byte for byte. The matching SoXR source
includes libsoxr 0.1.3-14-ga66f3ee; PyZMQ's source/build evidence identifies libzmq
4.3.5. Their locked source URLs and build evidence are preserved in
`source-delivery/`. NumPy's bundled libgfortran exact GCC source/build identity is
still unknown; NumPy's sdist and the image's Ubuntu GCC source do not settle it.
Source availability and delivery are distinct checks. The relevant upstream terms
remain [GPL](https://www.gnu.org/licenses/gpl-3.0.html),
[LGPL](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html) and
[MPL 2.0](https://www.mozilla.org/en-US/MPL/2.0/).

Current Intel August 2024 EULAs (including their original text encodings), installed
third-party notices and the compiler's 2026.1.1 designation list are now retained.
Within the inspected wheel ELF/library scope, 19 of 59 paths have a basename in
that list; this is only a coverage screen, not a binary/designation attestation.
The remaining SYCL/UR entries need release-specific designation evidence. The
supplied Intel terms and their recipient arrangement must be reconciled for actual
executable distribution. Historical 2026.0 findings are not carried forward as
clearance. See the [Intel Developer Tools agreement](https://cdrdv2-public.intel.com/777700/intel-end-user-license-agreement-for-developer-tools-version-august-2024.pdf)
and the actual installed notice copies; no general relicensing of separately
licensed third-party code is asserted.

Fifteen of sixteen modified upstream vLLM files lack an explicit modification
header in the inspected first twenty lines. Exact paths/hashes are in
`source-delivery/modification-notice-review.json`. The GDN corrected headers and
kernel Python wrapper also need a final changed-file notice review. Preserve the
accepted source as an immutable comparison target; any future notice-only export
revision must have separate hashes. An external provenance ledger alone does not
settle the [Apache 2.0 changed-file notice requirement](https://www.apache.org/licenses/LICENSE-2.0).
Original upstream notices are retained, and no new personal author is invented.

SYCL-TLA is pinned at `87f6850680a580654b9ea2c80dbc01aeb36ad231`; its complete
5,703-file archive includes embedded third-party notices. Preserve its BSD and
per-file terms. Preserve Qwen Community License 1.0 for the frontend/configuration,
and incoai's pinned Apache-2.0 drafter model-card attribution. Model weights are
acquired independently at the pinned revisions.

`UNRESOLVED.json` is the machine-readable list. A passing offline reconstruction
does not clear distribution or prove independent native/image reproduction.
