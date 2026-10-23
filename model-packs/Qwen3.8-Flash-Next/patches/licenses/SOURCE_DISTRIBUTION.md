# Distribution status — publication blocked

2026-10-22, America/Phoenix. Source-export work is complete for the 57 reviewed
production derivatives. The accepted runtime is frozen. This report establishes
technical findings and release decisions; it grants no distribution clearance.

| Item | Verified facts | Outstanding evidence / Wulf or legal decision |
| --- | --- | --- |
| Production source notices | 31 Python, 23 native and 3 GDN header derivatives add only recorded comments. All 15 previously missing upstream vLLM notices are covered; the existing model_state.py notice and every original byte/notice are retained. Materialized export and accepted-to-annotated mapping are in `../production/manifests/source-export.json` and its archive. | Eventual executable delivery must receive applicable notices without describing it as the frozen qualified image. Legal must settle archival-original treatment; `source-delivery/archival-notice-review.json` screens 241 indexed records, 194 without explicit modification wording in the first 24 lines. This is a heuristic, not a compliance verdict. |
| Stock XPU wheel | PyPI wheel `vllm_xpu_kernels-0.1.15.4-cp38-abi3-manylinux_2_28_x86_64.whl`, SHA256 `4ec262f7afdd07c62defc8286dc9f357569627b996c09b410f7972b653696fe7`; all 12 accepted stock native libraries match. Installed METADATA/WHEEL also match. | Exact source/submodule/build attestation remains missing. No sdist or 0.1.15.4 tag was found in inspected sources; per-file PyPI provenance endpoint returned 404. `v0.1.15` at `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45` is only a candidate. Apache-2.0 metadata itself does not require complete source delivery; separately licensed embedded dependencies need their own review. |
| Companion delivery | 34 source assets and ten Ubuntu .dsc member sets pass exact hash checks; a separate local source-only companion archive is staged. SoXR sdist includes libsoxr; PyZMQ/libzmq 4.3.5 source and build records are retained. | Choose actual maintained source hosting, image/digest-to-source mapping, retention and maintenance owner. Complete inherited-image source/license scope review; ten families do not cover every inherited executable. No public URL or written source offer exists. |
| NumPy libgfortran | Accepted NumPy native hash is locked. NumPy v2.2.6 requirements pin scipy-openblas64 0.3.29.0.0; its libgfortran shares build ID `5bbe74eb6855e0a2c043c0bec2f484bf3e9f14c0` and 23 ELF sections. Only .dynamic, .dynstr and .gnu.version_r differ. | Exact GCC source/build patches and immutable manylinux build identity remain unknown. Neither NumPy sdist nor Ubuntu GCC source proves them. Legal must assess GPLv3 / GCC Runtime Library Exception duties for this separately bundled runtime. |
| Intel 2026.1.1 | All 55 inspected EULA-governed runtime paths match explicitly designated files in the pinned compiler image byte for byte. Four additional PTI paths carry MIT notices. This resolves the old 19/59 installed-list screen for this scope. | Implement compliant recipient terms or obtain alternate Intel authorization; no agreement/permission is evidenced. Resolve the April-2023 credist-preamble / August-2024 installed-EULA discrepancy with legal/vendor review. No compiler redistribution or exhaustive image clearance is inferred. |
| Independent build and runtime | Prior Phase 3 custom-native byte matches and Phase 4 live evidence are retained unchanged. Accepted source replay, image payload hashes and archive acquisition/import are rechecked without compilation or inference. | Stock vendor stack was inherited; no new annotated executable is qualified. Controller compatibility/publication/registry/clean-install decisions remain. |

## Intel recipient and authorization requirements

The controlling retained wheel text is **Intel Developer Tools EULA, August
2024**, SHA256 `febfb69a5b058f55892dc7f341108a9ca29d50e9fe85723a57eebbb826436958`.
Section 1.I defines designated Redistributables. Section 2.1.D permits conditional
redistribution as part of Your Product; D(2) limits Intel-provided executable
Redistributables to executable distribution subject to an agreement prohibiting
reverse engineering, decompiling or disassembly **of those Redistributables**.
This is a recipient-license requirement, not a notice-file requirement. A public
container pull or a copied EULA does not demonstrate that the required arrangement
exists. Legal/Wulf must choose and implement that arrangement or obtain alternate
written Intel permission covering the intended distribution.

D(3) limits sublicensing to necessary Intel copyrights within Your Product.
D(4) allocates support/liability, restricts certification/marketing claims and
requires additional restrictions and indemnity. Sections 2.2 and 3.1 require
separate treatment of third-party terms and restrict uses beyond the grant,
including source redistribution and reciprocal-license combinations. Do not impose
Intel's recipient restriction on separately licensed open-source components.
Section 3.2 requires a written Intel waiver for otherwise prohibited prerelease
uses; no waiver is presumed, and no finding here identifies these inspected
2026.1.1 files as prerelease. Legal should assess relevance rather than infer it.

The full compiler designation list and exact byte/path mapping are retained in
`source-delivery/intel-2026.1.1-review.json`. Historical installed-list screening
is preserved in `intel-2026.1.1-scope.json`; it is superseded only for this inspected
scope. Matching designation does not satisfy recipient or other license conditions.

Detailed findings: `source-delivery/kernel-wheel-provenance.json`,
`numpy-libgfortran-provenance.json`, `companion-delivery.json`,
`archival-notice-review.json`, and `UNRESOLVED.json`.
Original licenses and historical evidence remain unchanged.

Authoritative references: [PyPI wheel record](https://pypi.org/project/vllm-xpu-kernels/0.1.15.4/#files),
[upstream release](https://github.com/vllm-project/vllm-xpu-kernels/releases/tag/v0.1.15),
[NumPy pinned build requirements](https://github.com/numpy/numpy/blob/v2.2.6/requirements/ci_requirements.txt),
[OpenBLAS wheel build source](https://github.com/MacPython/openblas-libs/tree/510428188c2a87260a0593796af9ce425f86ad8e),
[Intel August 2024 agreement](https://cdrdv2-public.intel.com/777700/intel-end-user-license-agreement-for-developer-tools-version-august-2024.pdf),
[Apache 2.0 section 4](https://www.apache.org/licenses/LICENSE-2.0),
[GCC Runtime Library Exception](https://www.gnu.org/licenses/gcc-exception-3.1.html).

The approved model-pack release version is 1.0.5; Controller remains 1.0.1.
The external writable-work mount is unnecessary; the unchanged accepted image
serves through installed Controller 1.0.1 using its ordinary writable filesystem.
Before publication Wulf must authorize actual image/source publication with real
identities and arrange the later public clean-install test.
No release, registry artifact, GitHub push or publication is performed here.
