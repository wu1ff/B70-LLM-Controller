# Distribution status — publication blocked

2026-10-22, America/Phoenix. This is local staging. No public runtime/reference,
legal clearance or complete corresponding-source delivery is claimed.

| Finding | Evidence and current status | Remaining decision/work |
| --- | --- | --- |
| Changed-file notices | `notice-export.json` stages all 15 missing vLLM headers, the modified kernel wrapper, 3 GDN conv headers and custom sources as comment-only derivatives. Exact accepted bytes remain comparison inputs. | Review final distribution/archive notices and incorporate applicable notices in eventual executable/source delivery. Frozen image remains unchanged. |
| Stock kernel wheel source | Vendor digest pins vllm-xpu-kernels 0.1.15.4 bytes. Intel reference commits/header trees reconstruct custom kernels; they do not attest the stock wheel. | Obtain matching stock-wheel source/build provenance or decide an adequate distribution route. |
| Companion source access | Ten Ubuntu source families match actual accepted versions; wheel native hashes and source/build references retained in `source-delivery/`. [Directions](SOURCE_ACCESS.md) name exact references and limits. | Maintain matching source/build assets with durable access adjacent to eventual image distribution. Current bundle excludes large companion source archives and does not settle every inherited executable. |
| NumPy libgfortran | NumPy 2.2.6 native hash matches retained wheel evidence. | Exact GCC source/build identity remains unknown; neither Ubuntu GCC source nor NumPy sdist proves it. |
| Intel 2026.1.1 | Installed August 2024 terms, notices and compiler `credist.txt` retained. Only 19 of 59 inspected wheel ELF/library paths match designation basenames. | Reconcile release-specific SYCL/UR designation and executable recipient arrangements; basename screening is not attestation. |
| Independent build/live behavior | Phase 3 independently rebuilt 12/12 matching custom native outputs and all non-bytecode runtime files. Phase 4 live equivalence passed. | OCI bytes differ; vendor stack is inherited. Comment-only exports do not extend executable qualification. |

`UNRESOLVED.json` is the machine-readable status. Applicable original legal text
is retained in `upstream/`, `accepted-vendor/`, `compiler-input/`, model notices and
embedded source archives. Historical records retain their original dates/statuses;
this table supersedes their current-status assertions without rewriting them.

Do not infer clearance from a source verifier, build, public download URL or model
card alone. Publication also needs release-version ratification, an actual GHCR
artifact/digest, a normalized source archive, and a separately authorized clean
installation test after publication prerequisites are resolved.
