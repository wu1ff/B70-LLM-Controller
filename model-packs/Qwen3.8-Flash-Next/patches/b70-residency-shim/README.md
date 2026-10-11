> Historical Vanilla stage. Current Disk uses the cumulative source and exact native recipes in [../disk/](../disk/README.md). The old shipping identity and application instructions below describe Vanilla; they are not the current Disk build contract.

# B70 Level Zero peer-residency shim

- **Purpose:** B70 Level Zero peer-residency shim.
- **Category:** resource.
- **Pinned upstream base:** Standalone retained C source; built inside the pinned vLLM parent image.
- **Files affected:** `l0_peer_residency_shim.c`.
- **Applied runtime stage:** `base-c5`.
- **Currently shipping:** Yes, in `ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0` (`sha256:85512b52c09fa660a2e7fe441417129e7c47fac727fd85f66ccea6b65e0a9122`).
- **Technical explanation:** The preload shim controls Level Zero peer-residency policy for B70 allocations while preserving the required oneCCL and small-allocation paths. It is independent of the vLLM source patches.
- **Reproduction/application:** Build the retained C file using `BUILD.md`; production loads `/opt/b70-residency-shim/libl0_peer_residency_shim.so`.
- **Retained provenance:** `build/manifest.json`; `shim/l0_peer_residency_shim.c`; `docs/BASE_BRINGUP.md`; `Desktop/shim/rrshim.md`.
