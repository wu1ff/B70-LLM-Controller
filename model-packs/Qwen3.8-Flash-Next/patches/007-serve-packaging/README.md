> Historical Vanilla stage. Current Disk uses the cumulative source and exact native recipes in [../disk/](../disk/README.md). The old shipping identity and application instructions below describe Vanilla; they are not the current Disk build contract.

# Dense-QSA serving configuration and preparator

- **Purpose:** Publish the exact final packaging-layer source.
- **Category:** correctness.
- **Pinned upstream base:** Model revision `40b8f18df4d4a32cb6e687a51c78207e5e438522` of `devan-carlin/Qwen3.8-Flash-Next-W4A16` (`config.json` SHA-256 `b9ef7d7d97a9c0bf046616470fc1db9e69d80c8b1998dc4a732ed6b0b4abf954`); runtime parent `sha256:ad6d44d7a8aee41a4d419befbcaf78c7d73e0acd7f7aaaf597b5ff283601b448`.
- **Files affected:** In-image `/opt/b70-flashnext/serve-config.json`, `/opt/b70-flashnext/prepare-serve.sh`, and `ENTRYPOINT`; model snapshot remains read-only.
- **Applied runtime stage:** production pack packaging layer `pack-1.0.0`.
- **Currently shipping:** Yes, in `ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0` (`sha256:85512b52c09fa660a2e7fe441417129e7c47fac727fd85f66ccea6b65e0a9122`).
- **Technical explanation:** The checkpoint has no indexer weights, so the serving config removes exactly five `text_config.indexer_*` keys. The entrypoint creates a writable serve directory with symlinks to the mounted snapshot, installs that config, and starts vLLM.
- **Reproduction/application:** Apply `dense-qsa-serve-config.patch` to the pinned model `config.json`; it yields the included `serve-config.json` byte-for-byte. `Dockerfile` and `prepare-serve.sh` are the retained production packaging source. This bundle documents the original layer; pack 1.0.2 does not rebuild it.
- **Retained provenance:** `packaging/PROMOTION.json`; `packaging/Dockerfile`; `packaging/serve-config.json`; `packaging/prepare-serve.sh`.
