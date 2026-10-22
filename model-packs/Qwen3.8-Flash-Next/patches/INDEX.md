# DFlash2 source index

- [Production](production/README.md): exact accepted source reconstruction; 16 ordered
  patches, 12 native families, frontend, residency shim, helpers and immutable inputs.
- [Campaign](archive/reconnaissance/CAMPAIGN.md): selected, superseded, rejected and
  diagnostic work with reasons. Original path/hash indices: `archive/patch-index.json`
  (97 patch paths / 86 unique contents), `archive/native-variants-index.json` (113
  originals), `archive/python-variants.json` (120 paths / 53 unique variants).
  `archive/objects/` stores immutable content; `archive/originals/` retains sources.
- [Qualification](archive/qualification/README.md): Phase 2–4 source/build/live receipts
  and accepted stock BetterBench/serving measurements. They are historical evidence.
- [Notices](licenses/NOTICE.md), [source access](licenses/SOURCE_ACCESS.md),
  [distribution findings](licenses/SOURCE_DISTRIBUTION.md).
- [Visual assets](assets/README.md): reserved for reviewed showcase material.

Do not apply archived patches over production. Some original patch names mention
MTP investigations: their selected effects are already in the accepted DFlash2
chain; they do not expose a Controller MTP mode. Historical published sources stay
available at the 1.0.4 release recorded in `archive/published-1.0.4.json`.
`SHA256SUMS` covers every regular file below this directory except itself.
