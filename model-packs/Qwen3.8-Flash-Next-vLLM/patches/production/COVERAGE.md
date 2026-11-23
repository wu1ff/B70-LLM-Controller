# Required v6 coverage and application order

Start from the exact vendor foundation/pristine vLLM v0.33 input, install generated
version metadata, then apply `manifests/patch-series.json` in ascending order.

| Patch | Covered changes |
| --- | --- |
| 01 | Accepted v0.32 changes ported to v0.33: DFlash2 draft/feature capture/speculator and serving integration, common disk BF16 PLE, Intel checkpoint preparation interfaces, residency consumers, GDN metadata B1, graph infrastructure, HC/projection/MoE dispatch, current prefill and FA2 interfaces |
| 02 | Shared all-group GDN metadata route gated to XPU DFlash2 |
| 03 | Owned draft-query graph and exact ordered native HC projection dispatch |
| 04 | BF16 repaired graph/verification shape handling, original STANDARD verification, score configuration and accepted thinking EOS guard |
| 05 | Calibrated proposal integration and exact BF16 grouped-convolution fusion; exact calibration bytes installed separately |
| 06 | Compiled target-graph input update and fused graph-input copy |
| 07 | Parallel gather from the same disk PLE mmap, exact GDN BA, HC schedule at 32 rows |
| 08 | MoE glue integration and retained v1 native library (historical image payload) |
| 09 | Deferred PLE preparation and per-block MoE stash clear |
| 10 | Corrected exact MoE v3 and 64-byte alignment guard |

Next build/install every library in `manifests/native-builds.json`, preserving
fallbacks and the legacy v1 image artifact. Install `support/b70-flashnext`, the
14 exact `image-support` mirror deltas, calibration tensor and image configuration.
The native libraries are independent outputs over the same pinned environment;
the manifest order is the historical family order followed by the later additions.

Finally supply the unchanged external profile and its two diagnostic worker
modules. It preserves TP4/EP4, four slots, context 262144, MBT1024, utilization
.91, BF16 target and draft, disk BF16 PLE with eight gather threads, STANDARD,
seven proposals, draft TP4, query8/support16, scale1, canonical reductions,
EOS guard, calibrated proposal, BF16 convolution, target/query graphs, compiled
update, fused copy, exact BA, HC20/split32=10, corrected MoE and deferred PLE.
It also retains APC, tools (`qwen3_xml`), reasoning (`qwen3`) and two images/no
video. Generic `--enforce-eager` coexists with the custom XPU graph paths.

This table inventories the existing selected implementation. It adds no feature,
tuning change, output-quality claim or renewed qualification result.
