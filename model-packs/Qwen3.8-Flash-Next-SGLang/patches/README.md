# DFlash2 SGLang 1.0.0 release patches

The accepted OCI image is `local/flash-next-sglang@sha256:e834b059c63f40ff4b3478b775cac07cb753e04d0d8add0c547c56c053fa1a66`, from recipe revision `e83385902fbbe47999527a22a85364649b02f70b`. The [OCI chain](oci-chain.tsv) records the selected build order, full parent and config digests, and recipe revisions. The [native input manifest](native-inputs.tsv) records staged source hashes and installed native artifacts.

## Pinned bases and patch order

Full reproduction requires inputs beyond this folder. Obtain the three pinned checkouts below, Intel's XPU base patches for the pinned SGLang and sgl-kernel-xpu bases, the staged `entrypoint.sh`, and the published `qwen38-flashnext-b70-1.0.6.tar.gz` for donor inputs. The publication set excludes `apply.sh`, the internal application script.

| Base or input | Revision | Application |
| --- | --- | --- |
| [SGLang](https://github.com/sgl-project/sglang/commit/ecd3d75a2a790cdb3acba0ecdb4786ff4d4ab25d) | `ecd3d75a2a790cdb3acba0ecdb4786ff4d4ab25d` | Apply the 22 `????-intel-*.patch` files in filename order, then release patches 0001–0004. |
| [sgl-kernel-xpu](https://github.com/sgl-project/sgl-kernel-xpu/commit/555588379c04d213388efaa043828560718a70f4) | `555588379c04d213388efaa043828560718a70f4` | Apply `0001-xpu-flashnext-bf16-fmha.patch` (BF16 FMHA) and `0002-xpu-unique-gdn-kernel-name.patch` (GDN functor identity), in order. |
| [SYCL-TLA](https://github.com/intel/sycl-tla/commit/7b56b936b87f799f80dde60c2fd8d11c1128cebb) | `7b56b936b87f799f80dde60c2fd8d11c1128cebb` | Unmodified source for the XPU wheel. |
| Staged `entrypoint.sh` | Recipe `a13bee645c780dff715d8cbd5ed1692a6644c6ca` | Stage the pinned file at mode 0755 and apply release patch 0005 in its directory. |

| Order | Release patch | Purpose |
| --- | --- | --- |
| 0001 | [Model, attention and native integration](0001-model-attention-and-native-integration.patch) | Target BF16/INT4 model and native dispatch, bounded cyclic packing, attention, HC, GDN and disk PLE. |
| 0002 | [Prefill, APC and admission](0002-prefill-apc-and-admission.patch) | Checkpoint ownership, KV pools and admissions, short FFN and HC classes, wide-prefill memory and scheduler integration. |
| 0003 | [DFlash2 graphs and STANDARD](0003-dflash2-graphs-and-standard.patch) | Packed target/draft replay, proposal transport and calibration, original actual-q and exact terminal sampling, rollback and allocator transition in the worker. |
| 0004 | [Serving, vision and scoring](0004-serving-vision-and-scoring.patch) | Rank startup and request receive overlap, score stripes, launch integration and multimodal placement. |
| 0005 | [OCI entrypoint](0005-oci-entrypoint.patch) | 1,024-token prefill, 276,480 KV rows and the selected state-cost APC policy. |

Start from three clean, disposable Git checkouts detached at the commits above and use a new output directory for the launch file. Check and apply each source patch with Git's index, then compare the output of `git write-tree` at each checkpoint. Verify the staged entrypoint after patch 0005 with SHA-256.

| Verification checkpoint | Expected value |
| --- | --- |
| SGLang Git tree after the base patches | `1b7dbe8263fd2af9d319e418c231ab2291e06071` |
| sgl-kernel-xpu Git tree after the base patches | `8d0989dd7480899b43192fa36edebb8a3a820d9c` |
| SGLang Git tree after release patches 0001–0004 | `1933fc7403a310166cc747b7e31243f17fd7642b` |
| Staged entrypoint SHA-256 after release patch 0005 | `e52b295908c51f806fabb21c23fac7de57995d72264d4603bd1530744471d7aa` |

SGLang source lives under `python/sglang`. The original build installs the locked wheels in `/opt/flashnext/site` and updates the installed SGLang package at that path. The release patches give the net source of that update. For a file-by-file installed projection, use the locked SGLang wheel's package membership and modes, add the 48 new Python files from the final source tree, and use setuptools-scm's generated `_version.py` at version `0.5.24.dev0+gecd3d75`. New files install at 0644 except `gdn_admission.py`, `gdn_factors.py`, `greedy_head.py`, `head_admission.py`, `owned_standard.py`, `pages_transfer.py` and `wide_hc.py`, which the source overlays install at 0664. The native proposal binding installs separately into the same package at the location in the native input manifest.

This is the release source application path. The historical image recipes already perform the corresponding patching and overlays in OCI layer order. Apply each path to its declared base. The historical recipes are retained for exact image reproduction.

## Build and launch dependencies

- The wheel recipe selects Xe2 `bmg`, BF16-only Flash Next FMHA and GDN, and disables Rust extensions and the generic MLA, QSA and MoE build options. It uses the compiler from `intel/deep-learning-essentials@sha256:f5a3127b3bc4e856defe185dffa827aee4bc359b8705e766ce3d0f578a969477`. The checked `icpx` SHA-256 is `be7fa7c3c74c2ba2b8e1b50b0a61e37986441727899e25f3545636bde157c083`. The dependency lock, base source inputs and callback-only shim source complete the foundation build inputs.
- The v6 runtime, Torch/XPU ABI, device user space, serving-view adapter and installed donor libraries come from `ghcr.io/wu1ff/qwen38-flashnext-b70@sha256:2fa91d1ac8c17ad866d9ae7eaea83ac1b1666ffd0ccceb7685564f55528c47af`. The amd64 manifest is in the first OCI chain row. The v6 production release documents its source provenance. [v6 production release](https://github.com/wu1ff/B70-LLM-Controller/releases/tag/qwen3.8-flash-next-v1.0.6)
- For native build recipes that accept a donor path, give them the extracted `patches/production` directory from the published `qwen38-flashnext-b70-1.0.6.tar.gz` (SHA-256 `03ca4467223cd18af3f669c7c53e08f673f4a611d60181c2c67759d14ac2ecb8`). The recipes validate the unmodified donor files, headers and SYCL-TLA snapshot, then stage their listed source and native patches. The [native input manifest](native-inputs.tsv) follows dependency order. For multi-file staged sources, its source hash is from `LC_ALL=C sha256sum ./* | sha256sum` run in the builder's `source` directory with C-locale glob order. Single-file inputs are hashed directly. The compiler, headers and ABI are separate pins. The manifest records the selected ordered-GDN library, which replaced the earlier library at the same installed path.
- Release patch 0003 includes the final generated tiled-filter source. The original build uses the digest-checked v6 top-k/top-p donor (SHA-256 `6a5ed7e80cb9a054cfe4251cf70e9f3dedf7c39febb5345c96d04056726cb658`), the generator, and the later position-owner changes. The embedded BF16-draft selector's 7×2 FP32 calibration matches the v6 `quality_calibration.pt` (SHA-256 `f6a104dd32e6a41754a89a2bb1cd3a3f35ec04ef6083c68ffd17e73a75375624`).
- Use the pinned checkpoints and the B70 launcher for the accepted image. The OCI loads `sglang.launch_server` from `/opt/flashnext/site`. Live verification recorded the effective environment, target/draft mounts, model-view hashes, and the LAN port override applied outside the image. Build and launch recipes also record the BF16/INT4, APC, graph, vision, and context settings.

## Verification boundary

Verification applied the release in clean checkouts and compared 6,661 source-package files against an independent replay of the selected build-source assembly. The 86 post-base SGLang changes include 48 additions and zero deletions. The launch file matched at mode 0755. A separate check replayed historical OCI source installation and verified all 13,314 files under `/opt/flashnext` against the pinned layers, including the four historical `.orig` patch backups and saved recipe inputs. Those backups and compiled objects are outside the source patch payload. [Clean-checkout and source-package replay](../TUNING_GUIDE.md#10-the-selected-sglang-implementation)

Earlier uncached OCI builds reproduced all five manifests from the pinned parent. Source verification checked source assembly, input pins, installed native hashes and OCI ancestry. The inherited v6 native binaries and ABI came from the verified parent. Recompiling those binaries and rebuilding the image were outside the scope of source verification.
