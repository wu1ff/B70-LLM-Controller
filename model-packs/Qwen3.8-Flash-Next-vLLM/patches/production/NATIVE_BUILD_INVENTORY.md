# Native sources and build inventory

All 17 retained outputs match the selected image. Sources and recipes are
included; generated libraries remain external comparison evidence. All compiler
flags and link options are preserved in each family’s `build.sh`.

| Family | Installed output | Compiler | SHA256 |
| --- | --- | --- | --- |
| native | `libgdn_index64.so` | oneAPI 2026.0.0.20260331 | `101e6a2555449dc92d94b9acc4b79c9581e458f95c4866ce7446bb03518ffa87` |
| residency-shim | `libl0_peer_residency_shim.so` | vendor GCC 13.3.0 | `ae2c82f549d97393268cbd7b90dba7cf38fd00dbccae54f090b1a545e5613b3c` |
| astra500-native | `libastra_moe.so` | oneAPI 2026.0.0.20260331 | `fd704477a62d24bb96f573564894e4723b765620e86ff1f760abaed994d755c8` |
| astra500-structural-hc | `libastra_hc.so` | oneAPI 2026.0.0.20260331 | `2fdda680027db5d730ae9b9ed0098f0db4b97d83a6bc2fe27e55331e1756c44c` |
| astra500-structural-moe | `libastra_direct.so` | oneAPI 2026.0.0.20260331 | `ffefd9b53777893a56224cacbefbd77c09c6baa44be5c64304b27fdd846b5103` |
| astra500-ownership-groups | `libastra_groups.so` | oneAPI 2026.0.0.20260331 | `e8d9887f2918046f6fb45b86a2d8737633f4e42fd3a7e734fe1e5977c9729418` |
| astra500-ownership-batch | `libastra_batch.so` | oneAPI 2026.0.0.20260331 | `551f15c64119aff9905b651f880d7c1313bf0ade3fb3fcb4dc31a7432c50f77a` |
| astra500-beyond-prefill | `libbeyond_prefill.so` | oneAPI 2026.0.0.20260331 | `f2bafe516962f7109d5ec14fce47583db2289fd8830dfb7f571c14c6cf9a66d4` |
| astra500-dominant-native | `libdominant_w4_layout.so` | oneAPI 2026.0.0.20260331 | `1a2e8f582e76e27bbc478a9c2ffa6f26f6e2283673a4059a5c5c0564d2c9d673` |
| prefill-native | `libserving_prefill.so` | oneAPI 2026.0.0.20260331 | `c1c0190a5fd1290d057dabf11f26e3ce66bf01cdd9d32f8fcb7e315a842f35ea` |
| long-hc-selected | `liblong_hc.so` | oneAPI 2026.0.0.20260331 | `959439265e485be1434b038b2050aab775c7d421646bc77806415552749499c5` |
| long-prefill-native | `liblong_prefill_v3.so` | oneAPI 2026.0.0.20260331 | `fe7ba008bd1251ffc1cd6bac7bc08b93fb76b61ec78bed1b66a737300f20c55f` |
| ordered-hc-v2 | `libb70_ordered_hc_v2.so` | oneAPI 2026.0.0.20260331 | `4ddb9a938ce6f415b9b0ae2c0486e04e3f74a358d9c3448381c52672f6db1ce6` |
| exact-v1 | `libb70_exact_v1.so` | oneAPI 2026.0.0.20260331 | `be4e6e756afd5852cefee188f5cce56a18d5caf5eccc3d26b0e6c099123ad1fa` |
| exact-moe-v1 | `libb70_exact_moe_v1.so` | oneAPI 2026.0.0.20260331 | `892e1d86b1e0deb87538936b99d296f2c16865650997ba2565c37dc11395cb6a` |
| exact-moe-v3 | `libb70_exact_moe_v3.so` | oneAPI 2026.0.0.20260331 | `20691aa32d8b61d9e1f907be92f0115960199a6a296ad14762396b218654aaaa` |
| ple-gather | `libple_gather.so` | vendor GCC 13.3.0 | `b3011fac3b7f0687b4ce77a26bad8f20e1e3e9eb21b05bcb3219d8ecafab52f1` |

GDN uses `dependencies/gdn-headers`; other SYCL families use
`dependencies/common-native-headers` and pinned TLA. The shim and PLE gather
use vendor GCC. Keep the legacy MoE v1 library to preserve the final image
payload; corrected MoE v3 is the selected consumer path.

The original GDN compilation produced its object but the first link encountered
an output path that was a directory. The retained link-only retry succeeded;
both logs and commands are preserved. The consolidated GDN build script performs
that same compile and link in a new output directory.

The original PLE GCC command was not retained. A reconstructed `-O2 -fPIC
-shared -pthread` recipe under the pinned vendor compiler reproduced SHA256
`b3011fac3b7f0687b4ce77a26bad8f20e1e3e9eb21b05bcb3219d8ecafab52f1`
exactly. That result closes the source/recipe gap without a serving run.

`manifests/native-dependency-audit.json` resolves 3,348 packaged/toolchain
dependency references and 4,802 external vendor header paths with no gaps.
`manifests/toolchain-files.json` and `compiler-lock.json` pin the complete
retained compiler tree and its 15 package inputs. No arbitrary host compiler
or local structural-builder image is used.

Independent SYCL recompilation of all families was not performed in this task.
