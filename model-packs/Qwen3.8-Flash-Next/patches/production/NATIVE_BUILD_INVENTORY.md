# Complete native source/build inventory

All twelve installed libraries are required for payload reconstruction, including fallback consumers. Build inputs are pinned by `immutable-inputs.json` and `input-lock.json`. No native binary is included in this bundle.

| Installed library | Translation units | Build script | Headers | Output before install |
| --- | --- | --- | --- | --- |
| vllm_xpu_kernels/libgdn_index64.so | native/native/gdn_index64.cpp | native/native/build.sh | gdn-headers | libgdn_index64.so |
| b70-residency-shim/libl0_peer_residency_shim.so | native/residency-shim/l0_peer_residency_shim.c | vendor GCC command in build/Dockerfile.native | vendor Level Zero + libc | libl0_peer_residency_shim.so |
| vllm_xpu_kernels/libastra_moe.so | native/astra500-native/astra_moe.cpp | native/astra500-native/build.sh | common-native-headers | libastra_moe.so |
| vllm_xpu_kernels/libastra_hc.so | native/astra500-structural-hc/astra_hc_v3.cpp | native/astra500-structural-hc/build-v3.sh | common-native-headers | libastra_hc_v3.so |
| vllm_xpu_kernels/libastra_direct.so | native/astra500-structural-moe/astra_direct.cpp | native/astra500-structural-moe/build.sh | common-native-headers | libastra_direct.so |
| vllm_xpu_kernels/libastra_groups.so | native/astra500-ownership-groups/astra_direct.cpp, native/astra500-ownership-groups/ownership_remap.cpp, native/astra500-ownership-groups/ownership_gather.cpp | native/astra500-ownership-groups/build.sh | common-native-headers | libastra_groups.so |
| vllm_xpu_kernels/libastra_batch.so | native/astra500-ownership-batch/astra_batch.cpp | native/astra500-ownership-batch/build.sh | common-native-headers | libastra_batch.so |
| vllm_xpu_kernels/libbeyond_prefill.so | native/astra500-beyond-prefill/beyond_prefill.cpp | native/astra500-beyond-prefill/build.sh | common-native-headers | libbeyond_prefill.so |
| vllm_xpu_kernels/libdominant_w4_layout.so | native/astra500-dominant-native/w4_layout.cpp | native/astra500-dominant-native/build.sh | common-native-headers | libdominant_w4_layout.so |
| vllm_xpu_kernels/libserving_prefill.so | native/prefill-native/prefill_small.cpp | native/prefill-native/build.sh | common-native-headers | libserving_prefill.so |
| vllm_xpu_kernels/liblong_hc.so | native/long-hc-selected/hc.cpp | native/long-hc-selected/build.sh | common-native-headers | liblong_hc.so |
| vllm_xpu_kernels/liblong_prefill_v3.so | native/long-prefill-native/prefill_small.cpp | native/long-prefill-native/build.sh | common-native-headers | liblong_prefill_v3.so |

The scripts preserve the exact C++20/O3/NDEBUG/PIC/hidden-visibility/SYCL spir64 options, XE2 and CUTLASS definitions, include order, SPIR-V extension flags, symbol binding, Torch libraries and relative rpath. GDN links the pinned vendor delta-rule library; the shim retains its C11/O2/Wall/Wextra/Werror/ze_loader/dl/pthread command. HC v3 is installed as libastra_hc.so. Direct v2 uses its retained normalized astra_direct.cpp/direct_gemm.hpp filenames and libastra_direct.so export.

The old structural-builder image is the vendor layer prefix plus the copied compiler and an unrelated four-file native build context/output. Its environment equals the vendor environment. The replacement copies the same pinned compiler into the same /toolchain path and supplies all selected include trees directly. See native-environment-closure.json and phase2-evidence for inspected ancestry; the local image is not an input.

Required vendor headers/loaders were inspected without compilation. Kernel, TLA and compiler source/header file hashes and references are distinct from accepted .so hashes. Independent compilation and binary comparison remain the next step.
