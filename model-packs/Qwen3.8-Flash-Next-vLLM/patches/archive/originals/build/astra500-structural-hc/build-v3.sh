#!/usr/bin/env bash
set -euo pipefail
cd /build
COMPILER=/toolchain/oneapi/opt/intel/oneapi/compiler/2026.1
export PATH="$COMPILER/bin:/opt/venv/bin:$PATH"
export LD_LIBRARY_PATH="$COMPILER/lib:/toolchain/oneapi/opt/intel/oneapi/tbb/latest/lib:${LD_LIBRARY_PATH:-}"
TORCH=/opt/venv/lib/python3.12/site-packages/torch
icpx -std=c++20 -O3 -DNDEBUG -fPIC -fvisibility=hidden -fsycl \
  -fsycl-targets=spir64 -DVLLM_XPU_ENABLE_XE2 \
  -DCUTLASS_ENABLE_HEADERS_ONLY -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET \
  -I/build/source -I/build/source/csrc \
  -I/build/source/csrc/xpu/grouped_gemm/xe_2 \
  -I/build/tla/include -I/build/tla/tools/util/include -I/build/tla/examples/common \
  -I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include" \
  -I/usr/include/python3.12 -I"$COMPILER/include/syclcompat" \
  -include /build/source/csrc/sycl_first.h \
  -c astra_hc_v3.cpp -o astra_hc_v3.o
icpx -shared -fsycl -fsycl-targets=spir64 -fsycl-max-parallel-link-jobs=4 \
  -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate \
  astra_hc_v3.o -o libastra_hc_v3.so \
  -L"$TORCH/lib" -Wl,-rpath,'$ORIGIN/../torch/lib:$ORIGIN' \
  -Wl,-Bsymbolic -ltorch -ltorch_cpu -ltorch_xpu -lc10 -lc10_xpu
sha256sum libastra_hc_v3.so
