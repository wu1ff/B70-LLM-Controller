#!/usr/bin/env bash
set -euo pipefail
cd /build
ATTN_COMPILER=/toolchain/oneapi/opt/intel/oneapi/compiler/2026.0
export PATH="$ATTN_COMPILER/bin:/opt/venv/bin:$PATH"
export LD_LIBRARY_PATH="$ATTN_COMPILER/lib:/toolchain/oneapi/opt/intel/oneapi/tbb/2023.0/lib:${LD_LIBRARY_PATH:-}"
ATTN_TORCH=/opt/venv/lib/python3.12/site-packages/torch
ATTN_TLA=/build/sycl-tla-87f6850680a580654b9ea2c80dbc01aeb36ad231
icpx --version
icpx -std=c++20 -O3 -DNDEBUG -fPIC -fvisibility=hidden -fsycl -fsycl-targets=spir64 -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate -fno-sycl-instrument-device-code -DCUTLASS_ENABLE_HEADERS_ONLY -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET -DCUTLASS_VERSIONS_GENERATED -I/build -I/build/csrc -I/build/csrc/xpu/attn/xe_2 -I"$ATTN_TLA/include" -I"$ATTN_TLA/tools/util/include" -I"$ATTN_TLA/applications" -I"$ATTN_TORCH/include" -I"$ATTN_TORCH/include/torch/csrc/api/include" -I/usr/include/python3.12 -I"$ATTN_COMPILER/include/syclcompat" -include /build/csrc/sycl_first.h -c native.cpp -o native.o
icpx -shared -fsycl -fsycl-targets=spir64 -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate -fsycl-max-parallel-link-jobs=2 native.o -o libb70_attn_reduce.so -L"$ATTN_TORCH/lib" -Wl,-rpath,'$ORIGIN/../torch/lib:$ORIGIN' -Wl,-Bsymbolic -ltorch -ltorch_cpu -ltorch_xpu -lc10 -lc10_xpu
sha256sum libb70_attn_reduce.so
