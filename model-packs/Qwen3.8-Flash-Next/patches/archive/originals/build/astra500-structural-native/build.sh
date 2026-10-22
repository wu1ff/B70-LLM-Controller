#!/usr/bin/env bash
set -euo pipefail
cd /build
COMPILER=/toolchain/oneapi/opt/intel/oneapi/compiler/2026.1
export PATH="$COMPILER/bin:/opt/venv/bin:$PATH"
export LD_LIBRARY_PATH="$COMPILER/lib:/toolchain/oneapi/opt/intel/oneapi/tbb/latest/lib:${LD_LIBRARY_PATH:-}"
TORCH=/opt/venv/lib/python3.12/site-packages/torch
icpx -std=c++20 -O3 -DNDEBUG -fPIC -fvisibility=hidden -fsycl -fsycl-targets=spir64 \
 -I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include" -I/usr/include/python3.12 \
 -I"$COMPILER/include/syclcompat" -shared astra_ar4.cpp -o libastra_ar4_v10.so \
 -L"$TORCH/lib" -Wl,-rpath,'$ORIGIN/../torch/lib:$ORIGIN' \
 -Wl,-Bsymbolic -ltorch -ltorch_cpu -ltorch_xpu -lc10 -lc10_xpu -lze_loader
sha256sum libastra_ar4_v10.so
