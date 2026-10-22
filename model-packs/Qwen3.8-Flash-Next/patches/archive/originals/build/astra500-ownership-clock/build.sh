#!/usr/bin/env bash
set -euo pipefail
cd /diag
COMPILER=/toolchain/oneapi/opt/intel/oneapi/compiler/2026.1
export PATH="$COMPILER/bin:/opt/venv/bin:$PATH"
export LD_LIBRARY_PATH="$COMPILER/lib:/toolchain/oneapi/opt/intel/oneapi/tbb/latest/lib:${LD_LIBRARY_PATH:-}"
TORCH=/opt/venv/lib/python3.12/site-packages/torch
icpx -std=c++20 -O3 -DNDEBUG -fPIC -fvisibility=hidden -fsycl -fsycl-targets=spir64 \
 -I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include" \
 -I/usr/include/python3.12 -I/usr/include/level_zero \
 -c clock.cpp -o clock.o
icpx -shared -fsycl -fsycl-targets=spir64 clock.o -o libastra_clock.so \
 -L"$TORCH/lib" -Wl,-rpath,'$ORIGIN/../torch/lib:$ORIGIN' \
 -Wl,-Bsymbolic -ltorch -ltorch_cpu -ltorch_xpu -lc10 -lc10_xpu -ldl
g++ -std=c++17 -O2 -shared -fPIC trace.cpp -o libownership_trace_final.so -ldl -pthread
