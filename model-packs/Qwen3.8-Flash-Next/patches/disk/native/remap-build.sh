#!/usr/bin/env bash
set -euo pipefail
cd /build
REMAP_COMPILER=/toolchain/oneapi/opt/intel/oneapi/compiler/2026.0
export PATH="$REMAP_COMPILER/bin:/opt/venv/bin:$PATH"
export LD_LIBRARY_PATH="$REMAP_COMPILER/lib:/toolchain/oneapi/opt/intel/oneapi/tbb/2023.0/lib:${LD_LIBRARY_PATH:-}"
REMAP_TORCH=/opt/venv/lib/python3.12/site-packages/torch
icpx --version
python -c 'import torch; print(torch.__version__); assert not torch.xpu.is_available()'
icpx -std=c++20 -O3 -DNDEBUG -fPIC -fvisibility=hidden -fsycl -fsycl-targets=spir64 -I/build -I/build/csrc -I"$REMAP_TORCH/include" -I"$REMAP_TORCH/include/torch/csrc/api/include" -I/usr/include/python3.12 -I"$REMAP_COMPILER/include/syclcompat" -include /build/csrc/sycl_first.h -c remap_guard.cpp -o remap_guard.o
icpx -shared -fsycl -fsycl-targets=spir64 -fsycl-max-parallel-link-jobs=4 remap_guard.o -o libmoe_remap_guard.so -L"$REMAP_TORCH/lib" -Wl,-rpath,'$ORIGIN/../torch/lib:$ORIGIN' -Wl,-Bsymbolic -ltorch -ltorch_cpu -ltorch_xpu -lc10 -lc10_xpu
sha256sum libmoe_remap_guard.so
