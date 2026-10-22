#!/usr/bin/env bash
set -euo pipefail
cd /build
COMPILER=/toolchain/oneapi/opt/intel/oneapi/compiler/2026.1
export PATH="$COMPILER/bin:/opt/venv/bin:$PATH"
export LD_LIBRARY_PATH="$COMPILER/lib:/toolchain/oneapi/opt/intel/oneapi/tbb/latest/lib:${LD_LIBRARY_PATH:-}"
TORCH=/opt/venv/lib/python3.12/site-packages/torch
KERNELS=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
icpx --version
/opt/venv/bin/python -c 'import torch; print(torch.__version__); assert not torch.xpu.is_available()'
# Only the GDN interface TU is rebuilt. No compiler/runtime packages are
# installed in the serving image. The existing delta-rule library is unchanged.
icpx -std=c++20 -O3 -DNDEBUG -fPIC -fvisibility=hidden -fsycl \
  -fsycl-targets=spir64 -DVLLM_XPU_ENABLE_XE2 \
  -I/build/source -I/build/source/csrc \
  -I/build/source/csrc/xpu/gdn_attn \
  -I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include" \
  -I/usr/include/python3.12 -I"$COMPILER/include/syclcompat" \
  -include /build/source/csrc/sycl_first.h \
  -c astra_gdn.cpp -o astra_gdn.o
icpx -shared -fsycl -fsycl-targets=spir64 -fsycl-max-parallel-link-jobs=4 \
  astra_gdn.o -o libastra_gdn.so \
  -L"$TORCH/lib" \
  -Wl,-rpath,'$ORIGIN/../torch/lib:$ORIGIN' \
  -Wl,-Bsymbolic -ltorch -ltorch_cpu -ltorch_xpu -lc10 -lc10_xpu

sha256sum libastra_gdn.so
