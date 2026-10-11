#!/usr/bin/env bash
set -euo pipefail
cd /build
# Preserve the historical absolute compiler/include/source paths for byte identity.
sha256sum -c moe-inputs.sha256
bash /build/gdn-build.sh
cp -a source/csrc ./csrc
sed 's@source/csrc/@csrc/@' moe-inputs.sha256 | sha256sum -c -
bash /build/remap-build.sh
bash /build/attention-build.sh
gcc --version
gcc -std=c11 -O2 -fPIC -shared -Wall -Wextra -Werror l0_peer_residency_shim.c -o libl0_peer_residency_shim.so -lze_loader -ldl -pthread
mkdir -p /out/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels /out/opt/b70-residency-shim
cp libgdn_index64.so libmoe_remap_guard.so libb70_attn_reduce.so /out/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/
cp libl0_peer_residency_shim.so /out/opt/b70-residency-shim/
sha256sum /out/opt/b70-residency-shim/* /out/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/*
