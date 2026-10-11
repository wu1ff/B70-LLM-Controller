# Official vLLM 0.30.0 XPU linux/amd64 dependency image, pinned by local config ID.
# build.py verifies its vendor registry provenance before building offline.
FROM sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9 AS native
COPY dependencies/compiler/ /toolchain/debs/
RUN mkdir -p /toolchain/oneapi && for deb in /toolchain/debs/*.deb; do dpkg-deb -x "$deb" /toolchain/oneapi; done
COPY native/ /build/
COPY dependencies/sycl-tla.tar.gz /build/sycl-tla.tar.gz
RUN cd /build && tar -xzf sycl-tla.tar.gz && bash build.sh

FROM sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9
COPY overlay/ /
COPY --from=native /out/ /
WORKDIR /opt/venv
RUN chmod 0755 /opt/b70-flashnext/prepare-serve.sh && mkdir -p /work/tmp /work/hf-cache /work/triton-cache /work/vllm-cache /work/xdg-cache && /opt/venv/bin/python -m compileall -q /opt/venv/lib/python3.12/site-packages/vllm /opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
ENTRYPOINT ["/opt/b70-flashnext/prepare-serve.sh"]
