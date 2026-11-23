FROM vllm/vllm-openai-xpu@sha256:9dd913e9036633d55ace17265cbfc29f48b9b85fe4bc8b129debcf7e1b269ce7
COPY sources/vllm/vllm/ /opt/venv/lib/python3.12/site-packages/vllm/
COPY sources/release-1.0.4/native/l0_peer_residency_shim.c /tmp/l0_peer_residency_shim.c
RUN mkdir -p /opt/b70-residency-shim && gcc -std=c11 -O2 -fPIC -shared -Wall -Wextra -Werror /tmp/l0_peer_residency_shim.c -o /opt/b70-residency-shim/libl0_peer_residency_shim.so -lze_loader -ldl -pthread
COPY build/frontend/ /opt/b70-flashnext/
RUN /opt/venv/bin/python -m compileall -q /opt/venv/lib/python3.12/site-packages/vllm/models/qwen4_exp
ENTRYPOINT ["/opt/b70-flashnext/prepare-serve.sh"]
WORKDIR /opt/venv
COPY build/native-out/libgdn_index64.so /opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/libgdn_index64.so
