"""Assemble only the measured query graph and exact-order HC changes."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

W = Path(__file__).resolve().parents[1]
E = W / "evidence/optimization"
C = W / "build/optimization-candidate-v3"
C.mkdir(exist_ok=False)
site = C / "overlay/opt/venv/lib/python3.12/site-packages"
parent = "local/dflash2-v033:parity-shared-gdn"
expected = "sha256:b0262b274313ddd1127b95631157a6929073dd6123df55746bdac6927ec76a58"
original = json.loads(subprocess.check_output(["docker", "image", "inspect", parent]))[0]
assert original["Id"] == expected
source = (W / "build/optimization-inputs/dflash2_speculator.py").read_text()
source = source.replace("from typing import Any", "import os\nfrom typing import Any", 1)
needle = '''        self.candidate_sampler = CandidateSampler(
            self.max_num_reqs, self.num_speculative_steps, self.top_k, device
        )
'''
assert source.count(needle) == 1
source = source.replace(needle, needle + '''        if (device.type == "xpu" and type(self).__name__ == "DFlash2Speculator"
                and os.getenv("B70_DFLASH_QUERY_GRAPH", "0") == "1"):
            from .b70_graphs import install
            install(self)
''')
payloads = {"vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py": source}
query = (W / "build/optimization-inputs/draft_query_graph.py").read_text()
# Context and prepared-metadata experiments remain in research inputs only.
query = query[:query.index("\ndef context_snapshot")]
payloads["vllm/v1/worker/gpu/spec_decode/dflash2/b70_graphs.py"] = query
hc_input = W / "build/optimization-inputs/b70_hc_gemm.py"
subprocess.run(["docker", "cp", "dflash-opt-baseline-20261112:/opt/venv/lib/python3.12/site-packages/vllm/models/qwen4_exp/nvidia/b70_hc_gemm.py", str(hc_input)], check=True)
hc = hc_input.read_text()
needle = "_large_hc = None\n"
assert hc.count(needle) == 1
hc = hc.replace(needle, needle + '''_ordered_native_hc = int(os.getenv("B70_DFLASH_NATIVE_HC", "0"))
if _ordered_native_hc:
    if _ordered_native_hc not in (10, 20, 40):
        raise ValueError("Unsupported exact-order HC split")
    import vllm_xpu_kernels
    torch.ops.load_library(str(Path(vllm_xpu_kernels.__file__).with_name(
        "libb70_ordered_hc_v2.so")))
''')
needle = '''    rows, inner = x.shape
    columns = weight.shape[0]
    block_k = 64 if rows > 8 else 128
'''
assert hc.count(needle) == 1
hc = hc.replace(needle, '''    rows, inner = x.shape
    columns = weight.shape[0]
    if (_ordered_native_hc and 1 <= rows <= 32 and inner == 10240
            and x.is_contiguous() and weight.is_contiguous()
            and columns in (320, 336)):
        return torch.ops.b70_ordered_hc_v2.forward(
            x, weight, valid_columns, _ordered_native_hc)
    block_k = 64 if rows > 8 else 128
''')
payloads["vllm/models/qwen4_exp/nvidia/b70_hc_gemm.py"] = hc
manifest = []
for relative, value in payloads.items():
    path = site / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    compile(value, str(path), "exec")
    mirror = C / "overlay/workspace/vllm" / relative
    mirror.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, mirror)
    manifest.append(dict(path=relative, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
library = W / "build/optimization-ordered-hc-v2/libb70_ordered_hc_v2.so"
target = site / "vllm_xpu_kernels" / library.name
target.parent.mkdir(exist_ok=True)
shutil.copy2(library, target)
manifest.append(dict(path="vllm_xpu_kernels/" + library.name,
                     sha256=hashlib.sha256(library.read_bytes()).hexdigest()))
(C / "manifest.json").write_text(json.dumps(dict(parent=expected, files=manifest), indent=2))
(C / "Dockerfile").write_text("FROM " + parent + "\nCOPY overlay/ /\n")
profile = json.loads((W / "evidence/parity/recovered-profile.json").read_text())
profile["env"].update(B70_DFLASH_QUERY_GRAPH="1", B70_DFLASH_NATIVE_HC="20")
profile["optimization"] = "Owned draft query graph and exact-order target HC projection; context graph/prepared metadata/alias target/cyclic alternate disabled."
(E / "candidate-profile.json").write_text(json.dumps(profile, indent=2))
with (E / "candidate-build.log").open("w") as output:
    subprocess.run(["docker", "build", "--network=none", "-t",
                    "local/dflash2-v033:opt-query-ordered-hc-v3", str(C)],
                   stdout=output, stderr=subprocess.STDOUT, check=True)
image = json.loads(subprocess.check_output(["docker", "image", "inspect",
                    "local/dflash2-v033:opt-query-ordered-hc-v3"]))[0]
assert image["RootFS"]["Layers"][:len(original["RootFS"]["Layers"])] == original["RootFS"]["Layers"]
(E / "candidate-image.json").write_text(json.dumps(image, indent=2))
print(image["Id"], flush=True)
