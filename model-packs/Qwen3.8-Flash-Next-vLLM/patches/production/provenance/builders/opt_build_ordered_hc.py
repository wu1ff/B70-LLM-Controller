"""Build an exact-order HC projection using MTP's native tile infrastructure."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

W = Path(__file__).resolve().parents[1]
O = W / "build/optimization-ordered-hc"
O.mkdir(exist_ok=False)
M = Path("/home/wulf/Desktop/Qfn-Mtp-vllm33/astra/candidates/target-hc-native-v1")
shutil.copy2(W / "build/optimization-inputs/ordered_hc.cpp", O / "ordered_hc.cpp")
(O / "dense_fragment.hpp").write_text((M / "dense_fragment.hpp").read_text().replace(
    "AstraHCNativeV1", "B70OrderedHCNativeV1"))
s = (W / "build/optimization-fused-cyclic/build.sh").read_text()
s = s.replace("cyclic_fused.cpp", "ordered_hc.cpp").replace("cyclic_fused.o", "ordered_hc.o")
s = s.replace("libb70_cyclic_fused_v1.so", "libb70_ordered_hc_v1.so")
(O / "build.sh").write_text(s)
cmd = json.loads((W / "build/optimization-fused-cyclic/command.json").read_text())
cmd[cmd.index("--name") + 1] = "dflash-opt-build-ordered-hc-v1"
old_mount = str(W / "build/optimization-fused-cyclic") + ":/outputs:rw"
cmd[cmd.index(old_mount)] = str(O) + ":/outputs:rw"
(O / "command.json").write_text(json.dumps(cmd, indent=2))
(O / "source-hashes.json").write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (M / "dense_fragment.hpp", O / "ordered_hc.cpp", O / "dense_fragment.hpp")}, indent=2))
with (O / "build.log").open("w") as output:
    subprocess.run(cmd, stdout=output, stderr=subprocess.STDOUT, check=True)
print("Built", O / "libb70_ordered_hc_v1.so", flush=True)
