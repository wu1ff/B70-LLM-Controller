"""Extend exact-order native HC to the retained larger verification batches."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

W = Path(__file__).resolve().parents[1]
O = W / "build/optimization-ordered-hc-v2"
O.mkdir(exist_ok=False)
P = W / "build/optimization-ordered-hc"
shutil.copy2(W / "build/optimization-inputs/ordered_hc_v2.cpp", O / "ordered_hc_v2.cpp")
(O / "dense_fragment.hpp").write_text((P / "dense_fragment.hpp").read_text().replace(
    "B70OrderedHCNativeV1", "B70OrderedHCNativeV2"))
s = (P / "build.sh").read_text().replace("ordered_hc.cpp", "ordered_hc_v2.cpp")
s = s.replace("ordered_hc.o", "ordered_hc_v2.o").replace("libb70_ordered_hc_v1.so", "libb70_ordered_hc_v2.so")
(O / "build.sh").write_text(s)
cmd = json.loads((P / "command.json").read_text())
cmd[cmd.index("--name") + 1] = "dflash-opt-build-ordered-hc-v2"
cmd[cmd.index(str(P) + ":/outputs:rw")] = str(O) + ":/outputs:rw"
(O / "command.json").write_text(json.dumps(cmd, indent=2))
(O / "source-hashes.json").write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (O / "ordered_hc_v2.cpp", O / "dense_fragment.hpp")}, indent=2))
with (O / "build.log").open("w") as output:
    subprocess.run(cmd, stdout=output, stderr=subprocess.STDOUT, check=True)
print("Built", O / "libb70_ordered_hc_v2.so", flush=True)
