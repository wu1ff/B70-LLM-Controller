import json
from pathlib import Path
import sys
import torch

p = Path("/work/phase3-audit")
version = sys.argv[1]
branch = sys.argv[2] if len(sys.argv) > 2 else "first"
a = torch.load(
    p / f"phase3-serial-{branch}-{version}-probe-rank-0.pt", weights_only=False
)
b = torch.load(
    p / f"phase3-vector-{branch}-{version}-probe-rank-0.pt", weights_only=False
)
records = []
for name, rows in a.items():
    if name not in b:
        continue
    for part in range(len(rows[0])):
        try:
            x = torch.cat([v[part] for v in rows])
            y = torch.cat([v[part] for v in b[name]])
        except Exception:
            continue
        if x.shape != y.shape:
            continue
        d = x.float() - y.float()
        records.append(
            dict(
                name=name,
                part=part,
                equal=torch.equal(x, y),
                relative_l2=float(d.norm() / x.float().norm().clamp_min(1e-10)),
                max_abs=float(d.abs().max()),
                row0_equal=torch.equal(x[0], y[0]),
                row0_max_abs=float(d[0].abs().max()),
            )
        )
(p / f"probe-{branch}-{version}.json").write_text(json.dumps(records, indent=2))
print(json.dumps(records[:65], indent=2))
