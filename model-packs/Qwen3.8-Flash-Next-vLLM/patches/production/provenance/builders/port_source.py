"""Apply accepted local deltas to pinned pristine v0.33, retaining merge evidence."""
import hashlib, json, subprocess
from pathlib import Path
w=Path(__file__).resolve().parents[1]
d=Path('/home/wulf/Desktop/qwen3.8-flash-next-dflash2-v032')
u=Path('/home/wulf/Desktop/qwen3.8-flash-next-v033-recon/upstream-v033')
o=w/'source/vllm'
assert not o.exists()
subprocess.run(['git','clone','--local','--no-hardlinks',str(u),str(o)],check=True)
subprocess.run(['git','checkout','--detach','4ba787ec9959e566d3b587a796dff07673324da5'],cwd=o,check=True)
merge=w/'build/merge';merge.mkdir(parents=True,exist_ok=True)
results=[]
for x in json.loads((d/'release-prep/inventories/accepted-vllm-delta.json').read_text()):
    path=x['path']
    if path=='vllm/_version.py':continue
    accepted=(d/'release-prep/accepted-extract'/path).read_bytes()
    assert hashlib.sha256(accepted).hexdigest()==x['sha256']
    p=o/path;p.parent.mkdir(parents=True,exist_ok=True)
    base=subprocess.run(['git','show','5df63a7203cdb9328d824a27b682e77c7b9928be:'+path],cwd=u,capture_output=True)
    if base.returncode:
        assert not p.exists(),path
        p.write_bytes(accepted);status='added local module'
    else:
        pristine=p.read_bytes()
        if pristine==base.stdout:
            p.write_bytes(accepted);status='unchanged upstream, local delta'
        else:
            m=merge/path;m.parent.mkdir(parents=True,exist_ok=True)
            m.with_suffix('.accepted').write_bytes(accepted)
            m.with_suffix('.base').write_bytes(base.stdout)
            m.with_suffix('.upstream').write_bytes(pristine)
            r=subprocess.run(['git','merge-file','-p',str(m.with_suffix('.upstream')),str(m.with_suffix('.base')),str(m.with_suffix('.accepted'))],capture_output=True)
            p.write_bytes(r.stdout);status='merged' if r.returncode==0 else 'conflict'
    results.append({'path':path,'status':status})
(w/'evidence/source-initial-merge.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
