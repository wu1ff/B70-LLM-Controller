import hashlib
import json
from pathlib import Path

records=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:
        maps=(proc/'maps').read_text()
        if 'model-00016-of-00017.safetensors' not in maps:continue
        status={k:v.strip() for line in (proc/'status').read_text().splitlines() if ':' in line for k,v in [line.split(':',1)]}
        records.append({'pid':int(proc.name),'cmdline':(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'),
                        'status':{k:status[k] for k in ('Name','VmRSS','VmLck','VmPin','RssAnon','RssFile','RssShmem','Threads') if k in status},
                        'ple_maps':[line for line in maps.splitlines() if 'model-00016-of-00017.safetensors' in line]})
    except (FileNotFoundError,PermissionError,ProcessLookupError):pass
base=Path('/opt/venv/lib/python3.12/site-packages')
paths=[base/'vllm_xpu_kernels/libgdn_index64.so',Path('/opt/b70-residency-shim/libl0_peer_residency_shim.so')]
print(json.dumps({'workers':records,'native_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}},indent=2))
