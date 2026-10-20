#!/usr/bin/env python3
"""Verify exact final installed inventory and every distributed overlay."""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('image');p.add_argument('--output',required=True);args=p.parse_args()
root=Path(__file__).resolve().parent
program='''import hashlib,json
from pathlib import Path
paths=[]
for r in ['/opt/venv/lib/python3.12/site-packages/vllm','/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels','/opt/b70-flashnext','/opt/b70-residency-shim']:
 paths.extend(str(p) for p in Path(r).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
print(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(paths)}))
'''
a=json.loads((root/'manifests/accepted.json').read_text())
notices={}
for name,item in json.loads((root/'manifests/overlay.json').read_text()).items():
 data=(root/item['package']).read_bytes();actual=hashlib.sha256(data).hexdigest()
 if actual != item['sha256'] or a[name]!=actual:raise ValueError('Final overlay lock mismatch: '+name)
 if data.startswith(b'# Modified by Wulf for B70 Flash-Next Disk; upstream terms are retained.\n'):notices[name]=actual
b=json.loads(subprocess.check_output(['docker','run','--rm','--network=none','--entrypoint','/opt/venv/bin/python',args.image,'-B','-c',program]))
out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.with_name(out.stem+'-installed.json').write_text(json.dumps(b,indent=2)+'\n')
d={'image':args.image,'accepted_entries':len(a),'rebuilt_entries':len(b),'required_notice_files':len(notices),'exact_matches':sum(b.get(k)==v for k,v in a.items()),'missing':sorted(a.keys()-b.keys()),'extra':sorted(b.keys()-a.keys()),'nonmatching':{k:{'accepted':v,'rebuilt':b[k]} for k,v in a.items() if k in b and b[k]!=v},'exact':a==b};out.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d));sys.exit(0 if d['exact'] else 1)
