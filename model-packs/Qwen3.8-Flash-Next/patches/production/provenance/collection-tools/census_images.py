"""Read image sources/metadata without devices, network, imports, or serving."""
from pathlib import Path
import hashlib
import json
import subprocess
import tarfile

W = Path(__file__).resolve().parents[1]
E = Path((W / 'release-prep/active-provenance-path.txt').read_text().strip())
SITE = 'opt/venv/lib/python3.12/site-packages'
CODE = r'''
from pathlib import Path
import sys,tarfile,json,hashlib,subprocess,importlib.metadata as md,io
roots=['/opt/venv/lib/python3.12/site-packages/vllm','/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels','/opt/b70-flashnext','/opt/b70-residency-shim']
inventory=[]; sources=[]
for root in roots:
 for p in sorted(Path(root).rglob('*')):
  if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.pyc':continue
  data=p.read_bytes(); rel=str(p).lstrip('/')
  inventory.append(dict(path=rel,sha256=hashlib.sha256(data).hexdigest(),size=len(data),mode=p.stat().st_mode & 0o777,symlink=str(p.readlink()) if p.is_symlink() else None))
  if '.so' not in p.name and p.name!='vllm-rs': sources.append(p)
metadata={'distributions':sorted([{'name':d.metadata['Name'],'version':d.version,'direct_url':d.read_text('direct_url.json')} for d in md.distributions()],key=lambda x:(x['name'] or '')),'python':sys.version,'git_head':subprocess.run(['git','-C','/workspace/vllm','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),'git_status':subprocess.run(['git','-C','/workspace/vllm','status','--porcelain'],capture_output=True,text=True).stdout,'gcc':subprocess.run(['gcc','--version'],capture_output=True,text=True).stdout,'ld':subprocess.run(['ld','--version'],capture_output=True,text=True).stdout}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as t:
 for name,data in [('inventory.json',json.dumps(inventory,indent=2).encode()),('metadata.json',json.dumps(metadata,indent=2).encode()),('dpkg-status.txt',Path('/var/lib/dpkg/status').read_bytes())]:
  item=tarfile.TarInfo(name);item.size=len(data);t.addfile(item,io.BytesIO(data))
 for p in sources:t.add(p,arcname=str(p).lstrip('/'),recursive=False)
'''

for info in json.loads((E / 'image-ancestry.json').read_text()):
    ident = info['Id'].split(':')[1]
    dest = E / 'snapshots' / ident
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / 'sources.tar'
    if not archive.exists():
        with archive.open('wb') as stream:
            subprocess.run(['docker','run','--rm','--pull','never','--network','none',
                            '--read-only','-e','LD_PRELOAD=','--entrypoint',
                            '/opt/venv/bin/python',info['Id'],'-c',CODE],
                           stdout=stream,check=True)
        with tarfile.open(archive) as t:
            t.extractall(dest, filter='data')
    print(info['RepoTags'], len(json.loads((dest/'inventory.json').read_text())), flush=True)
