#!/usr/bin/env python3
"""Render the dynamic MTP5 launch; execute only with --execute."""
import argparse,json,shlex,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('mode',choices=['mtp5-dynamic']);p.add_argument('--model-dir',required=True);p.add_argument('--port',type=int,default=18000);p.add_argument('--bind',default='127.0.0.1');p.add_argument('--name');p.add_argument('--image',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
model=Path(a.model_dir).resolve();assert model.is_dir(),'Model directory missing'
assert 1<=a.port<=65535
s=json.loads((Path(__file__).resolve().parent/'profiles'/f'{a.mode}.json').read_text())
cmd=['docker','run','-d','--name',a.name or f'qwen38-flashnext-disk-{a.mode}','--publish',f'{a.bind}:{a.port}:8000',*s['docker_options']]
for m in s['mounts']:cmd+=['--mount',m.format(model_dir=str(model))]
for k,v in s['env'].items():cmd+=['--env',k+'='+v]
cmd += [a.image,*s['command']]
print(shlex.join(cmd))
if a.execute:subprocess.run(cmd,check=True)
