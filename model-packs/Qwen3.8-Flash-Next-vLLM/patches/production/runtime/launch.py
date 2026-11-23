import argparse,json,re,subprocess,time
from pathlib import Path
w=Path(__file__).resolve().parents[1]
a=argparse.ArgumentParser();a.add_argument('image');a.add_argument('--name',required=True);a.add_argument('--port',type=int,default=18083);a.add_argument('--bind-host',choices=['127.0.0.1','0.0.0.0'],default='127.0.0.1');a.add_argument('--profile',type=Path,default=w/'source/profile.json');args=a.parse_args()
assert 1024<=args.port<=65535 and (args.port!=8000 or args.bind_host=='0.0.0.0')
active=subprocess.check_output(['docker','ps','--format','{{.Names}} {{json .HostConfig.Devices}}'],text=True) if False else subprocess.check_output(['docker','ps','-q'],text=True)
# CPU-only build containers may coexist. Every active container is inspected.
for cid in active.splitlines():
    c=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
    if c['HostConfig'].get('Devices'):raise RuntimeError('GPU container already active: '+c['Name'])
subprocess.run(['sudo','-n','xpu-smi','discovery'],check=True,stdout=(w/'evidence/host'/f'{args.name}-discovery.txt').open('w'))
owners=subprocess.check_output(['sudo','-n','xpu-smi','ps'],text=True)
(w/'evidence/host'/f'{args.name}-gpu-owners.txt').write_text(owners)
assert not re.search(r'^\s*\d+\s+\S+', owners, re.MULTILINE), 'GPU process already active; inspect owner receipt'
p=json.loads(args.profile.read_text());work=w/'runtime'/args.name;work.mkdir(parents=True,exist_ok=False)
cmd=['docker','run','-d','--pull','never','--name',args.name,'--device','/dev/dri','--ipc','private','--shm-size','32g','--security-opt','seccomp=unconfined','-p',f'{args.bind_host}:{args.port}:8000','-v','/mnt/models/intel-qwen38-flash-next-4c67bf686b7f/checkpoint:/models:ro','-v','/home/wulf/Desktop/qwen3.8-flash-next-dflash2-v032/runtime/draft:/draft:ro','-v',str(work)+':/work']
for k,v in p['env'].items():cmd+=['-e',k+'='+v]
for host_path, container_path in p.get('diagnostic_binds', {}).items():
    assert Path(host_path).is_file()
    cmd += ['--mount', f'type=bind,src={host_path},dst={container_path},readonly']
cmd += [args.image,*p['command']]
receipt={'time':time.time(),'profile':p,'image':json.loads(subprocess.check_output(['docker','image','inspect',args.image]))[0]['Id'],'command':cmd,'port':args.port,'name':args.name}
(w/'evidence'/f'{args.name}-launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
r=subprocess.run(cmd,check=True,capture_output=True,text=True);print(r.stdout.strip())
