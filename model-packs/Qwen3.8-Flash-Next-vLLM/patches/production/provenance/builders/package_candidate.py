import hashlib,json,shutil,subprocess,argparse
parser=argparse.ArgumentParser();parser.add_argument("--fused-gdn",action="store_true");args=parser.parse_args()
from pathlib import Path
w=Path(__file__).resolve().parents[1];v=w/'source/vllm';root=w/'build/context/overlay';root.mkdir(parents=True,exist_ok=True)
site=root/'opt/venv/lib/python3.12/site-packages';site.mkdir(parents=True,exist_ok=True)
paths=sorted(set(subprocess.check_output(['git','diff','--name-only','4ba787ec9959e566d3b587a796dff07673324da5'],cwd=v,text=True).splitlines()) | set(subprocess.check_output(['git','ls-files','-m','-o','--exclude-standard'],cwd=v,text=True).splitlines()))
for rel in paths:
    if not rel.startswith('vllm/') or not rel.endswith('.py'):continue
    dest=site/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(v/rel,dest)
shutil.copytree(w/'source/kernel-overlay/vllm_xpu_kernels',site/'vllm_xpu_kernels',dirs_exist_ok=True)
p=Path('/home/wulf/Desktop/qwen3.8-flash-next-dflash2-v032/release-prep/publication-1.0.5/final-pack/patches/production')
for x in json.loads((p/'manifests/native-builds.json').read_text()):
    name=Path(x['installed_library']).name;src=w/'build/native'/('gdn-guard' if x['family']=='native' else x['family'])/name
    if not src.exists():raise ValueError('Build still missing: '+str(src))
    dest=root/('opt/'+x['installed_library'] if x['family']=='residency-shim' else 'opt/venv/lib/python3.12/site-packages/'+x['installed_library'])
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
assets=root/'opt/b70-flashnext';assets.mkdir(parents=True,exist_ok=True)
shutil.copytree(w/'source/assets',assets/'assets',dirs_exist_ok=True)
s=(w/'source/prepare_intel.py').read_text();a=s.index('LAYOUT = (');b=s.index('\n\n\ndef read_locked_json',a)
s=s[:a]+"LAYOUT = Path('/opt/venv/lib/python3.12/site-packages/vllm/models/qwen4_exp/intel_checkpoint.py')"+s[b:];(assets/'prepare-intel.py').write_text(s)
shutil.copyfile('/home/wulf/Desktop/Qfn-Mtp-vllm33/source/packaging/prepare-serve.sh',assets/'prepare-serve.sh');(assets/'prepare-serve.sh').chmod(0o755)
profile=json.loads((w/'evidence/host/b70ctl-before.json').read_text())[0]['Config']
env={x.split('=',1)[0]:x.split('=',1)[1] for x in profile['Env']};env.pop('PYTHONPATH',None);env['VLLM_XPU_INC_WNA16_BACKEND']='w4a16';env['TMPDIR']='/work/tmp';env['B70_DFLASH_LIVE_GRAPH_CHECKS']='1'
# Current vendor foundation; retain its defaults except explicit DFlash contract.
if args.fused_gdn: env['B70_DFLASH_FUSED_GDN_METADATA']='1'
keep=['B70_','CCL_','HF_','TRANSFORMERS_','TRITON_','VLLM_','XDG_','ZE_','LD_PRELOAD','TMPDIR','PLE_']
env={k:x for k,x in env.items() if any(k.startswith(y) for y in keep)}
cmd=profile['Cmd'];i=cmd.index('--num-gpu-blocks-override');cmd=cmd[:i]+cmd[i+2:]
result={'foundation':'sha256:010bdc441c72ee222f459d3b688bb41317525f1bd7223afe6218490b0c526832','vllm_commit':'4ba787ec9959e566d3b587a796dff07673324da5','env':env,'command':cmd,'kv_blocks':'automatic first measurement; 7785 historical only'}
(w/'source/profile.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['FROM local/dflash2-v033:vendor-exact','COPY overlay/ /']
for k,x in env.items():lines.append('ENV '+k+'='+json.dumps(x))
lines+=['WORKDIR /opt/venv','ENTRYPOINT ["/opt/b70-flashnext/prepare-serve.sh"]','CMD '+json.dumps(cmd)]
(w/'build/context/Dockerfile').write_text('\n'.join(lines)+'\n')
files=[{'path':'/'+str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(root.rglob('*')) if p.is_file()]
(w/'evidence/candidate-payload.json').write_text(json.dumps(files,indent=2)+'\n');print('packaged files',len(files))
