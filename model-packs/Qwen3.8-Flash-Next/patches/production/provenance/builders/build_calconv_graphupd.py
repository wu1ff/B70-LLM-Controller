"""Layer the exact compiled target-graph input update on frozen Cal+Conv; base untouched.
TUNING CANDIDATE ONLY. Same arithmetic, same checks/copies; host submission change."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--fused-copy',action='store_true');a=p.parse_args()
W=Path(__file__).resolve().parents[2];C=W/'astra/candidates'/a.name;C.mkdir(exist_ok=False)
BASE='sha256:f5ac30152869ae3a0fd7c619812d645031ffa67bbb92c89235ac35b4d490da23';TAG='local/dflash2-v033:bf16-calconv-frozen-v1'
assert json.loads(subprocess.check_output(['docker','image','inspect',TAG]))[0]['Id']==BASE
BP=W/'astra/candidates/bf16-calconv-frozen-v1/profile.json';pb=BP.read_bytes()
assert hashlib.sha256(pb).hexdigest()=='14ce587ad4dd3b9be7e82c90c953e008671c7edc4aa9bfd3b61cb265f70f30e2'
REF=W/'astra/candidates/bf16-reference-v1/overlay/opt/venv/lib/python3.12/site-packages/vllm'
rel='models/qwen4_exp/nvidia/b70_target_graph.py';src=(REF/rel).read_text()
assert hashlib.sha256(src.encode()).hexdigest()=='25bb190e041ac5b077824532b65979004043971eb7e92961caf6cb8c817060eb'
profile=json.loads(pb);manifest=[]
def put(rel,data):
 if isinstance(data,str):compile(data,rel,'exec');data=data.encode()
 for prefix in ['opt/venv/lib/python3.12/site-packages/vllm','workspace/vllm/vllm']:
  dest=C/'overlay'/prefix/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
 manifest.append(dict(path=rel,sha256=hashlib.sha256(data).hexdigest()))
tail='''

if os.getenv("B70_DFLASH_COMPILED_GRAPH_UPDATE", "0") == "1":
    # Exact: same walk order, checks, exception types and copied pairs; the
    # static destination tree is compiled once per captured entry.
    from .b70_compiled_graph_update import update as _b70_compiled_update
    _b70_original_update = update
    update = _b70_compiled_update
'''
assert src.endswith('\n') and 'B70_DFLASH_COMPILED_GRAPH_UPDATE' not in src
put(rel,src+tail)
put('models/qwen4_exp/nvidia/b70_compiled_graph_update.py',(W/'astra/candidates/compiled_graph_update.py').read_text())
profile['env']['B70_DFLASH_COMPILED_GRAPH_UPDATE']='1'
profile['env']['B70_DFLASH_FUSED_GRAPH_COPY']='1' if a.fused_copy else '0'
profile['optimization']=profile['optimization']+' TUNING CANDIDATE: exact compiled target-graph input update'+(' with single-launch byte copy of graph inputs' if a.fused_copy else '')+'; target/draft arithmetic unchanged.'
(C/'profile.json').write_text(json.dumps(profile,indent=2)+'\n');(C/'Dockerfile').write_text('FROM '+TAG+'\nCOPY overlay/ /\n')
(C/'source-manifest.json').write_text(json.dumps(dict(base=BASE,base_tag=TAG,base_profile_sha256=hashlib.sha256(pb).hexdigest(),files=manifest,fused_copy=a.fused_copy,builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),profile_sha256=hashlib.sha256((C/'profile.json').read_bytes()).hexdigest(),status='TUNING CANDIDATE, NOT PRODUCTION-QUALIFIED'),indent=2))
with (C/'build.log').open('w') as f:subprocess.run(['docker','build','--network=none','-t','local/dflash2-v033:'+a.name,str(C)],stdout=f,stderr=subprocess.STDOUT,check=True)
info=json.loads(subprocess.check_output(['docker','image','inspect','local/dflash2-v033:'+a.name]))[0];(C/'image.json').write_text(json.dumps(info,indent=2));print(info['Id'])
