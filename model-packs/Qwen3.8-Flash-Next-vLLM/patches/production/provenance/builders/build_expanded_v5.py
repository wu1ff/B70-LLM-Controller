"""Build expanded candidate v5 = v4 + deferred Disk-PLE prepare (host order only) + defensive
MoE-glue stash clear at block entry. TUNING CANDIDATE ONLY. usage: build_expanded_v5.py NAME"""
import argparse, hashlib, json, subprocess
from pathlib import Path

p = argparse.ArgumentParser(); p.add_argument('name'); a = p.parse_args()
W = Path(__file__).resolve().parents[2]
C = W / 'astra/candidates' / a.name; C.mkdir(exist_ok=False)
BASE = 'sha256:761f4e425466133c13a53198451177e64e9a08a97759655031c7cf96a00fd089'
TAG = 'local/dflash2-v033:expanded-exact-v4'
assert json.loads(subprocess.check_output(['docker', 'image', 'inspect', TAG]))[0]['Id'] == BASE
BP = W / 'astra/candidates/expanded-exact-v4/profile.json'; pb = BP.read_bytes()
assert hashlib.sha256(pb).hexdigest() == 'b7d2eef6fbb07ae5f661cb2b2976ec275d3ef3e132d7dbe749062a804201223f'
T = W / 'astra/candidates/expanded-tails'
SITE = 'opt/venv/lib/python3.12/site-packages'
manifest = []


def image_file(path):
    return subprocess.check_output(['docker', 'run', '--rm', '--pull', 'never', '--network', 'none', '-e', 'LD_PRELOAD=',
                                    '--entrypoint', '/bin/cat', BASE, path])


def put(rel, data):
    compile(data.decode(), rel, 'exec')
    d = C / 'overlay' / SITE / rel; d.parent.mkdir(parents=True, exist_ok=True); d.write_bytes(data)
    manifest.append(dict(path=SITE + '/' + rel, sha256=hashlib.sha256(data).hexdigest()))


rel = 'vllm/models/qwen4_exp/nvidia/model_state.py'
src = image_file('/' + SITE + '/' + rel)
assert hashlib.sha256(src).hexdigest() == '74a8213bc1c6250483bf5f4b4de2041b1dc3587fc6e31f45690da9e0a4fd9958'
put(rel, src + (T / 'model_state_tail.py').read_bytes())
put('vllm/models/qwen4_exp/nvidia/b70_defer_ple.py', (T / 'b70_defer_ple.py').read_bytes())
old_glue = image_file('/' + SITE + '/vllm/models/qwen4_exp/nvidia/b70_moe_glue.py')
assert hashlib.sha256(old_glue).hexdigest() == '398fae1127a82cc622e2b09af472a0b1e0864a442b0405a3f6c7e86922104bbe'
put('vllm/models/qwen4_exp/nvidia/b70_moe_glue.py', (T / 'b70_moe_glue.py').read_bytes())
profile = json.loads(pb)
profile['env']['B70_DFLASH_DEFER_PLE'] = '1'
profile['optimization'] += ' v5: C1-verify Disk-PLE prepare deferred past the target-graph input update (host order only; same device ops/stream); MoE-glue stash cleared at block entry.'
(C / 'profile.json').write_text(json.dumps(profile, indent=2) + '\n')
(C / 'Dockerfile').write_text('FROM ' + TAG + '\nCOPY overlay/ /\n')
srcs = [T / 'model_state_tail.py', T / 'b70_defer_ple.py', T / 'b70_moe_glue.py']
(C / 'source-manifest.json').write_text(json.dumps(dict(
    base=BASE, base_tag=TAG, base_profile_sha256=hashlib.sha256(pb).hexdigest(), files=manifest,
    sources={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in srcs},
    note='workspace/vllm source-tree model_state.py differs from site-packages in the base image and is left unchanged',
    builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    profile_sha256=hashlib.sha256((C / 'profile.json').read_bytes()).hexdigest(),
    status='TUNING CANDIDATE, NOT PRODUCTION-QUALIFIED'), indent=2))
with (C / 'build.log').open('w') as f:
    subprocess.run(['docker', 'build', '--network=none', '-t', 'local/dflash2-v033:' + a.name, str(C)],
                   stdout=f, stderr=subprocess.STDOUT, check=True)
info = json.loads(subprocess.check_output(['docker', 'image', 'inspect', 'local/dflash2-v033:' + a.name]))[0]
(C / 'image.json').write_text(json.dumps(info, indent=2)); print(info['Id'])
