"""Build expanded2 candidate v6 = v5 + MoE-glue egate arithmetic correction (b70_exact_moe_v3: measured
oneDNN N=1 order, parallel step partials) + 64-byte alignment guard. TUNING CANDIDATE ONLY.
usage: build_expanded2_v6.py NAME"""
import argparse, hashlib, json, subprocess
from pathlib import Path

p = argparse.ArgumentParser(); p.add_argument('name'); a = p.parse_args()
W = Path(__file__).resolve().parents[2]
C = W / 'astra/candidates' / a.name; C.mkdir(exist_ok=False)
BASE = 'sha256:848f48ff831ff244154bab6ef2938093bb0a588b2e2bb643a51c8a57b920789b'
TAG = 'local/dflash2-v033:expanded-exact-v5'
assert json.loads(subprocess.check_output(['docker', 'image', 'inspect', TAG]))[0]['Id'] == BASE
BP = W / 'astra/candidates/expanded-exact-v5/profile.json'; pb = BP.read_bytes()
assert hashlib.sha256(pb).hexdigest() == 'd893c5b4e170d080a5e7321cc546a2f1cfa540e75567f1338e384e7121b842ff'
T = W / 'astra/candidates/expanded2-tails'
K3 = W / 'astra/candidates/exact-kernels-v3'
SITE = 'opt/venv/lib/python3.12/site-packages'
manifest = []


def image_file(path):
    return subprocess.check_output(['docker', 'run', '--rm', '--pull', 'never', '--network', 'none', '-e', 'LD_PRELOAD=',
                                    '--entrypoint', '/bin/cat', BASE, path])


def put(rel, data, py=True):
    if py:
        compile(data.decode(), rel, 'exec')
    d = C / 'overlay' / SITE / rel; d.parent.mkdir(parents=True, exist_ok=True); d.write_bytes(data)
    manifest.append(dict(path=SITE + '/' + rel, sha256=hashlib.sha256(data).hexdigest()))


old_glue = image_file('/' + SITE + '/vllm/models/qwen4_exp/nvidia/b70_moe_glue.py')
assert hashlib.sha256(old_glue).hexdigest() == hashlib.sha256((W / 'astra/candidates/expanded-tails/b70_moe_glue.py').read_bytes()).hexdigest()
put('vllm/models/qwen4_exp/nvidia/b70_moe_glue.py', (T / 'b70_moe_glue.py').read_bytes())
lib = (K3 / 'libb70_exact_moe_v3.so').read_bytes()
assert hashlib.sha256(lib).hexdigest() == '20691aa32d8b61d9e1f907be92f0115960199a6a296ad14762396b218654aaaa'
put('vllm_xpu_kernels/libb70_exact_moe_v3.so', lib, py=False)
profile = json.loads(pb)
assert profile['env']['B70_DFLASH_MOE_GLUE'] == '1'
profile['optimization'] += (' v6 (expanded2): MoE-glue shared-expert gate logit now reproduces the measured oneDNN N=1 '
                            'arithmetic (v1 kernel differed by 1 BF16 ulp in ~2e-4 of rows); 16 step partials on 8 subgroups; '
                            '64-byte input alignment guard.')
(C / 'profile.json').write_text(json.dumps(profile, indent=2) + '\n')
(C / 'Dockerfile').write_text('FROM ' + TAG + '\nCOPY overlay/ /\n')
srcs = [T / 'b70_moe_glue.py', K3 / 'exact_moe_v3.cpp', K3 / 'dense_fragment.hpp', K3 / 'build_moe_v3.sh', K3 / 'command_moe_v3.json']
(C / 'source-manifest.json').write_text(json.dumps(dict(
    base=BASE, base_tag=TAG, base_profile_sha256=hashlib.sha256(pb).hexdigest(), files=manifest,
    sources={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in srcs},
    builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    profile_sha256=hashlib.sha256((C / 'profile.json').read_bytes()).hexdigest(),
    status='TUNING CANDIDATE, NOT PRODUCTION-QUALIFIED'), indent=2))
with (C / 'build.log').open('w') as f:
    subprocess.run(['docker', 'build', '--network=none', '-t', 'local/dflash2-v033:' + a.name, str(C)],
                   stdout=f, stderr=subprocess.STDOUT, check=True)
info = json.loads(subprocess.check_output(['docker', 'image', 'inspect', 'local/dflash2-v033:' + a.name]))[0]
(C / 'image.json').write_text(json.dumps(info, indent=2)); print(info['Id'])
