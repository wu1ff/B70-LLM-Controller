"""Build expanded candidate v4 = v3 + exact fused MoE glue (env-gated, default off in code).
TUNING CANDIDATE ONLY.  usage: build_expanded_v4.py NAME"""
import argparse, hashlib, json, subprocess
from pathlib import Path

p = argparse.ArgumentParser(); p.add_argument('name'); a = p.parse_args()
W = Path(__file__).resolve().parents[2]
C = W / 'astra/candidates' / a.name; C.mkdir(exist_ok=False)
BASE = 'sha256:cf2f4b846bd7b1c3843fd2739a68809d2bdc75cbc38d76c928012e960c9b1b14'
TAG = 'local/dflash2-v033:expanded-exact-v3'
assert json.loads(subprocess.check_output(['docker', 'image', 'inspect', TAG]))[0]['Id'] == BASE
BP = W / 'astra/candidates/expanded-exact-v3/profile.json'; pb = BP.read_bytes()
assert hashlib.sha256(pb).hexdigest() == '50f135523be3ae0b4c0ff4dab6baf9b41ecbe1c27128dfedecb8048792947c7c'
T = W / 'astra/candidates/expanded-tails'; K = W / 'astra/candidates/exact-kernels-v1'
SITE = 'opt/venv/lib/python3.12/site-packages'
manifest = []


def image_file(path):
    return subprocess.check_output(['docker', 'run', '--rm', '--pull', 'never', '--network', 'none', '-e', 'LD_PRELOAD=',
                                    '--entrypoint', '/bin/cat', BASE, path])


def put(rel, data):
    d = C / 'overlay' / SITE / rel; d.parent.mkdir(parents=True, exist_ok=True); d.write_bytes(data)
    manifest.append(dict(path=SITE + '/' + rel, sha256=hashlib.sha256(data).hexdigest()))


rel = 'vllm/models/qwen4_exp/nvidia/model.py'
src = image_file('/' + SITE + '/' + rel)
assert hashlib.sha256(src).hexdigest() == 'c101469b3d866adb0a6bf0f109f5187e3c502d2d57475b1fd873b9494b3a4332'
out = src + (T / 'model_tail.py').read_bytes(); compile(out.decode(), rel, 'exec'); put(rel, out)
glue = (T / 'b70_moe_glue.py').read_bytes(); compile(glue.decode(), 'b70_moe_glue.py', 'exec')
put('vllm/models/qwen4_exp/nvidia/b70_moe_glue.py', glue)
put('vllm_xpu_kernels/libb70_exact_moe_v1.so', (K / 'libb70_exact_moe_v1.so').read_bytes())
profile = json.loads(pb)
profile['env']['B70_DFLASH_MOE_GLUE'] = '1'
profile['optimization'] += ' v4: exact fused MoE glue (router+shared gate_up+silu_mul+expert gate; shared down+sigmoid gate) bit-identical to oneDNN/vllm ops.'
(C / 'profile.json').write_text(json.dumps(profile, indent=2) + '\n')
(C / 'Dockerfile').write_text('FROM ' + TAG + '\nCOPY overlay/ /\n')
srcs = [T / 'model_tail.py', T / 'b70_moe_glue.py', K / 'exact_moe.cpp', K / 'dense_fragment.hpp', K / 'build_moe.sh', K / 'command_moe.json']
(C / 'source-manifest.json').write_text(json.dumps(dict(
    base=BASE, base_tag=TAG, base_profile_sha256=hashlib.sha256(pb).hexdigest(), files=manifest,
    sources={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in srcs},
    note='workspace/vllm source-tree model.py differs from site-packages in the base image and is left unchanged',
    builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    profile_sha256=hashlib.sha256((C / 'profile.json').read_bytes()).hexdigest(),
    status='TUNING CANDIDATE, NOT PRODUCTION-QUALIFIED'), indent=2))
with (C / 'build.log').open('w') as f:
    subprocess.run(['docker', 'build', '--network=none', '-t', 'local/dflash2-v033:' + a.name, str(C)],
                   stdout=f, stderr=subprocess.STDOUT, check=True)
info = json.loads(subprocess.check_output(['docker', 'image', 'inspect', 'local/dflash2-v033:' + a.name]))[0]
(C / 'image.json').write_text(json.dumps(info, indent=2)); print(info['Id'])
