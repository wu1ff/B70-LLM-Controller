"""Build the expanded-campaign exact candidate on top of calconv-graphupd-v2.
TUNING CANDIDATE ONLY. Every change is env-gated and arithmetic-preserving:
  * B70_PLE_PARALLEL_GATHER: native parallel row gather from the same read-only PLE mmap.
  * B70_DFLASH_EXACT_BA: native chained-DPAS kernel for GDN in_proj_ba at 9..32 rows.
  * B70_DFLASH_HC_SPLIT32: existing ordered HC kernel with split 10 at exactly 32 rows.
usage: build_expanded_v3.py NAME"""
import argparse, hashlib, json, shutil, subprocess
from pathlib import Path

p = argparse.ArgumentParser(); p.add_argument('name'); a = p.parse_args()
W = Path(__file__).resolve().parents[2]
C = W / 'astra/candidates' / a.name; C.mkdir(exist_ok=False)
BASE = 'sha256:bc673822914e6137807cc5e87e6b8979195774b704e9f222a76bc9619e4e770b'
TAG = 'local/dflash2-v033:calconv-graphupd-v2'
assert json.loads(subprocess.check_output(['docker', 'image', 'inspect', TAG]))[0]['Id'] == BASE
BP = W / 'astra/candidates/calconv-graphupd-v2/profile.json'; pb = BP.read_bytes()
assert hashlib.sha256(pb).hexdigest() == 'cf1ae34d08d9ed04ae0b279ee1ae12502a652bf69b3276f5de204f27dd127c12'
T = W / 'astra/candidates/expanded-tails'
K = W / 'astra/candidates/exact-kernels-v1'
G = W / 'astra/evidence/expanded-20261116T000816Z/dev/ple_gather'


def image_file(path):
    return subprocess.check_output(['docker', 'run', '--rm', '--pull', 'never', '--network', 'none', '-e', 'LD_PRELOAD=',
                                    '--entrypoint', '/bin/cat', BASE, path])


SITE = 'opt/venv/lib/python3.12/site-packages'
originals = {
    'vllm/models/qwen4_exp/nvidia/b70_hc_gemm.py': 'a1471cffc4045a5222c775a4c93c45732d3ec5db91af761bedecf6a43fa37148',
    'vllm/models/qwen4_exp/common/disk_ple.py': '8332cfab7e3c2a0d2ea885ab0eb1040e2c91b48b742675c0270826d2cafaccb4',
}
tails = {
    'vllm/models/qwen4_exp/nvidia/b70_hc_gemm.py': T / 'b70_hc_gemm_tail.py',
    'vllm/models/qwen4_exp/common/disk_ple.py': T / 'disk_ple_tail.py',
}
manifest = []


def put(dest_rel, data, also_workspace=False):
    targets = [SITE + '/' + dest_rel] + (['workspace/vllm/' + dest_rel] if also_workspace else [])
    for t in targets:
        d = C / 'overlay' / t; d.parent.mkdir(parents=True, exist_ok=True); d.write_bytes(data)
    manifest.append(dict(path=dest_rel, targets=targets, sha256=hashlib.sha256(data).hexdigest()))


for rel, digest in originals.items():
    src = image_file('/' + SITE + '/' + rel)
    assert hashlib.sha256(src).hexdigest() == digest, rel
    tail = tails[rel].read_bytes()
    out = src + tail
    compile(out.decode(), rel, 'exec')
    put(rel, out, also_workspace=rel.endswith('b70_hc_gemm.py'))
put('vllm_xpu_kernels/libb70_exact_v1.so', (K / 'libb70_exact_v1.so').read_bytes())
put('vllm_xpu_kernels/libple_gather.so', (G / 'libple_gather.so').read_bytes())

profile = json.loads(pb)
profile['env']['B70_PLE_PARALLEL_GATHER'] = '8'
profile['env']['B70_DFLASH_EXACT_BA'] = '1'
profile['env']['B70_DFLASH_HC_SPLIT32'] = '10'
profile['optimization'] += (' EXPANDED CANDIDATE v3: native parallel PLE row gather (same mmap bytes);'
                            ' exact native chained-DPAS GDN in_proj_ba at 9..32 rows; ordered HC split 10 at 32 rows'
                            ' (bit-identical partials and sums); target/draft arithmetic unchanged.')
(C / 'profile.json').write_text(json.dumps(profile, indent=2) + '\n')
(C / 'Dockerfile').write_text('FROM ' + TAG + '\nCOPY overlay/ /\n')
src_files = [T / 'b70_hc_gemm_tail.py', T / 'disk_ple_tail.py', K / 'exact_gemm.cpp', K / 'dense_fragment.hpp',
             K / 'build.sh', K / 'command.json', G / 'ple_gather.c']
(C / 'source-manifest.json').write_text(json.dumps(dict(
    base=BASE, base_tag=TAG, base_profile_sha256=hashlib.sha256(pb).hexdigest(), files=manifest,
    sources={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in src_files},
    builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    profile_sha256=hashlib.sha256((C / 'profile.json').read_bytes()).hexdigest(),
    status='TUNING CANDIDATE, NOT PRODUCTION-QUALIFIED'), indent=2))
with (C / 'build.log').open('w') as f:
    subprocess.run(['docker', 'build', '--network=none', '-t', 'local/dflash2-v033:' + a.name, str(C)],
                   stdout=f, stderr=subprocess.STDOUT, check=True)
info = json.loads(subprocess.check_output(['docker', 'image', 'inspect', 'local/dflash2-v033:' + a.name]))[0]
(C / 'image.json').write_text(json.dumps(info, indent=2)); print(info['Id'])
