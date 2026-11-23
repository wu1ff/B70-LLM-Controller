"""Reassemble repaired DFlash from the pre-FP8 image, never an FP8 overlay.

All model arithmetic and proposal defaults are the protected BF16 implementation.
The existing, qualified EOS guard is copied byte-for-byte from repair-v2.
"""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(json.loads((ROOT / 'astra/current-bf16-campaign.json').read_text())['evidence'])
OUT = ROOT / 'astra/candidates/bf16-reference-v1'
BASE = 'sha256:e0c6ee4bdb33bc073a82b86c050aedfa2c93703c6ed80f772b873cafb0b791c6'
BASE_TAG = 'local/dflash2-v033:opt-query-ordered-hc-v3'
TAG = 'local/dflash2-v033:bf16-reference-v1'
assert not OUT.exists()
OUT.mkdir()
base_info = json.loads(subprocess.check_output(['docker', 'image', 'inspect', BASE]))[0]
assert json.loads(subprocess.check_output(['docker', 'image', 'inspect', BASE_TAG]))[0]['Id'] == BASE
manifest = []
# Keep repaired shape handling and score configuration, but exclude every
# experimental numeric dispatch, quantizer, native INT8 payload and fitted weight.
files = [
    'models/qwen4_exp/nvidia/b70_target_graph.py',
    'models/qwen4_exp/nvidia/c1_verify.py',
    'v1/attention/backends/gdn_attn.py',
    'v1/worker/gpu/sample/sampler.py',
    'v1/worker/gpu/sample/thinking_budget.py',
    'v1/worker/gpu/spec_decode/rejection_sampler_utils.py',
    'v1/worker/gpu/spec_decode/dflash2/speculator.py',
    'v1/worker/gpu/spec_decode/dflash/speculator.py',
]
for rel in files:
    data = (EVIDENCE / 'repaired/vllm' / rel).read_bytes()
    assert b'fp8' not in data.lower() and b'float8' not in data.lower()
    compile(data, rel, 'exec')
    for prefix in ['opt/venv/lib/python3.12/site-packages/vllm', 'workspace/vllm/vllm']:
        dest = OUT / 'overlay' / prefix / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    manifest.append(dict(path=rel, sha256=hashlib.sha256(data).hexdigest()))
profile = json.loads((ROOT / 'astra/candidates/conversation-repair-v2/profile.json').read_text())
for key in list(profile['env']):
    if 'FP8' in key or key in ['ASTRA_DFLASH_CYCLIC_W4A8', 'ASTRA_DFLASH_ACCEPT_ALPHA']:
        del profile['env'][key]
profile['optimization'] = ('Clean assembly from protected e0c6: original W4A16 target, BF16 draft/head and original selector; '
    'STANDARD acceptance, scale1, K7/query8/support16, canonical reductions, qualified HC20/cyclic W4A16, '
    'disk PLE, shared GDN and target/query graphs; byte-identical repair-v2 EOS guard. No campaign FP8 ancestry or payload. Qualification pending.')
(OUT / 'profile.json').write_text(json.dumps(profile, indent=2) + '\n')
(OUT / 'Dockerfile').write_text('FROM ' + BASE_TAG + '\nCOPY overlay/ /\n')
(OUT / 'source-manifest.json').write_text(json.dumps(dict(base=BASE, base_layers=base_info['RootFS']['Layers'], files=manifest,
    exclusions=['private FP8 head', 'attention input/output FP8 payload', 'learned selector weights/loader', 'experimental INT8 expert payload'],
    inherited_originals=['model_runner.py', 'qwen3_dflash2.py', 'astra_ownership.py']), indent=2))
with (OUT / 'build.log').open('w') as log:
    subprocess.run(['docker', 'build', '--network=none', '-t', TAG, str(OUT)], check=True, stdout=log, stderr=subprocess.STDOUT)
info = json.loads(subprocess.check_output(['docker', 'image', 'inspect', TAG]))[0]
assert info['RootFS']['Layers'][:-1] == base_info['RootFS']['Layers']
(OUT / 'image.json').write_text(json.dumps(info, indent=2))
print(info['Id'], flush=True)
