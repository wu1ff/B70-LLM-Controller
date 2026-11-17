"""Restore the original GDN reuse route in an isolated B1-derived candidate."""
import hashlib
import json
from pathlib import Path
import subprocess

w = Path(__file__).resolve().parents[1]
context = w/'build/parity-shared-context'
context.mkdir(exist_ok=False)
rel = Path('vllm/v1/attention/backends/gdn_attn.py')
original = (w/'evidence/parity/runtime-source-v033'/rel).read_text()
old = '''            and device.type == "cuda"
            # Not isinstance: KDA's RecoverSSM/checkpoint metadata is per group.
'''
new = '''            and (
                device.type == "cuda"
                or (
                    device.type == "xpu"
                    and os.getenv("B70_DFLASH_SHARED_GDN_METADATA", "0") == "1"
                    and self.speculative_config is not None
                    and self.speculative_config.method == "dflash"
                    and self.speculative_config.draft_model_config.architectures
                    == ["DFlash2DraftModel"]
                )
            )
            # Not isinstance: KDA's RecoverSSM/checkpoint metadata is per group.
'''
assert original.count(old) == 1
source = original.replace(old, new).replace(
    "# Opts into MRV2's CUDA-only aligned-index precompute.",
    "# Original DFlash2 XPU path: one all-group aligned-index launch.")
destination = context/'overlay/opt/venv/lib/python3.12/site-packages'/rel
destination.parent.mkdir(parents=True)
destination.write_text(source)
parent = 'sha256:ce670e4345c5476cc884052c90f3c1d3d802d6e0187d8b1f786612914a363071'
parent_ref = 'local/dflash2-v033:metadata-b1'
assert json.loads(subprocess.check_output(['docker', 'image', 'inspect', parent_ref]))[0]['Id'] == parent
(context/'Dockerfile').write_text(
    f'FROM {parent_ref}\nCOPY overlay/ /\nENV B70_DFLASH_SHARED_GDN_METADATA="1"\n')
manifest = dict(parent=parent, path=str(rel),
                before_sha256=hashlib.sha256(original.encode()).hexdigest(),
                after_sha256=hashlib.sha256(source.encode()).hexdigest(),
                scope='Original metadata sharing and all-group index gather, gated to XPU DFlash2.')
(w/'evidence/parity/shared-candidate-source.json').write_text(json.dumps(manifest, indent=2))
subprocess.run(['docker', 'build', '--network', 'none', '--pull=false', '-t',
                'local/dflash2-v033:parity-shared-gdn', str(context)], check=True)
(w/'evidence/parity/shared-candidate-image.json').write_bytes(
    subprocess.check_output(['docker', 'image', 'inspect', 'local/dflash2-v033:parity-shared-gdn']))
