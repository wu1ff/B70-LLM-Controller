"""Freeze tested components on the qualified clean BF16 image; baseline untouched."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--calibration',action='store_true');p.add_argument('--conv',action='store_true');p.add_argument('--hc',action='store_true');a=p.parse_args()
W=Path(__file__).resolve().parents[2];C=W/'astra/candidates'/a.name;C.mkdir(exist_ok=False)
BASE='sha256:8ef67bb4a18c5ed47c61f756e011f8c84994c7dc678d09157e2f2428c27a37fe';TAG='local/dflash2-v033:bf16-reference-v1'
assert json.loads(subprocess.check_output(['docker','image','inspect',TAG]))[0]['Id']==BASE
PROTECTED=W/'astra/evidence/bf16-quality-20261113T060934Z-9bbd092b/protected/vllm'
REF=W/'astra/candidates/bf16-reference-v1/overlay/opt/venv/lib/python3.12/site-packages/vllm'
profile=json.loads((W/'astra/candidates/bf16-reference-v1/profile.json').read_text());manifest=[]
def put(rel,data):
 if isinstance(data,str):compile(data,rel,'exec');data=data.encode()
 for prefix in ['opt/venv/lib/python3.12/site-packages/vllm','workspace/vllm/vllm']:
  dest=C/'overlay'/prefix/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
 manifest.append(dict(path=rel,sha256=hashlib.sha256(data).hexdigest()))
if a.calibration:
 rel='v1/worker/gpu/spec_decode/dflash2/speculator.py';source=(REF/rel).read_text()
 init='''        self._quality_calibration = None
        if os.getenv("B70_DFLASH_QUALITY_CALIBRATION", "0") == "1":
            from pathlib import Path
            import hashlib
            path = Path(__file__).with_name("quality_calibration.pt")
            if hashlib.sha256(path.read_bytes()).hexdigest() != "PARAM_HASH":
                raise RuntimeError("Calibration artifact changed")
            assert self.num_speculative_steps == 7 and self.top_k == 16 and self._astra_score_scale == 1.0
            self._quality_calibration = torch.load(path, map_location="cpu", weights_only=True).to(device)
            assert self._quality_calibration.shape == (7, 2) and bool(torch.isfinite(self._quality_calibration).all())
'''
 params=W/'runtime/dflash-bf16-reference-20261113/proposal-calibration-v1/params.pt';data=params.read_bytes();init=init.replace('PARAM_HASH',hashlib.sha256(data).hexdigest())
 needle='        self.candidate_sampler = CandidateSampler('
 assert source.count(needle)==1;source=source.replace(needle,init+needle)
 old='''        return self.model.model.candidate_selector(
            candidate_ids, unary_logits, hidden_states, anchor_token_ids
        ) * self._astra_score_scale'''
 new='''        result = self.model.model.candidate_selector(
            candidate_ids, unary_logits, hidden_states, anchor_token_ids
        ) * self._astra_score_scale
        if self._quality_calibration is not None:
            u = unary_logits[:, :, None, :]
            params = self._quality_calibration
            result = u * params[:, 0][None, :, None, None] + (result-u) * params[:, 1][None, :, None, None]
        return result'''
 assert source.count(old)==1;put(rel,source.replace(old,new));put('v1/worker/gpu/spec_decode/dflash2/quality_calibration.pt',data)
 profile['env']['B70_DFLASH_QUALITY_CALIBRATION']='1'
if a.conv:
 rel='model_executor/models/qwen3_dflash2.py';source=(PROTECTED/rel).read_text()
 source=source.replace('import torch\n','import os\nimport torch\n',1)
 source=source.replace('def _grouped_conv(\n','_QUALITY_BF16_CONV = os.getenv("B70_DFLASH_QUALITY_CONV", "0") == "1"\n\ndef _grouped_conv(\n',1)
 needle='''    if hidden_states.is_cuda:
        return dflash2_grouped_conv(hidden_states, delta, base, block_size, group_size)'''
 replacement='''    if _QUALITY_BF16_CONV and hidden_states.device.type == "xpu":
        from .quality_bf16_conv import fused
        return fused(hidden_states, delta, base, block_size, num_groups, group_size, taps)
'''+needle
 assert source.count(needle)==1;put(rel,source.replace(needle,replacement))
 conv=(W/'astra/candidates/bf16_conv_fusion.py').read_text().split('@torch.inference_mode()\ndef probe')[0];put('model_executor/models/quality_bf16_conv.py',conv)
 profile['env']['B70_DFLASH_QUALITY_CONV']='1'
if a.hc:
 rel='models/qwen4_exp/nvidia/b70_hyperconnection.py';source=(PROTECTED/rel).read_text()
 source=source.replace('import torch\n','import os\nfrom pathlib import Path\nimport torch\n',1)
 needle='logger = init_logger(__name__)';replacement=needle+'''
_QUALITY_NATIVE_HC = os.getenv("B70_DFLASH_QUALITY_HC", "0") == "1"
if _QUALITY_NATIVE_HC:
    torch.ops.load_library(str(Path(__file__).with_name("libastra_hc_native_v1.so")))
''';assert source.count(needle)==1;source=source.replace(needle,replacement)
 helper='''    def _quality_native_finish(self, xn):
        # Changed reduction order. Original upmix retained through row8.
        w = (self.input_mix_weight_down_block_inject if self.use_combine else self.input_mix_weight_down).weight
        low = xn.new_empty(xn.shape[0], 320)
        injection = xn.new_empty(xn.shape[0], 4)
        out = xn.new_empty(xn.shape[0], 2560)
        torch.ops.astra_hc_native_v1.down(xn, w, low, injection, 4)
        if xn.shape[0] <= 8:
            out = self._mix_up(low, xn)
        else:
            torch.ops.astra_hc_native_v1.upmix(low, self.input_mix_weight_up.weight, xn, out)
        return out, injection if self.use_combine else None

'''
 needle='    def _mix_eager(\n';assert source.count(needle)==1;source=source.replace(needle,helper+needle)
 needle='''        if self.use_combine:
            # produce injection logits for combine'''
 replacement='''        if _QUALITY_NATIVE_HC and xn.device.type == "xpu" and 1 <= xn.shape[0] <= 16:
            block_input, injection = self._quality_native_finish(xn)
            return hidden_states, block_input, injection

'''+needle
 assert source.count(needle)==2;put(rel,source.replace(needle,replacement))
 put('models/qwen4_exp/nvidia/libastra_hc_native_v1.so',(W/'runtime/dflash-bf16-reference-20261113/libastra_hc_native_v1.so').read_bytes())
 profile['env']['B70_DFLASH_QUALITY_HC']='1'
profile['optimization']='Quality-gated components on frozen clean BF16 reference; STANDARD and exact accepted EOS guard retained. Calibration='+str(a.calibration)+', exact BF16 convolution='+str(a.conv)+', changed target HC numerics='+str(a.hc)+'.'
(C/'profile.json').write_text(json.dumps(profile,indent=2)+'\n');(C/'Dockerfile').write_text('FROM '+TAG+'\nCOPY overlay/ /\n')
(C/'source-manifest.json').write_text(json.dumps(dict(base=BASE,files=manifest,components=vars(a),builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),profile_sha256=hashlib.sha256((C/'profile.json').read_bytes()).hexdigest()),indent=2))
with (C/'build.log').open('w') as f:subprocess.run(['docker','build','--network=none','-t','local/dflash2-v033:'+a.name,str(C)],stdout=f,stderr=subprocess.STDOUT,check=True)
info=json.loads(subprocess.check_output(['docker','image','inspect','local/dflash2-v033:'+a.name]))[0];(C/'image.json').write_text(json.dumps(info,indent=2));print(info['Id'])
