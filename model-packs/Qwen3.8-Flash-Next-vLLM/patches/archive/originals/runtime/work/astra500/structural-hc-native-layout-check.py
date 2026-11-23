"""Standalone loaded-checkpoint HC checks; run only with the service idle."""

from pathlib import Path
import json
import urllib.request

metrics = urllib.request.urlopen('http://127.0.0.1:8000/metrics',timeout=5).read().decode()
assert all(float(line.rsplit(' ',1)[1]) == 0 for line in metrics.splitlines()
           if line.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))), 'service is busy'

import torch
from safetensors import safe_open
from vllm.models.qwen4_exp.nvidia.ops.hc import hc_gate_mix

torch.set_num_threads(8)
torch.ops.load_library('/work/astra500/structural-native/libastra_hc_v3.so')
index = json.loads(Path('/models/model.safetensors.index.json').read_text())['weight_map']
names = ['model.language_model.layers.0.attn_hyper_connection.input_mix_weight_up.weight',
         'model.language_model.layers.24.attn_hyper_connection.input_mix_weight_up.weight',
         'model.language_model.hyper_connection_mixer.input_mix_weight_up.weight']
rows = []


def measure(fn):
    fn()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        for _ in range(8):
            value = fn()
    for _ in range(3): graph.replay()
    a,b = torch.xpu.Event(enable_timing=True),torch.xpu.Event(enable_timing=True)
    a.record()
    for _ in range(10):graph.replay()
    b.record();b.synchronize()
    return a.elapsed_time(b)/80


with torch.inference_mode():
    for name in names:
        with safe_open('/models/'+index[name],framework='pt',device='cpu') as checkpoint:
            cpu_weight = checkpoint.get_tensor(name)
        weight = cpu_weight.to('xpu')
        packed = weight.view(4,160,16,320).permute(1,0,2,3).contiguous().view(10240,320)
        assert torch.equal(packed.view(160,4,16,320).permute(1,0,2,3).reshape_as(weight),weight)
        for m in (1,2,4,7,8,32):
            torch.manual_seed(299+m)
            lora=torch.randn(m,320,device='xpu',dtype=torch.bfloat16)
            x=torch.randn(m,10240,device='xpu',dtype=torch.bfloat16)
            y=torch.empty(m,2560,device='xpu',dtype=torch.bfloat16)
            def base():return hc_gate_mix(x,torch.nn.functional.linear(lora,weight),4)
            def native():
                torch.ops.astra_hc_v3.up(lora,weight,x,y)
                return y
            expected=base()
            actual=native().clone()
            native()
            repeat_equal=torch.equal(actual,y)
            serial=torch.empty(1,2560,device='xpu',dtype=torch.bfloat16)
            torch.ops.astra_hc_v3.up(lora[-1:].contiguous(),weight,x[-1:].contiguous(),serial)
            gates=(lora[-1:].cpu().double()@cpu_weight.double().T).bfloat16().double()
            reference=(gates.sigmoid()*x[-1:].cpu().double()).reshape(1,4,2560).mean(1)
            errors={k:float((v[-1:].cpu().double()-reference).square().mean().sqrt()/reference.square().mean().sqrt())
                    for k,v in [('base',expected),('native',actual)]}
            row=dict(name=name,m=m,equal=torch.equal(expected,actual),repeat_equal=repeat_equal,
                     serial_equal=torch.equal(actual[-1:],serial),nrmse_fp64=errors,
                     max_abs=float((actual.float()-expected.float()).abs().max()),
                     baseline_ms=measure(base),native_ms=measure(native))
            rows.append(row)
            print(json.dumps(row),flush=True)
            Path('/work/astra500/structural-hc-native-layout.json').write_text(json.dumps(rows,indent=2))
