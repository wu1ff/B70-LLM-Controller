"""Focused native partial-state and >INT32 element-offset regressions."""
import json
from pathlib import Path
import torch
import vllm_xpu_kernels._xpu_C
import vllm_xpu_kernels
from oracle_gdn import fixture,reference

torch.set_num_threads(2)
torch.ops.load_library(str(Path(vllm_xpu_kernels.__file__).with_name('libgdn_index64.so')))
op=torch.ops._gdn_index64
results=[]

def run(widths,accepted,large=False):
    f,actual=fixture(widths,accepted)
    expected=reference(**f)
    dev=lambda x: x.to('xpu')
    ids=torch.tensor(f['ids'],dtype=torch.int32)
    conv=dev(f['conv'])
    if large:
        # The stride fits INT32; its product with the selected slot does not.
        per_slot=conv[0].numel()
        slot=(1<<31)//per_slot+2
        big=torch.empty((slot+2,*conv.shape[1:]),dtype=conv.dtype,device='xpu')
        big[slot].copy_(conv[1])
        big[slot-1].fill_(91)
        big[slot+1].fill_(91)
        conv=big
        ids[0,0]=slot
    z=torch.full((len(f['x']),f['nv'],f['d']),77.,dtype=torch.bfloat16,device='xpu')
    qsl=dev(torch.tensor(f['offsets'],dtype=torch.int32))
    token_map=dev(torch.tensor(f['token_map'],dtype=torch.int32))
    idsgpu=dev(ids)
    acceptedgpu=dev(torch.tensor(accepted,dtype=torch.int32))
    inter=op.causal_conv1d_spec(z,dev(f['x']),dev(f['ba']),16,48,128,128,
        conv,dev(f['w']),None,'silu',0,0,len(widths),qsl,token_map,idsgpu,acceptedgpu,actual,4,True)
    for got,want in zip(inter,expected[0]):
        torch.testing.assert_close(got.cpu(),want,atol=.0003,rtol=.02)
    torch.testing.assert_close(z.cpu(),expected[1],atol=0,rtol=0)
    if large:
        torch.testing.assert_close(conv[slot].cpu(),expected[3][1],rtol=0,atol=0)
        assert (conv[slot-1].cpu()==91).all() and (conv[slot+1].cpu()==91).all()
        results.append({'case':'large-conv-address','element_offset':slot*per_slot,
                        'allocated_bytes':conv.numel()*conv.element_size(),'passed':True})
    else:
        out=torch.full_like(z,77)
        ssm=dev(f['ssm'])
        op.gated_delta_rule_spec(out,*inter,48,128,dev(f['A']),dev(f['dt']),ssm,
            0,0,len(widths),qsl,token_map,idsgpu,acceptedgpu,actual,4)
        torch.testing.assert_close(out.cpu(),expected[2],atol=.0003,rtol=.02)
        torch.testing.assert_close(conv.cpu(),expected[3],rtol=0,atol=0)
        torch.testing.assert_close(ssm.cpu(),expected[4],atol=.00005,rtol=.015)
        results.append({'case':'partial-state','widths':widths,'accepted':accepted,'passed':True})
    torch.xpu.synchronize()

for widths,accepted in [([1],[4]),([3],[4]),([4],[1]),([2,4,1],[4,2,3])]:
    run(widths,accepted)
run([3],[4],large=True)
print(json.dumps({'passed':True,'cases':results},indent=2),flush=True)
