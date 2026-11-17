# SPDX-License-Identifier: Apache-2.0
# Derived from vLLM's DFlash2 grouped convolution (LightSeek TokenSpeed PR #1399).
"""Fuse XPU draft convolution while preserving every original BF16 boundary."""
import statistics
import torch
from vllm.triton_utils import triton, tl

ACTION = 'component'

@triton.jit
def kernel(X,D,W,O,XS:tl.constexpr,DS:tl.constexpr,DT:tl.constexpr,WS:tl.constexpr,
           C:tl.constexpr,G:tl.constexpr,T:tl.constexpr,Q:tl.constexpr,B:tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)*B + tl.arange(0,B)
    group = col//G
    pos = row%Q
    coef = (tl.load(W+col,col<C,0).to(tl.float32)+tl.load(D+row*DS+group,col<C,0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    x = tl.load(X+row*XS+col,col<C,0).to(tl.float32)
    value = (coef*x).to(tl.bfloat16).to(tl.float32)
    for tap in tl.static_range(1,T):
        coef = (tl.load(W+tap*WS+col,col<C,0).to(tl.float32)+tl.load(D+row*DS+tap*DT+group,col<C,0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        x = tl.load(X+(row-tap)*XS+col,(col<C)&(row>=tap),0).to(tl.float32)
        product = (coef*x).to(tl.bfloat16).to(tl.float32)
        product = product*tl.where(pos>=tap,1.,0.)
        value = (value+product).to(tl.bfloat16).to(tl.float32)
    tl.store(O+row*C+col,value,col<C)

def fused(x,delta,base,block_size,num_groups,group_size,taps):
    assert x.dtype==delta.dtype==base.dtype==torch.bfloat16
    assert x.stride(1)==delta.stride(2)==base.stride(1)==1
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    if x.shape[0]:
        kernel[(x.shape[0],triton.cdiv(x.shape[1],512))](
            x,delta,base,out,x.stride(0),delta.stride(0),delta.stride(1),base.stride(0),
            x.shape[1],group_size,taps,block_size,512,enable_fp_fusion=False)
    return out

