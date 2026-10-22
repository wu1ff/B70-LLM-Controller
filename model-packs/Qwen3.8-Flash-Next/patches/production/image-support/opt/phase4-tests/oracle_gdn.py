"""Independent sequential Torch oracle. No model loading or server control."""
import math
import torch
import torch.nn.functional as F

def projection(x,ba,nk,nv,d,reorder=True):
    if reorder:
        q,k,v,z=x.split([nk*d,nk*d,nv*d,nv*d],-1)
        b,a=ba.split(nv,-1)
    else:
        p=x.reshape(-1,nk,2*d+2*(nv//nk)*d)
        q,k,v,z=p.split([d,d,(nv//nk)*d,(nv//nk)*d],-1)
        q,k,v,z=[t.reshape(len(x),-1) for t in (q,k,v,z)]
        p=ba.reshape(-1,nk,2*(nv//nk));b,a=p.split(nv//nk,-1)
        b,a=b.reshape(-1,nv),a.reshape(-1,nv)
    return torch.cat([q,k,v],-1),z.reshape(-1,nv,d),b,a

def reference(x,ba,w,bias,conv,ssm,ids,accepted,offsets,token_map,A,dt,nk,nv,d,reorder=True):
    """Convolution by grouped conv; delta rule by matrix outer products.

    Returns all outputs and expected caches. Unprocessed state/padding remains
    exact. Internal recurrence stays FP32 between tokens, as the native kernel.
    """
    x,ba,w,conv,ssm,A,dt=[t.cpu().clone() for t in (x,ba,w,conv,ssm,A,dt)]
    bias=None if bias is None else bias.cpu().clone()
    qkv,z,b,a=projection(x,ba,nk,nv,d,reorder)
    cs,rs=conv.clone(),ssm.clone()
    out=torch.full((len(x),nv,d),77.,dtype=x.dtype)
    zout=torch.full_like(out,77.)
    qout=[];kout=[];vout=[];bout=[];aout=[]
    for row,(start,end) in enumerate(zip(offsets,offsets[1:])):
        length=end-start
        if not length:continue
        sel=accepted[row]-1
        hist=conv[ids[row][0],sel:sel+3].clone()
        assert len(hist)==3
        toks=token_map[start:end]
        series=torch.cat([hist,qkv[toks]],0)
        raw=F.conv1d(series.float().T[None],w.float()[:,None],None if bias is None else bias.float(),groups=w.shape[0])[0].T
        c=F.silu(raw).to(x.dtype)
        q,k,v=c.split([nk*d,nk*d,nv*d],-1)
        q,k,v=q.reshape(length,nk,d),k.reshape(length,nk,d),v.reshape(length,nv,d)
        bb,aa=b[toks],a[toks]
        qout.append(q);kout.append(k);vout.append(v);bout.append(bb);aout.append(aa)
        # All possible next acceptance windows need only this written prefix.
        cs[ids[row][0],:2+length]=series[1:]
        state=ssm[ids[row][sel]].float().clone()
        for t,glob in enumerate(toks):
            qq=q[t].float().repeat_interleave(nv//nk,0)
            kk=k[t].float().repeat_interleave(nv//nk,0)
            qq=qq/(qq.square().sum(-1,keepdim=True)+1e-6).sqrt()/math.sqrt(d)
            kk=kk/(kk.square().sum(-1,keepdim=True)+1e-6).sqrt()
            # dt_bias addition happens in FP32 after both BF16 loads.
            decay=(-A.float().exp()*F.softplus(aa[t].float()+dt.float())).exp()
            state=state*decay[:,None,None]
            residual=(v[t].float()-torch.einsum('hvk,hk->hv',state,kk))*bb[t].float().sigmoid()[:,None]
            state=state+residual[:,:,None]*kk[:,None,:]
            out[glob]=torch.einsum('hvk,hk->hv',state,qq).to(x.dtype)
            rs[ids[row][t]]=state.to(rs.dtype)
            zout[glob]=z[glob]
    inter=[torch.cat(t,0) for t in (qout,kout,vout,bout,aout)]
    return inter,zout,out,cs,rs

def fixture(widths,accepted,nk=4,nv=12,d=128,state_dtype=torch.float32,reorder=True,bias=False,padding=1):
    torch.manual_seed(832+sum(widths)*10+sum(accepted))
    n=sum(widths);slots=4*len(widths)+3
    offsets=[0]
    for l in widths:offsets.append(offsets[-1]+l)
    # Non-identity global map with gaps exercises packed vs global addressing.
    actual=n+2
    token_map=torch.randperm(actual)[:n].tolist()
    ids=[[1+4*r+i for i in range(4)] for r in range(len(widths))]
    rand=lambda shape:(torch.randn(shape)*.13).to(torch.bfloat16)
    conv=rand((slots,6,(2*nk+nv)*d));ssm=(torch.randn(slots,nv,d,d)*.01).to(state_dtype)
    x=rand((actual+padding,(2*nk+2*nv)*d));ba=rand((actual+padding,2*nv));w=rand(((2*nk+nv)*d,4))
    A=torch.randn(nv)*.1;dt=rand((nv,));cb=rand(((2*nk+nv)*d,)) if bias else None
    return dict(x=x,ba=ba,w=w,bias=cb,conv=conv,ssm=ssm,ids=ids,accepted=accepted,offsets=offsets,token_map=token_map,A=A,dt=dt,nk=nk,nv=nv,d=d,reorder=reorder),actual
