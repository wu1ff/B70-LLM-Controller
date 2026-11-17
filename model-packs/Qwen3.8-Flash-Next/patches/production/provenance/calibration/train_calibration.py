# SPDX-License-Identifier: Apache-2.0
"""Fit position-specific unary/edge scales using disjoint independent data."""
import json,hashlib,copy,time
from pathlib import Path
import torch

torch.set_num_threads(2);torch.manual_seed(20261112)
out=Path('/work/proposal-calibration-v1');out.mkdir(exist_ok=False)
root=Path('/work/proposal-pairs/proposal-data-v1')
rows={'train':[],'validation':[],'holdout':[]};requests={k:set() for k in rows}
for path in sorted(root.glob('*.pt')):
    for r in torch.load(path,map_location='cpu',weights_only=True):
        split=r['phase']['split'];assert split in rows
        assert abs(r['temperature']-.7)<1e-5
        assert r['unary'].shape==r['scores'].shape==r['target_probs'].shape==(7,16)
        rows[split].append(r);requests[split].add(r['phase']['id'])
assert [len(requests[k]) for k in rows]==[42,6,16],requests
D={}
for split,rr in rows.items():
    u=torch.stack([r['unary'] for r in rr]).float();sc=torch.stack([r['scores'] for r in rr]).float();p=torch.stack([r['target_probs'] for r in rr]).float()
    assert torch.isfinite(u).all() and torch.isfinite(sc).all() and torch.isfinite(p).all()
    D[split]=(u,sc-u,p)
weight=(.75**torch.arange(7)).reshape(1,7)
def predict(params,data):
    u,e,p=data;sc=u*params[:,0][None,:,None]+e*params[:,1][None,:,None]
    return (sc/.7).softmax(-1)
@torch.no_grad()
def evaluate(params,split):
    u,e,p=D[split];q=predict(params,D[split]);ov=torch.minimum(p,q).sum(-1)
    return dict(overlap=float(ov.mean()),weighted_overlap=float((ov*weight).mean()/weight.mean()),by_position=ov.mean(0).tolist(),support_upper=p.sum(-1).mean(0).tolist())
identity=torch.ones(7,2);baseline={k:evaluate(identity,k) for k in ('train','validation')}
best=dict(params=identity.clone(),score=baseline['validation']['weighted_overlap'],step=0,regularization=None);history=[]
started=time.time()
for regularization in (.001,.01,.1):
    params=torch.nn.Parameter(identity.clone());optimizer=torch.optim.Adam([params],lr=.01)
    for step in range(1,801):
        idx=torch.randint(len(D['train'][0]),(128,));batch=tuple(x[idx] for x in D['train']);q=predict(params,batch)
        loss=-(torch.minimum(batch[2],q).sum(-1)*weight).mean()/weight.mean()+regularization*(params-identity).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
        with torch.no_grad():params.clamp_(.25,4.)
        if step%50==0:
            score=evaluate(params,'validation');row=dict(step=step,regularization=regularization,validation=score,params=params.detach().tolist(),elapsed_s=time.time()-started);history.append(row)
            if score['weighted_overlap']>best['score']:best=dict(params=params.detach().clone(),score=score['weighted_overlap'],step=step,regularization=regularization)
    print('Finished regularization',regularization,'best',best['score'],flush=True)
# Held-out data is evaluated only after all fitting/selection is finished.
result=dict(baseline={**baseline,'holdout':evaluate(identity,'holdout')},selected={k:evaluate(best['params'],k) for k in D},params=best['params'].tolist(),step=best['step'],regularization=best['regularization'],requests={k:sorted(v) for k,v in requests.items()},batches={k:len(v) for k,v in rows.items()},objective='Mean conditional overlap weighted .75^position; 14 scalars, identity regularization, bounds[.25,4]. Validation selects; heldout never fits or selects.',limits='Fixed proposed-prefix offline proxy; serving acceptance/distribution/performance still require tests.',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'results.json').write_text(json.dumps(result,indent=2));(out/'history.json').write_text(json.dumps(history,indent=2));torch.save(best['params'],out/'params.pt');print(json.dumps(result),flush=True)
