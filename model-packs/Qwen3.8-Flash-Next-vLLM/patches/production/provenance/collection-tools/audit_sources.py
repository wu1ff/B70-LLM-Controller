"""Cross-check retained contexts/manifests against immutable installed image bytes."""
from pathlib import Path
import hashlib,json,subprocess as sp,shutil,re
W=Path(__file__).resolve().parents[1];E=Path((W/'release-prep/active-provenance-path.txt').read_text().strip());P=W/'release-prep/v033-v6-source/patches/production'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
series=json.loads((P/'manifests/patch-series.json').read_text());cumulative={};context_checks=[];manifest_checks=[];source_checks=[];recoveries=[];by_hash={}
for entry in series:
    context=Path(entry['historical_context'])
    for f in context.rglob('*'):
        if f.is_file() and f.suffix not in ['.o','.so','.pyc'] and f.stat().st_size<10000000:by_hash.setdefault(sha(f),[]).append(f)
for root in [W/'astra/candidates/expanded-tails',W/'astra/candidates/expanded2-tails',W/'astra/candidates/exact-kernels-v1',W/'astra/candidates/exact-kernels-v3',W/'astra/scripts',W/'build/optimization-inputs']:
    for f in root.glob('*'):
        if f.is_file() and f.suffix in ['.py','.cpp','.hpp','.json','.sh']:by_hash.setdefault(sha(f),[]).append(f)
for entry in series:
    context=Path(entry['historical_context']);snapshot=E/'snapshots'/entry['image'].split(':')[1];installed={x['path']:x for x in json.loads((snapshot/'inventory.json').read_text())}
    for f in (context/'overlay').rglob('*'):
        if f.is_file():
            rel=str(f.relative_to(context/'overlay'));cumulative[rel]=dict(sha256=sha(f),file=f)
            if rel in installed:context_checks.append(dict(layer=entry['order'],path=rel,matches=sha(f)==installed[rel]['sha256']))
    mf=context/'source-manifest.json'
    if not mf.exists():mf=context/'manifest.json'
    if mf.exists():
        m=json.loads(mf.read_text())
        for item in m.get('files',[]):
            rel=item['path'];targets=item.get('targets') or [rel if rel.startswith('opt/') else 'opt/venv/lib/python3.12/site-packages/'+(rel if rel.startswith(('vllm/','vllm_xpu_kernels/')) else 'vllm/'+rel)]
            for t in targets:
                if t in installed:manifest_checks.append(dict(layer=entry['order'],path=t,matches=installed[t]['sha256']==item['sha256']))
        for name,expected in m.get('sources',{}).items():
            f=Path(name);current=sha(f) if f.is_file() else None;row=dict(layer=entry['order'],original=name,expected=expected,current=current,status='MATCH' if current==expected else 'SUPERSEDED_OR_MISSING')
            matches=by_hash.get(expected,[])
            if matches:
                source=matches[0];dest=P/'provenance/recovered-sources'/expected/Path(name).name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
                row.update(recovered_from=str(source),packaged=str(dest.relative_to(P)),recovery_sha256=sha(dest));row['status']='MATCH' if current==expected else 'RECOVERED_EXACT'
            elif current==expected:
                dest=P/'provenance/recovered-sources'/expected/f.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest);row['packaged']=str(dest.relative_to(P))
            else:row['status']='GAP'
            source_checks.append(row)
        if 'builder_sha256' in m:
            matches=by_hash.get(m['builder_sha256'],[]);source_checks.append(dict(layer=entry['order'],kind='builder',expected=m['builder_sha256'],matches=[str(x) for x in matches],status='MATCH' if matches else 'GAP'))
    if mf.name=='source-manifest.json' and 'profile_sha256' in m:
        assert sha(context/'profile.json')==m['profile_sha256'],str(context)
    if 'base' in (m if mf.exists() else {}):assert m['base']==entry['base'],entry['order']

# Read all cumulative overlay targets in the final image, including source mirrors.
paths=list(cumulative)
code='from pathlib import Path;import json,hashlib;paths='+repr(paths)+';print(json.dumps({p:hashlib.sha256(Path("/"+p).read_bytes()).hexdigest() if Path("/"+p).is_file() else None for p in paths}))'
actual=json.loads(sp.check_output(['docker','run','--rm','--pull','never','--network','none','--read-only','-e','LD_PRELOAD=','--entrypoint','/opt/venv/bin/python',series[-1]['image'],'-c',code]))
final_checks=[dict(path=k,expected=x['sha256'],actual=actual[k],matches=x['sha256']==actual[k]) for k,x in cumulative.items()]
dump(P/'manifests/image-overlay-audit.json',dict(context_files=context_checks,manifest_files=manifest_checks,source_references=source_checks,final_overlay_files=final_checks))
summary=dict(context_files=len(context_checks),context_mismatches=sum(not x['matches'] for x in context_checks),manifest_files=len(manifest_checks),manifest_mismatches=sum(not x['matches'] for x in manifest_checks),source_reference_count=len(source_checks),recovered_superseded=[x for x in source_checks if x['status']=='RECOVERED_EXACT'],source_gaps=[x for x in source_checks if x['status']=='GAP'],final_overlay_files=len(final_checks),final_mismatches=[x for x in final_checks if not x['matches']])
dump(E/'source-audit-summary.json',summary)
print(json.dumps(summary,indent=2))
