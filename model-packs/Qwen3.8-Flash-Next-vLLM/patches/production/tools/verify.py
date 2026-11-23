"""Offline source replay; no Docker, compiler, weights, or source workspace needed."""
from pathlib import Path
import argparse, hashlib, json, shutil, subprocess, tarfile

P=Path(__file__).resolve().parents[1]
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(name):return json.loads((P/name).read_text())
def verify_lock():
    lock=load('manifests/input-lock.json')
    for x in lock:
        p=P/x['path']
        assert p.is_file() and sha(p)==x['sha256'],x['path']
    return len(lock)
def compare(root,entries):
    for x in entries:
        p=root/x['path'];assert p.is_file() and sha(p)==x['sha256'],x['path']
def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--staging',type=Path,required=True);a.add_argument('--report',type=Path,required=True);args=a.parse_args()
    assert not args.staging.exists(),'staging directory must be new'
    count=verify_lock();root=args.staging/'reconstructed';root.mkdir(parents=True)
    for name in ['vllm-v033.tar.gz','vendor-kernel-python.tar.gz']:
        with tarfile.open(P/'inputs'/name) as t:t.extractall(root,filter='data')
    shutil.copy2(P/'generated/vllm/_version.py',root/'vllm/_version.py')
    compare(root,load('manifests/base-source.json'));patches=[]
    for x in load('manifests/patch-series.json'):
        cmd=['git','apply','--whitespace=nowarn','-p'+str(x['strip']),str(P/x['patch'])]
        subprocess.run(cmd[:2]+['--check']+cmd[2:],cwd=root,check=True)
        subprocess.run(cmd,cwd=root,check=True)
        compare(root,x['changed'])
        for name in x['removed']:assert not (root/name).exists(),name
        patches.append(dict(order=x['order'],patch=x['patch'],status='PASS'))
    expected=load('manifests/accepted-source.json');compare(root,expected)
    actual={str(p.relative_to(root)) for top in ['vllm','vllm_xpu_kernels'] for p in (root/top).rglob('*') if p.is_file()}
    assert actual=={x['path'] for x in expected},'source file set mismatch'
    for p in (P/'accepted-overlay').rglob('*'):
        if p.is_file():assert p.read_bytes()==(root/p.relative_to(P/'accepted-overlay')).read_bytes(),str(p)
    dest=root/'vllm/v1/worker/gpu/spec_decode/dflash2/quality_calibration.pt'
    shutil.copy2(P/'generated/quality_calibration.pt',dest)
    syntax=0
    for p in root.rglob('*.py'):
        compile(p.read_bytes(),str(p),'exec');syntax+=1
    native=load('manifests/native-builds.json')
    for x in native:
        for name in [x['recipe'],*x['source_files']]:assert (P/name).is_file(),name
        subprocess.run(['bash','-n',str(P/x['recipe'])],check=True)
    assert len(native)==17
    p=load('runtime/profile.json')
    assert sha(P/'runtime/profile.json')=='ed6e18ce9c7bf052757da399d94857f99ebff0def85c94bb725d83301b7a7a85'
    spec=json.loads(p['command'][p['command'].index('--speculative-config')+1])
    assert spec['rejection_sample_method']=='standard' and p['env']['B70_THINKING_EOS_GUARD']=='1'
    result=dict(status='PASS',bundle_files_verified=count,source_files=len(expected),syntax_files=syntax,patches=patches,native_families=len(native),calibration_sha256=sha(dest),staging=str(args.staging.resolve()),native_compilation='NOT RUN by verifier',full_image_build='NOT RUN',bit_identical_image='NOT DEMONSTRATED',external_dependencies='Pinned vendor image, compiler debs, runtime/model assets remain external; see immutable-inputs.json.')
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
