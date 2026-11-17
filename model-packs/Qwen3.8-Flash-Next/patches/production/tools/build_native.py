"""Plan or rebuild the 17 custom libraries in the pinned vendor environment."""
from pathlib import Path
import argparse, hashlib, json, shutil, subprocess, tarfile
P=Path(__file__).resolve().parents[1]
BASE='sha256:010bdc441c72ee222f459d3b688bb41317525f1bd7223afe6218490b0c526832'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--plan',action='store_true');a.add_argument('--output',type=Path);g=a.add_mutually_exclusive_group();g.add_argument('--toolchain',type=Path);g.add_argument('--compiler-debs',type=Path);a.add_argument('--family',action='append');args=a.parse_args()
    native=json.loads((P/'manifests/native-builds.json').read_text());selected=[x for x in native if not args.family or x['family'] in args.family]
    assert selected and (not args.family or set(args.family)<={x['family'] for x in native})
    if args.plan or not args.output:print(json.dumps(selected,indent=2));return
    assert not args.output.exists(),'output directory must be new'
    assert args.toolchain or args.compiler_debs,'supply pinned compiler debs or verified retained toolchain'
    info=json.loads(subprocess.check_output(['docker','image','inspect',BASE]))[0];assert info['Id']==BASE
    args.output.mkdir(parents=True);toolchain=args.toolchain
    if args.compiler_debs:
        toolchain=args.output/'toolchain';toolchain.mkdir()
        for name,expected in json.loads((P/'manifests/compiler-lock.json').read_text())['debs'].items():
            deb=args.compiler_debs/name;assert sha(deb)==expected,name
            subprocess.run(['dpkg-deb','-x',str(deb),str(toolchain)],check=True)
    for row in json.loads((P/'manifests/toolchain-files.json').read_text()):
        path=toolchain/row['path']
        if 'symlink' in row:assert path.is_symlink() and str(path.readlink())==row['symlink'],str(path)
        else:assert path.is_file() and sha(path)==row['sha256'],str(path)
    tla=args.output/'tla-input';tla.mkdir()
    with tarfile.open(P/'dependencies/sycl-tla.tar.gz') as t:t.extractall(tla,filter='data')
    tla=next(tla.iterdir());results=[]
    for x in selected:
        dest=args.output/x['family'];dest.mkdir()
        # Some recipes compile /native/<family>; later recipes compile /outputs.
        for name in x['source_files']:shutil.copy2(P/name,dest/Path(name).name)
        shutil.copy2(P/x['recipe'],dest/'build.sh')
        headers=P/'dependencies'/(x['header_tree'] or 'common-native-headers')
        cmd=['docker','run','--rm','--pull','never','--network','none','--ipc','private','--cpus','8','-e','LD_PRELOAD=','--entrypoint','/bin/bash','-v',str((P/'native').resolve())+':/native:ro','-v',str(headers.resolve())+':/headers:ro','-v',str(toolchain.resolve())+':/toolchain:ro','-v',str(tla.resolve())+':/tla:ro','-v',str(dest.resolve())+':/outputs:rw',BASE,'/outputs/build.sh']
        with (dest/'build.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        binary=dest/Path(x['installed_path']).name;actual=sha(binary) if binary.is_file() else None
        result=dict(family=x['family'],command=cmd,returncode=r.returncode,sha256=actual,expected_sha256=x['sha256'],matches=actual==x['sha256']);results.append(result)
        (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        assert r.returncode==0 and result['matches'],f"{x['family']}: build failed or binary differs; inspect results.json"
        print(x['family'],'PASS',flush=True)
if __name__=='__main__':main()
