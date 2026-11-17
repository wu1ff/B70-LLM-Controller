"""Validate retained compiler dependency files against packaged and pinned inputs."""
from pathlib import Path
import hashlib,json,posixpath,shlex,subprocess as sp,tarfile
W=Path(__file__).resolve().parents[1];E=Path((W/'release-prep/active-provenance-path.txt').read_text().strip());P=W/'release-prep/v033-v6-source/patches/production';M=Path('/home/wulf/Desktop/Qfn-Mtp-vllm33/build/phase6')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
tla=E/'tla-header-check';tla.mkdir(exist_ok=True)
with tarfile.open(P/'dependencies/sycl-tla.tar.gz') as t:t.extractall(tla,filter='data')
tla=next(tla.iterdir());external=set();checks=[];errors=[]
for row in json.loads((P/'manifests/native-builds.json').read_text()):
    family=row['family'];paths=set()
    for dep in (P/'provenance/native'/family).glob('*.o.d'):
        data=dep.read_text().replace('\\\n',' ');body=data.split(':',1)[1];paths.update(posixpath.normpath(x) for x in shlex.split(body) if x.startswith('/'))
    for path in sorted(paths):
        target=None;original=None
        if path.startswith('/headers/'):
            rel=path.removeprefix('/headers/');target=P/'dependencies'/row['header_tree']/rel
            original=(W/'source/native/gdn-headers'/rel if row['header_tree']=='gdn-headers' else Path('/home/wulf/Desktop/qwen3.8-flash-next-dflash2-v032/release-prep/publication-1.0.5/final-pack/patches/production/dependencies/common-native-headers')/rel)
        elif path.startswith('/tla/'):target=tla/path.removeprefix('/tla/');original=M/'sycl-tla'/path.removeprefix('/tla/')
        elif path.startswith('/native/'):target=P/'native'/path.removeprefix('/native/')
        elif path.startswith('/outputs/'):target=P/'native'/family/path.removeprefix('/outputs/')
        elif path.startswith('/toolchain/'):target=M/'toolchain'/path.removeprefix('/toolchain/')
        else:external.add(path)
        if target is not None:
            ok=target.is_file() and (original is None or original.is_file() and sha(target)==sha(original))
            checks.append(dict(family=family,include=path,sha256=sha(target) if target.is_file() else None,matches=ok))
            if not ok:errors.append(dict(family=family,path=path,target=str(target)))
    # Current TU/header bytes have exact copies and the retained library matches installed image hash.
    assert sha(row['retained_binary'])==row['sha256']
code='from pathlib import Path;import hashlib,json,sys;paths=json.load(sys.stdin);print(json.dumps([{\'path\':p,\'sha256\':hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).is_file() else None} for p in paths]))'
actual=json.loads(sp.check_output(['docker','run','-i','--rm','--pull','never','--network','none','--read-only','-e','LD_PRELOAD=','--entrypoint','/opt/venv/bin/python','sha256:010bdc441c72ee222f459d3b688bb41317525f1bd7223afe6218490b0c526832','-c',code],input=json.dumps(sorted(external)).encode()))
dump(P/'manifests/native-dependency-audit.json',dict(packaged_and_toolchain=checks,external_vendor_headers=actual,errors=errors))
summary=dict(native_families=17,packaged_or_toolchain_dependency_references=len(checks),vendor_dependency_files=len(actual),missing_vendor=[x for x in actual if x['sha256'] is None],errors=errors)
dump(E/'native-audit-summary.json',summary);print(json.dumps(summary,indent=2))
