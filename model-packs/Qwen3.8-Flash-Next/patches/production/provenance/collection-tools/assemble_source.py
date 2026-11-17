"""Consolidate the selected image's sources; never build/tag/start a server."""
from pathlib import Path
import hashlib, io, json, shutil, subprocess as sp, tarfile

W = Path(__file__).resolve().parents[1]
E = Path((W/'release-prep/active-provenance-path.txt').read_text().strip())
OUT = W/'release-prep/v033-v6-source'
P = OUT/'patches/production'
OLD = Path('/home/wulf/Desktop/qwen3.8-flash-next-dflash2-v032/release-prep/publication-1.0.5/final-pack/patches/production')
MTP = Path('/home/wulf/Desktop/Qfn-Mtp-vllm33')
UP = Path('/home/wulf/Desktop/qwen3.8-flash-next-v033-recon/upstream-v033')
HEAD = '4ba787ec9959e566d3b587a796dff07673324da5'
SITE = 'opt/venv/lib/python3.12/site-packages/'
CHAIN = json.loads((E/'image-ancestry.json').read_text())

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def dump(p, data):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,indent=2)+'\n')

copies=[]
def cp(src,dst):
    src=Path(src); dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dst)
    copies.append(dict(original=str(src),packaged=str(dst.relative_to(OUT)),sha256=sha(dst)))

def snapshot(image): return E/'snapshots'/image['Id'].split(':')[1]
def inv(image): return {x['path']:x for x in json.loads((snapshot(image)/'inventory.json').read_text())}
def files(root): return {str(p.relative_to(root)):p for p in root.rglob('*') if p.is_file()}
def source_inv(image): return {k:x for k,x in inv(image).items() if k.startswith(SITE) and (snapshot(image)/k).is_file() and not k.endswith('.pt')}
def norm(k): return k.removeprefix(SITE)

assert not OUT.exists(), 'use a new output, do not replace a package'
for d in ['inputs','patch-chain','accepted-overlay','native','dependencies','generated','support','image-support','build','manifests','provenance','tools','runtime']:
    (P/d).mkdir(parents=True)

# Clean upstream source, separate immutable vendor kernel baseline and generated metadata.
with (P/'inputs/vllm-v033.tar.gz').open('wb') as f:
    sp.run(['git','-C',str(UP),'archive','--format=tar.gz',HEAD,'vllm','LICENSE','pyproject.toml','setup.py','docker/Dockerfile.xpu','requirements/xpu.txt'],stdout=f,check=True)
base=snapshot(CHAIN[0]); final=snapshot(CHAIN[-1]); bi=source_inv(CHAIN[0]); fi=source_inv(CHAIN[-1])
with tarfile.open(P/'inputs/vendor-kernel-python.tar.gz','w:gz') as t:
    for name,p in files(base/SITE/'vllm_xpu_kernels').items(): t.add(p,arcname='vllm_xpu_kernels/'+name,recursive=False)
cp(base/SITE/'vllm/_version.py',P/'generated/vllm/_version.py')
cp(final/SITE/'vllm/v1/worker/gpu/spec_decode/dflash2/quality_calibration.pt',P/'generated/quality_calibration.pt')

# Derive ordered installed-source diffs from the actual immutable images.
scratch=E/'patch-generation';scratch.mkdir()
with tarfile.open(P/'inputs/vllm-v033.tar.gz') as t:t.extractall(scratch,filter='data')
with tarfile.open(P/'inputs/vendor-kernel-python.tar.gz') as t:t.extractall(scratch,filter='data')
shutil.copy2(P/'generated/vllm/_version.py',scratch/'vllm/_version.py')
sp.run(['git','init','-q',str(scratch)],check=True)
sp.run(['git','-C',str(scratch),'add','-f','.'],check=True)
names=['01-v033-port-metadata-b1','02-shared-gdn','03-query-ordered-hc','04-bf16-repair-standard-eos','05-calibration-bf16-conv','06-compiled-graph-update-copy','07-parallel-ple-exact-ba-hc','08-exact-moe-v1','09-deferred-ple','10-exact-moe-v3']
contexts=[W/'build/metadata-b1-context',W/'build/parity-shared-context',W/'build/optimization-candidate-v3',*[W/'astra/candidates'/n for n in ['bf16-reference-v1','bf16-calconv-frozen-v1','calconv-graphupd-v2','expanded-exact-v3','expanded-exact-v4','expanded-exact-v5','expanded2-exact-v6']]]
series=[]; prev=bi
for ix,(info,label,context) in enumerate(zip(CHAIN[1:],names,contexts),1):
    current=source_inv(info);changed=[k for k in sorted(current) if k not in prev or current[k]['sha256']!=prev[k]['sha256']]
    removed=sorted(set(prev)-set(current))
    for k in changed:
        d=scratch/norm(k);d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(snapshot(info)/k,d)
    for k in removed:(scratch/norm(k)).unlink()
    # Intent-to-add exposes added files without changing the prior index bytes.
    sp.run(['git','-C',str(scratch),'add','-N','vllm','vllm_xpu_kernels'],check=True)
    patch=sp.check_output(['git','-C',str(scratch),'diff','--binary','--no-ext-diff','--no-renames','--','vllm','vllm_xpu_kernels'])
    path=P/'patch-chain'/f'{label}.patch';path.write_bytes(patch)
    sp.run(['git','-C',str(scratch),'add','-f','vllm','vllm_xpu_kernels'],check=True)
    assert CHAIN[ix-1]['RootFS']['Layers']==info['RootFS']['Layers'][:len(CHAIN[ix-1]['RootFS']['Layers'])]
    series.append(dict(order=ix,patch=str(path.relative_to(P)),strip=1,base=CHAIN[ix-1]['Id'],image=info['Id'],tag=info['RepoTags'],changed=[dict(path=norm(k),sha256=current[k]['sha256']) for k in changed],removed=[norm(k) for k in removed],historical_context=str(context)))
    for f in context.iterdir():
        if f.is_file() and f.name in ['Dockerfile','source-manifest.json','manifest.json','profile.json','build.log']:
            cp(f,P/'provenance/layers'/label/f.name)
    prev=current
for k in sorted(fi):
    if k not in bi or fi[k]['sha256']!=bi[k]['sha256']:cp(final/k,P/'accepted-overlay'/norm(k))
dump(P/'manifests/patch-series.json',series)
dump(P/'manifests/accepted-source.json',[dict(path=norm(k),sha256=x['sha256'],size=x['size'],mode=x['mode']) for k,x in sorted(fi.items())])
dump(P/'manifests/base-source.json',[dict(path=norm(k),sha256=x['sha256'],size=x['size']) for k,x in sorted(bi.items())])

# Exact supporting assets and cumulative mirror payloads. Binary outputs are manifests, not source inputs.
for prefix in ['opt/b70-flashnext']:
    for rel,p in files(final/prefix).items():cp(p,P/'support'/prefix.removeprefix('opt/')/rel)
mirrors={}
for context in contexts:
    for rel,p in files(context/'overlay/workspace').items():
        if '.so' not in p.name and p.suffix!='.pyc':mirrors['workspace/'+rel]=p
for rel,p in mirrors.items():cp(p,P/'image-support'/rel)
cp(W/'astra/candidates/expanded2-exact-v6/profile.json',P/'runtime/profile.json')
for f in ['qualification/parity_worker.py','qualification/tuning_worker.py','scripts/launch.py']:
    cp(W/f,P/'runtime'/Path(f).name)
dump(P/'manifests/serving-contract.json',json.loads((E/'container-before.json').read_text())[0])
dump(P/'manifests/final-image.json',CHAIN[-1])
dump(P/'manifests/image-lineage.json',[dict(id=x['Id'],tags=x['RepoTags'],layers=x['RootFS']['Layers'],descriptor=x.get('Descriptor')) for x in CHAIN])
cp(final/'metadata.json',P/'manifests/installed-metadata.json');cp(final/'dpkg-status.txt',P/'manifests/installed-dpkg-status.txt')
cp(final/'inventory.json',P/'manifests/installed-artifacts.json')

# Sources/headers and actual v0.33 build recipes for all installed custom native outputs.
for name in ['common-native-headers']:
    shutil.copytree(OLD/'dependencies'/name,P/'dependencies'/name)
shutil.copytree(W/'source/native/gdn-headers',P/'dependencies/gdn-headers')
cp(MTP/'build/phase6/inputs/sycl-tla.tar.gz',P/'dependencies/sycl-tla.tar.gz')
cp(MTP/'upstream/mtp-frozen/patches/disk/manifests/compiler-lock.json',P/'manifests/compiler-lock.json')
cp(MTP/'upstream/mtp-frozen/patches/disk/manifests/sycl-tla.json',P/'manifests/sycl-tla.json')
native=[]; custom={k:x for k,x in inv(CHAIN[-1]).items() if ('.so' in Path(k).name) and k not in inv(CHAIN[0])}
def family(name,srcdir,recipe,output,source_files,header='common-native-headers',library=None):
    folder=P/'native'/name
    (P/'provenance/native'/name).mkdir(parents=True,exist_ok=True)
    for file in source_files:cp(srcdir/file,folder/file)
    cp(recipe,folder/'build.sh')
    for file in ['build.log',recipe.with_suffix('.log').name,*[x.name for x in srcdir.glob('*.o.d')]]:
        if (srcdir/file).is_file():cp(srcdir/file,P/'provenance/native'/name/file)
    lib=library or srcdir/output
    target='opt/b70-residency-shim/'+output if name=='residency-shim' else SITE+'vllm_xpu_kernels/'+output
    expected=inv(CHAIN[-1])[target]['sha256'];assert sha(lib)==expected,(name,sha(lib),expected)
    native.append(dict(family=name,installed_path=target,sha256=expected,recipe=f'native/{name}/build.sh',source_files=[f'native/{name}/{x}' for x in source_files],header_tree=header,retained_binary=str(lib),compiler='vendor GCC 13.3.0' if name in ['residency-shim','ple-gather'] else 'oneAPI 2026.0.0.20260331',original_source_directory=str(srcdir)))
    for flag,label in [('-d','dynamic'),('-n','notes'),('-p','comment')]:
        args=['readelf',flag]+(['.comment'] if flag=='-p' else [])+[str(lib)]
        result=sp.run(args,capture_output=True,text=True)
        (P/'provenance/native'/name/f'elf-{label}.txt').write_text(result.stdout+result.stderr)
    return folder
for old in json.loads((OLD/'manifests/native-builds.json').read_text()):
    name=old['family'];source=W/'source/native'/name
    sf=[str(p.relative_to(source)) for p in source.rglob('*') if p.is_file() and p.suffix in ['.cpp','.hpp','.c','.h']]
    output=Path(old['installed_library']).name;build=W/'build/native'/('gdn-guard' if name=='native' else name)
    family(name,source,build/'build.sh',output,sf,header='gdn-headers' if name=='native' else 'common-native-headers',library=build/output)
    for file in ['build.log','link.sh','link.log','gdn_index64.o.d']:
        if (build/file).exists():cp(build/file,P/'provenance/native'/name/file)
for name,src,script,output,sf in [
 ('ordered-hc-v2',W/'build/optimization-ordered-hc-v2','build.sh','libb70_ordered_hc_v2.so',['ordered_hc_v2.cpp','dense_fragment.hpp']),
 ('exact-v1',W/'astra/candidates/exact-kernels-v1','build.sh','libb70_exact_v1.so',['exact_gemm.cpp','dense_fragment.hpp']),
 ('exact-moe-v1',W/'astra/candidates/exact-kernels-v1','build_moe.sh','libb70_exact_moe_v1.so',['exact_moe.cpp','dense_fragment.hpp']),
 ('exact-moe-v3',W/'astra/candidates/exact-kernels-v3','build_moe_v3.sh','libb70_exact_moe_v3.so',['exact_moe_v3.cpp','dense_fragment.hpp'])]:
    family(name,src,src/script,output,sf)
    for file in src.glob('command*.json'):
        if file.name in ['command.json','command_moe.json','command_moe_v3.json']:cp(file,P/'provenance/native'/name/file.name)
ple=W/'astra/evidence/expanded-20261116T000816Z/dev/ple_gather';folder=P/'native/ple-gather';folder.mkdir();cp(ple/'ple_gather.c',folder/'ple_gather.c')
(folder/'build.sh').write_text('set -euxo pipefail\ncd /outputs\ngcc -O2 -fPIC -shared -pthread /outputs/ple_gather.c -o /outputs/libple_gather.so\n')
native.append(dict(family='ple-gather',installed_path=SITE+'vllm_xpu_kernels/libple_gather.so',sha256=sha(ple/'libple_gather.so'),recipe='native/ple-gather/build.sh',source_files=['native/ple-gather/ple_gather.c'],header_tree=None,compiler='vendor GCC 13.3.0',retained_binary=str(ple/'libple_gather.so'),recipe_origin='Reconstructed command; CPU rebuild byte-matched installed binary; original invocation unavailable.'))
assert {x['installed_path'] for x in native}==set(custom), (set(custom)-{x['installed_path'] for x in native})
dump(P/'manifests/native-builds.json',native)

# Original builders and acquisition evidence, read-only references to the old release.
builder_scripts=['scripts/port_source.py','scripts/adapt_interfaces.py','scripts/prepare_metadata_candidate.py','scripts/build_native.py','scripts/package_candidate.py','scripts/build_parity_shared.py','scripts/opt_build_candidate.py','scripts/opt_build_ordered_hc.py','scripts/opt_build_ordered_hc_v2.py','astra/scripts/build_bf16_reference.py','astra/scripts/build_bf16_tuned.py','astra/scripts/build_calconv_graphupd.py','astra/scripts/build_expanded_v3.py','astra/scripts/build_expanded_v4.py','astra/scripts/build_expanded_v5.py','astra/scripts/build_expanded2_v6.py']
for f in builder_scripts:cp(W/f,P/'provenance/builders'/Path(f).name)
for f in ['acquisition.json','xpu-index.json','xpu-manifest.json','vllm-release-comparison.json']:
    cp(MTP/'evidence/exact-artifact'/f,P/'provenance/vendor'/f)
for f in ['native-toolchain.json','implementation-decisions.json','source-initial-merge.json','native-builds.json','metadata-b1-payload.json','metadata-b1-image.json']:
    if (W/'evidence'/f).exists():cp(W/'evidence'/f,P/'provenance/initial'/f)
for f in ['native-build-containers.json','upstream-base-comparison.json','ple-build-check.json','git-state.json']:
    cp(E/f,P/'provenance'/f)
for f in (E/'images').iterdir():cp(f,P/'provenance/images'/f.name)
for f in ['README.md','BUNDLE.md','manifests/immutable-inputs.json','manifests/patch-series.json']:
    cp(OLD/f,P/'provenance/v032-reference'/f)
cp(OLD.parents[3]/'release-receipt.json',P/'provenance/v032-reference/release-receipt.json') if False else None
cp(Path('/home/wulf/Desktop/qwen3.8-flash-next-dflash2-v032/release-prep/publication-1.0.5/release-receipt.json'),P/'provenance/v032-reference/release-receipt.json')
dump(P/'manifests/copied-inputs.json',copies)
dump(E/'assembly-result.json',dict(package=str(OUT),patches=len(series),native_libraries=len(native),source_files=len(fi),source_delta=len(files(P/'accepted-overlay')),mirrors=len(mirrors)))
print((E/'assembly-result.json').read_text())

if __name__=='__main__': pass
