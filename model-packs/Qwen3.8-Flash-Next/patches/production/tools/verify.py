#!/usr/bin/env python3
"""Offline source reconstruction. Requires Python >=3.12, git and bash; no GPUs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

PRODUCTION = Path(__file__).resolve().parents[1]
PATCHES = PRODUCTION.parent

def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def read(name):
    return json.loads((PRODUCTION/'manifests'/name).read_text())

def require(condition, message):
    if not condition:
        raise ValueError(message)

def check_files(root, expected, exact=False):
    errors = []
    for name, h in expected.items():
        p = root/name
        if not p.is_file(): errors.append('missing: '+name)
        elif digest(p) != h: errors.append('hash mismatch: '+name)
    if exact:
        actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
        errors += ['unexpected: '+s for s in sorted(actual-set(expected))]
    require(not errors, '\n'.join(errors))
    return len(expected)

def unpack(archive, dst, strip=0):
    dst.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as t:
        for m in t:
            require(m.isfile() or m.isdir(), 'unsupported archive member: '+m.name)
            parts = Path(m.name).parts
            require(not Path(m.name).is_absolute() and '..' not in parts, 'unsafe archive member')
            if len(parts) <= strip: continue
            m.name = str(Path(*parts[strip:]))
            t.extract(m, dst, filter='data')

def run(args, cwd=None):
    env = dict(os.environ, GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull)
    p = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True)
    require(p.returncode == 0, 'command failed: '+repr(args)+'\n'+p.stdout+p.stderr)
    return p

def verify(dest):
    require(sys.version_info >= (3,12), 'Python 3.12+ required')
    for tool in ['git','bash']:
        require(shutil.which(tool), 'missing tool: '+tool)
    # Verify inventory before extracting anything. No working checkout is an input.
    sums = {}
    for line in (PATCHES/'SHA256SUMS').read_text().splitlines():
        h, name = line.split('  ',1)
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'unsafe manifest path')
        sums[name] = h
    actual = {str(p.relative_to(PATCHES)) for p in PATCHES.rglob('*') if p.is_file()}
    expected_files = set(sums)|{'SHA256SUMS'}
    require(actual == expected_files,
            'bundle file set differs from SHA256SUMS; missing: '+', '.join(sorted(expected_files-actual))+
            '; unexpected: '+', '.join(sorted(actual-expected_files)))
    file_count = check_files(PATCHES,sums)
    check_files(PRODUCTION,read('input-lock.json')['files'])
    require(not dest.exists(), 'staging directory must not exist: '+str(dest))
    require(not dest.is_relative_to(PATCHES), 'staging must be outside the bundle')
    dest.mkdir(parents=True)
    reconstructed = dest/'reconstructed'
    unpack(PRODUCTION/'inputs/vllm-v0.32.0-package.tar.gz',reconstructed)
    unpack(PRODUCTION/'inputs/vendor-kernel-python.tar.gz',reconstructed)
    check_files(reconstructed,read('baseline-source.json'))
    stages = []
    series = read('patch-series.json')
    require(len(series) == 16 and [s['order'] for s in series] == list(range(1,17)), 'invalid patch sequence')
    for s in series:
        patch = PRODUCTION/s['path']
        require(digest(patch) == s['sha256'], 'patch mismatch: '+str(patch))
        require(s['strip'] == (3 if patch.name == 'astra500-dominant.patch' else 1), 'invalid strip level')
        run(['git','apply','--check',f'-p{s["strip"]}',str(patch)], reconstructed)
        run(['git','apply',f'-p{s["strip"]}',str(patch)], reconstructed)
        stages.append({'order':s['order'],'patch':patch.name,'strip':s['strip'],'status':'PASS'})
    # Vendor-generated version metadata is an explicit input, never a seventeenth patch.
    shutil.copy2(PRODUCTION/'generated/vllm/_version.py',reconstructed/'vllm/_version.py')
    shutil.copy2(PRODUCTION/'generated/vllm_xpu_kernels/.gitkeep',reconstructed/'vllm_xpu_kernels/.gitkeep')
    source_count = check_files(reconstructed,read('accepted-python.json'))
    package_data_count = check_files(reconstructed,read('accepted-package-data.json'))
    overlay = {str(p.relative_to(PRODUCTION/'accepted-overlay')):digest(p)
               for p in (PRODUCTION/'accepted-overlay').rglob('*') if p.is_file()}
    check_files(reconstructed,overlay)
    for package in ['vllm','vllm_xpu_kernels']:
        found = {str(p.relative_to(reconstructed)) for p in (reconstructed/package).rglob('*.py')}
        expected = {n for n in read('accepted-python.json') if n.startswith(package+'/')}
        require(found == expected, package+' Python source file set differs from accepted image')
    deps = dest/'native-dependencies'
    dep_counts = {}
    for name in ['common-native-headers','gdn-headers']:
        shutil.copytree(PRODUCTION/'dependencies'/name,deps/name)
        dep_counts[name] = check_files(deps/name,read('dependency-trees.json')[name],True)
    unpack(PRODUCTION/'inputs/sycl-tla.tar.gz',deps/'sycl-tla',strip=1)
    dep_counts['sycl-tla'] = check_files(deps/'sycl-tla',read('dependency-trees.json')['sycl-tla'],True)
    native = read('native-builds.json')
    require(len(native) == 12 and len({r['installed_library'] for r in native}) == 12, 'expected twelve native libraries')
    native_inputs = 0
    for r in native:
        native_inputs += check_files(PRODUCTION,r['source_files'])
        require(r['translation_units'], 'no translation unit for '+r['family'])
        if r['build_script']:
            run(['bash','-n',str(PRODUCTION/r['build_script'])])
            script = (PRODUCTION/r['build_script']).read_text()
            require(r['build_output'] in script, 'output name not in script: '+r['family'])
            for tu in r['translation_units']:
                require(Path(tu).name in script, 'translation unit not compiled: '+tu)
        recipe = PRODUCTION/'build/Dockerfile.native'
        require(' AS '+r['recipe_stage']+'\n' in recipe.read_text(), 'missing recipe stage: '+r['family'])
        require(r['installed_library'] in recipe.read_text(), 'missing native installation: '+r['installed_library'])
    support = dest/'support'
    shutil.copytree(PRODUCTION/'support',support)
    supporting_count = check_files(support,read('accepted-support.json'),True)
    image_support_count = check_files(PRODUCTION/'image-support',read('accepted-image-support.json'),True)
    patch_index = json.loads((PATCHES/'archive/patch-index.json').read_text())
    require(len(patch_index)==97, 'historical patch inventory must retain all 97 original paths')
    for r in patch_index:
        check_files(PATCHES,{r['bundle_path']:r['sha256']})
    native_index = json.loads((PATCHES/'archive/native-variants-index.json').read_text())
    require(len(native_index)==113, 'historical native inventory must retain all 113 originals')
    for r in native_index:
        for path in r['bundle_paths']:
            check_files(PATCHES,{path:r['sha256']})
    python_index = json.loads((PATCHES/'archive/python-variants.json').read_text())
    for r in python_index:
        check_files(PATCHES,{r['bundle_path']:r['sha256']})
    findings = json.loads((PATCHES/'licenses/UNRESOLVED.json').read_text())
    return {'source_verification':'PASS', 'bundle_files_verified':file_count,
            'baseline_files':len(read('baseline-source.json')), 'patches':stages,
            'accepted_python_files':source_count,'accepted_overlay_files':len(overlay),
            'accepted_package_data_files':package_data_count,
            'dependency_trees':dep_counts,'native_libraries':len(native),
            'native_source_recipe_entries':native_inputs,'accepted_support_files':supporting_count,
            'accepted_baked_helper_files':image_support_count,
            'archive_patch_paths':len(patch_index),'archive_native_build_paths':len(native_index),
            'archive_python_variant_paths':len(python_index),
            'staging':str(dest),'native_compilation':'NOT RUN', 'binary_reproducibility':'NOT PROVED',
            'full_image_build':'NOT RUN', 'external_build_inputs':read('immutable-inputs.json')['images'],
            'external_build_input_availability':'NOT CHECKED by offline source verifier; native build preflight requires both pinned images locally.',
            'unresolved_findings':findings}

def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--staging',type=Path,required=True,help='new directory outside bundle')
    a.add_argument('--report',type=Path,help='JSON report outside bundle')
    opts = a.parse_args()
    try:
        result = verify(opts.staging.resolve())
        code = 0
    except (ValueError,OSError,tarfile.TarError) as e:
        result = {'source_verification':'FAIL','error':str(e)}
        code = 1
    text = json.dumps(result,indent=2)+'\n'
    if opts.report:
        require(not opts.report.resolve().is_relative_to(PATCHES), 'report must be outside bundle')
        opts.report.parent.mkdir(parents=True,exist_ok=True)
        opts.report.write_text(text)
    print(text,end='')
    return code

if __name__ == '__main__':
    sys.exit(main())
