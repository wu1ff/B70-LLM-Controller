#!/usr/bin/env python3
"""Compile twelve selected native libraries in the pinned, GPU-free build environment."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

P = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',action='store_true',help='show inputs, order and outputs without building')
    ap.add_argument('--output',type=Path,help='new output directory, outside bundle')
    args = ap.parse_args()
    inputs = json.loads((P/'manifests/immutable-inputs.json').read_text())
    native = json.loads((P/'manifests/native-builds.json').read_text())
    if args.plan:
        print(json.dumps({'images':inputs['images'],'libraries':native,'gpu_required':False},indent=2))
        return 0
    if not args.output:
        ap.error('--output is required to build; use --plan to inspect')
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(P.parent):
        ap.error('output must be a new directory outside bundle')
    with tempfile.TemporaryDirectory(prefix='dflash2-native-preflight-') as d:
        subprocess.run([sys.executable,str(P/'tools/verify.py'),'--staging',str(Path(d)/'clean')],check=True)
    # Fail clearly before build if either immutable OCI input is missing locally.
    for name,ref in inputs['images'].items():
        p = subprocess.run(['docker','image','inspect',ref],capture_output=True,text=True)
        if p.returncode:
            raise SystemExit(f'Missing pinned {name} image. Preload {ref} before an offline build.')
    output.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['docker','build','--platform=linux/amd64','--pull=false','--network=none',
                    '--provenance=false','--progress=plain','--target','export',
                    '--output','type=local,dest='+str(output),'-f',str(P/'build/Dockerfile.native'),str(P)],check=True)
    comparisons = []
    for r in native:
        p = output/r['installed_library']
        with p.open('rb') as f: actual = hashlib.file_digest(f,'sha256').hexdigest()
        comparisons.append({'path':r['installed_library'],'actual_sha256':actual,
                            'accepted_sha256':r['accepted_sha256'],'matches':actual == r['accepted_sha256']})
    report = {'native_binary_comparisons':comparisons,'full_image_reproducibility':'NOT PROVED'}
    (output/'binary-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if all(r['matches'] for r in comparisons) else 1

if __name__ == '__main__':
    sys.exit(main())
