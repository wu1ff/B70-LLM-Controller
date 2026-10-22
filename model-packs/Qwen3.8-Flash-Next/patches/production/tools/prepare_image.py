#!/usr/bin/env python3
"""Stage the Phase 3 image recipe from verified source/native outputs; never build."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

P = Path(__file__).resolve().parents[1]


def stage(source, native, output):
    if output.exists() or output.is_relative_to(P.parent):
        raise ValueError('output must be a new directory outside patches/')
    expected = json.loads((P / 'manifests/image-payload.json').read_text())
    origins = {}
    site = '/opt/venv/lib/python3.12/site-packages/'
    for package in ['vllm', 'vllm_xpu_kernels']:
        base = source / package
        if not base.is_dir():
            raise ValueError('missing reconstructed package: ' + str(base))
        for f in sorted(base.rglob('*')):
            if f.is_file():
                origins[site + str(f.relative_to(source))] = f
    for sub, destination in [('support/b70-flashnext', '/opt/b70-flashnext/'),
                             ('image-support', '/')]:
        for f in sorted((P / sub).rglob('*')):
            if f.is_file():
                origins[destination + str(f.relative_to(P / sub))] = f
    for row in json.loads((P / 'manifests/native-builds.json').read_text()):
        name = row['installed_library']
        destination = ('/opt/' if name.startswith('b70-') else site) + name
        origins[destination] = native / name
    origins['/tmp/l0_peer_residency_shim.c'] = P / 'native/residency-shim/l0_peer_residency_shim.c'
    if origins.keys() != expected.keys():
        raise ValueError('image payload file set differs from Phase 3')
    # Validate every input before writing any image context.
    for name, f in origins.items():
        with f.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected[name]['sha256']:
            raise ValueError('payload hash mismatch: ' + name)
    output.mkdir(parents=True)
    for name, f in origins.items():
        dst = output / 'rootfs' / name.lstrip('/')
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        dst.chmod(expected[name]['mode'])
    modes = json.loads((P / 'manifests/image-modes.json').read_text())
    for directory in (output / 'rootfs').rglob('*'):
        if directory.is_dir():
            mode = modes.get('/' + str(directory.relative_to(output / 'rootfs')))
            if mode is not None:
                directory.chmod(mode)
    shutil.copy2(P / 'manifests/image-modes.json', output / 'metadata-fixups.json')
    shutil.copy2(P / 'build/Dockerfile.runtime', output / 'Dockerfile')
    return {'context': str(output), 'payload_files': len(origins),
            'hashes_match_phase3': True, 'image_build_run': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='verified staging/reconstructed')
    parser.add_argument('--native', type=Path, required=True, help='independent native output directory')
    parser.add_argument('--output', type=Path, required=True, help='new image context directory')
    args = parser.parse_args()
    print(json.dumps(stage(args.source.resolve(), args.native.resolve(), args.output.resolve()), indent=2))
