#!/usr/bin/env python3
"""Source-only distribution build; obtain the 15 locked compiler debs separately."""
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def inspect(ref):
    result = subprocess.run(['docker', 'image', 'inspect', ref], capture_output=True, text=True)
    return json.loads(result.stdout)[0] if result.returncode == 0 else None

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def restore_qualified_sources(context):
    """Reconstruct frozen build inputs in a disposable context, never the export.

    Check historical locks after removing only the added notice for comparison.
    Keep all Python notices in the runtime overlay. Only native compiler inputs
    are reconstructed in this temporary context to preserve binary reproduction;
    their notice-bearing source remains in the distributed source package.
    """
    locked = json.loads((ROOT / 'manifests/native-source.json').read_text())
    locked.update({'overlay/' + name.lstrip('/'): item['sha256'] for name, item in
                   json.loads((ROOT / 'manifests/overlay.json').read_text()).items()})
    for name, expected in locked.items():
        path = context / name
        data = path.read_bytes()
        for prefix in (b'# ', b'// '):
            notice = prefix + b'Modified by Wulf for B70 Flash-Next Disk; upstream terms are retained.\n'
            if data.startswith(notice):
                data = data[len(notice):]
                if name.startswith('native/'):
                    path.write_bytes(data)
                break
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Qualified source reconstruction mismatch: ' + name)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler-dir', type=Path, required=True)
    parser.add_argument('--tag', default='qwen38-flashnext:disk-test')
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--assign-disk', action='store_true')
    args = parser.parse_args()
    vendor = 'sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9'
    dependency = inspect(vendor)
    if not dependency or (dependency['Id'], dependency['Architecture'], dependency['Os']) != (vendor, 'amd64', 'linux'):
        raise ValueError('Missing exact official vendor dependency; see README.md')
    if args.tag == 'qwen38-flashnext:disk' or inspect(args.tag):
        raise ValueError('Choose a fresh, isolated test tag')
    if args.assign_disk and inspect('qwen38-flashnext:disk'):
        raise ValueError('Refusing to overwrite qualified Disk tag')
    locked = {}
    for line in (ROOT / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        if digest(ROOT / name) != expected:
            raise ValueError('Source package hash mismatch: ' + name)
        locked[name] = expected
    actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and p.name != 'SHA256SUMS' and '__pycache__' not in p.parts}
    if actual != set(locked):
        raise ValueError('Source package has missing or unlisted files')
    compiler = json.loads((ROOT / 'manifests/compiler-lock.json').read_text())['debs']
    for name, expected in compiler.items():
        if digest(args.compiler_dir / name) != expected:
            raise ValueError('Compiler package hash mismatch: ' + name)
    out = args.evidence.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'vendor-image.json').write_text(json.dumps(dependency, indent=2) + '\n')
    # The compiler bytes enter a disposable local build context only, never the
    # distributed source tree or final runtime stage.
    with tempfile.TemporaryDirectory(prefix='disk-build-') as temporary:
        context = Path(temporary) / 'package'
        shutil.copytree(ROOT, context, ignore=shutil.ignore_patterns('__pycache__'))
        restore_qualified_sources(context)
        target = context / 'dependencies/compiler'
        target.mkdir(parents=True)
        for name in sorted(compiler):
            shutil.copy2(args.compiler_dir / name, target / name)
        inputs = {str(p.relative_to(context)): digest(p) for p in context.rglob('*') if p.is_file() and (p.name in ('Dockerfile', '.dockerignore') or str(p.relative_to(context)).startswith(('native/', 'overlay/', 'dependencies/')))}
        (out / 'build-input-manifest.json').write_text(json.dumps(inputs, indent=2, sort_keys=True) + '\n')
        command = ['docker', 'build', '--pull=false', '--no-cache', '--network=none', '-t', args.tag, str(context)]
        (out / 'build-command.json').write_text(json.dumps(command, indent=2) + '\n')
        environment = os.environ.copy()
        environment['DOCKER_BUILDKIT'] = '0'
        with (out / 'build.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        (out / 'build-result.json').write_text(json.dumps({'exit_code': result.returncode, 'tag': args.tag, 'DOCKER_BUILDKIT': '0'}, indent=2) + '\n')
        if result.returncode:
            raise SystemExit(result.returncode)
    rebuilt = inspect(args.tag)
    (out / 'image.json').write_text(json.dumps(rebuilt, indent=2) + '\n')
    subprocess.run([sys.executable, str(ROOT / 'verify.py'), rebuilt['Id'], '--output', str(out / 'comparison.json')], check=True)
    if args.assign_disk:
        if inspect('qwen38-flashnext:disk'):
            raise ValueError('Disk tag appeared; refusing overwrite')
        subprocess.run(['docker', 'tag', rebuilt['Id'], 'qwen38-flashnext:disk'], check=True)
    print('Built and verified ' + rebuilt['Id'])

if __name__ == '__main__':
    main()
