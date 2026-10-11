#!/usr/bin/env python3
"""Add required source notices to the qualified local image without compilation."""
import argparse, datetime, hashlib, io, json, subprocess, tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QUALIFIED = 'sha256:fad9d4cfae29a4fc3abd70e7339e4f2784c3589c988f7ca3f6751ffb46806316'
BASE = 'qwen38-flashnext:disk'
NOTICE = b'# Modified by Wulf for B70 Flash-Next Disk; upstream terms are retained.\n'

def inspect(ref):
    return json.loads(subprocess.check_output(['docker', 'image', 'inspect', ref]))[0]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default='qwen38-flashnext:disk-rc')
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    if args.tag == BASE or subprocess.run(['docker', 'image', 'inspect', args.tag], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        raise ValueError('Choose an unused candidate tag; never overwrite Disk')
    parent = inspect(BASE)
    if (parent['Id'], parent['Os'], parent['Architecture']) != (QUALIFIED, 'linux', 'amd64'):
        raise ValueError('Qualified local base image mismatch')
    for line in (ROOT / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Source package hash mismatch: ' + name)
    overlay = json.loads((ROOT / 'manifests/overlay.json').read_text())
    files = {name: (ROOT / 'overlay' / name.lstrip('/')).read_bytes() for name in overlay}
    files = {name: data for name, data in files.items() if data.startswith(NOTICE)}
    if len(files) != 15:
        raise ValueError('Expected exactly 15 modified upstream Python files')
    # Replacing files updates their parent directory timestamps. Restore those
    # timestamps in the same tar layer so independent builds stay deterministic.
    directories = sorted({str(Path(name).parent) for name in files})
    program = 'import json,os,stat; print(json.dumps({p: {"mode":stat.S_IMODE(s.st_mode),"uid":s.st_uid,"gid":s.st_gid,"mtime":int(s.st_mtime)} for p in ' + repr(directories) + ' for s in [os.stat(p)]}))'
    metadata = json.loads(subprocess.check_output(['docker', 'run', '--rm', '--network=none', '--read-only', '--entrypoint', '/opt/venv/bin/python', BASE, '-B', '-c', program]))
    out = args.evidence.resolve()
    out.mkdir(parents=True, exist_ok=True)
    context = out / 'notice-context'
    context.mkdir()
    (out / 'parent-image.json').write_text(json.dumps(parent, indent=2) + '\n')
    container = subprocess.check_output(['docker', 'create', '--network=none', '--entrypoint', '/bin/true', BASE], text=True).strip()
    changes = {}
    try:
        with tarfile.open(context / 'notices.tar', 'w', format=tarfile.PAX_FORMAT) as target:
            for name, data in sorted(files.items()):
                raw = subprocess.check_output(['docker', 'cp', container + ':' + name, '-'])
                with tarfile.open(fileobj=io.BytesIO(raw)) as original:
                    members = original.getmembers()
                    if len(members) != 1 or not members[0].isfile():
                        raise ValueError('Expected one regular installed file: ' + name)
                    member = members[0]
                    baseline = original.extractfile(member).read()
                if data[len(NOTICE):] != baseline or hashlib.sha256(baseline).hexdigest() != overlay[name]['sha256']:
                    raise ValueError('Unexpected installed/source drift: ' + name)
                member.name = name.lstrip('/')
                member.size = len(data)
                target.addfile(member, io.BytesIO(data))
                changes[name] = {'qualified_sha256': hashlib.sha256(baseline).hexdigest(), 'candidate_sha256': hashlib.sha256(data).hexdigest(), 'notice_removed_exact_bytes': True, 'mode': member.mode, 'uid': member.uid, 'gid': member.gid, 'mtime': member.mtime}
            for name in directories:
                member = tarfile.TarInfo(name.lstrip('/'))
                member.type = tarfile.DIRTYPE
                for key, value in metadata[name].items():
                    setattr(member, key, value)
                target.addfile(member)
    finally:
        subprocess.run(['docker', 'rm', container], check=True, stdout=subprocess.DEVNULL)
    (context / 'Dockerfile').write_text('FROM ' + BASE + '\nADD notices.tar /\n')
    epoch = int(datetime.datetime.fromisoformat(parent['Created'].replace('Z', '+00:00')).timestamp())
    command = ['docker', 'buildx', 'build', '--no-cache', '--load', '--provenance=false', '--pull=false', '--network=none', '--build-arg', 'SOURCE_DATE_EPOCH=' + str(epoch), '-t', args.tag, str(context)]
    (out / 'build-command.json').write_text(json.dumps(command, indent=2) + '\n')
    with (out / 'build.log').open('w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    after = inspect(BASE)
    for field in ('Id', 'Config', 'RootFS', 'Os', 'Architecture', 'Created'):
        if after.get(field) != parent.get(field):
            raise ValueError('Qualified base changed while building: ' + field)
    candidate = inspect(args.tag)
    if candidate['RootFS']['Layers'][:-1] != parent['RootFS']['Layers']:
        raise ValueError('Candidate must retain every parent layer')
    if candidate['Config'] != parent['Config']:
        raise ValueError('Effective image configuration changed')
    (out / 'candidate-image.json').write_text(json.dumps(candidate, indent=2) + '\n')
    (out / 'notice-changes.json').write_text(json.dumps(changes, indent=2) + '\n')
    print(candidate['Id'])

if __name__ == '__main__':
    main()
