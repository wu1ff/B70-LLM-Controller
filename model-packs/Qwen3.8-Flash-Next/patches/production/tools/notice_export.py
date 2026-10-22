#!/usr/bin/env python3
"""Export separately hashed comment-only source derivatives; no executable build."""
import argparse
import ast
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

P = Path(__file__).resolve().parents[1]
L = P.parent / 'licenses'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def export(source, output):
    if output.exists() or output.is_relative_to(P.parent):
        raise ValueError('output must be a new directory outside patches/')
    rows = json.loads((L / 'notice-export.json').read_text())['files']
    for row in rows:
        original = (source / row['path'].removeprefix('reconstructed/')) if row['path'].startswith('reconstructed/') else P / row['path']
        derivative = L / 'notice-overlay' / row['path']
        before, after = original.read_bytes(), derivative.read_bytes()
        if sha(original) != row['accepted_sha256'] or sha(derivative) != row['notice_sha256']:
            raise ValueError('notice identity mismatch: ' + row['path'])
        # Prove the exact only change is the recorded inserted comment bytes.
        offset = row['insert_offset']
        comment = row['comment'].encode()
        prefix = b'# ' if original.suffix == '.py' else b'// '
        if (not comment.endswith(b'\n') or
                not all(line.startswith(prefix) and not line.endswith(b'\\')
                        for line in comment.splitlines()) or
                offset not in (0, before.find(b'\n') + 1 if before.startswith(b'#!') else 0)):
            raise ValueError('invalid comment insertion: ' + row['path'])
        if after != before[:offset] + comment + before[offset:]:
            raise ValueError('non-notice change: ' + row['path'])
        if original.suffix == '.py' and ast.dump(ast.parse(before)) != ast.dump(ast.parse(after)):
            raise ValueError('Python AST changed: ' + row['path'])
    shutil.copytree(source, output / 'reconstructed')
    for name in ['native', 'dependencies', 'inputs', 'support', 'image-support']:
        shutil.copytree(P / name, output / name)
    shutil.copytree(L, output / 'licenses')
    for row in rows:
        shutil.copy2(L / 'notice-overlay' / row['path'], output / row['path'])
    inventory = {str(f.relative_to(output)): sha(f) for f in sorted(output.rglob('*')) if f.is_file()}
    report = {'notice_files': len(rows), 'original_source_unchanged': True,
              'differences': 'only recorded comment insertions; Python AST equality checked',
              'executable_image_changed': False, 'binary_equivalence_claimed': False,
              'publication_clearance': False,
              'accepted_reconstruction': 'The enclosing pack production/ tree and its original manifests remain the accepted reconstruction authority. This directory contains annotated distribution copies only.',
              'mapping': [{k: row[k] for k in ['path', 'accepted_sha256', 'notice_sha256', 'insert_offset', 'comment']} for row in rows],
              'files': inventory}
    (output / 'NOTICE_EXPORT.json').write_text(json.dumps(report, indent=2) + '\n')
    return {k: v for k, v in report.items() if k not in ('files', 'mapping')}


def archive(source, destination):
    """Write a normalized, source-only archive; no timestamps or host ownership."""
    with destination.open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode='w', format=tarfile.GNU_FORMAT) as tar:
                for path in sorted(source.rglob('*')):
                    if path.is_symlink():
                        raise ValueError('symlink in source export: ' + str(path))
                    if not path.is_file():
                        continue
                    data = path.read_bytes()
                    member = tarfile.TarInfo(str(path.relative_to(source)))
                    member.size, member.mode = len(data), 0o644
                    tar.addfile(member, io.BytesIO(data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='verified staging/reconstructed')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive', type=Path, help='new deterministic tar.gz outside the output tree')
    args = parser.parse_args()
    result = export(args.source.resolve(), args.output.resolve())
    if args.archive:
        destination = args.archive.resolve()
        if destination.is_relative_to(args.output.resolve()):
            raise ValueError('archive must be outside the export tree')
        archive(args.output.resolve(), destination)
        result['archive_sha256'] = sha(destination)
        result['archive_bytes'] = destination.stat().st_size
    print(json.dumps(result, indent=2))
