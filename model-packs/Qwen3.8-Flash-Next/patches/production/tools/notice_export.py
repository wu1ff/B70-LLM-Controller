#!/usr/bin/env python3
"""Export separately hashed comment-only source derivatives; no executable build."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

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
              'publication_clearance': False, 'files': inventory}
    (output / 'NOTICE_EXPORT.json').write_text(json.dumps(report, indent=2) + '\n')
    return {k: v for k, v in report.items() if k != 'files'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='verified staging/reconstructed')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.source.resolve(), args.output.resolve()), indent=2))
