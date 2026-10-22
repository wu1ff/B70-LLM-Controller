#!/usr/bin/env python3
"""Compare the distributed source archive to verified accepted inputs, without builds."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import tarfile

P = Path(__file__).resolve().parents[1]
L = P.parent / 'licenses'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify(source):
    descriptor = json.loads((P / 'manifests/source-export.json').read_text())
    archive = P / descriptor['archive']
    assert sha(archive.read_bytes()) == descriptor['sha256'], 'source archive identity'
    files = {}
    with tarfile.open(archive) as tar:
        for member in tar:
            assert member.isfile() and not member.name.startswith('/') and '..' not in Path(member.name).parts
            assert member.name not in files, 'duplicate member'
            assert (member.mode, member.uid, member.gid, member.mtime, member.uname, member.gname) == (0o644, 0, 0, 0, '', '')
            files[member.name] = tar.extractfile(member).read()
    raw_report = files.pop('NOTICE_EXPORT.json')
    report = json.loads(raw_report)
    assert report['files'] == {name: sha(data) for name, data in files.items()}, 'export inventory'
    assert sha(raw_report) == descriptor['notice_export_manifest_sha256']
    expected = {}
    for root, prefix in [(source, 'reconstructed'), (L, 'licenses')] + [(P / name, name) for name in ['native','dependencies','inputs','support','image-support']]:
        for path in root.rglob('*'):
            if path.is_file(): expected[prefix + '/' + str(path.relative_to(root))] = path.read_bytes()
    rows = json.loads((L / 'notice-export.json').read_text())['files']
    assert report['mapping'] == [{k:row[k] for k in ['path','accepted_sha256','notice_sha256','insert_offset','comment']} for row in rows]
    for row in rows:
        name = row['path']
        before = expected[name]
        after = files[name]
        offset, comment = row['insert_offset'], row['comment'].encode()
        assert sha(before) == row['accepted_sha256'] and sha(after) == row['notice_sha256'], name
        assert after == before[:offset] + comment + before[offset:], name
        if name.endswith('.py'):
            assert ast.dump(ast.parse(before)) == ast.dump(ast.parse(after)), name
        expected[name] = after
    assert expected.keys() == files.keys(), 'source export file set'
    assert all(files[name] == data for name, data in expected.items()), 'unexpected source export differences'
    return dict(source_export_verification='PASS', files=len(files)+1, notice_files=len(rows),
                python_ast_equality=True, exact_comment_insertions=True,
                other_export_bytes_unchanged=True, normalized_archive_metadata=True,
                accepted_reconstruction_unchanged=True, native_compilation=False,
                executable_image_changed=False, publication_clearance=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='verified accepted reconstructed directory')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = verify(args.source.resolve())
    text = json.dumps(result, indent=2) + '\n'
    if args.report: args.report.write_text(text)
    print(text, end='')
