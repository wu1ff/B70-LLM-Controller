#!/usr/bin/env python3
"""Prepare a writable Intel serving view from a read-only Controller snapshot."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT = Path('/opt/b70-flashnext')
spec = importlib.util.spec_from_file_location('intel_checkpoint', '/opt/venv/lib/python3.12/site-packages/vllm/models/qwen4_exp/intel_checkpoint.py')
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)
snapshot, target = map(Path, sys.argv[1:])
layout.ple_layout(snapshot / layout.PLE_FILE)
config = json.loads((snapshot / 'config.json').read_text())
qualified = json.loads((ROOT / 'serve-config.json').read_text())
removed = {k: config['text_config'].pop(k) for k in list(config['text_config']) if k.startswith('indexer_')}
if len(removed) != 5 or config != qualified:
    raise ValueError('Intel config differs from the qualified dense AutoRound config')
index = json.loads((snapshot / 'model.safetensors.index.json').read_text())
excluded = {layout.PLE_PREFIX + str(i) + '.weight' for i in range(layout.PLE_SHARDS)}
actual = {k for k in index['weight_map'] if '.ngram_embedding.shard_' in k}
if actual != excluded or any(index['weight_map'][k] != layout.PLE_FILE for k in excluded):
    raise ValueError('Intel PLE weight index differs from the qualified layout')
index['weight_map'] = {k: v for k, v in index['weight_map'].items() if k not in excluded}
index['metadata']['total_size'] = 78764834808
# Exclude shard16 entirely from the weight iterator, retaining persistent hash buffers.
shards = set(index['weight_map'].values())
if layout.PLE_FILE in shards:
    raise ValueError('PLE-only shard must not enter the ordinary model loader')
if any(Path(n).name != n or not (snapshot / n).is_file() for n in shards):
    raise ValueError('Serving index has unsafe or missing shard files')
# The directory is exclusively owned by this entrypoint. Recreate on every boot.
if target.exists():
    shutil.rmtree(target)
target.mkdir(parents=True)
for name in sorted(shards):
    (target / name).symlink_to(snapshot / name)
(target / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
(target / 'model.safetensors.index.json').write_text(json.dumps(index, indent=2) + '\n')
for source in (ROOT / 'frontend').iterdir():
    shutil.copyfile(source, target / source.name)
print('Intel-only serving view prepared: 128 PLE shards mapped directly; qualified frontend installed', flush=True)
