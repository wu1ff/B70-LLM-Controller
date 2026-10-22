#!/usr/bin/env bash
# Wulf: prepare Intel's pinned checkpoint without a second PLE table.
set -euo pipefail
if [ "$#" -lt 3 ] || [ "$1" != vllm ] || [ "$2" != serve ]; then
  echo "b70-flashnext: expected 'vllm serve <model-path> [flags...]'" >&2
  exit 2
fi
snapshot="$3"
serve_dir=/work/flashnext-serve
export PLE_TABLE_PATH="$snapshot/model-00016-of-00017.safetensors"
/opt/venv/bin/python /opt/b70-flashnext/prepare-intel.py "$snapshot" "$serve_dir"
exec /opt/venv/bin/vllm serve "$serve_dir" "${@:4}"
