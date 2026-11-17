# Copyright Wulf. Qwen model data remains under the Qwen Community License 1.0.
# Adapted for the v0.33 source port for Wulf with Codex.
"""Pinned Intel PLE layout and bounded private mapping; no model conversion."""

import json
import mmap
import os
import struct
from typing import BinaryIO

PLE_FILE = "model-00016-of-00017.safetensors"
PLE_PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_"
PLE_ROWS = 2500012
PLE_WIDTH = 160
PLE_SHARDS = 128
PLE_BYTES = PLE_ROWS * PLE_WIDTH * 2
PLE_OFFSET = 20736
PLE_FILE_BYTES = 102400512256


def _unique_header(pairs: list[tuple[str, object]]) -> dict:
    header: dict = {}
    for key, value in pairs:
        if key in header:
            raise ValueError("Duplicate Intel PLE header key: " + key)
        header[key] = value
    return header


def _ple_layout(stream: BinaryIO) -> tuple[int, int, int]:
    """Reject changed dtype, numeric shard order, gaps, overlaps and truncation."""
    if os.fstat(stream.fileno()).st_size != PLE_FILE_BYTES:
        raise ValueError("Intel PLE file size differs from the pinned layout")
    size = struct.unpack("<Q", stream.read(8))[0]
    if size != PLE_OFFSET - 8:
        raise ValueError("Intel PLE header size differs from the pinned layout")
    header = json.loads(stream.read(size), object_pairs_hook=_unique_header)
    names = {f"{PLE_PREFIX}{i}.weight" for i in range(PLE_SHARDS)}
    if set(header) - {"__metadata__"} != names:
        raise ValueError("Intel PLE tensor inventory differs from the pinned layout")
    for i in range(PLE_SHARDS):
        entry = header[f"{PLE_PREFIX}{i}.weight"]
        if (
            entry["dtype"] != "BF16"
            or entry["shape"] != [PLE_ROWS, PLE_WIDTH]
            or entry["data_offsets"] != [i * PLE_BYTES, (i + 1) * PLE_BYTES]
        ):
            raise ValueError("Invalid Intel PLE numeric shard layout: " + str(i))
    if PLE_OFFSET + PLE_SHARDS * PLE_BYTES != PLE_FILE_BYTES:
        raise ValueError("Intel PLE payload does not exactly fill the pinned file")
    return PLE_OFFSET, PLE_SHARDS * PLE_ROWS, PLE_WIDTH


def ple_layout(path: str | os.PathLike) -> tuple[int, int, int]:
    with open(path, "rb") as stream:
        return _ple_layout(stream)


def map_ple_table(path: str | os.PathLike) -> dict:
    """Return the tensor and mapping owner for immediate read-only protection.

    A private writable view lets torch.frombuffer construct the tensor. The
    caller must mprotect it before lookup; neither step touches table pages.
    """
    import torch

    with open(path, "rb") as stream:
        offset, rows, width = _ple_layout(stream)
        aligned = offset // mmap.ALLOCATIONGRANULARITY * mmap.ALLOCATIONGRANULARITY
        delta = offset - aligned
        owner = mmap.mmap(
            stream.fileno(),
            delta + rows * width * 2,
            flags=mmap.MAP_PRIVATE,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
            offset=aligned,
        )
        file_stat = os.fstat(stream.fileno())
    table = torch.frombuffer(
        owner, dtype=torch.bfloat16, count=rows * width, offset=delta
    ).view(rows, width)
    return {"mapping": owner, "table": table, "file_stat": file_stat}
