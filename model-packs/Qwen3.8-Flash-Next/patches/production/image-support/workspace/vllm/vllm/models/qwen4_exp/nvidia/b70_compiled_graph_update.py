# SPDX-License-Identifier: Apache-2.0
"""Compiled replacement for b70_target_graph.update(); identical semantics.

update(destination, source, seen) walks the STATIC captured destination tree,
asserting the live source schema and gathering device copies. The destination
tree of a captured entry never changes, so the walk order, the de-duplicated
node set and every check are fixed. This module compiles that walk once per
destination root into straight-line Python with the same expressions, the same
order, the same exception types (AssertionError/TypeError/AttributeError) and
the same final torch._foreach_copy_ submission.
"""
import dataclasses
import os

import torch

_CACHE = {}
# Optional single-launch byte copy for the gathered same-shape/same-dtype
# contiguous pairs. Off: the original torch._foreach_copy_ submission.
FUSED_COPY = os.getenv("B70_DFLASH_FUSED_GRAPH_COPY", "0") == "1"
_COPY = {}
_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        from vllm.triton_utils import tl, triton

        @triton.jit
        def copy_bytes(table, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            dst = tl.load(table + 3 * pid).to(tl.pointer_type(tl.uint8))
            src = tl.load(table + 3 * pid + 1).to(tl.pointer_type(tl.uint8))
            n = tl.load(table + 3 * pid + 2)
            for start in range(0, n, BLOCK):
                offsets = start + tl.arange(0, BLOCK)
                mask = offsets < n
                tl.store(dst + offsets, tl.load(src + offsets, mask=mask), mask=mask)

        _KERNEL = copy_bytes
    return _KERNEL


def _signed(address):
    # USM device addresses can exceed 2**63; store the same 64 bits as int64.
    return address - (1 << 64) if address >= (1 << 63) else address


def fused_copy(destination, destinations, sources):
    """Byte-identical to torch._foreach_copy_(destinations, sources) when every
    pair is contiguous with equal shape/dtype (asserted by the gather) and
    destinations do not overlap (separate static clones). Otherwise falls back.
    Any host-side failure before the launch falls back to _foreach_copy_ and
    disables the fused path for this destination root."""
    state = _COPY.get(id(destination))
    if state is None or state[0] is not destination or state[1] != len(destinations):
        copy = None
        try:
            if destinations and all(d.device == destinations[0].device for d in destinations):
                n = len(destinations)
                copy = dict(host=torch.zeros(3 * n, dtype=torch.int64, pin_memory=True),
                            table=torch.empty(3 * n, dtype=torch.int64, device=destinations[0].device),
                            key=None, event=torch.xpu.Event(), dsts=list(destinations),
                            dcontig=[d.is_contiguous() for d in destinations],
                            dptr=[_signed(d.data_ptr()) for d in destinations],
                            nbytes=[d.numel() * d.element_size() for d in destinations],
                            checked=False)
        except Exception:
            copy = None
        state = _COPY[id(destination)] = (destination, len(destinations), copy)
    copy = state[2]
    if copy is None or any(a is not b for a, b in zip(copy["dsts"], destinations)):
        torch._foreach_copy_(destinations, sources)
        return False
    try:
        # Contiguous pairs share one launch; any other pair keeps _foreach_copy_.
        fuse = [i for i, (c, s) in enumerate(zip(copy["dcontig"], sources)) if c and s.is_contiguous()]
        chosen = set(fuse)
        rest = [i for i in range(len(sources)) if i not in chosen]
        key = (tuple(fuse), tuple(sources[i].data_ptr() for i in fuse))
        if fuse and key != copy["key"]:
            copy["event"].synchronize()  # pinned source of any prior upload is free
            rows, host = len(fuse), copy["host"]
            host[0:3 * rows:3] = torch.tensor([copy["dptr"][i] for i in fuse], dtype=torch.int64)
            host[1:3 * rows:3] = torch.tensor([_signed(p) for p in key[1]], dtype=torch.int64)
            host[2:3 * rows:3] = torch.tensor([copy["nbytes"][i] for i in fuse], dtype=torch.int64)
            copy["table"][:3 * rows].copy_(host[:3 * rows], non_blocking=True)
            copy["event"].record()
            copy["key"] = key
        launch = _kernel() if fuse else None
    except Exception:
        _COPY[id(destination)] = (destination, len(destinations), None)
        torch._foreach_copy_(destinations, sources)
        return False
    if rest:
        torch._foreach_copy_([destinations[i] for i in rest], [sources[i] for i in rest])
    if not fuse:
        return False
    launch[(len(fuse),)](copy["table"], BLOCK=1024)
    if not copy["checked"]:
        # First use per root: exact byte check, else permanent fallback.
        torch.xpu.current_stream().synchronize()
        if not all(torch.equal(d, s) for d, s in zip(destinations, sources)):
            _COPY[id(destination)] = (destination, len(destinations), None)
            torch._foreach_copy_(destinations, sources)
            return False
        copy["checked"] = True
    return True


def _compile(destination):
    consts, lines, seen = [], [], set()
    counter = [0]

    def const(value):
        consts.append(value)
        return f"C[{len(consts) - 1}]"

    def reg():
        counter[0] += 1
        return f"s{counter[0]}"

    def visit(dst, s):
        if id(dst) in seen:
            return
        seen.add(id(dst))
        if isinstance(dst, torch.Tensor):
            d = const(dst)
            lines.append(f"assert {d}.shape == {s}.shape and {d}.dtype == {s}.dtype")
            lines.append(f"assert {d}.device == {s}.device")
            if dst.device.type == "xpu":
                lines.append(f"D.append({d}); S.append({s})")
            else:
                lines.append(f"{d}.copy_({s})")
        elif isinstance(dst, dict):
            lines.append(f"assert {const(dst)}.keys() == {s}.keys()")
            for key in dst:
                r = reg()
                lines.append(f"{r} = {s}[{const(key)}]")
                visit(dst[key], r)
        elif dataclasses.is_dataclass(dst):
            for field in dataclasses.fields(dst):
                if field.name == "max_seq_len":
                    lines.append(f"assert getattr({s}, 'max_seq_len') <= {const(dst)}.max_seq_len")
                    continue
                r = reg()
                lines.append(f"{r} = getattr({s}, {field.name!r})")
                visit(getattr(dst, field.name), r)
        elif isinstance(dst, (tuple, list)):
            lines.append(f"assert len({const(dst)}) == len({s})")
            for i, item in enumerate(dst):
                r = reg()
                # zip() order and pairing; length equality asserted above.
                lines.append(f"{r} = {s}[{i}]")
                visit(item, r)
        else:
            d = const(dst)
            lines.append(f"assert {d} == {s}, ({d}, {s})")

    visit(destination, "s0")
    body = "\n    ".join(lines) or "pass"
    source = f"def gather(s0, D, S):\n    {body}\n"
    namespace = {"C": consts}
    exec(compile(source, "<b70-compiled-graph-update>", "exec"), namespace)
    return namespace["gather"], len(lines), frozenset(seen)


def update(destination, source, seen):
    """Drop-in for b70_target_graph.update; ``seen`` must be a fresh empty set."""
    if seen:
        from vllm.models.qwen4_exp.nvidia import b70_target_graph as tg
        return tg._b70_original_update(destination, source, seen)
    entry = _CACHE.get(id(destination))
    if entry is None or entry[0] is not destination:
        gather, _, _ = _compile(destination)
        entry = _CACHE[id(destination)] = (destination, gather)
    destinations, sources = [], []
    entry[1](source, destinations, sources)
    if destinations:
        if FUSED_COPY:
            fused_copy(destination, destinations, sources)
        else:
            torch._foreach_copy_(destinations, sources)
