"""Observe the next committed state after consuming a bonus or recovery token."""

import importlib.util
import json
from pathlib import Path
import torch


def install(worker):
    if getattr(worker, "_phase3_followup_installed", False):
        return
    worker._phase3_followup_installed = True
    spec = worker.model_runner.speculator
    original = spec.propose
    saved = set()

    def propose(*args, **kwargs):
        ctrl = Path("/work/phase2-audit/control.json")
        c = json.loads(ctrl.read_text()) if ctrl.exists() else {}
        tag = c.get("tag")
        capture = bool(
            c.get("followup_capture")
            and tag in worker._audit_saved
            and tag not in saved
            and args
            and args[0].num_tokens == 4
            and c.get("request_id", "") in args[0].req_ids[0]
        )
        snapshot = None
        if capture:
            ms = importlib.util.spec_from_file_location(
                "phase3_followup_state", "/tests/phase2_state.py"
            )
            observer = importlib.util.module_from_spec(ms)
            ms.loader.exec_module(observer)
            batch, metadata, slots, hidden, features, sampled, rejected = args[:7]
            snapshot = observer._snapshot(
                worker, batch, metadata, hidden, sampled, rejected
            )
        result = original(*args, **kwargs)
        if snapshot is not None:
            snapshot["draft_context"] = observer.draft_context(
                worker, snapshot["length"]
            )
            torch.save(
                snapshot,
                Path("/work/phase2-audit") / f"{tag}-followup-rank-{worker.rank}.pt",
            )
            saved.add(tag)
        return result

    spec.propose = propose
