"""Refresh command methods without replacing the loaded model or caches."""

import importlib


def run(worker, command):
    import phase4_worker

    cls = importlib.reload(phase4_worker).Phase4Worker
    type(worker).phase4_command = cls.phase4_command
    return {"rank": worker.rank, "refreshed": True}
