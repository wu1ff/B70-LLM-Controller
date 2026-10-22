"""Low overhead synchronized cycle timing without per-kernel instrumentation."""

import json
from pathlib import Path
import time
import torch


def run(worker, command):
    if command.get("stop"):
        for obj, name, original in reversed(worker._phase4_cycle_restores):
            setattr(obj, name, original)
        del worker._phase4_cycle_restores
        return {"rank": worker.rank, "cycles_stopped": True}
    from vllm.models.qwen4_exp.nvidia import c1_verify

    runner = worker.model_runner
    state = {"active": False, "records": [], "last_end": None}
    restores = []
    path = Path(f"/work/phase4-audit/{command['label']}-rank-{worker.rank}.json")

    def wrap(obj, name, fn):
        original = getattr(obj, name)
        restores.append((obj, name, original))
        setattr(obj, name, fn(original))

    def execute(original):
        def call(*args, **kwargs):
            sched = args[0] if args else kwargs["scheduler_output"]
            state["active"] = sched.total_num_scheduled_tokens == 4
            if state["active"]:
                state["start"] = time.perf_counter()
                state["spans"] = {}
                state["gap_ms"] = (
                    None
                    if state["last_end"] is None
                    else 1000 * (state["start"] - state["last_end"])
                )
                state["event"] = torch.xpu.Event(enable_timing=True)
                state["event"].record()
            return original(*args, **kwargs)

        return call

    def phase(name, sync=False):
        def wrapped(original):
            def call(*args, **kwargs):
                if not state["active"]:
                    return original(*args, **kwargs)
                if sync:
                    torch.xpu.synchronize()
                a = torch.xpu.Event(enable_timing=True)
                b = torch.xpu.Event(enable_timing=True)
                a.record()
                before = time.perf_counter()
                result = original(*args, **kwargs)
                b.record()
                host = 1000 * (time.perf_counter() - before)
                if sync:
                    b.synchronize()
                state["spans"][name] = (
                    a,
                    b,
                    host,
                    1000 * (time.perf_counter() - before),
                )
                if name == "sampling":
                    state["counts"] = result[1:3]
                return result

            return call

        return wrapped

    def finish(original):
        def call(*args, **kwargs):
            result = original(*args, **kwargs)
            if state["active"]:
                e = torch.xpu.Event(enable_timing=True)
                e.record()
                e.synchronize()
                end = time.perf_counter()
                item = dict(
                    wall_ms=1000 * (end - state["start"]),
                    queue_ms=state["event"].elapsed_time(e),
                    scheduler_gap_ms=state["gap_ms"],
                    spans={
                        k: dict(
                            queue_ms=a.elapsed_time(b),
                            host_ms=h,
                            synchronized_wall_ms=s,
                        )
                        for k, (a, b, h, s) in state["spans"].items()
                    },
                    sampled=state["counts"][0].cpu().tolist(),
                    rejected=state["counts"][1].cpu().tolist(),
                )
                state["records"].append(item)
                path.write_text(json.dumps(state["records"], indent=2))
                state["last_end"] = time.perf_counter()
                state["active"] = False
            return result

        return call

    wrap(runner, "execute_model", execute)
    wrap(c1_verify, "verify", phase("target", True))
    wrap(runner, "sample", phase("sampling"))
    wrap(runner.speculator, "propose", phase("draft"))
    wrap(runner, "sample_tokens", finish)
    worker._phase4_cycle_restores = restores
    return {"rank": worker.rank, "label": command["label"]}
