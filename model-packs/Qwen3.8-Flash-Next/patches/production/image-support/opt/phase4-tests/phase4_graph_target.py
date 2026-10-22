"""Fixed C1 piecewise replay using upstream breakable capture and eager XCCL."""

import json
from pathlib import Path
import time
import traceback
import torch


def run(worker, command):
    from vllm.models.qwen4_exp.nvidia import c1_verify
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    from vllm.distributed import get_tp_group
    from vllm.forward_context import get_forward_context

    runner = worker.model_runner
    original = c1_verify.verify
    state = {"capture": None, "replays": 0, "fallbacks": 0, "disabled": False}
    pool = torch.xpu.graph_pool_handle()
    path = Path(f"/work/phase4-audit/target-graph-rank-{worker.rank}.json")

    def report(**extra):
        path.write_text(
            json.dumps(
                {k: v for k, v in state.items() if k != "capture"} | extra, indent=2
            )
        )

    def verify(runner, batch, metadata, inputs):
        if batch.num_tokens != 4 or state["disabled"]:
            state["fallbacks"] += 1
            return original(runner, batch, metadata, inputs)
        if state["capture"] is not None:
            for name, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    state["inputs"][name].copy_(value)
            state["capture"].replay()
            state["replays"] += 1
            if state["replays"] in (1, 2, 10, 50, 100):
                report(
                    graphs=state["capture"].num_graphs,
                    eager_breaks=state["capture"].num_eager_breaks,
                )
            return state["output"]
        saved = []
        context = runner.vllm_config.compilation_config.static_forward_context
        seen = set()
        for name, meta in metadata.items():
            cache = getattr(context[name], "kv_cache", None)
            if not isinstance(cache, (tuple, list)) or not cache:
                continue
            indices = getattr(meta, "spec_state_indices_tensor", None)
            if indices is None:
                continue
            indices = indices.flatten().unique().long()
            for tensor in cache:
                if tensor.data_ptr() in seen:
                    continue
                seen.add(tensor.data_ptr())
                saved.append((tensor, indices, tensor[indices].clone()))
        static = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        state["inputs"] = static
        comm = get_tp_group().device_communicator
        before = torch.xpu.memory_reserved()
        begin = time.perf_counter()
        try:
            with c1_verify.canonical_reductions(4):
                native = comm.all_reduce

                def reduce(x):
                    capture = BreakableCUDAGraphCapture.current()
                    if capture is None or not capture._capturing:
                        return native(x)
                    output = torch.empty_like(x)
                    capture.add_eager(lambda: output.copy_(native(x)))
                    return output

                comm.all_reduce = reduce
                try:
                    cap = BreakableCUDAGraphCapture(pool=pool)
                    with cap:
                        output = runner.model(**static)
                finally:
                    comm.all_reduce = native
            for tensor, indices, clone in saved:
                tensor[indices] = clone
            state.update(capture=cap, output=output)
            cap.replay()
            state["replays"] += 1
            torch.xpu.synchronize()
            # Tensors and callable objects are reported separately from JSON.
            state["capture_ms"] = 1000 * (time.perf_counter() - begin)
            state["pool_reserved_delta"] = torch.xpu.memory_reserved() - before
            report(graphs=cap.num_graphs, eager_breaks=cap.num_eager_breaks)
            return output
        except Exception:
            for tensor, indices, clone in saved:
                tensor[indices] = clone
            state.update(disabled=True, error=traceback.format_exc())
            report()
            return original(runner, batch, metadata, inputs)

    # Exclude tensors from the diagnostic JSON serializer.
    old_report = report

    def report(**extra):
        path.write_text(
            json.dumps(
                {
                    k: v
                    for k, v in state.items()
                    if k not in ("capture", "inputs", "output")
                }
                | extra,
                indent=2,
            )
        )

    c1_verify.verify = verify
    worker._phase4_target_graph = state
    return {
        "rank": worker.rank,
        "installed": "breakable C1 target; communication remains eager",
    }
