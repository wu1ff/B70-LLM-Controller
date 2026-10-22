"""Compact allocator and shared cache report from an existing serving worker."""

import json
from pathlib import Path
import torch


def capture(worker):
    runner = worker.model_runner
    storages = {}
    layers = {}
    for (
        name,
        module,
    ) in runner.vllm_config.compilation_config.static_forward_context.items():
        cache = getattr(module, "kv_cache", ())
        cache = (cache,) if isinstance(cache, torch.Tensor) else cache
        if not isinstance(cache, (tuple, list)):
            continue
        layers[name] = []
        for tensor in cache:
            if not isinstance(tensor, torch.Tensor) or not tensor.numel():
                continue
            storage = tensor.untyped_storage()
            key = str(storage.data_ptr())
            storages[key] = storage.nbytes()
            layers[name].append(
                dict(
                    shape=list(tensor.shape),
                    stride=list(tensor.stride()),
                    dtype=str(tensor.dtype),
                    storage=key,
                )
            )
    Path(f"/work/phase3-audit/resources-rank-{worker.rank}.json").write_text(
        json.dumps(
            dict(
                rank=worker.rank,
                unique_cache_storage_bytes=sum(storages.values()),
                num_blocks=runner.kv_cache_config.num_blocks,
                cache_config=repr(runner.kv_cache_config),
                layers=layers,
                allocated=torch.xpu.memory_allocated(),
                reserved=torch.xpu.memory_reserved(),
                peak_allocated=torch.xpu.max_memory_allocated(),
                peak_reserved=torch.xpu.max_memory_reserved(),
            ),
            indent=2,
        )
    )
    for key, hook in list(runner.model._forward_pre_hooks.items()):
        if (
            getattr(hook, "__module__", "") == "phase3_worker"
            and getattr(hook, "__name__", "") == "count"
        ):
            del runner.model._forward_pre_hooks[key]
            runner.model._forward_pre_hooks_with_kwargs.pop(key, None)
    if not getattr(worker, "_phase3_count_fixed", False):

        def count(module, args, kwargs):
            if worker._phase3_widths is not None:
                worker._phase3_widths.append(int(kwargs["input_ids"].numel()))

        runner.model.register_forward_pre_hook(count, with_kwargs=True)
        worker._phase3_count_fixed = True
    from phase3_probe import install

    install(worker)
    control = json.loads(Path("/work/phase2-audit/control.json").read_text())
    if control.get("phase4_resources"):
        from phase4_resources import capture as capture_phase4

        capture_phase4(worker)
    if control.get("phase3_linear_probe"):
        import importlib
        import phase3_linear_probe

        importlib.reload(phase3_linear_probe).run(worker)
    if control.get("canonical_reductions") and not getattr(
        worker, "_phase3_reductions", False
    ):
        from vllm.models.qwen4_exp.nvidia import c1_verify
        from phase3_vector_impl import canonical_reductions

        previous = c1_verify.verify

        def verify(runner, batch, metadata, inputs):
            with canonical_reductions(batch.num_tokens):
                return previous(runner, batch, metadata, inputs)

        c1_verify.verify = verify
        worker._phase3_reductions = True
    from phase3_followup import install as install_followup

    install_followup(worker)
    if control.get("phase3_perf"):
        worker._audit_steps = 10000
    if not getattr(worker, "_phase3_collective_observer", False):
        from vllm.distributed import get_tp_group

        comm = get_tp_group().device_communicator
        native = comm.all_reduce
        worker._phase3_collective_widths = []

        def reduce(input_):
            if worker._phase3_widths is not None:
                worker._phase3_collective_widths.append(list(input_.shape))
            return native(input_)

        comm.all_reduce = reduce
        worker._phase3_collective_observer = True
        worker._phase3_native_reduce = native
        previous = runner.speculator.propose

        def observe_collectives(*args, **kwargs):
            cycle = worker._phase3_cycle
            if cycle is not None:
                record = dict(
                    req_ids=cycle["req_ids"],
                    mode=cycle["mode"],
                    width=cycle["width"],
                    reductions=worker._phase3_collective_widths,
                )
                with Path(
                    f"/work/phase3-audit/collectives-rank-{worker.rank}.jsonl"
                ).open("a") as f:
                    f.write(json.dumps(record) + "\n")
                worker._phase3_collective_widths = []
            return previous(*args, **kwargs)

        runner.speculator.propose = observe_collectives
    if control.get("measure_peak") and not getattr(
        worker, "_phase3_peak_observer", False
    ):
        from vllm.models.qwen4_exp.nvidia import c1_verify

        previous_verify = c1_verify.verify

        def measured(runner, batch, metadata, inputs):
            torch.xpu.reset_peak_memory_stats()
            return previous_verify(runner, batch, metadata, inputs)

        c1_verify.verify = measured
        worker._phase3_peak_observer = True
