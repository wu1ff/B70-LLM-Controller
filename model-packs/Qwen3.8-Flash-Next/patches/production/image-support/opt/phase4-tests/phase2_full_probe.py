"""Repeat one real target forward after restoring its exact physical state."""

import json
from pathlib import Path
import torch
from vllm.forward_context import get_forward_context


def probe(worker, run):
    runner = worker.model_runner
    saved = {}
    layers = [[], []]
    mlp_inputs = [[], []]
    phase = [0]
    handles = []

    def caches():
        result = []
        metadata = get_forward_context().attn_metadata
        for name, meta in metadata.items():
            module = runner.vllm_config.compilation_config.static_forward_context[name]
            cache = getattr(module, "kv_cache", ())
            if isinstance(cache, (tuple, list)) and cache:
                index = (
                    int(meta.non_spec_state_indices_tensor.flatten()[0])
                    if getattr(meta, "non_spec_state_indices_tensor", None) is not None
                    else int(meta.state_indices_tensor.flatten()[0])
                )
                for j, tensor in enumerate(cache):
                    result.append((name + str(j), tensor[index], tensor[index].clone()))
        return result

    def before(model, args, kwargs):
        saved["before"] = caches()
        layers[0].clear()
        layers[1].clear()
        mlp_inputs[0].clear()
        mlp_inputs[1].clear()

    def after(model, args, kwargs, output):
        first = (output[0].clone(), [t.clone() for t in output[1]])
        final = caches()
        for name, tensor, value in saved["before"]:
            tensor.copy_(value)
        for module in runner.model_state._disk_ple_modules:
            ids = module.compute_ngram_ids(
                kwargs["input_ids"], kwargs["query_start_loc"], kwargs["ngram_context"]
            )
            module.ngram_embedding.prepare(ids, 1, 1)
        phase[0] = 1
        second = model.forward(*args, **kwargs)
        differences = []
        for index, (a, b) in enumerate(zip(layers[0], layers[1])):
            differences.append(
                {
                    "layer": index,
                    "max_abs": [
                        float((x.float() - y.float()).abs().max()) for x, y in zip(a, b)
                    ],
                    "equal": all(torch.equal(x, y) for x, y in zip(a, b)),
                }
            )
        state_changes = [
            name for name, tensor, value in final if not torch.equal(tensor, value)
        ]
        result = {
            "mlp_inputs": [
                {
                    "layer": i,
                    "equal": torch.equal(a, b),
                    "max_abs": float((a.float() - b.float()).abs().max()),
                }
                for i, (a, b) in enumerate(zip(mlp_inputs[0], mlp_inputs[1]))
            ],
            "hidden_relative": float(
                (first[0].float() - second[0].float()).norm() / first[0].float().norm()
            ),
            "hidden_equal": torch.equal(first[0], second[0]),
            "state_changes": state_changes,
            "layers": differences,
        }
        Path(f"/work/phase2-audit/full-probe-rank-{worker.rank}.json").write_text(
            json.dumps(result, indent=2)
        )
        for name, tensor, value in final:
            tensor.copy_(value)
        phase[0] = 0
        return first

    for layer in runner.model.language_model.model.layers:

        def input_hook(module, args, kwargs):
            mlp_inputs[phase[0]].append(args[0].clone())

        handles.append(
            layer.mlp.register_forward_pre_hook(input_hook, with_kwargs=True)
        )

        def layer_hook(module, args, kwargs, output):
            layers[phase[0]].append(tuple(t.clone() for t in output))

        handles.append(layer.register_forward_hook(layer_hook, with_kwargs=True))
    handles.append(runner.model.register_forward_pre_hook(before, with_kwargs=True))
    handles.append(runner.model.register_forward_hook(after, with_kwargs=True))
    try:
        return run()
    finally:
        for handle in handles:
            handle.remove()
