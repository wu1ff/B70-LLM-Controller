"""Read logical committed cache contents for comparison with fresh target replay."""

from pathlib import Path
import json

import torch


def attention_prefix(module, metadata, length):
    cache = module.kv_cache
    assert isinstance(cache, torch.Tensor), type(cache)
    if cache.ndim == 4:
        # FlashAttention's native XPU cache is [blocks, heads, tokens, 2*dim].
        block_size = cache.shape[2]
        positions = torch.arange(length, device=cache.device)
        blocks = metadata.block_table[0, positions // block_size].long()
        selected = cache.transpose(1, 2)[blocks, positions % block_size]
        return torch.stack(selected.chunk(2, dim=-1)).cpu()
    assert cache.ndim == 5, cache.shape
    if cache.shape[0] != 2:
        assert cache.shape[1] == 2, cache.shape
        cache = cache.movedim(1, 0)
    block_size = cache.shape[2]
    block_table = metadata.block_table[0]
    positions = torch.arange(length, device=cache.device)
    blocks = block_table[positions // block_size].long()
    return cache[:, blocks, positions % block_size].cpu()


def snapshot(worker, batch, metadata, hidden, sampled, rejected):
    control = json.loads(Path("/work/phase2-audit/control.json").read_text())
    if control.get("phase3_resources"):
        import importlib
        import phase3_resources

        importlib.reload(phase3_resources).capture(worker)
    if control.get("finish_audit"):
        from types import MethodType
        import importlib
        import gc
        from vllm.models.qwen4_exp.nvidia import c1_verify

        spec = worker.model_runner.speculator
        spec.propose = MethodType(type(spec).propose, spec)
        sampler = spec.candidate_sampler
        sampler.sample = MethodType(type(sampler).sample, sampler)
        importlib.reload(c1_verify)
        for module in worker.model_runner.model.modules():
            for key, hook in list(module._forward_hooks.items()):
                if getattr(hook, "__module__", "") == "phase2_worker":
                    del module._forward_hooks[key]
                    module._forward_hooks_with_kwargs.pop(key, None)
                    module._forward_hooks_always_called.pop(key, None)
        for module in worker.model_runner.model.modules():
            for key, hook in list(module._forward_pre_hooks.items()):
                if getattr(hook, "__module__", "").startswith("phase3_"):
                    del module._forward_pre_hooks[key]
                    module._forward_pre_hooks_with_kwargs.pop(key, None)
            for key, hook in list(module._forward_hooks.items()):
                if getattr(hook, "__module__", "").startswith("phase3_"):
                    del module._forward_hooks[key]
                    module._forward_hooks_with_kwargs.pop(key, None)
                    module._forward_hooks_always_called.pop(key, None)
        if hasattr(worker, "_audit_current_features"):
            del worker._audit_current_features
        worker._pending_layers.clear()
        gc.collect()
        Path(f"/work/phase2-audit/audit-finished-rank-{worker.rank}.json").write_text(
            json.dumps(
                {
                    "rank": worker.rank,
                    "propose": spec.propose.__qualname__,
                    "sampler": sampler.sample.__qualname__,
                    "verifier": c1_verify.verify.__module__,
                    "debug_layer_hooks_removed": True,
                },
                indent=2,
            )
        )
        return _snapshot(worker, batch, metadata, hidden, sampled, rejected)
    if control.get("ordered_hc"):
        from types import MethodType
        from phase2_ordered_hc import ordered_hc_gemm

        def down(module, x):
            linear = (
                module.input_mix_weight_down_block_inject
                if module.use_combine
                else module.input_mix_weight_down
            )
            valid = module.lora_rank + (module.hc_count if module.use_combine else 0)
            return ordered_hc_gemm(x, linear.weight, valid)

        for module in worker.model_runner.model.modules():
            if hasattr(module, "_down_gemm"):
                module._down_gemm = MethodType(down, module)
    if control.get("split_hc_all_widths"):
        from vllm.models.qwen4_exp.nvidia import b70_hyperconnection

        b70_hyperconnection._HC_KSPLIT_MAX_STOCK_WIDTH = 0
    if control.get("restore_collectives"):
        from types import MethodType
        from vllm.distributed import get_tp_group

        comm = get_tp_group().device_communicator
        comm.all_reduce = MethodType(type(comm).all_reduce, comm)
    if control.get("collective_probe"):
        from phase2_collective_probe import run

        run(worker)
    cache_report = Path(f"/work/phase2-audit/cache-rank-{worker.rank}.json")
    if not cache_report.exists():
        runner = worker.model_runner
        storages = {}
        layers = {}
        for (
            name,
            module,
        ) in runner.vllm_config.compilation_config.static_forward_context.items():
            cache = getattr(module, "kv_cache", ())
            tensors = (cache,) if isinstance(cache, torch.Tensor) else cache
            if not isinstance(tensors, (tuple, list)):
                continue
            layers[name] = []
            for tensor in tensors:
                if not isinstance(tensor, torch.Tensor) or not tensor.numel():
                    continue
                storage = tensor.untyped_storage()
                key = str(storage.data_ptr())
                storages.setdefault(key, {"bytes": storage.nbytes(), "owners": []})[
                    "owners"
                ].append(name)
                layers[name].append(
                    {
                        "shape": list(tensor.shape),
                        "stride": list(tensor.stride()),
                        "dtype": str(tensor.dtype),
                        "storage": key,
                    }
                )
        cache_report.write_text(
            json.dumps(
                {
                    "rank": worker.rank,
                    "config": repr(runner.kv_cache_config),
                    "unique_cache_storage_bytes": sum(
                        s["bytes"] for s in storages.values()
                    ),
                    "storages": storages,
                    "layers": layers,
                    "allocated": torch.xpu.memory_allocated(),
                    "reserved": torch.xpu.memory_reserved(),
                    "peak_allocated": torch.xpu.max_memory_allocated(),
                },
                indent=2,
            )
        )
    from vllm.models.qwen4_exp.nvidia import c1_verify

    if not getattr(c1_verify, "_observed", False):
        original_verify = c1_verify.verify

        def verify(runner, input_batch, attn_metadata, model_inputs):
            result = original_verify(runner, input_batch, attn_metadata, model_inputs)
            if not Path("/work/phase2-audit/audit-complete").exists():
                with Path(
                    f"/work/phase2-audit/canonical-rank-{worker.rank}.jsonl"
                ).open("a") as file:
                    file.write(
                        json.dumps(
                            {
                                "positions": input_batch.positions.cpu().tolist(),
                                "hidden_shape": list(result[0].shape),
                                "features": [list(t.shape) for t in result[1]],
                            }
                        )
                        + "\n"
                    )
            return result

        c1_verify.verify = verify
        c1_verify._observed = True
    spec = worker.model_runner.speculator
    if not getattr(spec, "_feature_observer", False):
        original = spec.propose

        def observe(*args, **kwargs):
            if args:
                worker._audit_current_features = args[4]
            return original(*args, **kwargs)

        spec.propose = observe
        spec._feature_observer = True
    control = json.loads(Path("/work/phase2-audit/control.json").read_text())
    if control.get("incremental_prefix"):
        from phase2_incremental import replay

        if control.get("hc_probe"):
            import importlib

            probe = importlib.reload(importlib.import_module("phase2_hc_probe")).probe
            result, features = probe(
                worker, lambda: replay(worker, batch, control, _snapshot)
            )
        elif control.get("full_probe"):
            import importlib

            probe = importlib.reload(importlib.import_module("phase2_full_probe")).probe
            result, features = probe(
                worker, lambda: replay(worker, batch, control, _snapshot)
            )
        elif control.get("moe_probe"):
            from phase2_moe_probe import probe

            result, features = probe(
                worker, lambda: replay(worker, batch, control, _snapshot)
            )
        else:
            result, features = replay(worker, batch, control, _snapshot)
        global _incremental_aux
        _incremental_aux = features
        return result
    result = _snapshot(worker, batch, metadata, hidden, sampled, rejected)
    if control.get("draft_mutation_probe"):
        global _draft_mutation_probe
        _draft_mutation_probe = (batch, metadata, hidden, sampled, rejected, result)
    return result


def _snapshot(worker, batch, metadata, hidden, sampled, rejected):
    runner = worker.model_runner
    control = json.loads(Path("/work/phase2-audit/control.json").read_text())
    if control.get("request_id") and control["request_id"] not in batch.req_ids[0]:
        raise ValueError("Delayed previous request: skip this fixture snapshot")
    sampler = runner.speculator.candidate_sampler
    if not getattr(sampler, "_audited", False):
        original_sample = sampler.sample
        sampler._audited = True
        sampler._audit_count = 0

        def sample(*args, **kwargs):
            result = original_sample(*args, **kwargs)
            if Path("/work/phase2-audit/audit-complete").exists():
                return result
            (
                candidates,
                scores,
                num_reqs,
                positions,
                mapping,
                temp,
                seeds,
                tokens,
                cache,
            ) = args[:9]
            for row in range(num_reqs):
                req = int(mapping[row * 3].item())
                if req < 0:
                    continue
                previous = 0
                for step in range(3):
                    realized = scores[row, step, previous].float()
                    torch.testing.assert_close(sampler.scores[row, step], realized)
                    torch.testing.assert_close(
                        cache[req, step, candidates[row, step]], realized
                    )
                    finite_ids = (
                        cache[req, step].isfinite().nonzero().flatten().sort().values
                    )
                    torch.testing.assert_close(
                        finite_ids, candidates[row, step].sort().values
                    )
                    previous = int(
                        (candidates[row, step] == tokens[row, step]).nonzero().item()
                    )
            if sampler._audit_count < 3:
                torch.save(
                    {
                        "candidates": candidates.cpu(),
                        "scores": scores.cpu(),
                        "realized_scores": sampler.scores[:num_reqs].cpu(),
                        "proposals": tokens[:num_reqs].cpu(),
                        "mapping": mapping.cpu(),
                        "temperature": temp.cpu(),
                        "positions": positions.cpu(),
                    },
                    Path("/work/phase2-audit")
                    / f"selector-rank-{worker.rank}-{sampler._audit_count}.pt",
                )
            sampler._audit_count += 1
            return result

        sampler.sample = sample
    assert batch.num_reqs == 1
    count = int(sampled[0].item())
    length = int(batch.positions[-1].item()) + 1 - int(rejected[0].item())
    row = batch.num_tokens - 1 - int(rejected[0].item())
    state = int(batch.idx_mapping[0].item())
    result = {
        "request_ids": batch.req_ids,
        "rank": worker.rank,
        "length": length,
        "accepted_inputs": count,
        "rejected": int(rejected[0].item()),
        "hidden": hidden[row].cpu(),
        "tokens": runner.req_states.all_token_ids.gpu[state, :length].cpu(),
        "ngram_context": runner.model_state.ngram_context.cpu(),
        "states": {},
        "attention": {},
    }
    for name, module in runner.model.named_modules():
        prefix = getattr(module, "prefix", getattr(module, "layer_name", None))
        if prefix not in metadata or not hasattr(module, "kv_cache"):
            continue
        meta = metadata[prefix]
        if isinstance(module.kv_cache, torch.Tensor):
            if module.kv_cache.ndim in (4, 5):
                result["attention"][name] = attention_prefix(module, meta, length)
            continue
        if not isinstance(module.kv_cache, (tuple, list)) or not module.kv_cache:
            continue
        indices = getattr(meta, "spec_state_indices_tensor", None)
        is_spec = indices is not None and getattr(meta, "num_spec_decodes", 0) > 0
        if is_spec:
            base = int(indices.flatten()[0].item())
            selected = int(indices[0, count - 1].item()) if indices.ndim == 2 else base
            offset = count - 1
        else:
            indices = getattr(meta, "non_spec_state_indices_tensor", None)
            if indices is None:
                indices = meta.state_indices_tensor
            base = selected = int(indices.flatten()[0].item())
            offset = 0
        conv = module.kv_cache[0][base]
        if conv.shape[0] > conv.shape[1]:
            conv = conv.T
        window = getattr(module, "conv_state_len", 3)
        # PLE takes the trailing, unpadded state-capacity view before dispatch.
        if hasattr(module, "conv_state_len"):
            conv = conv[-(window + module.num_spec_tokens) :]
        result["states"][name + ".conv"] = conv[offset : offset + window].cpu()
        if len(module.kv_cache) == 2:
            result["states"][name + ".ssm"] = module.kv_cache[1][selected].cpu()
    assert len(result["states"]) == 73, list(result["states"])
    assert len(result["attention"]) == 12, list(result["attention"])
    return result


def draft_context(worker, length):
    draft = worker.model_runner.speculator
    if "_draft_mutation_probe" in globals():
        batch, metadata, hidden, sampled, rejected, before = _draft_mutation_probe
        after = _snapshot(worker, batch, metadata, hidden, sampled, rejected)
        changed = [
            group + "/" + name
            for group in ("states", "attention")
            for name, tensor in before[group].items()
            if not torch.equal(tensor, after[group][name])
        ]
        Path(f"/work/phase2-audit/draft-mutation-rank-{worker.rank}.json").write_text(
            json.dumps(changed)
        )
    if "_incremental_aux" in globals():
        features = [torch.cat(parts) for parts in zip(*_incremental_aux)]
        states = draft.model.combine_hidden_states(torch.cat(features, -1))
        positions = torch.arange(length, device=states.device)
        mappings = []
        for gid in draft.draft_kv_cache_group_ids:
            block_size = draft.block_tables.block_sizes[gid]
            table = draft.block_tables.input_block_tables[gid][0]
            mappings.append(
                table[positions // block_size].long() * block_size
                + positions % block_size
            )
        draft.model.precompute_and_store_context_kv(states, positions, mappings)
    result = {}
    # Draft metadata may use a different KV group and block size than the target.
    for layer in draft.model.model.layers:
        attn = layer.self_attn.attn
        gid = next(
            gid
            for gid in draft.draft_kv_cache_group_ids
            if attn.layer_name in draft.kv_cache_config.kv_cache_groups[gid].layer_names
        )
        block_table = draft.block_tables.input_block_tables[gid]

        class Metadata:
            pass

        metadata = Metadata()
        metadata.block_table = block_table
        result[attn.layer_name] = attention_prefix(attn, metadata, length)
    return result
