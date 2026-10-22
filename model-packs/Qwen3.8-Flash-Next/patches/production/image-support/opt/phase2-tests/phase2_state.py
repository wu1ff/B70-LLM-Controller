"""Read logical committed cache contents for comparison with fresh target replay."""

from pathlib import Path

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
    runner = worker.model_runner
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
