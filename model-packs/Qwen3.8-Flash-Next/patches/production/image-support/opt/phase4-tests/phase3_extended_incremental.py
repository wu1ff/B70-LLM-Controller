"""Controlled one-token target replay from an identical real prefill boundary."""

from dataclasses import replace

import numpy as np
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer


def replay(worker, batch, control, capture):
    runner = worker.model_runner
    req = int(batch.idx_mapping[0].item())
    tokens = control["incremental_prefix"]
    start = batch.num_tokens
    assert batch.num_reqs == 1 and batch.has_prefill and start == 13
    assert 14 <= len(tokens) <= 40
    torch.testing.assert_close(
        batch.input_ids.cpu(), torch.tensor(tokens[:start], dtype=batch.input_ids.dtype)
    )
    features = [[t.clone() for t in worker._audit_current_features]]
    computed_before = runner.req_states.num_computed_tokens.gpu[req].clone()
    accepted_before = runner.model_state.num_accepted_tokens_gpu[req].clone()
    history_before = runner.req_states.all_token_ids.gpu[req, : len(tokens)].clone()
    runner.req_states.all_token_ids.gpu[req, : len(tokens)] = torch.tensor(
        tokens, device=runner.device
    )
    result = None
    try:
        for position in range(start, len(tokens)):
            runner.req_states.num_computed_tokens.gpu[req] = position
            runner.model_state.num_accepted_tokens_gpu[req] = 1
            qsl = torch.tensor([0, 1], device=runner.device, dtype=torch.int32)
            positions = torch.tensor(
                [position], device=runner.device, dtype=torch.int64
            )
            single = replace(
                batch,
                num_tokens=1,
                num_tokens_after_padding=1,
                num_scheduled_tokens=np.array([1], np.int32),
                num_draft_tokens=0,
                num_draft_tokens_per_req=np.array([0], np.int32),
                query_start_loc=qsl,
                query_start_loc_np=np.array([0, 1], np.int32),
                seq_lens=torch.tensor(
                    [position + 1], device=runner.device, dtype=torch.int32
                ),
                seq_lens_cpu_upper_bound=torch.tensor(
                    [position + 1], dtype=torch.int32
                ),
                num_computed_tokens_np=np.array([position], np.int32),
                num_computed_prefill_tokens_np=np.array([start], np.int32),
                is_prefilling_np=np.array([False]),
                has_prefill=False,
                decode_graph_eligible=True,
                input_ids=torch.tensor(
                    [tokens[position]], device=runner.device, dtype=torch.int32
                ),
                positions=positions,
                is_padding=torch.tensor([False], device=runner.device),
                logits_indices=torch.tensor(
                    [0], device=runner.device, dtype=torch.int64
                ),
                cu_num_logits=qsl,
                cu_num_logits_np=np.array([0, 1], np.int32),
                max_query_len=1,
                prefill_runs_as_decode_np=None,
            )
            tables = runner.block_tables.gather_block_tables(single.idx_mapping, 1)
            slot_storage = torch.empty_like(runner.block_tables.slot_mappings)
            slots = runner.block_tables.compute_slot_mappings(
                single.idx_mapping, qsl, positions, 1, out=slot_storage
            )
            runner.model_state.preprocess_state(
                single,
                tables,
                runner.kv_cache_config,
                runner.req_states.num_computed_tokens.gpu,
            )
            # Build a true target-only decode row, with no speculative metadata.
            spec_config = runner.vllm_config.speculative_config
            try:
                runner.vllm_config.speculative_config = None
                metadata = runner.model_state.prepare_attn(
                    single,
                    CUDAGraphMode.NONE,
                    tables,
                    slots,
                    runner.attn_groups,
                    runner.kv_cache_config,
                )
            finally:
                runner.vllm_config.speculative_config = spec_config
            inputs = {
                "input_ids": single.input_ids,
                "positions": positions,
                "inputs_embeds": None,
                "intermediate_tensors": None,
                **runner.model_state.prepare_inputs(single, runner.req_states),
            }
            runner.eplb.prepare_forward(runner.model_config, 1, None)
            with set_forward_context(
                metadata,
                runner.vllm_config,
                num_tokens=1,
                batch_descriptor=BatchDescriptor(num_tokens=1),
                slot_mapping=build_slot_mappings_by_layer(
                    slots, runner.kv_cache_config
                ),
                is_padding=single.is_padding,
            ):
                hidden, aux = runner.model(**inputs)
            features.append([t.clone() for t in aux])
            runner.req_states.num_computed_tokens.gpu[req] = position + 1
            one = torch.ones(1, dtype=torch.int32, device=runner.device)
            zero = torch.zeros_like(one)
            runner.model_state.postprocess_state(
                single.idx_mapping, one, runner.req_states.num_computed_tokens.gpu
            )
            result = capture(worker, single, metadata, hidden, one, zero)
        result["oracle"] = (
            "real prefill13 followed by non-speculative one-token target forwards"
        )
        return result, features
    finally:
        runner.req_states.num_computed_tokens.gpu[req] = computed_before
        runner.model_state.num_accepted_tokens_gpu[req] = accepted_before
        runner.req_states.all_token_ids.gpu[req, : len(tokens)] = history_before
