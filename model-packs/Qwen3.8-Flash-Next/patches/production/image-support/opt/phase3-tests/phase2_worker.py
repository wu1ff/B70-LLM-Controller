"""Instrumented campaign worker. Controlled fixtures require a local control file."""

import importlib.util
import json
from pathlib import Path
import traceback

import torch
from safetensors import safe_open
from vllm.v1.worker.xpu_worker import XPUWorker

DIRECTORY = Path("/work/phase2-audit")


class Phase2Worker(XPUWorker):
    def load_model(self, **kwargs):
        super().load_model(**kwargs)
        DIRECTORY.mkdir(exist_ok=True)
        runner = self.model_runner
        draft = runner.speculator.model
        target = runner.model.language_model
        assert type(runner.speculator).__name__ == "DFlash2Speculator"
        assert type(draft).__name__ == "DFlash2Qwen3ForCausalLM"
        assert draft.model.embed_tokens is target.model.embed_tokens
        assert draft.lm_head is target.lm_head
        assert draft.config.vocab_size == target.config.vocab_size == 248320
        assert draft.config.bos_token_id == target.config.bos_token_id == 248044
        assert draft.config.eos_token_id == target.config.eos_token_id == 248044
        checked = []
        params = dict(draft.named_parameters())
        with safe_open("/draft/model.safetensors", framework="pt") as checkpoint:
            for name in checkpoint.keys():
                dest = "model." + name
                expected = checkpoint.get_tensor(name)
                if ".q_proj." in name or ".k_proj." in name or ".v_proj." in name:
                    part = name.split(".")[-2][0]
                    dest = dest.replace("." + part + "_proj.", ".qkv_proj.")
                    widths = {"q": 1024, "k": 256, "v": 256}
                    starts = {"q": 0, "k": 1024, "v": 1280}
                    width = widths[part]
                    actual = params[dest][starts[part] : starts[part] + width]
                    expected = expected[self.rank * width : (self.rank + 1) * width]
                elif ".gate_proj." in name or ".up_proj." in name:
                    part = name.split(".")[-2]
                    dest = dest.replace("." + part + ".", ".gate_up_proj.")
                    offset = 0 if part == "gate_proj" else 1920
                    actual = params[dest][offset : offset + 1920]
                    expected = expected[self.rank * 1920 : (self.rank + 1) * 1920]
                elif ".o_proj." in name or ".down_proj." in name:
                    actual = params[dest]
                    width = expected.shape[1] // 4
                    expected = expected[:, self.rank * width : (self.rank + 1) * width]
                else:
                    actual = params[dest]
                assert actual.dtype == expected.dtype == torch.bfloat16, name
                torch.testing.assert_close(
                    actual.cpu(), expected, rtol=0, atol=0, msg=name
                )
                checked.append(name)
        spec = runner.speculator
        hc_modules = [m for m in target.modules() if hasattr(m, "_kdown_lo")]
        assert len(hc_modules) == 97
        assert all(m._ordered_hc for m in hc_modules)
        assert all(m._kdown_lo is None and m._kdown_hi is None for m in hc_modules)
        report = {
            "rank": self.rank,
            "hc_ordered_modules": len(hc_modules),
            "hc_ordered_all_widths": True,
            "draft_class": type(draft).__module__ + "." + type(draft).__name__,
            "speculator_class": type(spec).__module__ + "." + type(spec).__name__,
            "checkpoint_tensors_bitwise_verified": checked,
            "dtype": "bfloat16",
            "embedding_shared": True,
            "lm_head_shared": True,
            "vocab_size": 248320,
            "draft_id_mapping": draft.draft_id_to_target_id is not None,
            "proposal_dtype": str(spec.draft_logits.dtype),
            "proposal_buffer_bytes": spec.draft_logits.numel()
            * spec.draft_logits.element_size(),
            "parameters_bytes": sum(
                p.numel() * p.element_size()
                for n, p in params.items()
                if n not in ("model.embed_tokens.weight", "lm_head.weight")
            ),
            "allocated_after_load": torch.xpu.memory_allocated(),
            "reserved_after_load": torch.xpu.memory_reserved(),
            "draft_causal": draft.get_draft_attn_causal(),
            "sliding_windows": [
                layer.self_attn.attn.impl.sliding_window for layer in draft.model.layers
            ],
            "conv_block_sizes": [
                layer.attention_conv.block_size for layer in draft.model.layers
            ],
            "target_layer_types": target.config.layer_types,
        }
        (DIRECTORY / f"load-rank-{self.rank}.json").write_text(
            json.dumps(report, indent=2)
        )
        print("PHASE2_DRAFT_LOADED " + json.dumps(report), flush=True)
        self._audit_saved = set()
        self._audit_steps = 0
        self._audit_live = False
        self._pending_layers = {}
        decoder = target.model
        for index in (5, 15, 25, 35, 45):

            def hook(layer, args, kwargs, output, index=index):
                if self._audit_live and self._audit_steps < 2:
                    self._pending_layers.setdefault(index, []).append(
                        tuple(t.detach().clone() for t in output)
                    )

            decoder.layers[index].register_forward_hook(hook, with_kwargs=True)
        original = spec.propose

        def propose(*args, **kwargs):
            if kwargs.get("dummy_run", False):
                return original(*args, **kwargs)
            batch, metadata, slots, hidden, features, sampled, rejected = args[:7]
            control_path = DIRECTORY / "control.json"
            control = (
                json.loads(control_path.read_text()) if control_path.exists() else {}
            )
            if self._audit_steps < 2:
                assert len(features) == 5
                for index, feature in zip((5, 15, 25, 35, 45), features):
                    references = []
                    for residual, block, injection in self._pending_layers[index]:
                        completed = (
                            residual.float().view(-1, 4, 2560)
                            + block.float()[:, None, :]
                            * (2 * torch.sigmoid(injection.float() / 4))[:, :, None]
                        ).to(residual.dtype)
                        references.append(completed.mean(1))
                    torch.testing.assert_close(
                        feature, torch.cat(references), rtol=0.008, atol=0.0078125
                    )
                torch.save(
                    {
                        "features": [t.cpu() for t in features],
                        "positions": batch.positions.cpu(),
                    },
                    DIRECTORY / f"features-rank-{self.rank}-{self._audit_steps}.pt",
                )
                self._pending_layers.clear()
            tag = control.get("tag")
            mode = control.get("mode")
            capture = (
                tag
                and tag not in self._audit_saved
                and batch.num_reqs == 1
                and (
                    (mode == "replay" and batch.num_tokens == control["length"])
                    or (
                        mode != "replay"
                        and batch.num_tokens == 4
                        and not batch.has_prefill
                    )
                )
            )
            snapshot = None
            if capture:
                # Reload the observer so audit fixes do not require a model restart.
                module_spec = importlib.util.spec_from_file_location(
                    "phase2_state", "/tests/phase2_state.py"
                )
                observer = importlib.util.module_from_spec(module_spec)
                module_spec.loader.exec_module(observer)
                try:
                    snapshot = observer.snapshot(
                        self, batch, metadata, hidden, sampled, rejected
                    )
                except Exception:
                    (
                        DIRECTORY / f"{tag}-rank-{self.rank}-observer-error.txt"
                    ).write_text(traceback.format_exc())
                    capture = False
            result = original(*args, **kwargs)
            if capture:
                try:
                    snapshot["draft_context"] = observer.draft_context(
                        self, snapshot["length"]
                    )
                    torch.save(snapshot, DIRECTORY / f"{tag}-rank-{self.rank}.pt")
                    self._audit_saved.add(tag)
                except Exception:
                    (
                        DIRECTORY / f"{tag}-rank-{self.rank}-observer-error.txt"
                    ).write_text(traceback.format_exc())
            if mode in ("all", "first", "middle"):
                assert batch.num_reqs == 1
                length = int(batch.positions[-1].item()) + 1 - int(rejected[0].item())
                tokens = control["target_tokens"]
                proposal = tokens[length + 1 : length + 4]
                if len(proposal) == 3:
                    if mode in ("first", "middle"):
                        position = 0 if mode == "first" else 1
                        proposal[position] = (proposal[position] + 137) % 248000
                    result[0].copy_(torch.tensor(proposal, device=result.device))
                    state = int(batch.idx_mapping[0].item())
                    spec.draft_logits[state].fill_(-float("inf"))
                    for step, token in enumerate(proposal):
                        spec.draft_logits[state, step, token] = 0
                    # Fixture support replaces the sparse cache too, avoiding stale mass.
                    spec.candidate_sampler.cached_candidate_ids[state].copy_(
                        torch.tensor(proposal, device=result.device)[:, None].expand(
                            -1, spec.top_k
                        )
                    )
            if self._audit_steps < 80 or control:
                record = {
                    "step": self._audit_steps,
                    "tag": tag,
                    "mode": mode,
                    "req_ids": batch.req_ids,
                    "input_ids": batch.input_ids.cpu().tolist(),
                    "positions": batch.positions.cpu().tolist(),
                    "sampled": sampled.cpu().tolist(),
                    "rejected": rejected.cpu().tolist(),
                    "proposals": result.cpu().tolist(),
                    "features": [list(t.shape) for t in features],
                    "context_positions": spec.context_positions[: batch.num_tokens]
                    .cpu()
                    .tolist(),
                    "context_slots": spec._context_slot_mappings[:, : batch.num_tokens]
                    .cpu()
                    .tolist(),
                    "query_positions": spec.input_buffers.positions[
                        : 4 * batch.num_reqs
                    ]
                    .cpu()
                    .tolist(),
                    "allocated": torch.xpu.memory_allocated(),
                    "reserved": torch.xpu.memory_reserved(),
                }
                with (DIRECTORY / f"trace-rank-{self.rank}.jsonl").open("a") as file:
                    file.write(json.dumps(record) + "\n")
            self._audit_steps += 1
            return result

        spec.propose = propose

    def execute_model(self, *args, **kwargs):
        self._audit_live = True
        try:
            return super().execute_model(*args, **kwargs)
        finally:
            self._audit_live = False
