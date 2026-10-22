"""Extend Phase 2's replay observer with verification-width and timing evidence."""

import importlib.util
import json
from pathlib import Path
import time

import torch
from phase2_worker import Phase2Worker

DIRECTORY = Path("/work/phase3-audit")
CONTROL = Path("/work/phase2-audit/control.json")


class Phase3Worker(Phase2Worker):
    def load_model(self, **kwargs):
        super().load_model(**kwargs)
        DIRECTORY.mkdir(exist_ok=True)
        from vllm.models.qwen4_exp.nvidia import c1_verify

        serial_spec = importlib.util.spec_from_file_location(
            "phase3_serial_reference", "/tests/phase3_serial_reference.py"
        )
        serial = importlib.util.module_from_spec(serial_spec)
        serial_spec.loader.exec_module(serial)
        vector_verify = c1_verify.verify
        runner = self.model_runner
        self._phase3_widths = None
        self._phase3_cycle = None
        self._phase3_saved = set()

        def count(module, args, kwargs):
            if self._phase3_widths is not None:
                self._phase3_widths.append(int(kwargs["input_ids"].numel()))

        runner.model.register_forward_pre_hook(count, with_kwargs=True)

        def verify(runner, batch, metadata, inputs):
            control = json.loads(CONTROL.read_text()) if CONTROL.exists() else {}
            mode = control.get("verify", "vector")
            self._phase3_widths = []
            torch.xpu.synchronize()
            allocated = torch.xpu.memory_allocated()
            start = torch.xpu.Event(enable_timing=True)
            end = torch.xpu.Event(enable_timing=True)
            start.record()
            before = time.perf_counter()
            result = (serial.verify if mode == "serial" else vector_verify)(
                runner, batch, metadata, inputs
            )
            end.record()
            end.synchronize()
            elapsed = time.perf_counter() - before
            widths = self._phase3_widths
            self._phase3_widths = None
            self._phase3_cycle = dict(
                mode=mode,
                req_ids=batch.req_ids,
                width=batch.num_tokens,
                model_forward_widths=widths,
                target_invocations=len(widths),
                wall_s=elapsed,
                device_ms=start.elapsed_time(end),
                allocated_before=allocated,
                allocated_after=torch.xpu.memory_allocated(),
                reserved=torch.xpu.memory_reserved(),
                peak_allocated=torch.xpu.max_memory_allocated(),
            )
            tag = control.get("tag")
            if tag and control.get("capture_logits") and tag not in self._phase3_saved:
                logits = runner.model.compute_logits(result[0])
                torch.save(
                    dict(
                        logits=logits.cpu(),
                        hidden=result[0].cpu(),
                        features=[x.cpu() for x in result[1]],
                        input_ids=batch.input_ids.cpu(),
                        positions=batch.positions.cpu(),
                    ),
                    DIRECTORY / f"{tag}-logits-rank-{self.rank}.pt",
                )
                self._phase3_saved.add(tag)
            return result

        c1_verify.verify = verify
        original = runner.speculator.propose

        def propose(*args, **kwargs):
            if kwargs.get("dummy_run", False):
                return original(*args, **kwargs)
            batch, metadata, slots, hidden, features, sampled, rejected = args[:7]
            cycle = self._phase3_cycle
            self._phase3_cycle = None
            if cycle is not None:
                cycle.update(
                    sampled=sampled.cpu().tolist(),
                    rejected=rejected.cpu().tolist(),
                    input_ids=batch.input_ids.cpu().tolist(),
                    positions=batch.positions.cpu().tolist(),
                )
                with (DIRECTORY / f"cycles-rank-{self.rank}.jsonl").open("a") as f:
                    f.write(json.dumps(cycle) + "\n")
            return original(*args, **kwargs)

        runner.speculator.propose = propose
