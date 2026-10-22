"""Event spans report queue elapsed time; Kineto reports kernel execution."""

import json
from pathlib import Path
import time

import torch


class Profile:
    def __init__(self, worker, config):
        self.worker, self.config = worker, config
        self.active = False
        self.step = 0
        self.records, self.spans, self.restores, self.hooks = [], [], [], []
        self.directory = Path("/work/phase4-audit")
        self.directory.mkdir(exist_ok=True)
        runner = worker.model_runner
        self.wrap(runner, "execute_model", "execute")
        self.wrap(runner, "sample_tokens", "sample_tokens")
        self.wrap(runner, "sample", "target_sample")
        self.wrap(runner.model_state, "prepare_inputs", "state_inputs_disk_ple")
        self.wrap(runner.model_state, "prepare_attn", "state_attention_metadata")
        self.wrap(runner.model_state, "preprocess_state", "state_rollback")
        from vllm.models.qwen4_exp.nvidia import c1_verify

        self.wrap(c1_verify, "verify", "target_verify")
        from vllm.distributed import get_tp_group

        if not config.get("coarse"):
            self.wrap(get_tp_group().device_communicator, "all_reduce", "xccl_native")
        spec = runner.speculator
        self.wrap(spec, "propose", "draft_total")
        self.wrap(spec, "_run_model", "draft_backbone")
        self.wrap(spec, "_precompute_context_kv", "draft_context_kv")
        self.wrap(spec.model, "combine_hidden_states", "draft_context_projection")
        self.wrap(spec.model, "compute_candidates", "draft_candidates")
        self.wrap(spec, "_score_candidates", "draft_selector")
        self.wrap(spec.candidate_sampler, "sample", "draft_walk_cache")
        for name, module in (
            [] if config.get("coarse") else runner.model.named_modules()
        ):
            typ = type(module).__name__
            if hasattr(module, "_down_gemm"):
                self.wrap(module, "_down_gemm", "hc_down")
            if typ == "GatedResidual":
                self.wrap(module, "mix", "hc_mix")
                self.wrap(module, "combine_and_mix", "hc_combine_mix")
            category = next(
                (
                    label
                    for suffix, label in [
                        (".linear_attn", "gdn"),
                        (".self_attn", "attention"),
                        (".mlp", "moe"),
                        (".ple", "ple"),
                    ]
                    if name.endswith(suffix) and ".layers." in name
                ),
                None,
            )
            if category:
                self.wrap(module, "forward", category)
        self.original_execute = runner.execute_model

        def execute(*args, **kwargs):
            sched = args[0] if args else kwargs.get("scheduler_output")
            width = getattr(sched, "total_num_scheduled_tokens", 0)
            self.active = width == 4 and self.step < config.get("cycles", 5)
            self.spans = []
            if self.active:
                torch.xpu.synchronize()
                if config.get("kineto") and worker.rank == 0 and self.step == 0:
                    self.kineto = torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.XPU,
                        ],
                        record_shapes=True,
                    )
                    self.kineto.start()
                self.cycle_start = time.perf_counter()
                self.cycle_event = torch.xpu.Event(enable_timing=True)
                self.cycle_event.record()
            return self.original_execute(*args, **kwargs)

        runner.execute_model = execute
        self.restores.append((runner, "execute_model", self.original_execute))
        self.original_sample = runner.sample_tokens

        def sample(*args, **kwargs):
            result = self.original_sample(*args, **kwargs)
            if self.active:
                end = torch.xpu.Event(enable_timing=True)
                end.record()
                end.synchronize()
                record = dict(
                    cycle=self.step,
                    wall_ms=1000 * (time.perf_counter() - self.cycle_start),
                    queue_ms=self.cycle_event.elapsed_time(end),
                    spans=[
                        dict(name=n, host_ms=h, queue_ms=s.elapsed_time(e))
                        for n, h, s, e in self.spans
                    ],
                )
                if hasattr(self, "counts"):
                    record.update(
                        sampled=self.counts[0].cpu().tolist(),
                        rejected=self.counts[1].cpu().tolist(),
                    )
                self.records.append(record)
                (
                    self.directory / f"{config['label']}-rank-{worker.rank}.json"
                ).write_text(json.dumps(self.records, indent=2))
                self.step += 1
                self.active = False
                if hasattr(self, "kineto"):
                    self.kineto.stop()
                    self.kineto.export_chrome_trace(
                        str(self.directory / f"{config['label']}-trace.json")
                    )
                    (self.directory / f"{config['label']}-kernels.txt").write_text(
                        self.kineto.key_averages().table(
                            sort_by="self_device_time_total", row_limit=60
                        )
                    )
                    del self.kineto
            return result

        runner.sample_tokens = sample
        self.restores.append((runner, "sample_tokens", self.original_sample))

    def wrap(self, obj, attr, name):
        original = getattr(obj, attr)

        def call(*args, **kwargs):
            if not self.active:
                return original(*args, **kwargs)
            start = torch.xpu.Event(enable_timing=True)
            end = torch.xpu.Event(enable_timing=True)
            start.record()
            before = time.perf_counter()
            with torch.profiler.record_function("phase4::" + name):
                result = original(*args, **kwargs)
            if name == "target_sample":
                self.counts = result[1:3]
            host = 1000 * (time.perf_counter() - before)
            end.record()
            self.spans.append((name, host, start, end))
            return result

        setattr(obj, attr, call)
        self.restores.append((obj, attr, original))

    def close(self):
        for obj, attr, original in reversed(self.restores):
            setattr(obj, attr, original)
        for hook in self.hooks:
            hook.remove()


def run(worker, command):
    worker._phase4_profiler.close()
    del worker._phase4_profiler
    return {"rank": worker.rank, "profile_stopped": True}
