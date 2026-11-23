"""Bounded native event timing via the engine's existing worker extension RPC.

No hooks or per-step work are installed until explicitly armed. Disarm restores
every original method. This module is a diagnostic bind, outside image payloads.
"""

import functools
import json
import time

import torch


class TuningWorker:
    def set_dflash_hc_epilogue(self, enabled="1"):
        import importlib
        import tuning_mutations
        return importlib.reload(tuning_mutations).set_hc(self, enabled)

    def probe_dflash_fast_metadata(self):
        from types import SimpleNamespace as NS
        from metadata_fast import eligible
        import vllm.v1.attention.backends.gdn_spec_metadata as metadata
        original = getattr(self, "_dflash_original_eligibility", metadata.eligible)
        b = NS(use_spec_decode=True, decode_cudagraph_max_bs=32, num_spec=7,
               kv_cache_spec=NS(num_speculative_blocks=7),
               vllm_config=NS(cache_config=NS(mamba_cache_mode="align")))
        cases = 0
        for n in range(1,5):
            m = NS(num_reqs=n, block_table_tensor=torch.empty((n,400),device=self.device,dtype=torch.int32),
                   seq_lens=torch.empty(n,device=self.device,dtype=torch.int32),
                   query_start_loc=torch.empty(n+1,device=self.device,dtype=torch.int32))
            for width in range(1,9):
                for active in range(n+1):
                    dc = [width-1]*active+[-1]*(n-active)
                    q = [min(i,active)*width for i in range(n+1)]
                    variants = [(dc,q),(dc[::-1],q),([x+1 for x in dc],q),
                                (dc,[x+1 for x in q]),(dc,q[:-1]),(dc[:-1],q)]
                    for counts, starts in variants:
                        d = torch.tensor(counts,dtype=torch.int32)
                        m.query_start_loc_cpu = torch.tensor(starts,dtype=torch.int32)
                        for budget in (0,active*width-1,active*width,active*width+1,32):
                            m.num_actual_tokens = budget
                            assert original(b,m,d) == eligible(b,m,d)
                            cases += 1
        m.num_actual_tokens = 32
        m.query_start_loc_cpu = torch.tensor([0,8,16,24,32],dtype=torch.int32)
        d = torch.full((4,),7,dtype=torch.int32)
        timing = {}
        for name, fn in [("B1",original),("host_list",eligible)]:
            start = time.perf_counter()
            for _ in range(4000):
                assert fn(b,m,d) == (4,8)
            timing[name] = 1000*(time.perf_counter()-start)/4000
        return dict(passed=True,cases=cases,host_ms_per_call=timing)

    def arm_dflash_timing(self, label="timing", samples="16", warmup="8"):
        assert not getattr(self, "_dflash_timing_restore", [])
        r = self.model_runner
        self._dflash_timing_restore = []
        self._dflash_timing_samples = []
        self._dflash_timing_current = None
        self._dflash_timing_steps = 0
        count, skip = int(samples), int(warmup)
        self._dflash_timing_label = label

        def mark(name):
            current = self._dflash_timing_current
            if current is not None:
                event = torch.xpu.Event(enable_timing=True)
                event.record()
                current["events"].append((name, event, time.perf_counter()))

        def wrap(obj, name, stage):
            original = getattr(obj, name)
            self._dflash_timing_restore.append((obj, name, original))

            @functools.wraps(original)
            def timed(*args, **kwargs):
                mark(stage + "_start")
                result = original(*args, **kwargs)
                mark(stage + "_end")
                return result

            setattr(obj, name, timed)

        original_execute = r.execute_model
        self._dflash_timing_restore.append((r, "execute_model", original_execute))

        @functools.wraps(original_execute)
        def execute(scheduler_output, *args, **kwargs):
            self._dflash_timing_steps += 1
            step = self._dflash_timing_steps
            if skip < step <= skip + count:
                self._dflash_timing_current = dict(
                    step=step,
                    tokens=scheduler_output.total_num_scheduled_tokens,
                    requests=len(scheduler_output.num_scheduled_tokens),
                    events=[],
                )
            else:
                self._dflash_timing_current = None
            mark("step_start")
            result = original_execute(scheduler_output, *args, **kwargs)
            mark("execute_end")
            return result

        r.execute_model = execute
        original_sample = r.sample_tokens
        self._dflash_timing_restore.append((r, "sample_tokens", original_sample))

        @functools.wraps(original_sample)
        def sample_tokens(*args, **kwargs):
            result = original_sample(*args, **kwargs)
            mark("step_end")
            if self._dflash_timing_current is not None:
                self._dflash_timing_samples.append(self._dflash_timing_current)
                self._dflash_timing_current = None
            return result

        r.sample_tokens = sample_tokens
        wrap(r, "prepare_attn", "block_tables")
        wrap(r.model_state, "prepare_attn", "metadata")
        wrap(r.model_state, "prepare_inputs", "model_inputs_and_ple")
        for module in r.model_state._disk_ple_modules:
            wrap(module.ngram_embedding, "prepare", "ple")
        wrap(r, "sample", "head_and_verification")
        wrap(r.rejection_sampler, "_verify", "verification_kernel")
        from vllm.models.qwen4_exp.nvidia import c1_verify

        wrap(c1_verify, "verify", "target")
        owner = getattr(r.model, "language_model", r.model)
        wrap(owner.logits_processor, "forward", "target_head")
        wrap(r.speculator, "propose", "draft")
        wrap(r.speculator, "_run_model", "draft_model")
        wrap(r.speculator.model, "compute_candidates", "draft_candidates")
        wrap(r.speculator, "_score_candidates", "draft_selector")
        wrap(r.speculator.candidate_sampler, "sample", "draft_sampling")
        return dict(armed=True, samples=count, warmup=skip, label=label)

    def disarm_dflash_timing(self):
        for obj, name, original in reversed(self._dflash_timing_restore):
            setattr(obj, name, original)
        self._dflash_timing_restore = []
        samples = self._dflash_timing_samples
        if samples:
            samples[-1]["events"][-1][1].synchronize()
        rows = []
        for sample in samples:
            events = sample.pop("events")
            spans = {}
            for name, event, host in events:
                if name.endswith("_start"):
                    spans[name[:-6]] = (event, host)
            elapsed = {}
            for name, event, host in events:
                if name.endswith("_end") and name[:-4] in spans:
                    first, begin = spans[name[:-4]]
                    elapsed[name[:-4]] = dict(
                        device_ms=first.elapsed_time(event),
                        host_ms=1000 * (host - begin),
                    )
            elapsed["step"] = dict(
                device_ms=events[0][1].elapsed_time(events[-1][1]),
                host_ms=1000 * (events[-1][2] - events[0][2]),
            )
            sample["spans"] = elapsed
            rows.append(sample)
        import torch.distributed as dist

        path = f"/work/{self._dflash_timing_label}-rank{dist.get_rank()}.json"
        with open(path, "w") as f:
            json.dump(rows, f, indent=2)
        return dict(disarmed=True, samples=len(rows), path=path)

    def set_dflash_tuning_mode(self, head="1", metadata="1", verify="0", target="1"):
        import os

        r = self.model_runner
        if not hasattr(self, "_dflash_tuning_head"):
            self._dflash_tuning_head = r.dflash_head_graphs
        h = self._dflash_tuning_head
        assert h is not None
        h.verify = verify == "1"
        r.dflash_head_graphs = h if head == "1" else None
        import vllm.v1.attention.backends.gdn_attn as gdn

        gdn._DFLASH_FUSED_METADATA = metadata == "1"
        os.environ["VLLM_XPU_C1_TARGET_GRAPH"] = target
        return dict(
            head=head,
            metadata=metadata,
            verify=verify,
            target=target,
            allocated=torch.xpu.memory_allocated(),
            head_replays={k: e["replays"] for k, e in h.entries.items()},
        )

    @torch.inference_mode()
    def probe_dflash_hc_epilogue(self):
        from tuning_kernels import hc_down_silu
        from vllm.models.qwen4_exp.nvidia.b70_hc_gemm import ordered_hc_gemm
        from vllm.models.qwen4_exp.nvidia.b70_hyperconnection import GatedResidual
        from vllm.models.qwen4_exp.nvidia.ops.hc import hc_silu

        modules = [
            m
            for m in self.model_runner.model.modules()
            if isinstance(m, GatedResidual) and m._ordered_hc
        ]
        rows = []
        for mi in (0, len(modules) // 2, len(modules) - 1):
            module = modules[mi]
            linear = (
                module.input_mix_weight_down_block_inject
                if module.use_combine
                else module.input_mix_weight_down
            )
            weight = linear.weight
            valid = module.lora_rank + (module.hc_count if module.use_combine else 0)
            for width in (1, 8, 16, 24, 32):
                x = torch.zeros(
                    (width, weight.shape[1]), device=weight.device, dtype=weight.dtype
                )

                def reference():
                    down = ordered_hc_gemm(x, weight, valid)
                    return hc_silu(down[:, : module.lora_rank], module.hc_count), down[
                        :, module.lora_rank : valid
                    ]

                def candidate():
                    return hc_down_silu(
                        x, weight, valid, module.lora_rank, module.hc_count
                    )

                reference()
                candidate()
                torch.xpu.synchronize()
                graphs, outputs = [], []
                for fn in (reference, candidate):
                    graph = torch.xpu.XPUGraph()
                    with torch.xpu.graph(graph):
                        output = fn()
                    graphs.append(graph)
                    outputs.append(output)
                exact = True
                for variant in range(4):
                    x.copy_(
                        (
                            torch.arange(x.numel(), device=x.device)
                            .reshape_as(x)
                            .remainder(257)
                            .float()
                            / 64
                            - 2
                            + variant * 0.25
                        ).to(x.dtype)
                    )
                    for graph in graphs:
                        graph.replay()
                    exact &= all(torch.equal(a, b) for a, b in zip(*outputs))
                    exact &= torch.equal(
                        module._mix_up(outputs[0][0], x),
                        module._mix_up(outputs[1][0], x),
                    )
                    assert outputs[1][0].is_contiguous()
                times = []
                for graph in graphs:
                    start, end = (
                        torch.xpu.Event(enable_timing=True),
                        torch.xpu.Event(enable_timing=True),
                    )
                    start.record()
                    for _ in range(48):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    times.append(start.elapsed_time(end) / 48)
                rows.append(
                    dict(
                        module=mi,
                        width=width,
                        exact=exact,
                        reference_ms=times[0],
                        candidate_ms=times[1],
                    )
                )
        import torch.distributed as dist

        path = f"/work/hc-epilogue-rank{dist.get_rank()}.json"
        with open(path, "w") as f:
            json.dump(rows, f, indent=2)
        return dict(path=path, exact=all(row["exact"] for row in rows), cases=len(rows))

    @torch.inference_mode()
    def probe_dflash_cyclic_experts(self):
        from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe

        pending = list(self.model_runner.model.modules())
        seen, found = set(), []
        while pending:
            obj = pending.pop()
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            if isinstance(obj, XpuFusedMoe) and obj._astra_groups is not None:
                found.append(obj)
            for name in (
                "quant_method",
                "scheme",
                "moe_kernel",
                "impl",
                "fused_experts",
                "fused_moe_impl",
            ):
                child = getattr(obj, name, None)
                if child is not None:
                    pending.append(child)
        assert len(found) == 48, len(found)
        records = []
        for mi in (0, 24, 47):
            module = found[mi]
            local = module._astra_groups
            for width in (8, 16, 24, 32):
                x = torch.zeros(
                    (width, 2560), device=local.w13.device, dtype=torch.bfloat16
                )
                ids = torch.arange(
                    width * 10, device=x.device, dtype=torch.int64
                ).reshape(width, 10)
                ids.copy_((ids * 17 + mi * 11).remainder(512))
                weights = torch.full((width, 10), 0.1, device=x.device)
                args = (
                    x,
                    local.w13,
                    local.gemm1_wei_scales,
                    local.w2,
                    local.gemm2_wei_scales,
                    ids,
                    weights,
                    module.ep_rank,
                )
                variants = {}
                if width == 8:
                    variants["retained"] = (
                        lambda out: torch.ops.dominant_w4_layout.forward(*args, out, 1)
                    )
                    transposed = local.gemm1_wei_scales.transpose(1, 2).contiguous()
                    trans_args = (
                        x,
                        local.w13,
                        transposed,
                        local.w2,
                        local.gemm2_wei_scales,
                        ids,
                        weights,
                        module.ep_rank,
                    )
                    variants["transposed_scales"] = (
                        lambda out: torch.ops.dominant_w4_layout.forward(
                            *trans_args, out, 2
                        )
                    )
                else:
                    variants["retained"] = lambda out: torch.ops.astra_batch.forward(
                        *args, out
                    )
                variants["prefill8"] = lambda out: torch.ops.serving_prefill8.forward(
                    *args, out
                )
                variants["long_prefill"] = (
                    lambda out: torch.ops.long_prefill_v3.forward(*args, out, 0)
                )
                graphs, outputs = {}, {}
                for label, fn in variants.items():
                    out = torch.empty_like(x)
                    fn(out)
                    torch.xpu.synchronize()
                    graph = torch.xpu.XPUGraph()
                    with torch.xpu.graph(graph):
                        fn(out)
                    graphs[label], outputs[label] = graph, out
                exact = {label: True for label in variants}
                for variant in range(3):
                    x.copy_(
                        (
                            torch.arange(x.numel(), device=x.device)
                            .reshape_as(x)
                            .remainder(251)
                            .float()
                            / 128
                            - 1
                            + variant * 0.25
                        ).to(x.dtype)
                    )
                    for graph in graphs.values():
                        graph.replay()
                    for label in variants:
                        exact[label] &= torch.equal(outputs[label], outputs["retained"])
                for label, graph in graphs.items():
                    start, end = (
                        torch.xpu.Event(enable_timing=True),
                        torch.xpu.Event(enable_timing=True),
                    )
                    start.record()
                    for _ in range(32):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    records.append(
                        dict(
                            module=mi,
                            width=width,
                            implementation=label,
                            exact=exact[label],
                            device_ms=start.elapsed_time(end) / 32,
                        )
                    )
        import torch.distributed as dist

        path = f"/work/cyclic-expert-probe-rank{dist.get_rank()}.json"
        with open(path, "w") as f:
            json.dump(records, f, indent=2)
        return dict(path=path, cases=len(records), modules=len(found))

    @torch.inference_mode()
    def qualify_dflash_cyclic_all_layers(self):
        from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe
        pending = list(self.model_runner.model.modules())
        seen, found = set(), []
        while pending:
            obj = pending.pop()
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            if isinstance(obj,XpuFusedMoe) and obj._astra_groups is not None:
                found.append(obj)
            for name in ("quant_method","scheme","moe_kernel","impl","fused_experts","fused_moe_impl"):
                child = getattr(obj,name,None)
                if child is not None:
                    pending.append(child)
        assert len(found) == 48
        cases = 0
        for mi,module in enumerate(found):
            local = module._astra_groups
            for width in (8,16,24,32):
                for variant in range(2):
                    x = (torch.arange(width*2560,device=self.device).reshape(width,2560)
                        .remainder(251).float()/128 - 1 + variant*.25).to(torch.bfloat16)
                    ids = torch.arange(width*10,device=self.device).reshape(width,10)
                    ids = ((ids*17+mi*11+variant*5).remainder(512)).contiguous()
                    if variant:
                        ids[::3,-1] = -1
                        ids[:,1] = ids[:,0]
                    weights = torch.full((width,10),.1,device=self.device)
                    args = (x,local.w13,local.gemm1_wei_scales,local.w2,local.gemm2_wei_scales,ids,weights,module.ep_rank)
                    a,b = torch.empty_like(x),torch.empty_like(x)
                    if width == 8:
                        torch.ops.dominant_w4_layout.forward(*args,a,1)
                    else:
                        torch.ops.astra_batch.forward(*args,a)
                    torch.ops.serving_prefill8.forward(*args,b)
                    assert torch.equal(a,b), (mi,width,variant)
                    cases += 1
        return dict(passed=True,layers=48,comparisons=cases,widths=[8,16,24,32])
