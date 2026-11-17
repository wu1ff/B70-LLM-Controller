"""Explicit, removable diagnostics; no model hooks are installed at startup."""
import hashlib
from pathlib import Path

import torch

from tuning_worker import TuningWorker


class ParityWorker(TuningWorker):
    def parity_snapshot(self):
        r = self.model_runner
        groups = []
        for i, layer_groups in enumerate(r.attn_groups):
            for group in layer_groups:
                b = group.get_metadata_builder(0)
                groups.append(dict(index=i, builder=type(b).__name__,
                                   layers=group.layer_names,
                                   shared=b.supports_update_block_table,
                                   aligned=hasattr(b, 'mamba_aligned_state_indices'),
                                   prefill_backend=getattr(b, 'gdn_prefill_backend', None)))
        graphs = getattr(r, '_c1_target_graphs', None)
        graph = dict(entries=[str(k) for k in graphs.entries],
                     replays=graphs.replays, fallbacks=graphs.fallbacks,
                     reasons={str(k): v for k, v in graphs.fallback_reasons.items()},
                     live_checks={str(k): v for k, v in getattr(graphs, 'live_checks', {}).items()}) if graphs else None
        ple = []
        for m in r.model_state._disk_ple_modules:
            e = m.ngram_embedding
            ple.append(dict(capacity=e._capacity, stage_bytes=e._stage.numel()*e._stage.element_size(),
                            stage_pinned=e._stage.is_pinned(),
                            table_shape=list(e._table.shape),
                            pending=e._pending, prepared=e._prepared_tokens,
                            failed=e._failed, closed=e._closed))
        from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe
        pending, seen, experts = list(r.model.modules()), set(), []
        while pending:
            m = pending.pop()
            if id(m) in seen:
                continue
            seen.add(id(m))
            if isinstance(m, XpuFusedMoe) and getattr(m, '_astra_groups', None) is not None:
                experts.append(m)
            for name in ('quant_method', 'scheme', 'moe_kernel', 'impl',
                         'fused_experts', 'fused_moe_impl'):
                child = getattr(m, name, None)
                if child is not None:
                    pending.append(child)
        s = r.speculator
        libs = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                       if '/' in line and '.so' in line and any(x in line for x in ['astra', 'dominant', 'long_', 'serving_prefill', 'gdn_index', 'residency'])})
        return dict(rank=self.rank, groups=groups, graph=graph,
                    verify_shapes=[list(x) for x in sorted(getattr(r, '_dflash_verify_shapes', []))],
                    draft_class=type(s).__name__, draft_query_width=s.num_query_per_req,
                    draft_graph_mode=str(s.query_cudagraph_manager.cudagraph_mode),
                    draft_aux_staging=getattr(s, '_aux_staging', None) is not None,
                    ple=ple, cyclic_modules=len(experts),
                    libraries={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in libs if Path(p).is_file()},
                    allocated=torch.xpu.memory_allocated(), reserved=torch.xpu.memory_reserved())

    def set_parity_shared(self, enabled='1'):
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
        from vllm.models.qwen4_exp.nvidia.b70_target_graph import TargetGraphs
        r = self.model_runner
        torch.xpu.synchronize()
        if not hasattr(self, '_parity_graph_families'):
            self._parity_graph_families = {'0': getattr(r, '_c1_target_graphs', None)}
        if enabled not in self._parity_graph_families:
            self._parity_graph_families[enabled] = TargetGraphs()
        r._c1_target_graphs = self._parity_graph_families[enabled]
        changed = 0
        for layer_groups in r.attn_groups:
            for group in layer_groups:
                b = group.get_metadata_builder(0)
                if type(b) is GDNAttentionMetadataBuilder:
                    if enabled == '1' and not hasattr(b, 'mamba_aligned_state_indices'):
                        b.mamba_aligned_state_indices = None
                    elif enabled != '1' and hasattr(b, 'mamba_aligned_state_indices'):
                        del b.mamba_aligned_state_indices
                    b.supports_update_block_table = enabled == '1'
                    changed += 1
        return dict(rank=self.rank, enabled=enabled, builders=changed,
                    graph_families=list(self._parity_graph_families))

    def arm_parity_timing(self, label='parity', samples='16', warmup='8'):
        return self.arm_dflash_timing(label, samples, warmup)

    def disarm_parity_timing(self):
        return self.disarm_dflash_timing()

    def run_parity_probe(self, filename='parity_probe.py'):
        import runpy
        import traceback
        assert '/' not in filename and filename.endswith('.py')
        try:
            return runpy.run_path('/work/'+filename)['probe'](self)
        except Exception as exc:
            # Keep all rank responses consumable even if a diagnostic fails.
            return dict(rank=self.rank, passed=False, error=repr(exc),
                        traceback=traceback.format_exc())

    @torch.inference_mode()
    def probe_parity_shared_metadata(self):
        """Compare reused groups to independent live-version builders."""
        import dataclasses
        from vllm.v1.attention.backend import CommonAttentionMetadata
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
        from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor

        r = self.model_runner
        live = [g.get_metadata_builder(0) for gs in r.attn_groups for g in gs
                if type(g.get_metadata_builder(0)) is GDNAttentionMetadataBuilder]
        builders = [GDNAttentionMetadataBuilder(live[0].kv_cache_spec,
                    [f'parity.synthetic.{i}'], self.vllm_config, self.device)
                    for i in range(4)]
        for b in builders:
            b.supports_update_block_table = False
        checks = 0

        def freeze(value):
            if isinstance(value, torch.Tensor):
                return value.clone()
            if dataclasses.is_dataclass(value):
                fields = {f.name: freeze(getattr(value, f.name))
                          for f in dataclasses.fields(value)}
                if 'num_spec_decodes' in fields and fields['spec_state_indices_tensor'] is not None:
                    # Consumers read exactly the active speculative rows.
                    fields['spec_state_indices_tensor'] = fields['spec_state_indices_tensor'][:fields['num_spec_decodes']]
                return fields
            if isinstance(value, dict):
                return {k: freeze(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [freeze(v) for v in value]
            return value

        def equal(a, b, path=''):
            if isinstance(a, torch.Tensor):
                assert isinstance(b, torch.Tensor) and torch.equal(a, b), path
            elif isinstance(a, dict):
                assert a.keys() == b.keys(), path
                for k in a:
                    equal(a[k], b[k], path+'.'+str(k))
            elif isinstance(a, list):
                assert len(a) == len(b), path
                for i, (x, y) in enumerate(zip(a, b)):
                    equal(x, y, path+'.'+str(i))
            else:
                assert a == b, (path, a, b)

        shapes = [([q], [q-1]) for q in range(1, 9)]
        shapes += [([8]*n, [7]*n) for n in (2, 3, 4)]
        shapes += [([8]*n+[0]*(4-n), [7]*n+[-1]*(4-n)) for n in (1, 2, 3)]
        shapes += [([1, 33], [0, -1]), ([8, 33, 0, 0], [7, -1, -1, -1]),
                   ([64], [-1]), ([1, 1, 1, 1], [0]*4),
                   ([1, 3, 5, 7], [0, 2, 4, 6])]
        for qs, drafts in shapes:
            n = len(qs)
            for depth in (832, 833, 1664, 1665, 65536, 262144):
                qcpu = torch.tensor([0]+qs, dtype=torch.int32).cumsum(0).to(torch.int32)
                seqcpu = torch.tensor([max(depth-i*17, q) for i, q in enumerate(qs)], dtype=torch.int32)
                dc = torch.tensor(drafts, dtype=torch.int32)
                for accepted in (1, 4, 8):
                    acc = torch.full((n,), accepted, dtype=torch.int32, device=self.device)
                    first = None
                    for i, b in enumerate(builders):
                        bt = (torch.arange(n*324, dtype=torch.int32, device=self.device)
                              .reshape(n, 324)+i*100000+accepted*19+self.rank*1000000)
                        slots = torch.arange(sum(qs), dtype=torch.int64, device=self.device)
                        m = CommonAttentionMetadata(
                            query_start_loc=qcpu.to(self.device), query_start_loc_cpu=qcpu,
                            seq_lens=seqcpu.to(self.device), seq_lens_cpu_upper_bound=seqcpu,
                            num_reqs=n, num_actual_tokens=sum(qs), max_query_len=max(qs),
                            max_seq_len=int(seqcpu.max()), block_table_tensor=bt,
                            slot_mapping=slots, causal=True,
                            is_prefilling=torch.tensor([d<0 and q>0 for d, q in zip(drafts, qs)],
                                                       dtype=torch.bool, device='cpu'))
                        b.mamba_aligned_state_indices = None
                        reference = freeze(b.build(0, m, acc, dc))
                        if first is None:
                            first = b.build(0, m, acc, dc)
                            equal(reference, freeze(first))
                        else:
                            b.mamba_aligned_state_indices = mamba_get_block_table_tensor(
                                bt, m.seq_lens, b.kv_cache_spec, 'align')
                            reused = b.update_block_table(first, bt, slots)
                            equal(reference, freeze(reused))
                        checks += 1
        torch.xpu.synchronize()
        return dict(rank=self.rank, passed=True, group_comparisons=checks,
                    shapes=len(shapes), distinct_group_ids=True,
                    reserved_state_columns=8)
