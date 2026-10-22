"""Temporary TP4 audit worker; target decoding remains non-speculative."""
import json
from pathlib import Path

import torch
from vllm.models.qwen4_exp.nvidia.dflash2_features import AUX_LAYER_BOUNDARIES, TARGET_LAYER_IDS
from vllm.v1.worker.xpu_worker import XPUWorker


class FeatureAuditWorker(XPUWorker):
    def load_model(self, **kwargs):
        super().load_model(**kwargs)
        runner=self.model_runner
        target=runner.model
        target.set_aux_hidden_state_layers(AUX_LAYER_BOUNDARIES)
        runner.use_aux_hidden_state_outputs=True
        decoder=target.language_model.model
        self._feature_decoder=decoder
        self._feature_live=False
        self._feature_captures=0
        self._feature_dummy=0
        self._feature_pending={}
        directory=Path('/work/feature-audit')
        directory.mkdir(exist_ok=True)
        disk=next(m for m in target.modules() if getattr(m,'disk_bf16',False))
        maps=Path('/proc/self/maps').read_text()
        self._feature_info={
            'rank':self.rank,'model_class':type(target).__module__+'.'+type(target).__name__,
            'decoder_class':type(decoder).__module__+'.'+type(decoder).__name__,
            'disk_class':type(disk).__module__+'.'+type(disk).__name__,
            'stage_bytes':disk._stage.numel()*disk._stage.element_size(),
            'table_weight_elements':disk.weight.numel(),'table_pinned':disk._table.is_pinned(),
            'tp_range':[disk.shard_indices.org_vocab_start_index,disk.shard_indices.org_vocab_end_index],
            'residency_shim_mapped':'libl0_peer_residency_shim.so' in maps,
            'gdn_library_mapped':'libgdn_index64.so' in maps,
            'disk_mappings':[line for line in maps.splitlines() if 'model-00016-' in line],
            'boundary_ids':list(decoder.aux_hidden_state_layers),
        }
        (directory/f'rank-{self.rank}-dispatch.json').write_text(json.dumps(self._feature_info,indent=2)+'\n')

        def layer_hook(index):
            def capture(layer,args,kwargs,output):
                if self._feature_live and self._feature_captures<2:
                    self._feature_pending[index]=tuple(t.detach().clone() if t is not None else None for t in output)
            return capture
        for index in TARGET_LAYER_IDS:
            decoder.layers[index].register_forward_hook(layer_hook(index),with_kwargs=True)

        def audit(model,args,kwargs,output):
            if not self._feature_live:
                self._feature_dummy+=1
                return
            if self._feature_captures>=2:
                return
            hidden,features=output
            references=[]
            errors=[]
            for index,feature in zip(TARGET_LAYER_IDS,features):
                residual,block,injection=self._feature_pending[index]
                shape=(*residual.shape[:-1],4,2560)
                completed=(residual.float().view(shape)+block.float().unsqueeze(-2)*
                           (2*torch.sigmoid(injection.float()/4)).unsqueeze(-1)).to(residual.dtype)
                reference=completed.mean(dim=-2)
                torch.testing.assert_close(feature.cpu(),reference.cpu(),atol=.0078125,rtol=.008)
                references.append(reference.cpu())
                errors.append(float((feature.float()-reference.float()).abs().max().item()))
            packed=torch.cat(features,-1)
            assert packed.shape[-1]==12800 and len(features)==5
            payload={
                'features':[t.cpu() for t in features], 'reference':references,
                'concatenated':packed.cpu(),
                'positions':kwargs['positions'].cpu(),
                'input_ids':kwargs['input_ids'].cpu() if kwargs['input_ids'] is not None else None,
                'max_abs_errors':errors,'dummy_profile_calls_before':self._feature_dummy,
                'disk_generation':disk._generation,
                'torch_allocated':torch.xpu.memory_allocated(),'dispatch':self._feature_info,
            }
            torch.save(payload,directory/f'rank-{self.rank}-capture-{self._feature_captures}.pt')
            print(f'PHASE1_DFLASH2_AUDIT rank={self.rank} capture={self._feature_captures} shape={tuple(packed.shape)} max_abs={errors} dummy_profile_calls={self._feature_dummy}',flush=True)
            self._feature_captures+=1
            self._feature_pending.clear()
        decoder.register_forward_hook(audit,with_kwargs=True)

    def execute_model(self,*args,**kwargs):
        self._feature_live=True
        try:
            return super().execute_model(*args,**kwargs)
        finally:
            self._feature_live=False
