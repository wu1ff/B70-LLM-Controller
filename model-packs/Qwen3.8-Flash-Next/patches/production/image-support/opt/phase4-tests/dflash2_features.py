"""Numerical completed-layer, token-order, wrapper and snapshot-lifetime checks."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from vllm.models.qwen4_exp.nvidia import model as module
from vllm.models.qwen4_exp.nvidia.dflash2_features import AUX_LAYER_BOUNDARIES
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import get_eagle3_aux_layers_from_config


class Mixer(nn.Module):
    def combine(self, hidden, block, injection):
        if block is None:
            return hidden
        return (hidden.float().view(-1,4,2560) + block.float()[:,None,:] *
                (2*torch.sigmoid(injection.float()/4))[:,:,None]).flatten(1).to(hidden.dtype)

    def combine_and_mix(self, hidden, block, injection):
        completed = self.combine(hidden, block, injection)
        return completed, completed.view(-1,4,2560).mean(1), None


class Layer(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index=index
        self.mlp_hyper_connection=Mixer()
        self.ple=None

    def forward(self, hidden_states, prev_block_output, prev_injection, **kwargs):
        hidden=self.mlp_hyper_connection.combine(hidden_states, prev_block_output, prev_injection)
        # Reuse the input storage, as an optimized native workspace may do.
        streams=torch.arange(4,device=hidden.device)[None,:,None]
        hidden.add_((streams+(self.index+1)/32).expand(hidden.shape[0],4,2560).flatten(1))
        tokens=torch.arange(hidden.shape[0],device=hidden.device)[:,None]
        columns=torch.arange(2560,device=hidden.device)[None,:]
        block=(tokens/64+columns/8192+(self.index+1)/16).to(hidden.dtype)
        injection=torch.tensor([-2.,-0.5,1.,3.],device=hidden.device)[None,:].expand(hidden.shape[0],-1)
        return hidden, block, injection


def decoder():
    obj=module.Qwen4ExpModel.__new__(module.Qwen4ExpModel)
    nn.Module.__init__(obj)
    obj.config=SimpleNamespace(hc_count=4,hidden_size=2560)
    obj.use_hc_sequence_parallel=obj.use_sequence_parallel=False
    obj.start_layer,obj.end_layer=0,48
    obj.layers=nn.ModuleList([Layer(i) for i in range(48)])
    obj.hyper_connection_mixer=Mixer()
    obj._mtp_hidden_buffer=None
    return obj


class Features(unittest.TestCase):
    def test_completed_layers_and_token_order_survive_workspace_reuse(self):
        for rows in (1,4,7,256):
            obj=decoder()
            config=SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(
                dflash_config={'target_layer_ids':[5,15,25,35,45]})))
            boundaries=get_eagle3_aux_layers_from_config(config)
            self.assertEqual(boundaries,AUX_LAYER_BOUNDARIES)
            obj._set_aux_hidden_state_layers(boundaries)
            inputs=(torch.arange(rows)[:,None]/8+torch.arange(2560)[None,:]/4096).float()
            # Independent recurrence: explicit float64 materialization at every layer.
            expected=inputs.double()[:,None,:].expand(rows,4,2560).clone()
            snapshots=[]
            for i in range(48):
                expected += torch.arange(4).double()[None,:,None]+(i+1)/32
                block=torch.arange(rows).double()[:,None]/64+torch.arange(2560).double()[None,:]/8192+(i+1)/16
                gate=2*torch.sigmoid(torch.tensor([-2.,-.5,1.,3.]).double()/4)
                expected += block[:,None,:]*gate[None,:,None]
                # The live path rounds each completed boundary to FP32.
                expected=expected.float().double()
                if i in (5,15,25,35,45):
                    snapshots.append(expected.mean(1).float())
            pp=SimpleNamespace(is_first_rank=True,is_last_rank=True)
            with patch.object(module,'get_pp_group',return_value=pp):
                final,features=obj(input_ids=None, positions=torch.arange(rows), inputs_embeds=inputs)
            self.assertEqual(len(features),5)
            for got,want in zip(features,snapshots):
                torch.testing.assert_close(got,want,rtol=2e-6,atol=2e-4)
            packed=torch.cat(features,-1)
            torch.testing.assert_close(packed,torch.cat(snapshots,-1),rtol=2e-6,atol=2e-4)
            self.assertEqual(packed.shape,(rows,12800))
            self.assertEqual(len({t.data_ptr() for t in features}),5)
            self.assertFalse(torch.equal(features[0],features[-1]))

    def test_causal_and_multimodal_wrappers_use_boundary_setter(self):
        for cls in (module.Qwen4ExpForCausalLM,module.Qwen4ExpForConditionalGeneration):
            wrapper=cls.__new__(cls)
            nn.Module.__init__(wrapper)
            obj=decoder()
            if cls is module.Qwen4ExpForCausalLM:
                wrapper.model=obj
            else:
                causal=module.Qwen4ExpForCausalLM.__new__(module.Qwen4ExpForCausalLM)
                nn.Module.__init__(causal)
                causal.model=obj
                wrapper.language_model=causal
                wrapper._language_model_names=['language_model']
            self.assertEqual(wrapper.get_eagle3_default_aux_hidden_state_layers(),AUX_LAYER_BOUNDARIES)
            wrapper.set_aux_hidden_state_layers(AUX_LAYER_BOUNDARIES)
            self.assertEqual(obj.aux_hidden_state_layers,AUX_LAYER_BOUNDARIES)
            with self.assertRaisesRegex(ValueError,'boundaries'):
                wrapper.set_aux_hidden_state_layers((5,15,25,35,45))
            obj.use_hc_sequence_parallel=True
            with self.assertRaisesRegex(ValueError,'unsharded'):
                wrapper.set_aux_hidden_state_layers(AUX_LAYER_BOUNDARIES)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
