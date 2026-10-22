"""Numeric metadata checks for actual query length versus maximum state depth."""
import importlib.util
import sys
import unittest

import torch
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import MambaSpec

spec = importlib.util.spec_from_file_location('gdn_helpers', '/upstream-tests/v1/attention/utils.py')
u = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = u
spec.loader.exec_module(u)


class StateSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = u.create_vllm_config(
            model_name='/work/flashnext-serve', dtype='bfloat16',
            max_num_seqs=4, max_num_batched_tokens=256)
        cls.config.cache_config.mamba_cache_mode = 'none'
        cls.config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        cls.config.speculative_config = SpeculativeConfig(method='ngram', num_speculative_tokens=5)

    def test_actual_uniform_lengths_keep_maximum_depth_states(self):
        for actual in (1, 4, 6):
            builder = GDNAttentionMetadataBuilder(
                MambaSpec(block_size=16, shapes=((16, 64),),
                          dtypes=(torch.bfloat16,), num_speculative_blocks=5),
                ['layer.0'], self.config, torch.device('cpu'))
            common = u.create_common_attn_metadata(
                u.BatchSpec(seq_lens=[128, 128], query_lens=[actual, actual]),
                16, torch.device('cpu'), arange_block_indices=True)
            accepted = torch.tensor([4, 2], dtype=torch.int32)
            meta = builder.build(0, common, accepted,
                                 torch.tensor([actual-1, actual-1], dtype=torch.int32))
            self.assertEqual(meta.uniform_spec_sequence_length, actual)
            torch.testing.assert_close(meta.spec_token_indx, torch.arange(2*actual, dtype=torch.int32))
            torch.testing.assert_close(meta.spec_state_indices_tensor, common.block_table_tensor[:, :6])
            torch.testing.assert_close(meta.num_accepted_tokens, accepted)
            state_values = torch.arange(16, dtype=torch.float32) * 13 + 7
            selected = state_values[meta.spec_state_indices_tensor[torch.arange(2), accepted.long()-1]]
            expected = state_values[torch.tensor([3, 9])]
            torch.testing.assert_close(selected, expected, rtol=0, atol=0)
            self.assertFalse(torch.equal(selected, state_values[torch.tensor([0, 8])]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
