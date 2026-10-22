"""Focused target prerequisites before loading the 181 GB checkpoint."""
import json
import os
import threading
import unittest
from types import SimpleNamespace

import torch
from vllm.models.qwen4_exp.nvidia.b70_disk import Qwen4ExpPLEDiskBF16Embedding
from vllm.models.qwen4_exp.intel_checkpoint import ple_layout
from vllm.models.qwen4_exp import Qwen4ExpForConditionalGeneration


class DiskTests(unittest.TestCase):
    def test_real_checkpoint_layout(self):
        self.assertEqual(ple_layout(os.environ['PLE_TABLE_PATH']), (20736, 320001536, 160))

    def stage(self, rank, device='cpu'):
        obj = Qwen4ExpPLEDiskBF16Embedding.__new__(Qwen4ExpPLEDiskBF16Embedding)
        torch.nn.Module.__init__(obj)
        obj.org_vocab_size, obj.embedding_dim = 16, 3
        obj.shard_indices = SimpleNamespace(org_vocab_start_index=rank*4, org_vocab_end_index=(rank+1)*4)
        obj._stage = torch.empty((4, 2, 3), dtype=torch.bfloat16, pin_memory=device=='xpu')
        obj._table = (torch.arange(48).reshape(16, 3) / 4).to(torch.bfloat16)
        obj._closed = True  # Only the row-gather method is under test here.
        return obj

    def test_tp4_owned_rows_padding_duplicates_and_values(self):
        ids = torch.tensor([[0, 15], [4, 4], [9, 2], [11, 5]])
        stages = []
        for rank in range(4):
            obj = self.stage(rank)
            obj._fill_stage(ids, 3)
            stages.append(obj._stage.clone())
            owned = (ids[:3] >= rank*4) & (ids[:3] < (rank+1)*4)
            self.assertTrue(torch.equal(obj._stage[:3][~owned], torch.zeros_like(obj._stage[:3][~owned])))
            self.assertEqual(obj._stage[3].count_nonzero().item(), 0)
        result = sum(stages)
        expected = torch.zeros_like(result)
        expected[:3] = self.stage(0)._table[ids[:3]]
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'out-of-table'):
            self.stage(0)._fill_stage(torch.tensor([[16, 0]]), 1)

    def test_xpu_dma_stage_event_and_ownership(self):
        obj = self.stage(1, 'xpu')
        obj._stream_mod = torch.xpu
        obj._capacity, obj._heads = 4, 2
        obj._prefetch_buffer = torch.empty_like(obj._stage, device='xpu')
        obj._pending = obj._closed = obj._failed = False
        obj._copy_event = torch.xpu.Event()
        obj._owner_stream = obj._owner_thread = None
        obj._lock = threading.Lock()
        obj._generation = obj._prepared_tokens = 0
        ids = torch.tensor([[4, 8], [7, 7], [0, 6], [15, 5]], device='xpu')
        for _ in range(3):
            obj.prepare(ids, 3, 4)
            torch.testing.assert_close(obj._prefetch_buffer.cpu(), obj._stage, rtol=0, atol=0)
        obj.prepare(None, 0, 4)
        self.assertEqual(obj._prefetch_buffer.cpu().count_nonzero().item(), 0)
        with self.assertRaisesRegex(ValueError, 'overflow'):
            obj.prepare(None, 0, 5)
        obj.close()


if __name__ == '__main__':
    print(json.dumps({'model_dispatch':Qwen4ExpForConditionalGeneration.__module__, 'torch':torch.__version__, 'xpu_count':torch.xpu.device_count()}), flush=True)
    unittest.main(verbosity=2)
