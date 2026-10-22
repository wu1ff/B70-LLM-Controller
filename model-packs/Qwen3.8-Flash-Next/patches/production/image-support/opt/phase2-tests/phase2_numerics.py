"""Small independent oracles before loading the TP4 drafter.

CPU checks cover block boundaries, selector edges and the rejection identity.
XPU checks exercise the actual selector/cache, rejection and PLE state kernels.
"""

import itertools
import json
import sys
import unittest
from types import SimpleNamespace

import torch

from vllm.model_executor.models.qwen3_dflash2 import _grouped_conv, _score_edges
from vllm.models.qwen4_exp.nvidia.model_state import Qwen4ExpModelState

DEVICE = "xpu" if "--xpu" in sys.argv else "cpu"
sys.argv = [x for x in sys.argv if x != "--xpu"]


class Numerics(unittest.TestCase):
    def test_four_token_convolution_has_no_request_leakage(self):
        torch.manual_seed(41)
        x = torch.randn(12, 8)
        delta = torch.randn(12, 2, 4)
        base = torch.randn(2, 8)
        expected = torch.zeros_like(x)
        for row in range(12):
            for tap in range(min(2, row % 4 + 1)):
                for col in range(8):
                    expected[row, col] += (
                        base[tap, col] + delta[row, tap, col // 2]
                    ) * x[row - tap, col]
        actual = _grouped_conv(
            x.to(DEVICE), delta.to(DEVICE), base.to(DEVICE), 4, 4, 2, 2
        )
        torch.testing.assert_close(actual.cpu(), expected)

    def test_selector_edges_include_anchor_and_realized_predecessor(self):
        torch.manual_seed(8)
        pred, succ = torch.randn(13, 5), torch.randn(13, 5)
        ids = torch.tensor([[[2, 3], [5, 6], [8, 9]]])
        unary, hidden = torch.randn(1, 3, 2), torch.randn(1, 3, 5)
        actual = _score_edges(pred, succ, ids, unary, hidden, torch.tensor([1]), 2)
        for step, previous, candidate in itertools.product(
            range(3), range(2), range(2)
        ):
            predecessor = 1 if step == 0 else ids[0, step - 1, previous]
            expected = unary[0, step, candidate] + sum(
                pred[predecessor, r]
                * hidden[0, step, r]
                * succ[ids[0, step, candidate], r]
                for r in range(5)
            )
            torch.testing.assert_close(actual[0, step, previous, candidate], expected)

    def test_rejection_residual_recovers_target_for_each_conditional_prefix(self):
        # Exact finite enumeration: the proposal depends on the accepted prefix.
        for prefix in itertools.product(range(3), repeat=2):
            p = torch.tensor([1 + prefix[0], 2 + prefix[1], 5.0], dtype=torch.float64)
            q = torch.tensor([6.0, 1 + prefix[1], 1 + prefix[0]], dtype=torch.float64)
            p, q = p / p.sum(), q / q.sum()
            acceptance = torch.minimum(p / q, torch.ones_like(p))
            residual = (p - q).clamp_min(0)
            recovered = (
                q * acceptance
                + (1 - (q * acceptance).sum()) * residual / residual.sum()
            )
            torch.testing.assert_close(recovered, p, rtol=1e-14, atol=1e-14)
            self.assertAlmostEqual(
                float(residual.sum()), float((q - p).clamp_min(0).sum())
            )

    def test_ngram_context_reads_only_committed_prefix_after_reordering(self):
        state = SimpleNamespace(
            ngram_context=torch.empty(4, 2, dtype=torch.int32),
            ngram_eos_token_id=99,
            ngram_context_offsets=torch.tensor([-2, -1]),
        )
        reqs = SimpleNamespace(
            num_computed_tokens=SimpleNamespace(gpu=torch.tensor([3, 0, 5, 1])),
            all_token_ids=SimpleNamespace(
                gpu=torch.tensor(
                    [
                        [10, 11, 12, 90, 91, 92],
                        [0] * 6,
                        [20, 21, 22, 23, 24, 93],
                        [30, 94, 95, 96, 97, 98],
                    ]
                )
            ),
        )
        batch = SimpleNamespace(num_reqs=3, idx_mapping=torch.tensor([2, 0, 3]))
        got = Qwen4ExpModelState._prepare_ngram_context(state, batch, reqs)
        self.assertEqual(got.tolist(), [[23, 24], [11, 12], [99, 30], [99, 99]])


@unittest.skipUnless(DEVICE == "xpu", "XPU kernel checks requested separately")
class Kernels(unittest.TestCase):
    def test_ple_committed_history_matches_sequential_replay_after_rejection(self):
        from vllm.models.qwen4_exp.common.ops.ple import ple_conv

        torch.manual_seed(18)
        channels, window = 8, 9
        history = torch.randn(window, channels)
        inputs = torch.randn(4, channels)
        recovery = torch.randn(1, channels)
        weights = torch.randn(channels, 4)
        for committed in (1, 2, 4):
            state = torch.zeros(2, channels, window + 3, device=DEVICE)
            state[1, :, :window] = history.T.to(DEVICE)
            residual = torch.zeros_like(inputs, device=DEVICE)
            ple_conv(
                inputs.to(DEVICE),
                residual,
                state,
                weights.to(DEVICE),
                torch.tensor([1], device=DEVICE),
                None,
                mode="spec",
                dilation=3,
                query_start_loc=torch.tensor([0, 4], device=DEVICE),
                num_accepted_tokens=torch.tensor([1], device=DEVICE),
                spec_query_len=4,
            )
            merged = torch.cat((history, inputs))
            expected = torch.stack(
                [
                    torch.nn.functional.silu(
                        sum(weights[:, k] * merged[row + 3 * k] for k in range(4))
                    )
                    for row in range(4)
                ]
            )
            torch.testing.assert_close(residual.cpu(), expected)
            torch.testing.assert_close(
                state[1, :, committed - 1 : committed - 1 + window].T.cpu(),
                torch.cat((history, inputs[:committed]))[-window:],
            )
            recovered = torch.zeros_like(recovery, device=DEVICE)
            ple_conv(
                recovery.to(DEVICE),
                recovered,
                state,
                weights.to(DEVICE),
                torch.tensor([1], device=DEVICE),
                None,
                mode="spec",
                dilation=3,
                query_start_loc=torch.tensor([0, 1], device=DEVICE),
                num_accepted_tokens=torch.tensor([committed], device=DEVICE),
                spec_query_len=4,
            )
            merged = torch.cat((history, inputs[:committed], recovery))
            expected = torch.nn.functional.silu(
                sum(weights[:, k] * merged[-window - 1 + 3 * k] for k in range(4))
            )
            torch.testing.assert_close(recovered[0].cpu(), expected)
            torch.testing.assert_close(state[1, :, :window].T.cpu(), merged[-window:])

    def test_conditional_selector_cache_support_replacement_and_padding(self):
        from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import CandidateSampler

        torch.manual_seed(4)
        sampler = CandidateSampler(3, 3, 4, torch.device(DEVICE))
        ids = torch.arange(12, device=DEVICE).view(1, 3, 4).repeat(3, 1, 1)
        scores = torch.randn(3, 3, 4, 4, device=DEVICE)
        tokens = torch.full((3, 3), -1, device=DEVICE, dtype=torch.long)
        cache = torch.full((3, 3, 32), -float("inf"), device=DEVICE)
        mapping = torch.tensor([2] * 3 + [0] * 3 + [-1] * 3, device=DEVICE)
        temp = torch.tensor([0.0, 1.0, 0.8], device=DEVICE)
        seeds = torch.tensor([11, 22, 33], device=DEVICE)
        for offset in (0, 12):
            candidates = ids + offset
            sampler.sample(
                candidates,
                scores,
                3,
                torch.arange(9, device=DEVICE),
                mapping,
                temp,
                seeds,
                tokens,
                cache,
                False,
            )
            for row, req in ((0, 2), (1, 0)):
                previous = 0
                for step in range(3):
                    index = (
                        (candidates[row, step] == tokens[row, step]).nonzero().item()
                    )
                    realized = scores[row, step, previous]
                    torch.testing.assert_close(sampler.scores[row, step], realized)
                    torch.testing.assert_close(
                        cache[req, step, candidates[row, step]], realized
                    )
                    support = torch.isfinite(cache[req, step]).nonzero().flatten()
                    torch.testing.assert_close(support, candidates[row, step])
                    previous = index
            self.assertTrue((tokens[2] == -1).all())
            self.assertTrue(cache[1].isneginf().all())

    def rejection(self, p, q, drafts, temp=1.0, trials=1):
        from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
            rejection_sample,
        )

        K, V = 3, len(p[0])
        target = torch.tensor(p, device=DEVICE).log().repeat(trials, 1)
        draft = torch.tensor(q, device=DEVICE).log().expand(trials, K, V).contiguous()
        tokens = torch.tensor(drafts, device=DEVICE).expand(trials, K)
        inputs = torch.cat(
            (torch.zeros(trials, 1, device=DEVICE, dtype=torch.long), tokens), 1
        ).flatten()
        mapping = torch.arange(trials, dtype=torch.int32, device=DEVICE)
        return rejection_sample(
            target,
            draft,
            inputs,
            torch.arange(trials + 1, dtype=torch.int32, device=DEVICE) * 4,
            torch.arange(4, dtype=torch.int32, device=DEVICE).repeat(trials),
            mapping,
            mapping.repeat_interleave(4),
            torch.arange(4, dtype=torch.int32, device=DEVICE).repeat(trials),
            torch.full((trials,), temp, device=DEVICE),
            torch.arange(trials, dtype=torch.long, device=DEVICE) + 42,
            K,
        )

    def test_forced_all_accepted_reject_first_reject_middle_and_recovery(self):
        q = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        for temp in (0.0, 1.0):
            for reject in (0, 1, 3):
                p = [row[:] for row in q] + [[0.0, 0.0, 0.0, 1.0]]
                if reject < 3:
                    p[reject] = [0.0, 0.0, 0.0, 1.0]
                sampled, count = self.rejection(p, q, [0, 1, 2], temp=temp)
                self.assertEqual(count.item(), reject + 1)
                self.assertEqual(
                    sampled[0, : reject + 1].cpu().tolist(), [0, 1, 2][:reject] + [3]
                )

    def test_sampled_rejection_matches_small_probability_oracle(self):
        from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
            rejection_sample,
        )

        # Independent proposal draws; distribution of first committed token is p.
        trials, K, V = 12000, 3, 4
        p = torch.tensor([0.15, 0.25, 0.35, 0.25], device=DEVICE)
        q = torch.tensor([0.65, 0.25, 0.10, 0.0], device=DEVICE)
        tokens = torch.multinomial(q.expand(trials, -1), K, replacement=True)
        mapping = torch.arange(trials, dtype=torch.int32, device=DEVICE)
        sampled, count = rejection_sample(
            p.log().expand(trials * 4, V).contiguous(),
            q.log().expand(trials, K, V).contiguous(),
            torch.cat(
                (torch.zeros(trials, 1, device=DEVICE, dtype=torch.long), tokens), 1
            ).flatten(),
            torch.arange(trials + 1, dtype=torch.int32, device=DEVICE) * 4,
            torch.arange(4, dtype=torch.int32, device=DEVICE).repeat(trials),
            mapping,
            mapping.repeat_interleave(4),
            torch.arange(4, dtype=torch.int32, device=DEVICE).repeat(trials),
            torch.ones(trials, device=DEVICE),
            mapping.long() + 700,
            K,
        )
        observed = torch.bincount(sampled[:, 0], minlength=V).float() / trials
        torch.testing.assert_close(observed, p, rtol=0, atol=0.016)
        print(
            json.dumps(
                {
                    "target": p.cpu().tolist(),
                    "observed": observed.cpu().tolist(),
                    "acceptance_counts": torch.bincount(count.long()).cpu().tolist(),
                }
            )
        )


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
