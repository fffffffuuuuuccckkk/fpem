import unittest

import torch
from torch import nn

from models.predictive_env.direct_invariant_forecaster import DirectInvariantForecaster
from models.predictive_env.prototype_invariance_losses import (
    prototype_risk_penalty, shared_optimum_penalty)
from models.predictive_env.dual_pattern_memory.forecast_fusion import (
    prototype_segment_reliability, variation_forecast)
from tools.run_fpem_f03_r3_invariance import (
    consensus_prediction, learn_consensus_weight, weighted_prediction)


class F03R3InvarianceTests(unittest.TestCase):
    def test_a0_head_is_exact_at_initialization(self):
        torch.manual_seed(4)
        hidden = torch.randn(3, 2, 5, 7)
        reference = nn.Linear(35, 11)
        f03 = DirectInvariantForecaster(5, 7, 11, rank=0,
                                        pretrained_head=reference)
        old = reference(hidden.permute(0, 1, 3, 2).flatten(-2))
        self.assertTrue(torch.equal(old, f03(hidden)))
        self.assertIsNone(f03.projection)

    def test_i1_i2_detach_router_and_reach_head(self):
        torch.manual_seed(6)
        head = DirectInvariantForecaster(3, 4, 6)
        hidden = torch.randn(5, 2, 3, 4)
        truth = torch.randn(5, 2, 6)
        q = torch.rand(5, 2, 3, requires_grad=True)
        for kind in ('I1', 'I2'):
            head.zero_grad(set_to_none=True)
            pred = head(hidden)
            if kind == 'I1':
                penalty, audit = prototype_risk_penalty(pred, truth, q)
            else:
                penalty, audit = shared_optimum_penalty(
                    pred, truth, q, head.head.weight)
            (pred.square().mean() + 0.01 * penalty).backward()
            self.assertTrue(torch.isfinite(penalty))
            self.assertTrue(torch.isfinite(head.head.weight.grad).all())
            self.assertIsNone(q.grad)
            self.assertEqual(audit['effective'].numel(), 3)

    def test_zero_reliability_preserves_anchor(self):
        anchor = torch.randn(4, 2, 8)
        candidates = torch.randn(4, 2, 3, 8)
        q = torch.rand(4, 2, 3).softmax(-1) * .8
        reliability = torch.zeros(3, 4)
        output, beta = variation_forecast(anchor, candidates, q, reliability,
                                          torch.empty(8, 0), True, 4)
        self.assertTrue(torch.equal(anchor, output))
        self.assertTrue(torch.equal(beta, torch.zeros_like(beta)))
        target = torch.randn_like(anchor)
        estimated, audit = prototype_segment_reliability(
            anchor, candidates, target, q, 4)
        self.assertEqual(tuple(estimated.shape), (3, 4))
        self.assertEqual(len(audit), 12)

    def test_learned_consensus_blend_uses_train_and_preserves_f03_fallback(self):
        torch.manual_seed(7)
        head = DirectInvariantForecaster(2, 2, 3)
        hidden = torch.randn(40, 1, 2, 2)
        teacher = {"weight": torch.randn(4, 3), "bias": torch.randn(3),
                   "hold": torch.arange(40) >= 20}
        data = {"hidden": hidden, "mean": torch.zeros(40, 1, 1),
                "std": torch.ones(40, 1, 1)}
        base = weighted_prediction(head, data, "cpu", teacher, 0.0)
        consensus = consensus_prediction(data, teacher)
        data["truth"] = base + 0.4 * (consensus - base)
        val = {key: value.clone() for key, value in data.items()}
        alpha, audit = learn_consensus_weight(head, data, val, teacher, "cpu")
        self.assertTrue(torch.equal(base, weighted_prediction(head, data, "cpu", teacher, 0.0)))
        self.assertGreater(alpha, 0.25)
        self.assertLess(alpha, 0.55)
        self.assertLess(audit["selected_val_mse"], audit["baseline_val_mse"])
        data["truth"] = torch.randn_like(data["truth"])
        self.assertTrue(torch.equal(consensus, consensus_prediction(data, teacher)))


if __name__ == '__main__':
    unittest.main()
