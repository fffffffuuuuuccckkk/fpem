import unittest

import torch

from models.predictive_env.swap_residual import (
    ConditionalResidualScale, VariantCorrection, select_swap_pairs,
    swap_loss, variant_difference_loss,
)


class SwapResidualTest(unittest.TestCase):
    def test_matching_excludes_self_and_overlapping_windows(self):
        z = torch.randn(4, 2, 3, 8)
        z[1] = z[0]  # This tempting neighbour is temporally overlapping.
        pairs, _ = select_swap_pairs(z, torch.tensor([0, 1, 500, 1000]),
                                     seq_len=96, pred_len=96, top_k=3,
                                     max_distance=2.1)
        self.assertFalse(any(i == j for i, j in pairs.tolist()))
        self.assertNotIn([0, 1], pairs.tolist())

    def test_bidirectional_swap_targets_recipient(self):
        z = torch.tensor([1., 3.]).view(2, 1, 1, 1).requires_grad_()
        y = torch.tensor([2., 6.]).view(2, 1, 1)
        def forecast(_, hidden, mean, std, cycle, head):
            return hidden.reshape(-1, 1, 1) * std + mean
        loss, _ = swap_loss(None, forecast, z, y, torch.zeros_like(y),
                            torch.ones_like(y), torch.zeros(2, 1, dtype=torch.long),
                            torch.tensor([[0, 1]]), torch.zeros(1), criterion="mse")
        self.assertAlmostEqual(float(loss), ((3-2)**2 + (1-6)**2)/2)
        loss.backward()
        self.assertIsNotNone(z.grad)

    def test_zero_correction_and_detached_difference_target(self):
        correction = VariantCorrection(3, 8, 5)
        z = torch.randn(4, 2, 3, 8, requires_grad=True)
        delta = correction(z, torch.ones(4, 1, 2))
        self.assertEqual(delta.shape, (4, 5, 2))
        self.assertEqual(float(delta.abs().max()), 0.)
        y = torch.randn_like(delta)
        inv = torch.randn_like(delta, requires_grad=True)
        loss = variant_difference_loss(delta, y, inv, torch.tensor([[0, 2]]))
        loss.backward()
        self.assertIsNone(inv.grad)

    def test_residual_target_stops_gradient_and_sampling_repeats(self):
        head = ConditionalResidualScale(8, 7)
        inv = torch.randn(3, 2, 1, 8, requires_grad=True)
        var = torch.randn_like(inv, requires_grad=True)
        mu = torch.randn(3, 7, 2, requires_grad=True)
        target = torch.randn_like(mu)
        std = torch.ones(3, 1, 2)
        sigma = head(inv, var)
        loss = head.nll(sigma, (target-mu).detach(), std)
        loss.backward()
        self.assertIsNone(mu.grad)
        self.assertIsNone(inv.grad)
        self.assertIsNone(var.grad)
        self.assertTrue(any(p.grad is not None for p in head.parameters()))
        first = head.sample(mu.detach(), sigma.detach(), std, 8,
                            torch.Generator().manual_seed(13))
        second = head.sample(mu.detach(), sigma.detach(), std, 8,
                             torch.Generator().manual_seed(13))
        self.assertEqual(first.shape, (3, 8, 7, 2))
        self.assertTrue(torch.equal(first, second))
        self.assertGreater(float((first[:, 0]-first[:, 1]).abs().mean()), 0.)


if __name__ == "__main__":
    unittest.main()
