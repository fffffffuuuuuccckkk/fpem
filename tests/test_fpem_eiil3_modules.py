"""Small numerical checks for the K=3 head, weighted Ridge and dynamic U."""
import unittest

import numpy as np
import torch

from models.predictive_env.eiil3_modules import (
    soft_risks_and_full_head_penalty, weighted_residual_ridge,
    predict_residual_candidates,
)
from models.predictive_env.relation_prototypes import train_router, route


class EIIL3ModulesTest(unittest.TestCase):
    def test_full_head_gradient_matches_autograd(self):
        torch.manual_seed(3)
        head = torch.nn.Linear(5, 4, dtype=torch.float64)
        x = torch.randn(11, 5, dtype=torch.float64)
        y = torch.randn(11, 4, dtype=torch.float64)
        ratio = torch.rand(11, 1, dtype=torch.float64) + .5
        q = torch.softmax(torch.randn(11, 3, dtype=torch.float64), -1)
        risks, penalty, norms, pred = soft_risks_and_full_head_penalty(head, x, y, ratio, q)
        manual = []
        for k in range(3):
            risk = (q[:, k] * ((pred-y)**2).mean(1)).sum()/q[:, k].sum()
            gw, gb = torch.autograd.grad(risk, (head.weight, head.bias), create_graph=True)
            manual.append(gw.square().sum()+gb.square().sum())
            self.assertTrue(torch.allclose(risks[k], risk, atol=1e-10))
        self.assertTrue(torch.allclose(penalty, torch.stack(manual).mean(), atol=1e-10))
        self.assertTrue(torch.isfinite(norms).all())
        penalty.backward()
        self.assertTrue(torch.isfinite(head.weight.grad).all())
        self.assertGreater(float(head.weight.grad.norm()), 0)

    def test_weighted_ridge_and_dynamic_u(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(60, 6)).astype("f")
        q = rng.dirichlet([2, 2, 2], size=60).astype("f")
        y = x @ rng.normal(size=(6, 12)).astype("f") + .2*rng.normal(size=(60, 12))
        fitted = weighted_residual_ridge(x, y, q, alpha=2.)
        candidates, correction, u = predict_residual_candidates(fitted, x, q)
        self.assertEqual(candidates.shape, (60, 3, 12))
        self.assertEqual(u.shape, (60, 12))
        self.assertTrue(np.isfinite(u).all())
        self.assertTrue((u >= -1e-7).all())
        self.assertTrue(np.allclose(correction, np.einsum("nk,nkt->nt", q, candidates), atol=1e-5))
        same = {"weight": np.repeat(fitted["weight"][0:1], 3, axis=0),
                "bias": np.repeat(fitted["bias"][0:1], 3, axis=0)}
        self.assertLess(np.abs(predict_residual_candidates(same, x, q)[2]).max(), 1e-8)

    def test_router_three_and_legacy_eight(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(16, 64)).astype("f")
        q3 = rng.dirichlet([1, 1, 1], size=16).astype("f")
        net, _, _ = train_router(x, q3, seed=1, epochs=1, num_env=3)
        self.assertEqual(route(net, x).shape, (16, 3))
        q8 = rng.dirichlet([1]*8, size=16).astype("f")
        old, _, _ = train_router(x, q8, seed=1, epochs=1)
        self.assertEqual(route(old, x).shape, (16, 8))


if __name__ == "__main__":
    unittest.main()
