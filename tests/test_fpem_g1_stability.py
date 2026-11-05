"""Small deterministic checks for the isolated G1 stability estimators."""
import unittest

import torch

from tools.fpem_g1_stability_methods import G1Correction, fit, proxy_groups, registry_rows
from tools.run_fpem_dppm_g1_r3_full import residual_ridge, ridge_query


class G1StabilityTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2021)
        self.key = torch.randn(36, 2, 64)
        self.residual = .1*self.key[..., :12]+.01*torch.randn(36, 2, 12)
        self.metadata = {"starts": torch.arange(36)*32}

    def test_ridge_regression_compatibility(self):
        fitted = fit("RIDGE", self.key, self.residual, self.metadata, {"ridge": .1})
        legacy = ridge_query(self.key, residual_ridge(self.key, self.residual, .1))
        self.assertLess(float((fitted.predict(self.key)-legacy).abs().max()), 1e-5)

    def test_zero_and_shape(self):
        for method in ("ZERO", "U01", "U06", "U16", "U17", "U20", "U26", "U31", "G07"):
            fitted = fit(method, self.key, self.residual, self.metadata,
                         {"ridge": .1, "repeats": 3, "proxy": "time", "rank": 4})
            prediction = fitted.predict(self.key[:1])
            self.assertEqual(tuple(prediction.shape), (1, 2, 12))
            self.assertTrue(torch.isfinite(prediction).all(), method)
            restored = G1Correction.load(fitted.export())
            self.assertTrue(torch.allclose(prediction, restored.predict(self.key[:1])), method)
        zero = fit("ZERO", self.key, self.residual, self.metadata, {})
        self.assertEqual(float(zero.predict(self.key).abs().max()), 0.)

    def test_query_never_needs_targets(self):
        model = fit("U01", self.key, self.residual, self.metadata, {"ridge": .1})
        first = model.predict(self.key[:2])
        second = model.predict(self.key[:2])
        self.assertTrue(torch.equal(first, second))

    def test_proxy_group_membership_train_only(self):
        for kind in ("time", "key", "dynamics", "residual"):
            labels = proxy_groups(self.key, self.residual, self.metadata, kind, 4)
            self.assertEqual(tuple(labels.shape), (36,))
            self.assertGreaterEqual(int(labels.min()), 0)
            self.assertLess(int(labels.max()), 4)

    def test_registry_honestly_marks_deferred(self):
        registry = {r["method_id"]: r for r in registry_rows()}
        self.assertEqual(registry["U01"]["status"], "implemented")
        self.assertEqual(registry["U13"]["status"], "mathematical_duplicate")
        self.assertEqual(registry["G05"]["status"], "deferred")


if __name__ == "__main__":
    unittest.main()
