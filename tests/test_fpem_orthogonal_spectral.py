import unittest

import torch

from models.predictive_env.orthogonal_spectral_decomposition import (
    OrthogonalComplementaryDecomposer, PredictiveSpectralDecomposer,
    StreamingPredictiveSpectrum, orthogonal_complement,
)


class OrthogonalSpectralTest(unittest.TestCase):
    def test_complementarity_orthogonality_and_gradients(self):
        for shape in ((8, 3, 4, 16), (8, 5, 1, 16)):
            h = torch.randn(*shape)
            model = OrthogonalComplementaryDecomposer(16, 4, coverage=True)
            out = model(h)
            inv, var = out["z_inv_tokens"], out["z_var_tokens"]
            self.assertTrue(torch.allclose(inv+var, h, atol=1e-6))
            self.assertLess(float((inv*var).sum(-1).abs().max()), 1e-5)
            q = model.basis()
            self.assertLess(float((q.T@q-torch.eye(4)).norm()), 1e-5)
            (inv.square().mean()+sum(out["aux_losses"].values())).backward()
            self.assertTrue(torch.isfinite(model.raw_basis.grad).all())
            self.assertGreater(float(model.raw_basis.grad.norm()), 0)

    def test_coverage_alone_has_basis_gradient(self):
        torch.manual_seed(31)
        h = torch.randn(12, 2, 3, 8)
        h[:4] *= 3
        model = OrthogonalComplementaryDecomposer(
            8, 2, coverage=True, shared_target=1.0,
            partial_low=0.0, partial_high=0.1)
        losses = model(h)["aux_losses"]
        sum(losses.values()).backward()
        self.assertGreater(float(model.raw_basis.grad.norm()), 0)

    def test_train_only_psd_and_ac(self):
        torch.manual_seed(4)
        stats = StreamingPredictiveSpectrum(8, 5)
        h = torch.randn(10, 3, 2, 8)
        y = torch.randn(10, 5, 3)
        with self.assertRaises(ValueError):
            stats.add(h, y, source_split="test")
        stats.add(h, y, source_split="train")
        q, info = stats.fit(6)
        self.assertEqual(info["train_rows"], 30)
        spectral = PredictiveSpectralDecomposer(8, 6)
        with self.assertRaises(ValueError):
            spectral.fit(q, info, source_split="val")
        spectral.fit(q, info, source_split="train")
        out = spectral(h)
        self.assertTrue(torch.allclose(out["z_inv_tokens"]+out["z_var_tokens"], h, atol=1e-6))
        self.assertLess(float((out["z_inv_tokens"]*out["z_var_tokens"]).sum(-1).abs().max()), 1e-5)
        complement = orthogonal_complement(q)
        self.assertEqual(complement.shape, (8, 2))
        self.assertLess(float((q.T@complement).norm()), 1e-5)
        ac = OrthogonalComplementaryDecomposer(8, 2)
        ac.initialize_variation_basis(complement)
        self.assertTrue(torch.allclose(ac(h)["z_inv_tokens"], out["z_inv_tokens"], atol=1e-5))


if __name__ == "__main__":
    unittest.main()
