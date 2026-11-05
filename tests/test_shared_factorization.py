"""Shape, algebra, environment, gradient and checkpoint tests for S1-S8."""

import io
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from models.predictive_env.shared_factorization import (
    EnvironmentPartitionAdapter, StreamingHiddenStatistics,
    make_shared_factor_decomposer,
)
from models.predictive_env.shared_factorization.covariance_utils import soft_projectors


class TinyModel(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width

    def encode(self, x, cycle):
        return x, None, None


class FactorizationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.hidden = torch.randn(12, 3, 4, 16)
        self.q = torch.softmax(torch.randn(12, 3), -1)
        self.stats = StreamingHiddenStatistics(16, 3)
        self.stats.add(self.hidden[:6], self.q[:6])
        self.stats.add(self.hidden[6:], self.q[6:])
        self.stats = self.stats.finish()

    def test_all_shapes_and_finite(self):
        methods = ("common_subspace", "shared_spectral", "robust_lowrank_sparse",
                   "supervised_joint_private", "shared_private_lite",
                   "pairwise_partial_shared", "sample_prevalence",
                   "subspace_recurrence", "subspace_recurrence_poly")
        for method in methods:
            for shape in ((4, 3, 4, 16), (5, 1, 1, 16)):
                with self.subTest(method=method, shape=shape):
                    module = make_shared_factor_decomposer(method, 16, seq_len=3)
                    h = torch.randn(shape, requires_grad=True)
                    result = module(h, sample_indices=torch.arange(shape[0]) * 4,
                                    train_env_q=torch.softmax(torch.randn(shape[0], 3), -1))
                    self.assertEqual(tuple(result["z_inv_tokens"].shape), shape)
                    self.assertEqual(tuple(result["z_var_tokens"].shape), shape)
                    self.assertTrue(torch.isfinite(result["z_inv_tokens"]).all())
                    self.assertTrue(torch.isfinite(result["z_var_tokens"]).all())
                    loss = result["z_inv_tokens"].square().mean() + sum(result["aux_losses"].values())
                    loss.backward()
                    self.assertTrue(torch.isfinite(h.grad).all())
                    if method in {"supervised_joint_private", "shared_private_lite", "pairwise_partial_shared"}:
                        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                            for p in module.parameters()))

    def test_streamed_soft_covariance_and_environment_controls(self):
        projected, ridge = soft_projectors(self.stats)
        self.assertGreater(ridge, 0)
        self.assertLess((projected - projected.mT).abs().max(), 1e-9)
        self.assertGreaterEqual(torch.linalg.eigvalsh(projected).min(), -1e-9)
        controller = EnvironmentPartitionAdapter(12, 3, "shuffled", seed=1)
        shuffled = controller.refresh_after_environment_update(self.q)
        self.assertTrue(torch.allclose(shuffled.sum(0), self.q.sum(0)))
        self.assertFalse(torch.allclose(shuffled, self.q))
        self.assertTrue(torch.equal(controller.get_assignments([0, 3]), shuffled[[0, 3]]))
        uniform = EnvironmentPartitionAdapter(12, 3, "uniform")
        uniform.refresh_after_environment_update(self.q)
        stream = StreamingHiddenStatistics(16, 3)
        stream.add(self.hidden, uniform.q)
        recurrence = make_shared_factor_decomposer("subspace_recurrence", 16)
        recurrence.fit(stream.finish())
        self.assertLess(recurrence.var_operator.abs().max(), 1e-5)

    def test_fitted_methods_and_reconstruction(self):
        for method in ("common_subspace", "shared_spectral", "robust_lowrank_sparse",
                       "subspace_recurrence", "subspace_recurrence_poly"):
            with self.subTest(method=method):
                module = make_shared_factor_decomposer(method, 16)
                info = module.fit(self.stats)
                result = module(self.hidden)
                self.assertTrue(torch.isfinite(result["z_inv_tokens"]).all())
                self.assertTrue(info)
                if method in {"shared_spectral", "robust_lowrank_sparse",
                              "subspace_recurrence", "subspace_recurrence_poly"}:
                    self.assertLess((result["reconstruction"] - self.hidden).abs().max(), 1e-4)
                buffer = io.BytesIO()
                torch.save(module.state_dict(), buffer)
                buffer.seek(0)
                copy = make_shared_factor_decomposer(method, 16)
                copy.load_state_dict(torch.load(buffer, weights_only=True))
                self.assertTrue(torch.allclose(module(self.hidden)["z_inv_tokens"],
                                               copy(self.hidden)["z_inv_tokens"]))

    def test_environment_label_permutation(self):
        perm = torch.tensor([2, 0, 1])
        a = StreamingHiddenStatistics(16, 3)
        b = StreamingHiddenStatistics(16, 3)
        a.add(self.hidden, self.q)
        b.add(self.hidden, self.q[:, perm])
        for method in ("common_subspace", "shared_spectral", "subspace_recurrence"):
            left = make_shared_factor_decomposer(method, 16)
            right = make_shared_factor_decomposer(method, 16)
            left.fit(a.finish()); right.fit(b.finish())
            self.assertTrue(torch.allclose(left.inv_operator, right.inv_operator, atol=1e-5))

    def test_sample_prevalence_and_identity_wrapper(self):
        dataset = [(self.hidden[i:i+1].squeeze(0), torch.zeros(1), torch.tensor(i),
                    torch.tensor(0)) for i in range(12)]
        loader = DataLoader(dataset, batch_size=4)
        module = make_shared_factor_decomposer("sample_prevalence", 16)
        module.fit(self.stats, model=TinyModel(16), loader=loader, device=torch.device("cpu"), q=self.q)
        result = module(self.hidden)
        self.assertLess((result["reconstruction"] - self.hidden).abs().max(), 1e-4)
        wrapped = make_shared_factor_decomposer("common_subspace", 16,
                                                function_preserving_init=True)
        wrapped.fit(self.stats)
        initial = wrapped(self.hidden)
        self.assertTrue(torch.equal(initial["z_inv_tokens"], self.hidden))
        self.assertTrue(torch.equal(initial["z_var_tokens"], torch.zeros_like(self.hidden)))


if __name__ == "__main__":
    unittest.main()
