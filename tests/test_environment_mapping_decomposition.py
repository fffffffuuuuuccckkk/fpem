"""Small deterministic tests for the EPMD mapping and spectral operator."""

import numpy as np
import pytest
import torch

from models.predictive_env.environment_mapping_decomposition import (
    EnvironmentMappingDecomposer,
    EnvironmentMappingEstimator,
    flatten_forecast_order,
    mapping_operator,
)


@pytest.mark.parametrize("patches,width,horizon", [(1, 8, 3), (3, 4, 5), (6, 2, 7)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_estimator_and_decomposition(patches, width, horizon, dtype):
    generator = torch.Generator().manual_seed(13)
    hidden = torch.randn(48, 2, patches, width, generator=generator, dtype=dtype)
    flat = flatten_forecast_order(hidden)
    target = flat @ torch.randn(patches * width, horizon, generator=generator, dtype=dtype)
    target = target + 0.1 * torch.randn(target.shape, generator=generator, dtype=dtype)
    q = torch.rand(48, 3, generator=generator, dtype=dtype)
    q = q / q.sum(-1, keepdim=True)
    fit = EnvironmentMappingEstimator(patches * width, horizon, 3)
    fit.add(hidden, target, q)
    basis, eigenvalues, mean, masses, diagnostics = fit.finish(np.zeros((patches * width, horizon)))
    assert basis.shape == (patches * width, len(eigenvalues))
    assert np.all(eigenvalues >= 0) and np.all(eigenvalues <= 1 + 1e-6)
    assert len(diagnostics["environments"]) == 3
    decomposer = EnvironmentMappingDecomposer(patches * width).to(dtype=dtype)
    decomposer.update_mapping(basis, eigenvalues, mean, diagnostics)
    invariant, variant, _ = decomposer(hidden)
    assert torch.allclose(invariant + variant, hidden, rtol=1e-5, atol=1e-6)
    assert invariant.shape == variant.shape == hidden.shape
    (invariant.square().mean() + variant.square().mean()).backward()
    assert decomposer.alpha_logit.grad is not None
    assert torch.isfinite(decomposer.alpha_logit.grad)
    clone = EnvironmentMappingDecomposer(patches * width).to(dtype=dtype)
    clone.load_state_dict(decomposer.state_dict())
    assert torch.allclose(clone(hidden)[0], invariant)


def test_uniform_environment_has_zero_variation():
    torch.manual_seed(41)
    hidden = torch.randn(30, 1, 2, 3)
    future = torch.randn(30, 1, 4)
    fit = EnvironmentMappingEstimator(6, 4, 3)
    fit.add(hidden, future, torch.full((30, 3), 1 / 3))
    basis, eigenvalues, mean, _, _ = fit.finish(np.zeros((6, 4)))
    assert eigenvalues.size == 0
    module = EnvironmentMappingDecomposer(6)
    module.update_mapping(basis, eigenvalues, mean, {})
    invariant, variant, _ = module(hidden)
    assert torch.equal(invariant, hidden)
    assert torch.count_nonzero(variant) == 0


def test_operator_exact_small_matrix():
    mappings = np.array([
        [[1., 0.], [0., 1.], [0., 0.]],
        [[1., 0.], [0., 0.], [1., 0.]],
    ])
    basis, values, diagnostics = mapping_operator(mappings, np.array([4., 6.]))
    g = (basis * values) @ basis.T
    assert np.allclose(g, g.T, atol=1e-10)
    assert np.linalg.eigvalsh(g).min() >= -1e-10
    assert np.linalg.eigvalsh(g).max() <= 1 + 1e-10
    assert diagnostics["B"]["trace"] > 0
    zero_basis, zero_values, _ = mapping_operator(np.stack([mappings[0]] * 2), np.ones(2))
    assert zero_basis.shape == (3, 0) and not len(zero_values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_forward_buffers():
    module = EnvironmentMappingDecomposer(12).cuda()
    module.update_mapping(np.eye(12, 2), np.array([0.1, 0.9]), np.zeros(12), {})
    hidden = torch.randn(4, 3, 3, 4, device="cuda")
    invariant, variant, _ = module(hidden)
    assert invariant.is_cuda and variant.is_cuda
    assert torch.allclose(invariant + variant, hidden, atol=1e-6)
