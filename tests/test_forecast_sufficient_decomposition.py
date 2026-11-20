import torch
from torch import nn

from models.predictive_env.forecast_sufficient_decomposition import (
    ForecastSufficientDecomposer, flatten_forecast_hidden,
    spectral_candidates, svd_reference_head,
)


def test_full_projection_preserves_loaded_linear_forecast():
    torch.manual_seed(7)
    hidden = torch.randn(2, 3, 5, 8)
    linear = nn.Linear(40, 9)
    u, singular, rank = svd_reference_head(linear)
    assert rank == 9
    z_inv, z_var, _ = ForecastSufficientDecomposer(u, rank)(hidden)
    assert torch.allclose(z_inv + z_var, hidden, atol=1e-6)
    assert torch.allclose(linear(flatten_forecast_hidden(z_inv)),
                          linear(flatten_forecast_hidden(hidden)), atol=2e-6)
    assert (z_inv * z_var).sum().abs() < 1e-4
    assert spectral_candidates(singular, rank)[-1] == rank


def test_full_row_rank_is_identity_and_truncated_is_complementary():
    torch.manual_seed(8)
    hidden = torch.randn(2, 1, 1, 6)
    linear = nn.Linear(6, 12)
    u, _, rank = svd_reference_head(linear)
    assert rank == 6
    full_inv, full_var, _ = ForecastSufficientDecomposer(u, rank)(hidden)
    assert torch.allclose(full_inv, hidden, atol=1e-6)
    assert full_var.abs().max() < 1e-6
    inv, var, _ = ForecastSufficientDecomposer(u, 2)(hidden)
    assert torch.allclose(inv + var, hidden)
    assert var.abs().max() > 0


def test_gradient_and_checkpoint_roundtrip():
    hidden = torch.randn(2, 2, 3, 4, requires_grad=True)
    linear = nn.Linear(12, 3)
    u, _, rank = svd_reference_head(linear)
    projection = ForecastSufficientDecomposer(u, rank)
    inv, _, _ = projection(hidden)
    inv.square().mean().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    restored = ForecastSufficientDecomposer(torch.empty_like(u), rank)
    restored.load_state_dict(projection.state_dict())
    assert torch.allclose(restored(hidden.detach())[0], inv.detach())
