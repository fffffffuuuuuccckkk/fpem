import pytest
import torch

from models.predictive_env.statistical_decomposition import make_statistical_decomposer
from models.predictive_env.statistical_decomposition.factorizers import (
    HouseholderCoordinates, grouped_statistics,
)


@pytest.mark.parametrize("shape", [(3, 7, 12, 64), (2, 7, 1, 64), (2, 1, 1, 64)])
@pytest.mark.parametrize("method", ["t1_group", "t2_orthogonal", "c3_dual", "t0_stnorm",
                                     "c1_covariance", "t4_learned", "c8_mixed"])
def test_forward_backward_and_descriptor(shape, method):
    torch.manual_seed(3)
    hidden = torch.randn(shape, requires_grad=True)
    decomposer = make_statistical_decomposer(method, shape[-1], 4, shape[1], shape[2])
    result = decomposer(hidden)
    assert set(result) == {"z_inv", "z_var_stats", "diagnostics"}
    assert result["z_inv"].shape == hidden.shape
    assert result["z_var_stats"] is not None
    assert torch.isfinite(result["z_inv"]).all()
    result["z_inv"].square().mean().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    if method in ("t1_group", "t2_orthogonal", "c1_covariance", "t4_learned"):
        assert result["diagnostics"]["reconstruction_error"] < 3e-4
    if method == "t0_stnorm":
        assert decomposer.views == 1 + (shape[1] > 1) + (shape[2] > 1)


def test_group_stats_use_exact_epsilon_adjusted_std_for_reconstruction():
    hidden = torch.ones(2, 3, 1, 32)
    normalized, stats, mean, std = grouped_statistics(hidden, 4)
    recovered = (normalized.reshape(2, 3, 1, 4, 8) * std + mean).reshape_as(hidden)
    assert torch.allclose(recovered, hidden)
    assert torch.allclose(stats[..., 4:].exp().unsqueeze(-1), std)


def test_orthogonal_coordinates_identity_at_init_and_learnable():
    coords = HouseholderCoordinates(32)
    x = torch.randn(2, 4, 1, 32, requires_grad=True)
    y = coords(x)
    assert torch.allclose(y, x, atol=1e-5)
    assert coords.orthogonality_error() < 1e-5
    loss = (y * torch.randn_like(y)).sum()
    loss.backward()
    assert coords.vectors.grad is not None
    assert coords.vectors.grad.abs().sum() > 0


def test_checkpoint_round_trip():
    x = torch.randn(2, 4, 3, 64)
    original = make_statistical_decomposer("t2_orthogonal", 64, 4, 4, 3)
    restored = make_statistical_decomposer("t2_orthogonal", 64, 4, 4, 3)
    restored.load_state_dict(original.state_dict())
    assert torch.allclose(original(x)["z_inv"], restored(x)["z_inv"])
