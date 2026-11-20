import torch

from models.predictive_env.direct_invariant_forecaster import (
    DirectInvariantForecaster, direct_ridge, group_risk)


def test_direct_head_preserves_reference_function():
    torch.manual_seed(2)
    head = torch.nn.Linear(12, 7)
    model = DirectInvariantForecaster(3, 4, 7, pretrained_head=head)
    hidden = torch.randn(2, 5, 3, 4)
    flat = hidden.permute(0, 1, 3, 2).flatten(-2)
    torch.testing.assert_close(model(hidden), head(flat), atol=0, rtol=0)
    assert model(hidden).shape == (2, 5, 7)


def test_projected_head_and_ridge_have_finite_gradients():
    for p, d, horizon in ((1, 8, 24), (4, 16, 48), (7, 32, 720)):
        hidden = torch.randn(3, 2, p, d)
        model = DirectInvariantForecaster(p, d, horizon, rank=min(8, d))
        loss = model(hidden).square().mean() + model.orthogonality_loss()
        loss.backward()
        assert torch.isfinite(model.projection.weight.grad).all()
    x = torch.randn(40, 12)
    y = x[:, :3] @ torch.randn(3, 6)
    w, b, rows = direct_ridge(x, y, 0.01)
    assert rows == 40 and (x @ w + b - y).square().mean() < 1e-3


def test_group_risk_is_direct_future_supervision():
    pred = torch.randn(8, 6, 2, requires_grad=True)
    true = torch.randn_like(pred)
    group = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    for method in ("vrex", "dro"):
        loss = group_risk(pred, true, group, method, 0.1)
        assert torch.isfinite(loss)
        loss.backward(retain_graph=True)
    assert torch.isfinite(pred.grad).all()
