import numpy as np
import torch

from models.predictive_env.eiil3_linear_correction import (
    EnvironmentLinearCorrection, correction_design, observed_context, weighted_environment_loss,
)


def test_history_only_context_and_shapes():
    history = np.arange(4 * 64, dtype=np.float32).reshape(4, 64)
    base = np.ones((4, 24), dtype=np.float32)
    assert observed_context(history).shape == (4, 11)
    assert correction_design(base, history).shape == (4, 24 + 64 + 11)
    model = EnvironmentLinearCorrection(24 + 64 + 11, 24)
    q = torch.full((4, 3), 1 / 3)
    candidates, correction, uncertainty = model(torch.from_numpy(correction_design(base, history)), q)
    assert candidates.shape == (4, 3, 24)
    assert correction.shape == uncertainty.shape == (4, 24)
    assert torch.equal(correction, torch.zeros_like(correction))
    assert torch.equal(uncertainty, torch.zeros_like(uncertainty))


def test_gate_and_experts_learn_without_future_at_inference():
    torch.manual_seed(7)
    model = EnvironmentLinearCorrection(5, 3)
    features = torch.randn(20, 5)
    truth = torch.randn(20, 3)
    anchor = torch.zeros_like(truth)
    q = torch.softmax(torch.randn(20, 3), 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    initial = model(features, q)[1].detach().clone()
    for _ in range(3):
        candidates, _, _ = model(features, q)
        loss = weighted_environment_loss(candidates, anchor, truth, q)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.allclose(model(features, q)[1], initial)
    assert any(layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0
               for layer in model.experts)
    assert model.g.grad is not None and model.g.grad.abs().sum() > 0
