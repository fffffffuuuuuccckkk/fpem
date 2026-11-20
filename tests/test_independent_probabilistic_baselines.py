import pytest
import torch

from models.predictive_env.independent_probabilistic_baselines import METHODS, history_scale


@pytest.mark.parametrize('method', ('timegrad', 'tsflow', 'tsdiff'))
@pytest.mark.parametrize('channels', (1, 7))
@pytest.mark.parametrize('horizon', (24, 48))
def test_independent_training_and_sampling(method, channels, horizon):
    torch.manual_seed(19)
    model = METHODS[method](channels, 96, horizon, hidden=16, steps=3)
    past = torch.randn(2, 96, channels)
    future = torch.randn(2, horizon, channels)
    loss = model.training_loss(past, future)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
    assert all(g is None or torch.isfinite(g).all() for g in grads)
    model.eval()
    one = model.sample(past, 3, 2, torch.Generator().manual_seed(3407))
    two = model.sample(past, 3, 2, torch.Generator().manual_seed(3407))
    assert one.shape == (2, 3, horizon, channels)
    assert torch.isfinite(one).all()
    assert torch.allclose(one, two)
    assert not torch.allclose(one[:, 0], one[:, 1])


def test_tsflow_minimum_scale_for_flat_history():
    past = torch.ones(2, 96, 7)
    _, scale = history_scale(past, minimum=1.0)
    assert torch.all(scale == 1.0)
    model = METHODS['tsflow'](7, 96, 48, hidden=16, steps=3)
    future = past[:, :48] + 0.5
    loss = model.training_loss(past, future)
    assert torch.isfinite(loss) and loss < 100
