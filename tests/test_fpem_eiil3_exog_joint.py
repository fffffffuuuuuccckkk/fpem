"""Small state/gradient tests for the joint-head exogenous search."""
from __future__ import annotations

import torch
from torch import nn

from tools.run_fpem_eiil3_exog_joint import joint_state, restore_joint_state


class _ToyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.h_forecast_head = nn.Linear(4, 2)


class _ToyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = _ToyBase()
        self.proj = nn.Linear(3, 4, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.base.encoder.requires_grad_(False)
        self.base.h_forecast_head.requires_grad_(True)

    def forward(self, x, calendar):
        return self.base.h_forecast_head(self.base.encoder(x) + self.proj(calendar))


def test_joint_head_gradient_and_restoration():
    torch.manual_seed(4)
    adapter = _ToyAdapter()
    x = torch.randn(7, 3)
    calendar = torch.randn(7, 3)
    target = torch.randn(7, 2)
    initial = joint_state(adapter)
    original = adapter(x, calendar).detach().clone()
    optimizer = torch.optim.AdamW([
        {"params": adapter.base.h_forecast_head.parameters(), "lr": 1e-3},
        {"params": adapter.proj.parameters(), "lr": 1e-3},
    ])
    loss = (adapter(x, calendar) - target).square().mean()
    loss.backward()
    assert adapter.base.encoder.weight.grad is None
    assert adapter.base.h_forecast_head.weight.grad is not None
    assert adapter.proj.weight.grad is not None
    assert adapter.proj.weight.grad.norm() > 0
    optimizer.step()
    assert not torch.equal(adapter.proj.weight, initial["proj"]["weight"])
    assert not torch.equal(adapter.base.h_forecast_head.weight,
                           initial["head"]["weight"])
    restore_joint_state(adapter, initial)
    torch.testing.assert_close(adapter(x, calendar), original, atol=0, rtol=0)


def test_head_only_keeps_projection_zero():
    torch.manual_seed(5)
    adapter = _ToyAdapter()
    adapter.proj.requires_grad_(False)
    before = joint_state(adapter)
    x, calendar, target = (torch.randn(7, 3), torch.randn(7, 3),
                           torch.randn(7, 2))
    loss = (adapter(x, calendar) - target).square().mean()
    loss.backward()
    assert adapter.proj.weight.grad is None
    assert adapter.base.h_forecast_head.weight.grad is not None
    assert torch.equal(adapter.proj.weight, before["proj"]["weight"])
