from types import SimpleNamespace

import torch

from tools.fpem_probabilistic_training import (
    HistoricalEMAMaturity, probability_loss_weight,
)


def test_historical_ema_matches_legacy_update_and_is_detached():
    controller = HistoricalEMAMaturity()
    first = controller.update(torch.tensor(2.0, requires_grad=True))
    assert first == {
        "maturity": 0.0, "inv_loss_ema": 2.0,
        "inv_loss_relative_change": 0.0,
    }
    second = controller.update(torch.tensor(1.0))
    assert abs(second["inv_loss_ema"] - 1.98) < 1e-8
    assert abs(second["inv_loss_relative_change"] - 0.01) < 1e-8
    assert second["maturity"] == 0.0
    third = controller.update(torch.tensor(1.98))
    assert 0.0 < third["maturity"] < 1.0


def test_historical_ema_weight_has_floor_without_gradient():
    controller = HistoricalEMAMaturity()
    args = SimpleNamespace(prob_loss_weight_mode="historical_ema_maturity",
                           prob_loss_min_weight=0.05)
    inv = torch.tensor(1.0, requires_grad=True)
    prob = torch.tensor(2.0, requires_grad=True)
    weight, terms = probability_loss_weight(
        inv, prob, (), (), args, 0, None, ema_controller=controller,
    )
    assert weight.item() == torch.tensor(0.05).item()
    assert not weight.requires_grad
    (weight * prob).backward()
    assert inv.grad is None
    assert prob.grad is not None
    assert terms["maturity"] == 0.0
