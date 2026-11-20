"""Regression test for the optional five-epoch decomposition delay."""

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from tools.fpem_probabilistic_training import train_epoch_prob


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.probabilistic_affine_dynamics = nn.Identity()
        self.probabilistic_affine_dynamics.register_buffer(
            "theta_scale", torch.ones(2)
        )
        self.environment_calls = 0

    def forward_components(self, x, cycle_index):
        prediction = x[:, :2] * self.weight
        return {"invariant_prediction": prediction,
                "z_inv": prediction, "z_var": prediction,
                "backbone_aux_loss": None}

    def environment_classification_loss(self, z_inv, z_var, q):
        self.environment_calls += 1
        loss = z_inv.square().mean()
        return loss, {"var_env_soft_ce": loss, "inv_env_soft_ce": loss}


def test_five_epoch_forecast_only_then_classification():
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.ones(2, 2, 1)
    y = torch.zeros(2, 2, 1)
    sample_id = torch.tensor([0, 1])
    cycle_index = torch.zeros(2, dtype=torch.long)
    batch = (x, y, sample_id, None, None, cycle_index, None)
    assignment = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float32)
    args = SimpleNamespace(prob_ablation="invariant_only", pred_len=2,
                           max_train_batches=0, lambda_prob_env=0.1,
                           lambda_timefilter_moe=0.0, prob_grad_clip=0.0,
                           use_stochastic_innovation=False,
                           decomposition_warmup_epochs=5)
    warm = train_epoch_prob(model, [batch], assignment, optimizer, args,
                            torch.device("cpu"), epoch_index=4)
    assert model.environment_calls == 0
    assert warm["decomposition_active"] == 0.0
    assert warm["loss/env"] == 0.0
    active = train_epoch_prob(model, [batch], assignment, optimizer, args,
                              torch.device("cpu"), epoch_index=5)
    assert model.environment_calls == 1
    assert active["decomposition_active"] == 1.0
    assert active["loss/env"] > 0.0
