"""Direct selective reuse of complementary Z_var without an adapter."""

import torch
from torch import nn
from torch.nn import functional as F


def variant_gain_loss(full_loss_per_sample, invariant_loss_per_sample, temperature):
    """Smoothly require full prediction to beat the detached invariant branch."""
    if full_loss_per_sample.ndim != 1 or invariant_loss_per_sample.ndim != 1:
        raise ValueError("variant gain losses must be sample-wise vectors")
    if full_loss_per_sample.shape != invariant_loss_per_sample.shape:
        raise ValueError("full and invariant sample losses must have equal shape")
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("variant gain temperature must be positive")
    difference = full_loss_per_sample - invariant_loss_per_sample.detach()
    return temperature * F.softplus(difference / temperature).mean()


class DirectGatedVariantFusion(nn.Module):
    """Z_final = Z_inv + sigmoid(gate(Z_inv,Z_var)) * Z_var."""

    def __init__(self, d_model, gate_type="token"):
        super().__init__()
        self.gate_type = str(gate_type)
        if self.gate_type not in ("token", "feature"):
            raise ValueError("gate_type must be token or feature")
        # Always instantiate the legacy token layer first, so feature gating
        # consumes exactly the same global initialization RNG as the baseline.
        # The larger experimental layer is then created in a forked RNG scope;
        # downstream classifiers/predictors keep identical initialization.
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, 1),
        )
        if self.gate_type == "feature":
            with torch.random.fork_rng(devices=[]):
                self.gate[-1] = nn.Linear(2 * d_model, int(d_model))
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def forward(self, z_inv, z_var):
        gate = torch.sigmoid(self.gate(torch.cat([z_inv, z_var], dim=-1)))
        variation = gate * z_var
        return z_inv + variation, gate, variation
