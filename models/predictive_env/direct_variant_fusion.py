"""Direct selective reuse of complementary Z_var without an adapter."""

import torch
from torch import nn


class DirectGatedVariantFusion(nn.Module):
    """Z_final = Z_inv + sigmoid(gate(Z_inv,Z_var)) * Z_var."""

    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, 1),
        )
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def forward(self, z_inv, z_var):
        gate = torch.sigmoid(self.gate(torch.cat([z_inv, z_var], dim=-1)))
        variation = gate * z_var
        return z_inv + variation, gate, variation
