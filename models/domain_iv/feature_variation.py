import torch
from torch import nn


class FeatureVariationAdapter(nn.Module):
    """Token-wise gated state variation derived only from Z_var."""
    def __init__(self, d_model, bottleneck=64):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, bottleneck), nn.GELU(),
            nn.Linear(bottleneck, d_model),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, 1),
        )
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def forward(self, z_inv, z_var, mode="gated"):
        variation = self.adapter(z_var)
        if mode == "off":
            gate = torch.zeros(*z_var.shape[:-1], 1, device=z_var.device)
        elif mode == "forced":
            gate = torch.ones(*z_var.shape[:-1], 1, device=z_var.device)
        elif mode == "gated":
            gate = torch.sigmoid(self.gate(torch.cat([z_inv, z_var], -1)))
        else:
            raise ValueError("feature variation mode must be off, forced, or gated")
        applied = gate * variation
        return z_inv + applied, gate, applied
