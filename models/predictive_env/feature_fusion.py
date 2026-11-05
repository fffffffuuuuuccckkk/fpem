"""Identity-initialized O3/O4 hidden-space fusion, with no Y-space bypass."""
from __future__ import annotations

import torch
from torch import nn


class FeatureFusion(nn.Module):
    def __init__(self, width: int, mode: str):
        super().__init__()
        if mode not in ("F1", "F2"):
            raise ValueError(f"unknown feature fusion mode: {mode}")
        self.mode = mode
        if mode == "F1":
            self.net = nn.Linear(2 * width, width)
            with torch.no_grad():
                self.net.weight.zero_()
                self.net.weight[:, :width].copy_(torch.eye(width))
                self.net.bias.zero_()
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(2 * width), nn.Linear(2 * width, width),
                nn.GELU(), nn.Linear(width, width),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_inv: torch.Tensor, z_var: torch.Tensor) -> torch.Tensor:
        if z_inv.shape != z_var.shape or z_inv.ndim != 4:
            raise ValueError("Zinv/Zvar must have identical [B,C,P,D] shapes")
        combined = torch.cat((z_inv, z_var), dim=-1)
        return self.net(combined) if self.mode == "F1" else z_inv + self.net(combined)
