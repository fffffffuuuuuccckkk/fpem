import torch
from torch import nn


class PatternAdapter(nn.Module):
    """Zero-initialized bottleneck residual adapter."""
    def __init__(self, d_model: int, bottleneck: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, d_model),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([hidden, context], dim=-1))
