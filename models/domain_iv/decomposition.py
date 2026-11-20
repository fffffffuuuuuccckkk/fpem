import torch
from torch import nn


class _Branch(nn.Module):
    def __init__(self, d_model, bottleneck, residual):
        super().__init__()
        self.residual = residual
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, d_model),
        )
        if residual:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, hidden):
        output = self.net(hidden)
        return hidden + output if self.residual else output


class DomainRepresentationDecomposer(nn.Module):
    """Independent invariant and variant projections of PatchTST tokens."""
    def __init__(self, d_model, bottleneck=64):
        super().__init__()
        self.invariant = _Branch(d_model, bottleneck, residual=True)
        self.variant = _Branch(d_model, bottleneck, residual=False)

    def forward(self, hidden):
        return self.invariant(hidden), self.variant(hidden)
