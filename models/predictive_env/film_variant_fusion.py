"""Feature-wise FiLM fusion conditioned only on the variant representation."""

import torch
from torch import nn


class FilmVariantFusion(nn.Module):
    """Z_final = (1 + gamma(Z_var)) * Z_inv + beta(Z_var)."""

    def __init__(
        self,
        d_model,
        bottleneck=64,
        gamma_scale=0.1,
        beta_scale=0.1,
    ):
        super().__init__()
        self.gamma_scale = float(gamma_scale)
        self.beta_scale = float(beta_scale)
        if self.gamma_scale <= 0 or self.beta_scale <= 0:
            raise ValueError("FiLM gamma/beta scales must be positive")
        self.generator = nn.Sequential(
            nn.LayerNorm(int(d_model)),
            nn.Linear(int(d_model), int(bottleneck)),
            nn.GELU(),
            nn.Linear(int(bottleneck), 2 * int(d_model)),
        )
        # Stage-2 starts from the invariant-only predictor. Both modulation
        # paths can then grow only when the forecasting objective supports it.
        nn.init.zeros_(self.generator[-1].weight)
        nn.init.zeros_(self.generator[-1].bias)

    def forward(self, z_inv, z_var):
        if z_inv.shape != z_var.shape:
            raise ValueError("FiLM requires Z_inv and Z_var with equal shape")
        gamma_raw, beta_raw = self.generator(z_var).chunk(2, dim=-1)
        gamma = self.gamma_scale * torch.tanh(gamma_raw)
        beta = self.beta_scale * torch.tanh(beta_raw)
        variation = gamma * z_inv + beta
        final = z_inv + variation
        # Existing sparsity/utility plumbing expects a feature-wise strength.
        # This is diagnostic/regularization magnitude, not an additive Z_var gate.
        strength = 0.5 * (
            gamma.abs() / self.gamma_scale + beta.abs() / self.beta_scale
        )
        return final, strength, variation, gamma, beta

