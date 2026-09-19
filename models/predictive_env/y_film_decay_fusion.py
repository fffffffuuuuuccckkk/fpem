"""Output-space FiLM with sample/channel-specific monotonic horizon decay."""

import torch
import torch.nn.functional as F
from torch import nn


class YFilmDecayFusion(nn.Module):
    """Condition a normalized invariant forecast on pooled ``Z_var``.

    The module never changes the latent representation.  It produces bounded
    scale/shift curves in normalized prediction space and multiplies them by
    ``exp(-rho * h/H)``.  Zero initialization makes the initial full forecast
    exactly equal to the invariant-only forecast.
    """

    def __init__(
        self,
        d_model,
        pred_len,
        bottleneck=64,
        gamma_scale=0.1,
        beta_scale=0.1,
        decay_bias=-4.0,
    ):
        super().__init__()
        self.pred_len = int(pred_len)
        self.gamma_scale = float(gamma_scale)
        self.beta_scale = float(beta_scale)
        if self.gamma_scale <= 0 or self.beta_scale <= 0:
            raise ValueError("Y-FiLM gamma/beta scales must be positive")
        self.generator = nn.Sequential(
            nn.LayerNorm(int(d_model)),
            nn.Linear(int(d_model), int(bottleneck)),
            nn.GELU(),
            nn.Linear(int(bottleneck), 2 * self.pred_len),
        )
        self.decay_head = nn.Sequential(
            nn.LayerNorm(int(d_model)),
            nn.Linear(int(d_model), 1),
        )
        nn.init.zeros_(self.generator[-1].weight)
        nn.init.zeros_(self.generator[-1].bias)
        nn.init.zeros_(self.decay_head[-1].weight)
        nn.init.constant_(self.decay_head[-1].bias, float(decay_bias))
        horizon = torch.arange(1, self.pred_len + 1, dtype=torch.float32)
        self.register_buffer("normalized_horizon", horizon / self.pred_len)

    def forward(self, invariant_prediction, z_var):
        if invariant_prediction.ndim != 3:
            raise ValueError("Y-FiLM expects invariant prediction [B,C,H]")
        if z_var.ndim != 4:
            raise ValueError("Y-FiLM expects Z_var [B,C,P,D]")
        pooled = z_var.mean(dim=2)
        gamma_raw, beta_raw = self.generator(pooled).chunk(2, dim=-1)
        gamma = self.gamma_scale * torch.tanh(gamma_raw)
        beta = self.beta_scale * torch.tanh(beta_raw)
        rho = F.softplus(self.decay_head(pooled))
        decay = torch.exp(
            -rho * self.normalized_horizon.to(dtype=rho.dtype).view(1, 1, -1)
        )
        effective_gamma = decay * gamma
        effective_beta = decay * beta
        variation = effective_gamma * invariant_prediction + effective_beta
        full_prediction = invariant_prediction + variation
        strength = 0.5 * (
            effective_gamma.abs() / self.gamma_scale
            + effective_beta.abs() / self.beta_scale
        )
        return (
            full_prediction,
            strength,
            variation,
            effective_gamma,
            effective_beta,
            rho,
            decay,
        )
