"""Latent-environment pooling, modulation, and reliability calibration."""

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TypedVariationFusion(nn.Module):
    """Apply raw-defined shape, scale, and shift with type-matched operators.

    Variation extraction is deliberately outside this module.  Encoders and
    gates consume only raw/pre-patch-normalization variation evidence; ``z_inv``
    is solely the representation being modulated.
    """

    COMPONENTS = {"shape", "geometry", "full"}

    def __init__(self, hidden_dim: int, pattern_dim: int, components: str = "full") -> None:
        super().__init__()
        if components not in self.COMPONENTS:
            raise ValueError("typed fusion components must be shape, geometry, or full")
        self.hidden_dim = int(hidden_dim)
        self.pattern_dim = int(pattern_dim)
        self.components = components
        self.shape_encoder = nn.Sequential(
            nn.LayerNorm(pattern_dim),
            nn.Linear(pattern_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.shape_gate = nn.Linear(2, 1)
        self.scale_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.scale_gate = nn.Linear(2, 1)
        self.shift_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.shift_gate = nn.Linear(2, 1)
        for encoder in (self.shape_encoder, self.scale_encoder, self.shift_encoder):
            nn.init.zeros_(encoder[-1].weight)
            nn.init.zeros_(encoder[-1].bias)
        for gate in (self.shape_gate, self.scale_gate, self.shift_gate):
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, -2.0)

    def forward(
        self,
        z_inv: Tensor,
        d_shape: Tensor,
        d_scale: Tensor,
        d_shift: Tensor,
        a_shape: Tensor,
        u_scale: Tensor,
        u_shift: Tensor,
        r_shape: Optional[Tensor] = None,
        r_scale: Optional[Tensor] = None,
        r_shift: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        prefix = z_inv.shape[:-1]
        if z_inv.ndim != 4 or z_inv.shape[-1] != self.hidden_dim:
            raise ValueError("z_inv must be [B,C,P,hidden_dim]")
        if d_shape.shape != (*prefix, self.pattern_dim):
            raise ValueError("d_shape must be [B,C,P,pattern_dim]")
        for name, value in (
            ("d_scale", d_scale), ("d_shift", d_shift), ("a_shape", a_shape),
            ("u_scale", u_scale), ("u_shift", u_shift),
        ):
            if value.shape != prefix:
                raise ValueError("{} must have shape [B,C,P]".format(name))
        ones = torch.ones_like(a_shape)
        r_shape = ones if r_shape is None else r_shape
        r_scale = ones if r_scale is None else r_scale
        r_shift = ones if r_shift is None else r_shift

        shape_enabled = self.components in {"shape", "full"}
        geometry_enabled = self.components in {"geometry", "full"}
        shape_magnitude = d_shape.float().square().mean(-1).sqrt().to(d_shape.dtype)
        if shape_enabled:
            shape_delta = self.shape_encoder(d_shape)
            shape_gate_learned = torch.sigmoid(
                self.shape_gate(torch.stack([a_shape, shape_magnitude], dim=-1))
            )
            shape_gate = a_shape.unsqueeze(-1) * r_shape.unsqueeze(-1) * shape_gate_learned
        else:
            shape_delta = torch.zeros_like(z_inv)
            shape_gate = z_inv.new_zeros(*prefix, 1)
        shape_correction = shape_gate * shape_delta
        z_after_shape = z_inv + shape_correction

        if geometry_enabled:
            scale_raw = self.scale_encoder(torch.stack([d_scale, u_scale], dim=-1))
            scale_gate_learned = torch.sigmoid(
                self.scale_gate(torch.stack([d_scale.abs(), u_scale], dim=-1))
            )
            scale_gate = u_scale.unsqueeze(-1) * r_scale.unsqueeze(-1) * scale_gate_learned
            log_scale_mod = scale_gate * torch.tanh(scale_raw)
            scale_factor = torch.exp(log_scale_mod)
        else:
            scale_raw = torch.zeros_like(z_inv)
            scale_gate = z_inv.new_zeros(*prefix, 1)
            log_scale_mod = torch.zeros_like(z_inv)
            scale_factor = torch.ones_like(z_inv)
        z_scale_only = scale_factor * z_inv
        z_after_scale = scale_factor * z_after_shape

        if geometry_enabled:
            shift_raw = self.shift_encoder(torch.stack([d_shift, u_shift], dim=-1))
            shift_gate_learned = torch.sigmoid(
                self.shift_gate(torch.stack([d_shift.abs(), u_shift], dim=-1))
            )
            shift_gate = u_shift.unsqueeze(-1) * r_shift.unsqueeze(-1) * shift_gate_learned
        else:
            shift_raw = torch.zeros_like(z_inv)
            shift_gate = z_inv.new_zeros(*prefix, 1)
        shift_bias = shift_gate * shift_raw
        z_shift_only = z_inv + shift_bias
        z_typed = z_after_scale + shift_bias
        return {
            "z_typed": z_typed,
            "z_after_shape": z_after_shape,
            "z_after_scale": z_after_scale,
            "z_scale_only": z_scale_only,
            "z_shift_only": z_shift_only,
            "shape_variation_norm": shape_magnitude,
            "shape_delta": shape_delta,
            "shape_gate": shape_gate,
            "shape_correction": shape_correction,
            "scale_raw": scale_raw,
            "scale_gate": scale_gate,
            "log_scale_mod": log_scale_mod,
            "scale_factor": scale_factor,
            "shift_raw": shift_raw,
            "shift_gate": shift_gate,
            "shift_bias": shift_bias,
            # Reserved type-specific reliability interface; fixed to one now.
            "r_shape": r_shape,
            "r_scale": r_scale,
            "r_shift": r_shift,
        }


class LatentEnvironmentEncoder(nn.Module):
    """Pool one latent environment from activated variant temporal tokens."""

    def __init__(self, hidden_dim: int, env_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1, bias=False)
        self.project = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, env_dim),
            nn.GELU(),
        )

    def forward(self, z_var: Tensor, activation: Tensor) -> Tensor:
        """Return ``e`` with shape ``[B, env_dim]`` from ``[B,C,P,D]``."""
        if z_var.ndim != 4 or activation.shape != z_var.shape[:-1]:
            raise ValueError("z_var must be [B,C,P,D] and activation [B,C,P]")
        logits = self.score(z_var).squeeze(-1)
        logits = logits + torch.log(activation.clamp_min(1e-6))
        weights = torch.softmax(logits.flatten(1), dim=-1).view_as(activation)
        pooled = torch.einsum("bcp,bcpd->bd", weights, z_var)
        return self.project(pooled)


class EnvironmentFusion(nn.Module):
    """Reliability-controlled representation modulation by a single latent e."""

    def __init__(self, hidden_dim: int, env_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.affine = nn.Linear(env_dim, hidden_dim * 2)
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(env_dim, hidden_dim, bias=False)
        self.scale = hidden_dim ** -0.5

    def candidate(self, z_inv: Tensor, environment: Tensor) -> Tuple[Tensor, Tensor]:
        """Return fully environment-conditioned representation and temporal gate."""
        gamma, beta = self.affine(environment).chunk(2, dim=-1)
        gamma = 1.0 + torch.tanh(gamma)
        z_env = gamma[:, None, None, :] * self.norm(z_inv) + beta[:, None, None, :]
        key = self.key(environment)[:, None, None, :]
        gate = torch.sigmoid((self.query(z_inv) * key).sum(-1) * self.scale)
        return z_env, gate

    def forward(self, z_inv: Tensor, environment: Tensor, reliability: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        z_env, temporal_gate = self.candidate(z_inv, environment)
        r = reliability.view(-1, 1, 1, 1).clamp(0.0, 1.0)
        z_fused = z_inv + r * temporal_gate.unsqueeze(-1) * (z_env - z_inv)
        return z_fused, z_env, temporal_gate


class EnvironmentReliability(nn.Module):
    """A tiny calibrator over activation and the two novelty signals."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.register_buffer("gain_count", torch.zeros((), dtype=torch.float64))
        self.register_buffer("gain_sum", torch.zeros((), dtype=torch.float64))
        self.register_buffer("gain_sum_sq", torch.zeros((), dtype=torch.float64))

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[-1] != 3:
            raise ValueError("reliability features must have shape [B,3]")
        return torch.sigmoid(self.linear(features)).squeeze(-1)

    @torch.no_grad()
    def reset_gain_statistics(self) -> None:
        self.gain_count.zero_()
        self.gain_sum.zero_()
        self.gain_sum_sq.zero_()

    @torch.no_grad()
    def update_gain_statistics(self, gain: Tensor) -> None:
        values = gain.detach().double().flatten()
        self.gain_count.add_(values.numel())
        self.gain_sum.add_(values.sum())
        self.gain_sum_sq.add_((values * values).sum())

    def gain_scale(self) -> Tensor:
        count = self.gain_count.clamp_min(2.0)
        mean = self.gain_sum / count
        variance = (self.gain_sum_sq / count - mean.square()).clamp_min(1e-6)
        return variance.sqrt().to(dtype=self.linear.weight.dtype)

    def target(self, gain: Tensor, update: bool) -> Tensor:
        if update:
            self.update_gain_statistics(gain)
        return torch.sigmoid(gain.detach() / self.gain_scale().detach().clamp_min(1e-3))

    @staticmethod
    def loss(prediction: Tensor, target: Tensor) -> Tensor:
        probability = prediction.float().clamp(1e-6, 1.0 - 1e-6)
        logits = torch.log(probability) - torch.log1p(-probability)
        return F.binary_cross_entropy_with_logits(logits, target.float())
