"""Small zero-init condition adapter for the existing residual Flow.

The retrieved values must be full-point chronological OOF residuals supplied
by the caller. This module never infers a current residual from future truth.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def _masked_median(values: torch.Tensor, valid: torch.Tensor):
    masked = values.masked_fill(~valid[..., None], float("nan"))
    result = torch.nanmedian(masked, dim=2).values
    return torch.nan_to_num(result, nan=0., posinf=0., neginf=0.)


def residual_difference_features(residual: torch.Tensor, valid: torch.Tensor,
                                 group_stable: torch.Tensor | None = None):
    """[B,C,K,H] + [B,C,K] -> [B,C,H,7] with zero for no OOF sources."""
    if residual.ndim != 4 or valid.shape != residual.shape[:3]:
        raise ValueError("residual [B,C,K,H], valid [B,C,K] required")
    if residual.shape[2] == 0:
        return residual.new_zeros(*residual.shape[:2], residual.shape[-1], 7)
    position = _masked_median(residual, valid)
    dispersion = _masked_median((residual - position[:, :, None]).abs(), valid)
    magnitude = _masked_median(residual.abs(), valid)
    low = _masked_median(residual.clamp(max=0), valid)
    high = _masked_median(residual.clamp(min=0), valid)
    if group_stable is None:
        gap = torch.zeros_like(position)
    else:
        stable = _masked_median(residual, valid & group_stable)
        under = _masked_median(residual, valid & ~group_stable)
        both = (valid & group_stable).any(-1) & (valid & ~group_stable).any(-1)
        gap = (stable - under) * both[..., None]
    support = (valid.float().sum(-1) / max(1, valid.shape[-1]))[..., None]
    support = support.expand_as(position)
    return torch.stack((position, dispersion, magnitude, low, high, gap,
                        support), -1)


class ResidualDifferenceCondition(nn.Module):
    """FiLM for [B,C,M,D] or a separately ablated output-scale adapter."""

    def __init__(self, feature_dim: int, condition_dim: int, patch_len: int = 16,
                 mode: str = "film"):
        super().__init__()
        if mode not in {"film", "output_scale"}:
            raise ValueError(mode)
        self.patch_len = int(patch_len)
        self.mode = mode
        output = 2 * condition_dim if mode == "film" else 1
        self.net = nn.Sequential(nn.LayerNorm(feature_dim),
                                 nn.Linear(feature_dim, max(16, condition_dim // 2)),
                                 nn.GELU(),
                                 nn.Linear(max(16, condition_dim // 2), output))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _patch(self, features, count):
        b, c, horizon, f = features.shape
        padded = count * self.patch_len
        if padded < horizon:
            raise ValueError("condition has too few future patches")
        x = F.pad(features, (0, 0, 0, padded - horizon))
        mask = features.new_zeros(padded)
        mask[:horizon] = 1
        mask = mask.view(1, 1, count, self.patch_len, 1)
        x = x.view(b, c, count, self.patch_len, f)
        return (x * mask).sum(3) / mask.sum(3).clamp_min(1)

    def forward(self, condition: torch.Tensor, features: torch.Tensor):
        if condition.ndim != 4 or features.ndim != 4 or condition.shape[:2] != features.shape[:2]:
            raise ValueError("condition [B,C,M,D], residual features [B,C,H,F] required")
        if self.mode != "film":
            raise RuntimeError("output_scale adapter uses scale_correction()")
        pooled = self._patch(features.detach(), condition.shape[2])
        a, b = self.net(pooled).chunk(2, -1)
        return (1 + a) * condition + b

    def scale_correction(self, features: torch.Tensor):
        if self.mode != "output_scale":
            raise RuntimeError("FiLM adapter has no output scale")
        return self.net(features.detach()).squeeze(-1).clamp(-2, 2).exp()


def select_features(features: torch.Tensor, mode: str):
    """All variants preserve the same shape/parameter count as full U5."""
    if mode == "U0":
        return torch.zeros_like(features)
    indices = {
        "U1": (0, 1, 2), "U2": (0,), "U3": (1,),
        "U4": (0, 1, 2), "U5": tuple(range(7)),
        "U6": tuple(range(7)), "U7": tuple(range(7)),
        "U8": tuple(range(7)),
    }[mode]
    mask = features.new_zeros(features.shape[-1])
    mask[list(indices)] = 1
    return features * mask
