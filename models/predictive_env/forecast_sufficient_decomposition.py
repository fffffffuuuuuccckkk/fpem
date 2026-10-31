"""Forecast-sufficient orthogonal projections of a frozen A0 linear head.

The predictor flattens [B,C,P,D] in D,P order.  The basis therefore lives
in that exact flattened coordinate system, not in the last D dimension.
"""

from __future__ import annotations

import torch
from torch import nn


def flatten_forecast_hidden(hidden: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 4:
        raise ValueError("hidden must have shape [B,C,P,D]")
    return hidden.permute(0, 1, 3, 2).reshape(hidden.shape[0], hidden.shape[1], -1)


def unflatten_forecast_hidden(flat: torch.Tensor, patches: int, width: int) -> torch.Tensor:
    return flat.reshape(flat.shape[0], flat.shape[1], width, patches).permute(0, 1, 3, 2)


def spectral_candidates(singular_values: torch.Tensor, full_rank: int) -> list[int]:
    if full_rank < 1:
        return [0]
    energy = singular_values[:full_rank].double().square()
    cumulative = energy.cumsum(0) / energy.sum().clamp_min(1e-30)
    ranks = {full_rank}
    for fraction in (0.5, 0.75, 0.9, 0.95, 0.99, 1.0):
        ranks.add(min(full_rank, int(torch.searchsorted(cumulative, fraction).item()) + 1))
    for fraction in (0.125, 0.25, 0.5):
        ranks.add(max(1, min(full_rank, round(full_rank * fraction))))
    return sorted(ranks)


class ForecastSufficientDecomposer(nn.Module):
    """Fixed, non-materialized orthogonal projection onto columns of U_r."""

    def __init__(self, basis: torch.Tensor, rank: int):
        super().__init__()
        if basis.ndim != 2 or not 0 <= rank <= basis.shape[1]:
            raise ValueError("basis/rank mismatch")
        self.register_buffer("basis", basis[:, :rank].contiguous().detach().float())
        self.rank = int(rank)

    def forward(self, hidden: torch.Tensor):
        flat = flatten_forecast_hidden(hidden)
        if flat.shape[-1] != self.basis.shape[0]:
            raise ValueError("basis does not match forecast-head flatten order")
        if self.rank:
            basis = self.basis.to(dtype=flat.dtype)
            z_inv_flat = (flat @ basis) @ basis.T
        else:
            z_inv_flat = torch.zeros_like(flat)
        z_inv = unflatten_forecast_hidden(z_inv_flat, hidden.shape[2], hidden.shape[3])
        z_var = hidden - z_inv
        return z_inv, z_var, {"rank": self.rank}


@torch.no_grad()
def svd_reference_head(linear: nn.Linear) -> tuple[torch.Tensor, torch.Tensor, int]:
    """SVD after loading A0. Return left vectors, singular values, numeric rank."""
    weight = linear.weight.detach().float().T.contiguous()
    u, singular, _ = torch.linalg.svd(weight, full_matrices=False)
    tolerance = torch.finfo(weight.dtype).eps * max(weight.shape) * singular[0]
    rank = int((singular > tolerance).sum().item())
    return u.detach(), singular.detach(), rank


@torch.no_grad()
def random_basis(features: int, rank: int, seed: int, device: torch.device):
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    matrix = torch.randn(features, rank, generator=generator, dtype=torch.float32)
    return torch.linalg.qr(matrix, mode="reduced").Q.to(device)
