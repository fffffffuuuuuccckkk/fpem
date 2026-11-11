"""Three environment-weighted one-layer output corrections (COSA-inspired).

This is an offline TRAIN-only FPEM adapter, not COSA's online test-time update.
Each expert predicts a *direct* residual; no Ridge solve or hidden MLP is used.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def observed_context(history_key):
    """History-only recent values and summary; never touches future truth."""
    x = np.asarray(history_key, np.float32)
    if x.ndim != 2 or x.shape[1] != 64:
        raise ValueError("expected history key [N,64]")
    recent = x[:, -16:]
    slope = (recent[:, -1] - recent[:, 0]) / 15.0
    return np.column_stack((x[:, -8:], recent.mean(1), recent.std(1), slope)).astype(np.float32)


def correction_design(stable_prediction, history_key):
    """[stable full future, observed history key, recent observed context]."""
    y = np.asarray(stable_prediction, np.float32)
    x = np.asarray(history_key, np.float32)
    if y.ndim != 2 or len(y) != len(x):
        raise ValueError("stable forecast and history rows differ")
    return np.concatenate((y, x, observed_context(x)), axis=1)


class EnvironmentLinearCorrection(nn.Module):
    """For each k: residual_k = tanh(g_k) * (W_k x + b_k)."""

    def __init__(self, input_dim: int, horizon: int, environments: int = 3):
        super().__init__()
        self.experts = nn.ModuleList(nn.Linear(input_dim, horizon) for _ in range(environments))
        self.g = nn.Parameter(torch.full((environments,), 0.1))
        for expert in self.experts:
            nn.init.zeros_(expert.weight)
            nn.init.zeros_(expert.bias)

    def forward(self, x, q):
        if q.ndim != 2 or q.shape != (len(x), len(self.experts)):
            raise ValueError("router weights must be [N,K]")
        candidates = torch.stack([layer(x) for layer in self.experts], dim=1)
        candidates = candidates * torch.tanh(self.g)[None, :, None]
        correction = (q.unsqueeze(-1) * candidates).sum(1)
        disagreement = (q.unsqueeze(-1) * (candidates - correction[:, None]).square()).sum(1)
        return candidates, correction, disagreement


def weighted_environment_loss(candidates, stable, truth, q):
    """Separate soft-environment risks, matching the previous Ridge supervision."""
    error = (stable[:, None, :] + candidates - truth[:, None, :]).square().mean(-1)
    mass = q.sum(0).clamp_min(1e-8)
    return ((q * error).sum(0) / mass).mean()
