"""Stable consensus and NULL-safe reliable local correction."""
from __future__ import annotations

import torch


def stable_consensus(candidate, q, stability, fallback):
    """Current-query mapped futures, not stored average future curves."""
    weighted = q * stability.to(q)[None, None]
    mass = weighted.sum(-1)
    estimate = (weighted[..., None] * candidate).sum(2) / mass[..., None].clamp_min(1e-8)
    return torch.where(mass[..., None] > 1e-8, estimate, fallback), mass


def reliable_variation(stable, candidate, q, reliability):
    """Use evidence-backed mapped candidates; zero evidence returns stable."""
    n, c, m, horizon = candidate.shape
    if stable.shape != (n, c, horizon) or q.shape != (n, c, m):
        raise ValueError("fusion shape mismatch")
    if reliability.shape[0] != m:
        raise ValueError("prototype reliability mismatch")
    segments = reliability.shape[1]
    per_time = torch.repeat_interleave(reliability.to(stable),
                                       torch.tensor([(s + 1) * horizon // segments - s * horizon // segments
                                                     for s in range(segments)], device=stable.device), dim=1)
    correction = (q[..., None] * per_time[None, None] * (candidate - stable[:, :, None])).sum(2)
    return stable + correction, correction
