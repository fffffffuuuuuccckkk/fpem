"""Compact prototype-conditional memory of complete point-forecast residuals."""
from __future__ import annotations

import torch

from models.predictive_env.dual_pattern_memory.uncertainty_memory import _weighted_quantile


@torch.no_grad()
def fit_residual_memory(residual, q, starts, span, rank=2):
    """One event per L+T; global fallback and low-rank trajectory summaries."""
    n, c, horizon = residual.shape
    if q.shape[:2] != (n, c) or len(starts) != n:
        raise ValueError("residual memory shape mismatch")
    rows, last = [], -span
    for i, start in enumerate(starts.tolist()):
        if start - last >= span:
            rows.append(i)
            last = start
    if len(rows) < 2:
        raise ValueError("insufficient independent residual events")
    x = residual[rows].reshape(-1, horizon).float().cpu()
    w = q[rows].reshape(-1, q.shape[-1]).float().cpu()
    def summarize(weight):
        weight = weight.clamp_min(0)
        total = weight.sum().clamp_min(1e-8)
        center = _weighted_quantile(x, weight[:, None].expand_as(x), .5)
        deviation = x - center
        mad = _weighted_quantile(deviation.abs(), weight[:, None].expand_as(x), .5)
        low = _weighted_quantile(x, weight[:, None].expand_as(x), .025)
        high = _weighted_quantile(x, weight[:, None].expand_as(x), .975)
        positive = (weight[:, None] * deviation.clamp_min(0)).sum(0) / total
        negative = (weight[:, None] * deviation.clamp_max(0)).sum(0) / total
        effective = float(total.square() / weight.square().sum().clamp_min(1e-8))
        # Numerical PCA of the weighted trajectory residual; no TRAIN-row
        # memory is retained and inference complexity is O(M*T*rank).
        matrix = deviation * (weight / total).sqrt()[:, None]
        try:
            _, _, vh = torch.linalg.svd(matrix, full_matrices=False)
            basis = vh[:rank]
        except RuntimeError:
            basis = x.new_zeros(0, horizon)
        if len(basis) < rank:
            basis = torch.cat((basis, x.new_zeros(rank-len(basis), horizon)))
        return {"center": center, "mad": mad, "q025": low, "q975": high,
                "positive": positive, "negative": negative,
                "basis": basis, "effective_support": effective}
    global_stats = summarize(torch.ones(len(x)))
    prototypes = [summarize(w[:, j]) for j in range(w.shape[-1])]
    return {"global": global_stats, "prototypes": prototypes,
            "independent_event_count": len(rows),
            "source": "later purged TRAIN downstream holdout; A0 saw TRAIN, not full-system OOF"}


def memory_features(memory, q, distance, history_scale, stability, reliability,
                    low_rank=True):
    """[B,C,M] -> [B,C,T,14], with continuous low-support fallback."""
    b, c, m = q.shape
    prototypes = memory["prototypes"]
    horizon = len(memory["global"]["center"])
    fields = ("center", "mad", "q025", "q975", "positive", "negative")
    pieces = []
    for field in fields:
        local = torch.stack([item[field] for item in prototypes]).to(q)
        global_field = memory["global"][field].to(q)
        support = q.new_tensor([item["effective_support"] for item in prototypes])
        shrink = support / (support + 8)
        local = shrink[:, None] * local + (1 - shrink[:, None]) * global_field[None]
        pieces.append(torch.einsum("bcm,mt->bct", q, local))
    for k in range(2):
        local = torch.stack([item["basis"][k] for item in prototypes]).to(q) if low_rank else q.new_zeros(m, horizon)
        pieces.append(torch.einsum("bcm,mt->bct", q, local))
    minimum = distance.min(-1).values
    entropy = -(q.clamp_min(1e-8).log() * q).sum(-1)
    eff = q.new_tensor([item["effective_support"] for item in prototypes])
    effective = torch.einsum("bcm,m->bc", q, eff)
    stable = torch.einsum("bcm,m->bc", q, stability.to(q))
    reliable = torch.einsum("bcm,m->bc", q, reliability.to(q).mean(-1))
    scale = history_scale.transpose(1, 2).squeeze(-1).clamp_min(1e-4)
    for value in (minimum, entropy, effective.log1p(), stable, reliable, scale.log()):
        pieces.append(value[..., None].expand(b, c, horizon))
    return torch.stack(pieces, -1)
