"""Compact, full-horizon descriptive statistics of final point OOF errors.

Stable and variation summaries describe the SAME residual events; they must
not be interpreted as independent variances or added together.
"""
from __future__ import annotations

import torch


def _weighted_quantile(values: torch.Tensor, weight: torch.Tensor, fraction: float):
    ordered, position = values.sort(dim=0)
    w = weight.gather(0, position)
    total = w.sum(0).clamp_min(1e-8)
    threshold = (w.cumsum(0) >= total * fraction).float().argmax(0)
    return ordered.gather(0, threshold[None]).squeeze(0)


@torch.no_grad()
def fit_uncertainty_memory(oof_error: torch.Tensor, weights: torch.Tensor,
                           max_examples: int = 4):
    """OOF errors [N,C,T], prototype weights [N,C,M], TRAIN only."""
    n, c, horizon = oof_error.shape
    if weights.shape[:2] != (n,c):
        raise ValueError("OOF/memory weight mismatch")
    flattened = oof_error.reshape(n*c,horizon).float()
    weight = weights.reshape(n*c,-1).float()
    fields = {name: [] for name in ("center", "mad", "q025", "q975",
                                      "negative_deviation", "positive_deviation",
                                      "support", "effective_support", "examples")}
    for j in range(weight.shape[-1]):
        w = weight[:,j]
        if float(w.sum()) < 1e-8:
            center = flattened.new_zeros(horizon)
            mad = center.clone()
            low = center.clone()
            high = center.clone()
            negative = center.clone()
            positive = center.clone()
            effective = 0.
            examples = center.new_zeros(0,horizon)
        else:
            center = _weighted_quantile(flattened, w[:,None].expand_as(flattened), .5)
            deviation = flattened - center[None]
            mad = _weighted_quantile(deviation.abs(), w[:,None].expand_as(flattened), .5)
            low = _weighted_quantile(flattened, w[:,None].expand_as(flattened), .025)
            high = _weighted_quantile(flattened, w[:,None].expand_as(flattened), .975)
            negative = (w[:,None]*deviation.clamp_max(0)).sum(0)/w.sum()
            positive = (w[:,None]*deviation.clamp_min(0)).sum(0)/w.sum()
            effective = float(w.sum().square()/w.square().sum().clamp_min(1e-8))
            examples = flattened[w.topk(min(max_examples,len(w))).indices]
        for name,value in (("center",center),("mad",mad),("q025",low),
                           ("q975",high),("negative_deviation",negative),
                           ("positive_deviation",positive),
                           ("support",int((w>0).sum())),
                           ("effective_support",effective),("examples",examples)):
            fields[name].append(value)
    return {key:(torch.stack(value) if key not in ("support","effective_support","examples")
                 else value) for key,value in fields.items()}
