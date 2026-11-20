"""TRAIN-memory-only future consensus. Query truth is never an input."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def _median(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    masked = values.masked_fill(~valid[..., None], float("nan"))
    return torch.nan_to_num(torch.nanmedian(masked, dim=2).values, nan=0.)


def future_consensus(future: torch.Tensor, valid: torch.Tensor,
                     typical_scale: float | torch.Tensor) -> dict:
    """All retrieved futures [B,C,K,H] in source-history normalized units.

    The count factor prevents one surviving neighbour from claiming perfect
    consensus merely because its median absolute deviation is zero.
    """
    if future.ndim != 4 or valid.shape != future.shape[:3]:
        raise ValueError("future [B,C,K,H] and valid [B,C,K] required")
    count = valid.sum(-1)
    if future.shape[2] == 0:
        center = future.new_zeros(*future.shape[:2], future.shape[-1])
        disagreement = torch.zeros_like(center)
    else:
        center = _median(future, valid)
        disagreement = _median((future - center[:, :, None]).abs(), valid)
    scale = torch.as_tensor(typical_scale, device=future.device,
                            dtype=future.dtype).clamp_min(1e-6)
    reliability = count.float() / (count.float() + 2.)
    reliability = torch.where(count > 1, reliability, torch.zeros_like(reliability))
    weight = scale / (scale + disagreement) * reliability[..., None]
    return {"center": center, "disagreement": disagreement,
            "weight": weight, "reliability": reliability, "count": count}


def future_difference_features(future: torch.Tensor, valid: torch.Tensor,
                               typical_scale: float | torch.Tensor,
                               p_well: torch.Tensor | None = None) -> torch.Tensor:
    """Seven causal retrieved-future features, shaped [B,C,H,7]."""
    view = future_consensus(future, valid, typical_scale)
    center, disagreement, weight = (view[k] for k in
                                    ("center", "disagreement", "weight"))
    difference = (future - center[:, :, None]) * valid[..., None]
    variation = (1 - weight[:, :, None]) * difference
    negative = _median(variation.clamp(max=0), valid) if future.shape[2] else center
    positive = _median(variation.clamp(min=0), valid) if future.shape[2] else center
    asymmetry = positive + negative
    temporal = F.pad(disagreement[..., 1:] - disagreement[..., :-1], (1, 0))
    strength = _median(variation.abs(), valid) if future.shape[2] else center
    if p_well is None or future.shape[2] == 0:
        group_gap = torch.zeros_like(center)
    else:
        well = valid & (p_well >= .5)
        poor = valid & ~well
        both = well.any(-1) & poor.any(-1)
        group_gap = (_median(difference, well) - _median(difference, poor)) * both[..., None]
    reliability = view["reliability"][..., None].expand_as(center)
    return torch.stack((disagreement, asymmetry, temporal, strength,
                        group_gap, reliability, weight), -1)
