"""Prediction-space projection and reliability-limited full-future selection."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def stable_direction_basis(memory_oof: torch.Tensor, base_oof: torch.Tensor,
                           truth_oof: torch.Tensor, blocks: torch.Tensor,
                           max_rank: int = 8):
    """Keep directions whose memory forecast improves every OOF time block.

    SVD is taken on *predicted* differences, not directly on real futures.
    This is an empirical transferable-prediction subspace, not a proof of
    theoretical invariance.
    """
    if memory_oof.shape != base_oof.shape or truth_oof.shape != base_oof.shape:
        raise ValueError("OOF forecast shapes differ")
    prediction_delta = (memory_oof - base_oof).reshape(-1, base_oof.shape[-1]).double()
    if len(prediction_delta) < 2:
        return base_oof.new_zeros(base_oof.shape[-1], 0), []
    _, _, right = torch.linalg.svd(prediction_delta, full_matrices=False)
    selected, audits = [], []
    block_names = blocks.unique(sorted=True)
    for direction in right[:min(max_rank * 4, len(right))]:
        component = ((memory_oof - base_oof).double() @ direction)[:, :, None] * direction
        gains = []
        for block in block_names:
            present = blocks == block
            if bool(present.any()):
                before = (base_oof[present] - truth_oof[present]).double().square().mean()
                after = (base_oof[present] + component[present] - truth_oof[present]).double().square().mean()
                gains.append(float(before - after))
        if not gains:
            continue
        audits.append({"direction": len(audits), "block_gains": gains,
                       "accepted": all(gain > 0 for gain in gains)})
        if all(gain > 0 for gain in gains):
            selected.append(direction.float())
        if len(selected) >= max_rank:
            break
    basis = torch.stack(selected, -1) if selected else base_oof.new_zeros(base_oof.shape[-1], 0)
    return basis.to(base_oof), audits


def stable_forecast(base: torch.Tensor, memory: torch.Tensor,
                    basis: torch.Tensor, eta: float):
    """Projection-selective fusion of two complete future predictions."""
    if basis.shape[-1] == 0 or eta == 0:
        return base.clone()
    delta = memory - base
    return base + float(eta) * ((delta @ basis) @ basis.T)


def variation_forecast(stable: torch.Tensor, candidates: torch.Tensor,
                       weight: torch.Tensor, reliability: torch.Tensor,
                       basis: torch.Tensor, use_segments: bool = True,
                       segment_count: int = 8):
    """Blend complete future candidates only outside stable-prediction space.

    reliability [M,S] is TRAIN-only OOF evidence. NULL mass is implicit in
    sum(weight) <= 1; it can make the entire correction identically zero.
    """
    b, c, m, horizon = candidates.shape
    if weight.shape != (b,c,m):
        raise ValueError("candidate/weight shape mismatch")
    if reliability.shape != (m, segment_count):
        raise ValueError("TRAIN OOF reliability shape mismatch")
    if basis.shape[-1]:
        offset = candidates - stable[:,:,None]
        offset = offset - (offset @ basis) @ basis.T
        candidates = stable[:,:,None] + offset
    if use_segments:
        span = torch.div(torch.arange(horizon, device=stable.device) * segment_count,
                         horizon, rounding_mode="floor").clamp_max(segment_count-1)
        reliability_t = reliability.to(stable)[:, span]
        # One-step local averaging prevents artificial hard segment seams.
        reliability_t = F.avg_pool1d(
            F.pad(reliability_t[:,None], (2, 2), mode="replicate"),
            5, stride=1)[:,0]
    else:
        reliability_t = reliability.to(stable).mean(-1, keepdim=True).expand(m, horizon)
    mass = weight[:,:,:,None] * reliability_t[None,None]
    beta = mass.sum(2).clamp(0,1)
    prediction = (1-beta) * stable + (mass * candidates).sum(2)
    return prediction, beta


def prototype_segment_reliability(stable: torch.Tensor, candidates: torch.Tensor,
                                  truth: torch.Tensor, weight: torch.Tensor,
                                  segments: int = 8):
    """OOF per-pattern, per-future-segment gain with disagreement shrinkage."""
    n, c, m, horizon = candidates.shape
    if truth.shape != (n,c,horizon):
        raise ValueError("OOF target shape mismatch")
    source = (stable[:,:,None] - truth[:,:,None]).square()
    adapted = (candidates - truth[:,:,None]).square()
    benefit = source - adapted
    reliability = benefit.new_zeros(m, segments)
    audit = []
    for j in range(m):
        w = weight[:,:,j].flatten().float()
        for p in range(segments):
            start = p*horizon//segments
            stop = (p+1)*horizon//segments
            sample_gain = benefit[:,:,j,start:stop].mean(-1).flatten()
            total = w.sum()
            if total < 1e-8:
                gain, dispersion, effective, value = 0., 0., 0., 0.
            else:
                gain_t = (w * sample_gain).sum()/total
                disp_t = (w * (sample_gain-gain_t).square()).sum().div(total).sqrt()
                effective_t = total.square()/w.square().sum().clamp_min(1e-8)
                value_t = gain_t.clamp_min(0)/(gain_t.abs()+disp_t+1e-8)
                value_t *= ((effective_t-1).clamp_min(0)/(effective_t+1))
                gain, dispersion = float(gain_t), float(disp_t)
                effective, value = float(effective_t), float(value_t)
            reliability[j,p] = value
            audit.append({"pattern":j, "segment":p, "oof_gain":gain,
                          "gain_dispersion":dispersion, "effective_support":effective,
                          "reliability":value})
    return reliability, audit
