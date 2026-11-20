"""Variable-level R3-prototype full-future constraints.

All target-dependent calculations are TRAIN losses. q is detached by design;
the invariant objective cannot collapse the R3 router to evade constraints.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def prototype_risks(pred: torch.Tensor, truth: torch.Tensor,
                    q: torch.Tensor, *, sample_level: bool = False):
    """Return risks, continuous effective-support weights, and support diagnostics."""
    if pred.shape != truth.shape or pred.ndim != 3 or q.shape[:2] != pred.shape[:2]:
        raise ValueError("expected prediction/target [B,C,T], route [B,C,M]")
    q = q.detach().float().clamp_min(0)
    loss = (pred - truth).square().mean(-1)
    if sample_level:
        q = q.mean(1)
        loss = loss.mean(1)
    flat_q = q.reshape(-1, q.shape[-1])
    flat_loss = loss.reshape(-1)
    mass = flat_q.sum(0)
    effective = mass.square() / flat_q.square().sum(0).clamp_min(1e-8)
    risks = (flat_q * flat_loss[:, None]).sum(0) / mass.clamp_min(1e-8)
    # Continuous support weighting; NULL has no prototype risk yet all rows
    # remain in the ordinary full-future forecasting objective.
    confidence = effective / (effective + 4.0)
    rho = mass * confidence
    rho = rho / rho.sum().clamp_min(1e-8)
    return risks, rho, {"mass": mass.detach(), "effective": effective.detach()}


def prototype_risk_penalty(pred, truth, q, kind="variance", strength=0.1,
                           base=None, sample_level=False):
    risks, rho, diagnostics = prototype_risks(pred, truth, q, sample_level=sample_level)
    if kind == "incremental":
        if base is None:
            raise ValueError("incremental risk requires a frozen A0 prediction")
        reference, _, _ = prototype_risks(base.detach(), truth, q, sample_level=sample_level)
        risks = risks - reference.detach()
    center = (rho * risks).sum()
    if kind in ("variance", "incremental"):
        penalty = (rho * (risks - center).square()).sum()
    elif kind == "dro":
        temp = max(float(strength), 1e-5)
        penalty = temp * torch.logsumexp(risks / temp + rho.clamp_min(1e-12).log(), 0)
    else:
        raise ValueError(kind)
    return penalty, {"risks": risks.detach(), "rho": rho.detach(), **diagnostics}


def shared_optimum_penalty(pred, truth, q, head_weight, variant="norm",
                           sample_level=False):
    """Differentiable group-head gradient penalties, not sample-backbone grads."""
    risks, rho, diagnostics = prototype_risks(pred, truth, q, sample_level=sample_level)
    grads = []
    for risk in risks:
        grad = torch.autograd.grad(risk, head_weight, retain_graph=True,
                                   create_graph=True, allow_unused=False)[0]
        grads.append(grad.flatten())
    g = torch.stack(grads)
    if variant == "norm":
        value = (rho[:, None] * g.square()).sum()
    elif variant == "variance":
        center = (rho[:, None] * g).sum(0)
        value = (rho[:, None] * (g - center).square()).sum()
    elif variant == "irm":
        # IRMv1-style scalar shared-head optimality check *after* a learned
        # H projection; explicitly distinct from full parameter gradient norm.
        scale = torch.tensor(1.0, device=pred.device, requires_grad=True)
        scaled = pred * scale
        group_scaled, _, _ = prototype_risks(scaled, truth, q, sample_level=sample_level)
        scalar_grads = [torch.autograd.grad(r, scale, create_graph=True,
                                           retain_graph=True)[0] for r in group_scaled]
        value = (rho * torch.stack(scalar_grads).square()).sum()
    else:
        raise ValueError(variant)
    return value, {"gradient_norms": g.detach().norm(dim=1), **diagnostics}


def conditional_agreement(pred, truth, q, feature, min_support=4.0):
    """Compare group residual mechanisms only inside common H support.

    Shared local support is approximated by a soft neighborhood around the
    common TRAIN feature center; different futures are never forced equal.
    """
    pooled = feature.detach().mean(1)
    center = pooled.mean(0)
    radius = (pooled - center).square().mean(-1)
    overlap = torch.exp(-radius / radius.median().clamp_min(1e-6))[:, None, None]
    weighted = q.detach() * overlap
    residual = truth - pred
    mass = weighted.sum((0, 1))
    valid = mass / (mass + min_support)
    mean_error = torch.einsum("bcm,bct->mt", weighted, residual) / mass[:, None].clamp_min(1e-8)
    common = (valid[:, None] * mean_error).sum(0) / valid.sum().clamp_min(1e-8)
    return (valid[:, None] * (mean_error - common).square()).sum() / valid.sum().clamp_min(1e-8)
