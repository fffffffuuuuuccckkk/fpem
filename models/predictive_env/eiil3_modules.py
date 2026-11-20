"""Minimal K=3 soft-IRM head and direct weighted residual Ridge utilities.

The q passed here must come from a history-only router for deployment. Teacher
q is permitted only for TRAIN supervision and diagnostics.
"""
from __future__ import annotations

import numpy as np
import torch


def soft_risks_and_full_head_penalty(head, design, truth, ratio, q):
    """Return R_k and exactly sum_k ||grad_(W,b) R_k||² / K.

    ``design`` is [N,D], ``truth`` is [N,T], and ``ratio`` is [N,1].
    This analytic linear-head gradient stays differentiable w.r.t. head.
    """
    if q.ndim != 2 or q.shape[0] != len(design) or q.shape[1] != 3:
        raise ValueError("K=3 q and design batch do not match")
    if truth.ndim != 2 or ratio.shape != (len(design), 1):
        raise ValueError("truth/ratio shape mismatch")
    q = q.detach()
    pred = head(design) * ratio
    error = pred - truth
    risk_by_sample = error.square().mean(dim=1)
    mass = q.sum(dim=0).clamp_min(1e-8)
    risks = (q * risk_by_sample[:, None]).sum(dim=0) / mass
    factor = 2.0 * (error * ratio) / truth.shape[1]
    grad_weight = torch.einsum("nk,nt,nd->ktd", q, factor, design) / mass[:, None, None]
    grad_bias = torch.einsum("nk,nt->kt", q, factor) / mass[:, None]
    per_env_norm_sq = grad_weight.square().sum(dim=(1, 2)) + grad_bias.square().sum(dim=1)
    penalty = per_env_norm_sq.mean()
    return risks, penalty, per_env_norm_sq.sqrt(), pred


def weighted_residual_ridge(x, residual, q, alpha):
    """Fit three independent weighted Ridge maps; the intercept is unpenalized.

    The objective is sum_i q_ik ||X_i W_k + b_k - r_i||² + alpha||W_k||².
    No shared map, low-rank truncation, or environment centering is used.
    """
    x = np.asarray(x, np.float64)
    residual = np.asarray(residual, np.float64)
    q = np.asarray(q, np.float64)
    if x.ndim != 2 or residual.ndim != 2 or q.shape != (len(x), 3):
        raise ValueError("expected x[N,D], residual[N,T], q[N,3]")
    if alpha < 0 or not np.isfinite(q).all() or np.min(q) < -1e-7:
        raise ValueError("invalid alpha or q")
    weights, biases, masses = [], [], []
    for k in range(3):
        sample_weight = np.clip(q[:, k], 0., None)
        mass = sample_weight.sum()
        if mass <= 1e-8:
            raise ValueError("environment has zero effective mass")
        xmean = (sample_weight[:, None] * x).sum(axis=0) / mass
        ymean = (sample_weight[:, None] * residual).sum(axis=0) / mass
        xc = x - xmean
        yc = residual - ymean
        gram = xc.T @ (sample_weight[:, None] * xc) + alpha * np.eye(x.shape[1])
        rhs = xc.T @ (sample_weight[:, None] * yc)
        try:
            coefficient = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            coefficient = np.linalg.lstsq(gram, rhs, rcond=None)[0]
        weights.append(coefficient)
        biases.append(ymean - xmean @ coefficient)
        masses.append(float(mass))
    return {"weight": np.asarray(weights, np.float32),
            "bias": np.asarray(biases, np.float32),
            "mass": masses, "alpha": float(alpha)}


def predict_residual_candidates(model, x, q):
    x = np.asarray(x, np.float32)
    q = np.asarray(q, np.float32)
    if q.shape != (len(x), 3):
        raise ValueError("router q must have shape [N,3]")
    candidates = np.einsum("nd,kdt->nkt", x, model["weight"]) + model["bias"][None]
    correction = np.einsum("nk,nkt->nt", q, candidates)
    disagreement = np.einsum("nk,nkt->nt", q, (candidates-correction[:, None]) ** 2)
    return candidates, correction, disagreement


def global_residual_ridge(x, residual, alpha):
    """Same linear solve with q=1, used as a capacity control."""
    x = np.asarray(x, np.float64)
    y = np.asarray(residual, np.float64)
    mean_x, mean_y = x.mean(0), y.mean(0)
    xc, yc = x - mean_x, y - mean_y
    weight = np.linalg.solve(xc.T @ xc + alpha*np.eye(x.shape[1]), xc.T @ yc)
    return {"weight": weight.astype(np.float32),
            "bias": (mean_y-mean_x@weight).astype(np.float32), "alpha": float(alpha)}
