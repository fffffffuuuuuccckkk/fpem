"""Prototype-weighted H-space full-future operators and shared consensus."""
from __future__ import annotations

import numpy as np
import torch

from models.predictive_env.environment_mapping_decomposition import mapping_operator


def _weighted_ridge(x, y, w, ridge):
    x, y, w = x.double(), y.double(), w.double().clamp_min(0)
    mass = w.sum().clamp_min(1e-8)
    mean_x, mean_y = (w[:, None] * x).sum(0) / mass, (w[:, None] * y).sum(0) / mass
    xc, yc = x - mean_x, y - mean_y
    design = xc * w.sqrt()[:, None]
    target = yc * w.sqrt()[:, None]
    dim = x.shape[1]
    gram = design.T @ design / mass + ridge * torch.eye(dim, dtype=x.dtype)
    weight = torch.linalg.solve(gram, design.T @ target / mass)
    bias = mean_y - mean_x @ weight
    return weight.float(), bias.float()


def fit_prototype_mappings(hidden: torch.Tensor, truth: torch.Tensor,
                           q: torch.Tensor, ridge: float = 0.1,
                           max_rows: int = 4096, seed: int = 2021):
    """All W_j share flattened H coordinates; target is complete future Y."""
    if hidden.shape[:2] != q.shape[:2] or truth.shape[:2] != q.shape[:2]:
        raise ValueError("variable-level shape mismatch")
    x = hidden.permute(0, 1, 3, 2).flatten(-2).reshape(-1, hidden.shape[2]*hidden.shape[3])
    y = truth.reshape(-1, truth.shape[-1])
    weight = q.reshape(-1, q.shape[-1]).detach()
    if len(x) > max_rows:
        indices = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))[:max_rows]
        x, y, weight = x[indices], y[indices], weight[indices]
    common_w, common_b = _weighted_ridge(x, y, weight.sum(-1).clamp_min(1e-3), ridge)
    maps, bias, mass = [], [], []
    for k in range(q.shape[-1]):
        w, b = _weighted_ridge(x, y, weight[:, k], ridge)
        maps.append(w)
        bias.append(b)
        mass.append(float(weight[:, k].sum()))
    maps, bias = torch.stack(maps), torch.stack(bias)
    masses = torch.tensor(mass).clamp_min(1e-8)
    consensus = (maps * masses[:, None, None]).sum(0) / masses.sum()
    consensus_bias = (bias * masses[:, None]).sum(0) / masses.sum()
    basis, eig, spectral = mapping_operator(maps.numpy(), masses.numpy())
    return {"global_weight": common_w, "global_bias": common_b,
            "weights": maps, "bias": bias, "masses": masses,
            "consensus_weight": consensus, "consensus_bias": consensus_bias,
            "variation_basis": torch.from_numpy(basis).float(),
            "variation_eigenvalues": torch.from_numpy(eig).float(),
            "spectral": spectral, "fit_rows": len(x)}


def predict_mapping(hidden: torch.Tensor, fitted: dict, kind: str,
                    shrink: float = 0.0):
    x = hidden.permute(0, 1, 3, 2).flatten(-2)
    if kind == "global":
        w, b = fitted["global_weight"], fitted["global_bias"]
    elif kind == "consensus":
        w, b = fitted["consensus_weight"], fitted["consensus_bias"]
    elif kind == "spectral":
        w, b = fitted["consensus_weight"], fitted["consensus_bias"]
        v = fitted["variation_basis"]
        if v.shape[-1]:
            attenuation = (fitted["variation_eigenvalues"] * shrink).clamp(0, 1)
            w = w - v @ (attenuation[:, None] * (v.T @ w))
    else:
        raise ValueError(kind)
    return x @ w + b
