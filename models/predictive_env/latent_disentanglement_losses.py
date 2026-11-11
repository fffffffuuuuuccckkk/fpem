"""Training-only optional losses for latent decomposition."""

import torch
from torch.nn import functional as F


def orthogonality_loss(z_inv, z_var, mode="cosine"):
    inv = z_inv.mean((1, 2)).float()
    var = z_var.mean((1, 2)).float()
    if mode == "cosine":
        return F.cosine_similarity(inv, var, dim=-1, eps=1e-8).square().mean()
    if mode == "cross_cov":
        if inv.shape[0] < 2:
            return (inv * 0).sum()
        inv = F.normalize(inv - inv.mean(0), dim=0)
        var = F.normalize(var - var.mean(0), dim=0)
        return ((inv.T @ var) / inv.shape[0]).square().mean()
    raise ValueError("orth_type must be cosine or cross_cov")
