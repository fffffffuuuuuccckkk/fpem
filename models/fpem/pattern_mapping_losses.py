"""Auxiliary losses for FPem-PMG."""

import torch
from torch import Tensor
import torch.nn.functional as F


def invariant_consensus_loss(z_inv: Tensor, context: Tensor, evidence: Tensor) -> Tensor:
    """Align high-evidence invariant tokens with stable graph context."""
    cosine_cost = 1.0 - F.cosine_similarity(z_inv.float(), context.float(), dim=-1)
    weights = evidence.detach().float()
    return (weights * cosine_cost).sum() / weights.sum().clamp_min(1.0)


def separation_loss(z_inv: Tensor, z_var: Tensor) -> Tensor:
    """Reduce redundancy without defining invariant/variant semantics."""
    cosine = F.cosine_similarity(z_inv.float(), z_var.float(), dim=-1)
    return cosine.square().mean()
