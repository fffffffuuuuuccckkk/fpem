"""Optional prototype-aware representation controls for the shared forecaster."""
from __future__ import annotations

import torch
from torch import nn

from models.predictive_env.prototype_guided_invariant import PrototypeGuidedSharedHead


class _Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, strength):
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.strength * grad, None


class PrototypeAdversarialHead(PrototypeGuidedSharedHead):
    """Predict Y while an auxiliary classifier tries to recover TRAIN-only q."""

    def __init__(self, tokens, width, horizon, *, rank=16, prototypes=8):
        super().__init__(tokens, width, horizon, rank=rank, prototypes=prototypes)
        self.prototype_classifier = nn.Sequential(
            nn.LayerNorm(rank or width), nn.Linear(rank or width, prototypes))

    def adversarial_logits(self, hidden, strength=1.0):
        shared = self.forecaster.projection(hidden) if self.forecaster.projection else hidden
        pooled = shared.mean(2)
        return self.prototype_classifier(_Reverse.apply(pooled, strength))


class PrototypeSharedPrivateHead(PrototypeGuidedSharedHead):
    """Minimal shared/private control: only shared coordinates forecast Y."""

    def __init__(self, tokens, width, horizon, *, rank=16, prototypes=8):
        super().__init__(tokens, width, horizon, rank=rank, prototypes=prototypes)
        self.shared_map = nn.Linear(width, width, bias=False)
        self.private_map = nn.Linear(width, width, bias=False)
        nn.init.eye_(self.shared_map.weight)
        nn.init.zeros_(self.private_map.weight)

    def split(self, hidden):
        return self.shared_map(hidden), self.private_map(hidden)

    def forward(self, hidden, q=None, base_normalized=None):
        shared, _ = self.split(hidden)
        return self.forecaster(shared)

    def split_penalty(self, hidden):
        shared, private = self.split(hidden)
        reconstruction = (shared + private - hidden).square().mean()
        si, vi = shared.mean(2), private.mean(2)
        # Normalization prevents a vanishing private branch from artificially
        # presenting a low raw inner product as disentanglement.
        decor = torch.nn.functional.cosine_similarity(si, vi, dim=-1).square().mean()
        return reconstruction + 0.01 * decor


def pairwise_prototype_error_loss(prediction, target, q, ids, embargo):
    """B3: match error *patterns*, never force same-q futures to be equal."""
    b, c, _ = prediction.shape
    error = prediction - target
    error = error - error.mean(-1, keepdim=True)
    normed = torch.nn.functional.normalize(error, dim=-1)
    similarity = torch.einsum('bcm,dcm->bcd', q.detach(), q.detach())
    separation = (ids[:, None] - ids[None, :]).abs() >= embargo
    separation.fill_diagonal_(False)
    weights = similarity * separation[:, None, :]
    if weights.sum() < 1e-8:
        return prediction.new_zeros(())
    # Shared error *direction* should not be prototype-specific. Penalize
    # within-prototype excess similarity relative to all eligible pairs.
    pair_cos = torch.einsum('bct,dct->bcd', normed, normed)
    eligible = separation[:, None, :].expand(b, c, b)
    background = pair_cos[eligible].mean()
    return ((pair_cos - background).square() * weights).sum() / weights.sum().clamp_min(1e-8)
