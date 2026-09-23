"""Training-only predictor for past-to-future variant representations."""

import torch
import torch.nn.functional as F
from torch import nn


class FutureVariantPredictor(nn.Module):
    """Small predictor mapping pooled past Zvar to pooled future Zvar."""

    def __init__(self, dimension):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
        )

    def forward(self, pooled_past_zvar):
        return self.network(pooled_past_zvar)


def future_variant_objective(prediction, target):
    """Return scalar cosine loss and per-sample teacher diagnostics."""
    target = target.detach()
    cosine = F.cosine_similarity(prediction, target, dim=-1)
    return {
        "loss": 1.0 - cosine.mean(),
        "cosine": cosine,
        "l2": (prediction - target).norm(dim=-1),
        "prediction_norm": prediction.norm(dim=-1),
        "target_norm": target.norm(dim=-1),
    }
