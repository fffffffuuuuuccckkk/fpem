"""Auxiliary conditional forecast head for testing Zvar utility."""

import torch
import torch.nn.functional as F
from torch import nn


class ConditionalVariantPredictor(nn.Module):
    """Predict each channel's horizon from stop-grad Zinv and trainable Zvar."""

    def __init__(self, d_model, pred_len):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, pred_len),
        )

    @staticmethod
    def pool(z_inv_tokens, z_var_tokens):
        # Pool patches but retain variables: [B,C,P,D] -> [B,C,D].
        invariant_condition = z_inv_tokens.detach().mean(dim=2)
        variant = z_var_tokens.mean(dim=2)
        return invariant_condition, variant

    def predict_pooled(self, invariant_condition, variant):
        features = torch.cat((invariant_condition, variant), dim=-1)
        return self.network(features).permute(0, 2, 1)

    def forward(self, z_inv_tokens, z_var_tokens):
        invariant_condition, variant = self.pool(z_inv_tokens, z_var_tokens)
        return self.predict_pooled(invariant_condition, variant)

    def paired_and_shuffled(self, z_inv_tokens, z_var_tokens):
        invariant_condition, variant = self.pool(z_inv_tokens, z_var_tokens)
        paired = self.predict_pooled(invariant_condition, variant)
        permutation = torch.roll(
            torch.arange(variant.shape[0], device=variant.device), shifts=1
        )
        shuffled = self.predict_pooled(invariant_condition, variant[permutation])
        return paired, shuffled, permutation


def reliability_weighted_utility_loss(reliability, full_loss, invariant_loss):
    """Penalize harmful Zvar use only where reliability elects to use it."""
    return (reliability * F.relu(full_loss - invariant_loss)).mean()


def conditional_gain_ranking_loss(pair_loss, shuffled_loss, margin=0.0):
    """Improve true pairs without training the negative branch to get worse."""
    return F.relu(pair_loss - shuffled_loss.detach() + float(margin)).mean()
