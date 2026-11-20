"""Portable FOIL and ShifTS adapters for the project's A0 forecasting backbones.

Both adapters use the existing PatchTST/ModernTCN A0 encoder and forecast head.
Training-only labels are never part of ``predict``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def backbone_forecast(backbone, x):
    """Return the ordinary A0 forecast and its encoded tokens, [B,H,C]."""
    tokens, mean, std = backbone.encode(x)
    pred = backbone.forecast_head(tokens).transpose(1, 2)
    return pred * std[:, :1] + mean[:, :1], tokens, mean, std


def horizon_shape(value, eps=1e-5):
    """FOIL's horizon-centered, scale-normalized predictive shape."""
    mean = value.mean(dim=-2, keepdim=True)
    scale = value.var(dim=-2, keepdim=True, unbiased=False).add(eps).sqrt()
    return (value - mean) / scale


class FOILAdapter(nn.Module):
    """FOIL environment-head discovery and its final invariant loss.

    The source method learns environment-specific forecast heads on a frozen
    pretrained representation, assigns each TRAIN sample to its best head,
    then trains a shared predictor with a robust average loss and a
    shape-error variance penalty.  This version uses the same mechanics on
    the project's channel-wise token contract, without importing Informer.
    """

    def __init__(self, backbone: nn.Module, d_model: int, pred_len: int, env_num=3):
        super().__init__()
        self.backbone = backbone
        self.env_num = int(env_num)
        self.env_heads = nn.Parameter(torch.empty(self.env_num, d_model, pred_len))
        nn.init.xavier_uniform_(self.env_heads)

    def predict(self, x):
        return backbone_forecast(self.backbone, x)[0]

    def environment_predictions(self, x):
        _, tokens, mean, std = backbone_forecast(self.backbone, x)
        pooled = tokens.mean(dim=2)  # [B,C,D]
        pred = torch.einsum("bcd,kdh->bkch", pooled, self.env_heads)
        return pred.transpose(-1, -2) * std[:, None, :1] + mean[:, None, :1]

    @torch.no_grad()
    def infer_environment(self, x, y):
        env_pred = self.environment_predictions(x)
        target_shape = horizon_shape(y).unsqueeze(1)
        shape_error = (horizon_shape(env_pred) - target_shape).square().mean((2, 3))
        return shape_error.argmin(dim=1)

    def environment_head_loss(self, x, y, source_reweight=0.2):
        with torch.no_grad():
            base = self.predict(x)
            base_error = (
                horizon_shape(base) - horizon_shape(y)
            ).square().mean((1, 2))
        env_pred = self.environment_predictions(x)
        env_error = (
            horizon_shape(env_pred) - horizon_shape(y).unsqueeze(1)
        ).square().mean((2, 3))
        best_error = env_error.min(dim=1).values
        weight = torch.where(
            best_error.detach() > base_error,
            torch.full_like(best_error, source_reweight),
            torch.ones_like(best_error),
        )
        return (weight * best_error).mean()

    def final_loss(self, x, y, env_labels, mean_weight=0.8):
        pred = self.predict(x)
        robust_mean = torch.log1p((pred - y).square()).mean()
        shape_error = (
            (pred - pred.mean(dim=1, keepdim=True))
            - (y - y.mean(dim=1, keepdim=True))
        ).square().mean(dim=1)  # [B,C], as in the source shape loss
        # Source FOIL computes variance over samples in sufficiently populated
        # discovered environments.  Keep the selection detached.
        counts = torch.bincount(env_labels, minlength=self.env_num)
        valid = counts[env_labels] >= 3
        shape_variance = (
            shape_error[valid].reshape(-1).var(unbiased=False)
            if bool(valid.any()) else robust_mean.new_zeros(())
        )
        loss = mean_weight * robust_mean + (1.0 - mean_weight) * shape_variance
        return loss, {"robust_mean": robust_mean.detach(),
                      "shape_variance": shape_variance.detach()}


class SimpleRevIN(nn.Module):
    """ShifTS' outer reversible normalization, distinct from the A0 internals."""

    def __init__(self, channels: int, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def normalize(self, x):
        mean = x.mean(dim=1, keepdim=True).detach()
        std = x.var(dim=1, keepdim=True, unbiased=False).add(self.eps).sqrt().detach()
        return (x - mean) / std * self.weight + self.bias, (mean, std)

    def denormalize(self, x, state):
        mean, std = state
        return (x - self.bias) / (self.weight + self.eps * self.eps) * std + mean


class ShifTSAdapter(nn.Module):
    """ShifTS lookback/horizon selector and cross-variable target translator.

    The upstream ShifTS code treats the last channel as the target and applies
    its causal auxiliary loss to the other channels.  We preserve that
    convention while keeping the project's multivariate forecast objective.
    ``x_all`` is TRAIN-only and contains aligned history/horizon windows.
    """

    def __init__(self, backbone: nn.Module, seq_len: int, channels: int):
        super().__init__()
        self.backbone = backbone
        self.seq_len = int(seq_len)
        self.channels = int(channels)
        self.revin = SimpleRevIN(channels)
        if channels > 1:
            self.mask = nn.Parameter(torch.rand(seq_len + 1, channels - 1))
            self.translator = nn.Linear(channels - 1, 1)
        else:
            self.register_parameter("mask", None)
            self.translator = None

    def selection_weights(self):
        if self.mask is None:
            return None
        weights = self.mask.softmax(dim=0)
        active = weights > weights.mean(dim=0, keepdim=True)
        selected = weights * active
        return selected / selected.sum(dim=0, keepdim=True).clamp_min(1e-8)

    def predict_with_aux(self, x, x_all=None):
        normalized, state = self.revin.normalize(x)
        base = backbone_forecast(self.backbone, normalized)[0]
        if self.translator is not None:
            target = base[:, :, -1:] + self.translator(base[:, :, :-1])
            adjusted = torch.cat((base[:, :, :-1], target), dim=-1)
        else:
            adjusted = base
        prediction = self.revin.denormalize(adjusted, state)
        if x_all is None or self.mask is None:
            return prediction, prediction.new_zeros(())
        if x_all.shape[1:3] != (self.seq_len + 1, prediction.shape[1]):
            raise ValueError("ShifTS TRAIN-only aligned windows have wrong shape")
        selected = torch.einsum(
            "bwhc,wc->bhc", x_all[:, :, :, :-1], self.selection_weights()
        )
        aux = F.mse_loss(selected, prediction[:, :, :-1])
        return prediction, aux

    def predict(self, x):
        return self.predict_with_aux(x)[0]


def train_only_aligned_windows(train_values, sample_ids, seq_len, pred_len):
    """ShifTS source windows; rejects any read across the TRAIN split."""
    ids = torch.as_tensor(sample_ids, dtype=torch.long, device=train_values.device)
    offsets = torch.arange(seq_len + 1, device=train_values.device)
    horizon = torch.arange(pred_len, device=train_values.device)
    index = ids[:, None, None] + offsets[None, :, None] + horizon[None, None, :]
    if int(index.min()) < 0 or int(index.max()) >= len(train_values):
        raise ValueError("ShifTS auxiliary windows would cross the TRAIN boundary")
    return train_values[index]
