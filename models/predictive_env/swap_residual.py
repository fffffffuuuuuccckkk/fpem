"""History-only swap supervision and lightweight online residual distribution.

The target residual is computed by the trainer from TRAIN labels and detached.
Neither module receives environment assignments or future labels at inference.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def select_swap_pairs(z_inv, sample_ids, *, seq_len, pred_len, top_k=2,
                      max_distance=0.5):
    """Return unique, nonoverlapping nearest-neighbour pairs [M,2]."""
    if z_inv.ndim != 4 or sample_ids.ndim != 1 or len(sample_ids) != len(z_inv):
        raise ValueError("z_inv [B,C,P,D] and sample_ids [B] required")
    if len(z_inv) < 2 or top_k < 1:
        return sample_ids.new_empty((0, 2)), z_inv.new_empty((0,))
    pooled = z_inv.detach().float().mean((1, 2))
    normalized = F.normalize(pooled, dim=-1)
    distance = torch.cdist(normalized, normalized)
    exclusion = (sample_ids[:, None] - sample_ids[None, :]).abs() < seq_len + pred_len
    distance.masked_fill_(exclusion, float("inf"))
    neighbours = distance.topk(min(top_k, len(z_inv) - 1), largest=False, dim=1)
    selected = set()
    for i in range(len(z_inv)):
        for j, d in zip(neighbours.indices[i].tolist(), neighbours.values[i].tolist()):
            if d <= max_distance:
                selected.add((min(i, j), max(i, j)))
    if not selected:
        return sample_ids.new_empty((0, 2)), z_inv.new_empty((0,))
    pair = torch.tensor(sorted(selected), device=z_inv.device, dtype=torch.long)
    return pair, distance[pair[:, 0], pair[:, 1]]


def swap_loss(model, prediction_fn, z_inv, y, mean, std, cycle, pairs,
              distance, *, temperature=0.25, criterion="huber"):
    """Both directions use the *recipient's* RevIN statistics and future label."""
    if pairs.numel() == 0:
        return z_inv.new_zeros(()), {"pair_fraction": 0., "target_difference": 0.,
                                     "mean_distance": 0.}
    i, j = pairs[:, 0], pairs[:, 1]
    pred_i_from_j = prediction_fn(model, z_inv[j], mean[i], std[i], cycle[i], head="active")
    pred_j_from_i = prediction_fn(model, z_inv[i], mean[j], std[j], cycle[j], head="active")
    def error(pred, truth):
        diff = pred - truth
        if criterion == "huber":
            return F.smooth_l1_loss(pred, truth, reduction="none").mean((1, 2))
        if criterion == "mse":
            return diff.square().mean((1, 2))
        raise ValueError(f"unknown swap criterion: {criterion}")
    weights = torch.exp(-distance.detach().square() / temperature)
    value = (weights * (error(pred_i_from_j, y[i]) +
                        error(pred_j_from_i, y[j]))).sum() / (2 * weights.sum().clamp_min(1e-8))
    diagnostics = {"pair_fraction": float(torch.unique(pairs).numel() / len(z_inv)),
                   "target_difference": float((y[i] - y[j]).detach().square().mean()),
                   "mean_distance": float(distance.mean())}
    return value, diagnostics


class VariantCorrection(nn.Module):
    """One existing-head-shaped linear correction, initialized at zero."""
    def __init__(self, tokens, width, horizon):
        super().__init__()
        self.linear = nn.Linear(tokens * width, horizon)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, z_var, input_std):
        b, c, _, _ = z_var.shape
        delta_norm = self.linear(z_var.reshape(b, c, -1))
        return delta_norm.permute(0, 2, 1) * input_std


def variant_difference_loss(delta, y, y_inv, pairs):
    if pairs.numel() == 0:
        return delta.new_zeros(())
    i, j = pairs[:, 0], pairs[:, 1]
    target = (y[i] - y[j] - (y_inv[i] - y_inv[j]).detach()).detach()
    return F.smooth_l1_loss(delta[i] - delta[j], target)


class ConditionalResidualScale(nn.Module):
    """Diagonal zero-mean Gaussian residual in historical RevIN scale."""
    def __init__(self, width, horizon, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(2 * width),
                                 nn.Linear(2 * width, hidden), nn.GELU(),
                                 nn.Linear(hidden, horizon))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -1.0)

    def forward(self, z_inv, z_var, *, detach_condition=True):
        features = torch.cat((z_inv.mean(2), z_var.mean(2)), dim=-1)
        if detach_condition:
            features = features.detach()
        return F.softplus(self.net(features)).clamp_min(1e-4).permute(0, 2, 1)

    @staticmethod
    def nll(sigma, residual_target, input_std):
        normalized = residual_target.detach() / input_std.detach().clamp_min(1e-4)
        return (sigma.log() + 0.5 * (normalized / sigma).square()).mean()

    @staticmethod
    def sample(mu, sigma, input_std, count, generator=None):
        if count < 2:
            raise ValueError("at least two trajectories required")
        noise = torch.randn((mu.shape[0], count, *mu.shape[1:]),
                            device=mu.device, dtype=mu.dtype, generator=generator)
        return mu[:, None] + noise * (sigma * input_std)[:, None]
