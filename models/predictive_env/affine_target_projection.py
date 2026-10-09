"""TRAIN-label-only affine projection and matching continuous decoding."""

import math

import torch


def patch_centers(pred_len, patch_len, device, dtype):
    starts = torch.arange(0, pred_len, patch_len, device=device, dtype=dtype)
    ends = torch.clamp(starts + patch_len, max=pred_len)
    return (starts + ends - 1) / 2


def interpolate_patches(values, pred_len, patch_len):
    """Linearly decode [B,C,M,D] at actual patch centers to [B,C,L,D]."""
    if values.ndim != 4:
        raise ValueError("patch values must be [B,C,M,D]")
    count = math.ceil(pred_len / patch_len)
    if values.shape[2] != count:
        raise ValueError("patch count does not match horizon")
    if count == 1:
        return values.expand(-1, -1, pred_len, -1)
    centers = patch_centers(pred_len, patch_len, values.device, values.dtype)
    time = torch.arange(pred_len, device=values.device, dtype=values.dtype)
    right = torch.searchsorted(centers.contiguous(), time.contiguous()).clamp(max=count - 1)
    left = (right - 1).clamp(min=0)
    weight = ((time - centers[left]) / (centers[right] - centers[left]).clamp_min(1e-8))
    weight = weight.clamp(0, 1).view(1, 1, pred_len, 1)
    return values.index_select(2, left) * (1 - weight) + values.index_select(2, right) * weight


def project_affine_targets(y_true, y_inv, input_scale, patch_len, ridge=1e-3):
    """Find per-patch regularized gamma/beta; no gradients enter Yinv/labels."""
    if y_true.shape != y_inv.shape or y_true.ndim != 3:
        raise ValueError("y_true/y_inv must align as [B,L,C]")
    horizon = y_true.shape[1]
    y = y_true.detach().transpose(1, 2)
    inv = y_inv.detach().transpose(1, 2)
    scale = input_scale.detach().transpose(1, 2).clamp_min(1e-4)
    theta, centers = [], []
    for start in range(0, horizon, patch_len):
        end = min(start + patch_len, horizon)
        inv_patch = inv[..., start:end]
        center = inv_patch.mean(-1, keepdim=True)
        u = (inv_patch - center) / scale
        residual = (y[..., start:end] - inv_patch) / scale
        beta = residual.mean(-1)
        gamma = (u * (residual - beta.unsqueeze(-1))).sum(-1) / (
            u.square().sum(-1) + float(ridge)
        )
        theta.append(torch.stack((gamma, beta), dim=-1))
        centers.append(center.squeeze(-1))
    return torch.stack(theta, dim=2), torch.stack(centers, dim=2)


def decode_affine(y_inv, input_scale, theta, center_patch, patch_len, innovation=None):
    """Decode physical-unit paths; theta is [B,S,C,M,2] or [B,C,M,2]."""
    if y_inv.ndim != 3:
        raise ValueError("y_inv must be [B,L,C]")
    if theta.ndim == 4:
        theta = theta.unsqueeze(1)
    batch, samples, channels, _, _ = theta.shape
    horizon = y_inv.shape[1]
    if (batch, channels) != (y_inv.shape[0], y_inv.shape[2]):
        raise ValueError("theta and Yinv batch/channel dimensions differ")
    flat = theta.reshape(batch * samples, channels, theta.shape[3], 2)
    point = interpolate_patches(flat, horizon, patch_len).reshape(batch, samples, channels, horizon, 2)
    center = interpolate_patches(center_patch.unsqueeze(-1), horizon, patch_len)[..., 0]
    inv = y_inv.detach().transpose(1, 2).unsqueeze(1)
    scale = input_scale.detach().transpose(1, 2).clamp_min(1e-4).unsqueeze(1)
    gamma, beta = point.unbind(-1)
    correction = gamma * (inv - center.unsqueeze(1)) + scale * beta
    if innovation is not None:
        if innovation.shape != correction.shape:
            raise ValueError("innovation must be [B,S,C,L]")
        correction = correction + scale * innovation
    return (inv + correction).permute(0, 1, 3, 2)


def oracle_affine_diagnostics(y_true, y_inv, input_scale, patch_len, ridge=1e-3):
    theta, centers = project_affine_targets(y_true, y_inv, input_scale, patch_len, ridge)
    reconstructed = decode_affine(y_inv, input_scale, theta, centers, patch_len)[:, 0]
    inv_error = (y_true - y_inv).square().mean()
    oracle_error = (y_true - reconstructed).square().mean()
    return {
        "oracle_affine_MSE": oracle_error,
        "oracle_explained_residual_ratio": 1 - oracle_error / inv_error.clamp_min(1e-8),
        "oracle_gamma_abs_mean": theta[..., 0].abs().mean(),
        "oracle_beta_abs_mean": theta[..., 1].abs().mean(),
    }
