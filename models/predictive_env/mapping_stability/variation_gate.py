"""Fixed history-variation diagnostics and analytic local ridge/LOO."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .stable_mapping import history_scale


def _normal_history(x: torch.Tensor, floor: torch.Tensor):
    mean, scale = history_scale(x, floor)
    return ((x.float() - mean) / scale).transpose(1, 2)


def variation_description(x: torch.Tensor, floor: torch.Tensor,
                          center: torch.Tensor | None = None,
                          spread: torch.Tensor | None = None):
    """No learned encoder: 32 local differences + 16 trend-slope samples."""
    series = _normal_history(x, floor)
    diff = series[..., 1:] - series[..., :-1]
    smooth = F.avg_pool1d(series.flatten(0, 1).unsqueeze(1), 9,
                          stride=1, padding=4).reshape_as(series)
    slope = smooth[..., 1:] - smooth[..., :-1]
    descriptor = torch.cat((
        F.interpolate(diff.flatten(0, 1)[:, None], size=32,
                      mode="linear", align_corners=False).reshape(*series.shape[:2], 32),
        F.interpolate(slope.flatten(0, 1)[:, None], size=16,
                      mode="linear", align_corners=False).reshape(*series.shape[:2], 16)), -1)
    if center is not None:
        descriptor = (descriptor - center[None]) / spread[None].clamp_min(1e-3)
    return descriptor.clamp(-20, 20)


def variation_activity(x: torch.Tensor, floor: torch.Tensor):
    series = _normal_history(x, floor)
    smooth = F.avg_pool1d(series.flatten(0, 1)[:, None], 9,
                          stride=1, padding=4).reshape_as(series)
    fluct = (series - smooth).square().mean(-1)
    slow = (smooth - smooth.mean(-1, keepdim=True)).square().mean(-1)
    return fluct / (fluct + slow + 1e-8)


def nearest_variation_distance(query: torch.Tensor, source: torch.Tensor,
                               valid: torch.Tensor):
    distance = (source - query[:, :, None]).square().mean(-1)
    minimum = distance.masked_fill(~valid, float("inf")).min(-1).values
    return minimum


def coverage_from_ecdf(distance: torch.Tensor, sorted_train: torch.Tensor,
                       valid: torch.Tensor):
    """1 - empirical CDF of causal TRAIN nearest distances, no hard cutoff."""
    if sorted_train.ndim != 2:
        raise ValueError("sorted TRAIN distance [C,N] required")
    rank = torch.searchsorted(sorted_train.contiguous(),
                              distance.transpose(0, 1).contiguous(),
                              right=True).transpose(0, 1)
    coverage = 1 - rank.float() / sorted_train.shape[-1]
    return torch.where(valid & torch.isfinite(distance), coverage.clamp(0, 1),
                       torch.zeros_like(coverage))


def local_variation_mapping(query: torch.Tensor, source: torch.Tensor,
                            target: torch.Tensor, valid: torch.Tensor,
                            ridge: float, tau: float = 1.):
    """One operator per query. Kernel-ridge analytic leave-one-out errors.

    Invalid/padded neighbours are zeroed before the KxK solve. LOO errors
    are alpha_i / diag((K + lambda I)^-1), not in-sample residuals.
    """
    if query.ndim != 3 or source.ndim != 4 or target.ndim != 4:
        raise ValueError("query [B,C,F], source [B,C,K,F], target [B,C,K,T]")
    if source.shape[:3] != valid.shape or target.shape[:3] != valid.shape:
        raise ValueError("local source/target/valid mismatch")
    b, c, k = valid.shape
    horizon = target.shape[-1]
    if k == 0:
        empty = query.new_zeros(b, c, horizon)
        return {"candidate": empty, "constant": empty,
                "map_reliability": empty, "gain": empty,
                "loo_residual": target, "map_loo_error": empty,
                "constant_loo_error": empty}
    x = source.float() * valid[..., None]
    y = target.float() * valid[..., None]
    gram = x @ x.transpose(-1, -2)
    eye = torch.eye(k, dtype=gram.dtype, device=gram.device)
    matrix = gram + max(float(ridge), 1e-5) * eye
    inverse = torch.linalg.solve(matrix, eye)
    dual = inverse @ y
    query_kernel = (query.float().unsqueeze(-2) @ x.transpose(-1, -2)).squeeze(-2)
    candidate = torch.einsum("bck,bckt->bct", query_kernel, dual)
    # [B,C,K,T] true out-of-neighbourhood residuals.
    loo_residual = dual / inverse.diagonal(dim1=-2, dim2=-1).clamp_min(1e-10)[..., None]
    counts = valid.sum(-1)
    other_mean = (y.sum(2, keepdim=True) - y) / (counts[..., None, None] - 1).clamp_min(1)
    constant_loo = y - other_mean
    weight = valid[..., None].float()
    map_error = (loo_residual.square() * weight).sum(2) / counts.clamp_min(1)[..., None]
    zero_error = (y.square() * weight).sum(2) / counts.clamp_min(1)[..., None]
    constant_error = (constant_loo.square() * weight).sum(2) / counts.clamp_min(1)[..., None]
    reference = torch.minimum(zero_error, constant_error)
    gain = reference - map_error
    reliability = torch.sigmoid(gain / max(float(tau), 1e-6))
    reliability *= torch.where(counts >= 2, (counts - 1).float() /
                               (counts + 1).float(), torch.zeros_like(counts.float()))[..., None]
    constant = y.sum(2) / counts.clamp_min(1)[..., None]
    candidate = torch.where((counts > 0)[..., None], candidate,
                            torch.zeros_like(candidate))
    return {"candidate": candidate, "constant": constant,
            "map_reliability": reliability, "gain": gain,
            "loo_residual": loo_residual * weight,
            "map_loo_error": map_error, "constant_loo_error": constant_error,
            "effective_count": counts}
