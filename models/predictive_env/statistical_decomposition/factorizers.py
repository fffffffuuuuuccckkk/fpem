"""Trainable and parameter-free latent statistical decomposers.

All outputs act on Backbone H=[B,C,P,D].  The descriptor is the *actual*
statistical factor, not an H-shaped placeholder.  Paper names denote ideas
adapted to H-space; these are not literal reproductions of their architectures.
"""

from __future__ import annotations

import math

import torch
from torch import nn


EPS = 1e-5


def grouped_statistics(hidden: torch.Tensor, groups: int):
    if hidden.ndim != 4 or hidden.shape[-1] % groups:
        raise ValueError("Expected [B,C,P,D] with D divisible by groups")
    size = hidden.shape[-1] // groups
    grouped = hidden.reshape(*hidden.shape[:-1], groups, size)
    mean = grouped.mean(dim=-1, keepdim=True)
    std = ((grouped - mean).square().mean(dim=-1, keepdim=True) + EPS).sqrt()
    normalized = (grouped - mean) / std
    descriptor = torch.cat((mean.squeeze(-1), std.log().squeeze(-1)), dim=-1)
    return normalized.reshape_as(hidden), descriptor, mean, std


def covariance_statistics(hidden: torch.Tensor, groups: int):
    """Treat aligned within-group feature coordinates as covariance observations."""
    if hidden.shape[-1] % groups:
        raise ValueError("D must be divisible by groups")
    width = hidden.shape[-1] // groups
    matrix = hidden.reshape(*hidden.shape[:-1], groups, width).float()
    mean = matrix.mean(-1, keepdim=True)
    centered = matrix - mean
    covariance = centered @ centered.transpose(-1, -2) / width
    eye = torch.eye(groups, dtype=matrix.dtype, device=matrix.device)
    covariance = covariance + EPS * eye
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(EPS)

    def transform(power):
        return (eigenvectors * eigenvalues.pow(power).unsqueeze(-2)) @ eigenvectors.transpose(-1, -2)

    whiten = transform(-0.5)
    color = transform(0.5)
    log_covariance = (eigenvectors * eigenvalues.log().unsqueeze(-2)) @ eigenvectors.transpose(-1, -2)
    normalized = whiten @ centered
    descriptor = torch.cat((mean.squeeze(-1), log_covariance.flatten(-2)), dim=-1)
    recovered = color @ normalized + mean
    diagnostics = {
        "reconstruction_error": (recovered - matrix).abs().mean(),
        "cov_condition": (eigenvalues.max(-1).values / eigenvalues.min(-1).values).mean(),
        "cov_min_eigenvalue": eigenvalues.min(-1).values.mean(),
        "cov_offdiag_abs": (covariance - torch.diag_embed(covariance.diagonal(dim1=-2, dim2=-1))).abs().mean(),
    }
    return normalized.reshape_as(hidden).to(hidden.dtype), descriptor.to(hidden.dtype), diagnostics


class StatisticalDecomposer(nn.Module):
    method: str

    def _result(self, z_inv, descriptor, diagnostics=None):
        if z_inv.ndim != 4:
            raise ValueError("z_inv must be [B,C,P,D]")
        return {"z_inv": z_inv, "z_var_stats": descriptor,
                "diagnostics": diagnostics or {}}


class GroupFactorization(StatisticalDecomposer):
    method = "t1_group"

    def __init__(self, width, groups):
        super().__init__()
        self.width, self.groups = int(width), int(groups)
        if width % groups:
            raise ValueError("D must be divisible by groups")

    def forward(self, hidden):
        inv, stats, mean, std = grouped_statistics(hidden, self.groups)
        reconstructed = (inv.reshape(*hidden.shape[:-1], self.groups, -1) * std + mean).reshape_as(hidden)
        return self._result(inv, stats, {
            "reconstruction_error": (reconstructed - hidden).abs().mean(),
            "group_mean_abs": mean.abs().mean(), "group_std_mean": std.mean(),
        })


class HouseholderCoordinates(nn.Module):
    """Paired reflections yield exact identity at init, with nonzero gradients."""

    def __init__(self, width, pairs=2):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(86123 + int(width))
            initial = torch.randn(pairs, width)
        self.vectors = nn.Parameter(initial.repeat_interleave(2, dim=0))

    def forward(self, x, inverse=False):
        vectors = torch.nn.functional.normalize(self.vectors.float(), dim=-1)
        vectors = torch.flip(vectors, dims=(0,)) if inverse else vectors
        result = x
        for vector in vectors:
            v = vector.to(result.dtype)
            result = result - 2 * (result * v).sum(dim=-1, keepdim=True) * v
        return result

    @torch.no_grad()
    def orthogonality_error(self):
        eye = torch.eye(self.vectors.shape[-1], device=self.vectors.device)
        q = self.forward(eye)
        return (q.T @ q - eye).square().mean().sqrt()


class OrthogonalFactorization(StatisticalDecomposer):
    method = "t2_orthogonal"

    def __init__(self, width, groups):
        super().__init__()
        self.groups = int(groups)
        self.coordinates = HouseholderCoordinates(width)

    def forward(self, hidden):
        rotated = self.coordinates(hidden)
        normalized, stats, mean, std = grouped_statistics(rotated, self.groups)
        inv = self.coordinates(normalized, inverse=True)
        recovered_rotated = normalized.reshape(*hidden.shape[:-1], self.groups, -1) * std + mean
        reconstructed = self.coordinates(recovered_rotated.reshape_as(hidden), inverse=True)
        return self._result(inv, stats, {
            "reconstruction_error": (reconstructed - hidden).abs().mean(),
            "group_mean_abs": mean.abs().mean(), "group_std_mean": std.mean(),
        })


class STNormLatent(StatisticalDecomposer):
    method = "t0_stnorm"

    def __init__(self, width, groups, channels, patches):
        super().__init__()
        self.groups = int(groups)
        self.use_token = patches > 1
        self.use_channel = channels > 1
        self.views = 1 + int(self.use_token) + int(self.use_channel)
        self.fusion = nn.Linear(self.views * width, width)
        with torch.no_grad():
            self.fusion.weight.zero_()
            self.fusion.bias.zero_()
            self.fusion.weight[:, :width] = torch.eye(width)

    def forward(self, hidden):
        group_view, group_stats, _, _ = grouped_statistics(hidden, self.groups)
        views = [group_view]
        stats = {"group": group_stats}
        if self.use_token:
            mean = hidden.mean(2, keepdim=True)
            std = ((hidden - mean).square().mean(2, keepdim=True) + EPS).sqrt()
            views.append((hidden - mean) / std)
            stats["token"] = torch.cat((mean, std.log()), dim=-1)
        if self.use_channel:
            mean = hidden.mean(1, keepdim=True)
            std = ((hidden - mean).square().mean(1, keepdim=True) + EPS).sqrt()
            views.append((hidden - mean) / std)
            stats["channel"] = torch.cat((mean, std.log()), dim=-1)
        return self._result(self.fusion(torch.cat(views, dim=-1)), stats,
                            {"active_views": self.views})


class LearnedStatistics(StatisticalDecomposer):
    method = "t4_learned"

    def __init__(self, width, groups):
        super().__init__()
        self.groups = int(groups)
        self.net = nn.Sequential(nn.LayerNorm(2 * groups), nn.Linear(2 * groups, max(16, 4 * groups)),
                                 nn.GELU(), nn.Linear(max(16, 4 * groups), 2 * groups))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, hidden):
        _, raw, _, _ = grouped_statistics(hidden, self.groups)
        delta = self.net(raw)
        learned_mean = raw[..., :self.groups] + delta[..., :self.groups]
        learned_log_std = (raw[..., self.groups:] + delta[..., self.groups:]).clamp(-8, 8)
        mean = learned_mean.unsqueeze(-1)
        std = learned_log_std.exp().unsqueeze(-1)
        grouped = hidden.reshape(*hidden.shape[:-1], self.groups, -1)
        normalized = (grouped - mean) / std
        reconstructed = normalized * std + mean
        stats = torch.cat((learned_mean, learned_log_std), dim=-1)
        return self._result(normalized.reshape_as(hidden), stats, {
            "reconstruction_error": (reconstructed - grouped).abs().mean(),
            "coefficient_change_abs": delta.abs().mean(),
            "true_stat_distance": (stats - raw).abs().mean(),
        })


class CovarianceFactorization(StatisticalDecomposer):
    method = "c1_covariance"

    def __init__(self, width, groups):
        super().__init__()
        self.groups = int(groups)

    def forward(self, hidden):
        whitened, stats, diagnostics = covariance_statistics(hidden, self.groups)
        return self._result(whitened, stats, diagnostics)


class DualViewFactorization(StatisticalDecomposer):
    method = "c3_dual"

    def __init__(self, width, groups):
        super().__init__()
        self.groups = int(groups)
        self.fusion = nn.Linear(2 * width, width)
        with torch.no_grad():
            self.fusion.weight.zero_()
            self.fusion.bias.zero_()
            self.fusion.weight[:, :width] = torch.eye(width)

    def forward(self, hidden):
        group, group_stats, _, _ = grouped_statistics(hidden, self.groups)
        whitened, covariance_stats, diagnostics = covariance_statistics(hidden, self.groups)
        inv = self.fusion(torch.cat((group, whitened), dim=-1))
        diagnostics = {"whitening_internal_reconstruction_error": diagnostics["reconstruction_error"],
                       **{key: value for key, value in diagnostics.items() if key != "reconstruction_error"}}
        return self._result(inv, {"group": group_stats, "covariance": covariance_stats}, diagnostics)


class MixedNormalization(StatisticalDecomposer):
    method = "c8_mixed"

    def __init__(self, width, groups):
        super().__init__()
        if width % 2 or (width // 2) % groups:
            raise ValueError("D/2 must be divisible by groups")
        self.groups = int(groups)
        self.half = width // 2
        self.batch_norm = nn.BatchNorm1d(self.half)

    def forward(self, hidden):
        first, second = hidden[..., :self.half], hidden[..., self.half:]
        norm_first, stats_first, _, _ = grouped_statistics(first, self.groups)
        flat = second.reshape(-1, self.half)
        norm_second = self.batch_norm(flat).reshape_as(second)
        _, stats_second, _, _ = grouped_statistics(second, self.groups)
        return self._result(torch.cat((norm_first, norm_second), dim=-1),
                            {"instance": stats_first, "batch_branch_instance_stats": stats_second},
                            {"bn_running_var_mean": self.batch_norm.running_var.mean()})


class LayerNormBaseline(StatisticalDecomposer):
    method = "layernorm"

    def __init__(self, width, groups):
        super().__init__()
        self.norm = nn.LayerNorm(width)

    def forward(self, hidden):
        _, stats, _, _ = grouped_statistics(hidden, 1)
        return self._result(self.norm(hidden), stats)


class LinearBaseline(StatisticalDecomposer):
    method = "linear"

    def __init__(self, width, groups):
        super().__init__()
        self.proj = nn.Linear(width, width)
        nn.init.eye_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden):
        return self._result(self.proj(hidden), None)


class MLPBaseline(StatisticalDecomposer):
    method = "mlp"

    def __init__(self, width, groups):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, hidden):
        return self._result(hidden + self.net(hidden), None)


METHODS = {
    "t1_group": GroupFactorization,
    "t2_orthogonal": OrthogonalFactorization,
    "t0_stnorm": STNormLatent,
    "t4_learned": LearnedStatistics,
    "c1_covariance": CovarianceFactorization,
    "c3_dual": DualViewFactorization,
    "c8_mixed": MixedNormalization,
    "layernorm": LayerNormBaseline,
    "groupnorm": GroupFactorization,
    "linear": LinearBaseline,
    "mlp": MLPBaseline,
}


def make_statistical_decomposer(method, width, groups, channels, patches):
    if method not in METHODS:
        raise ValueError(f"Unknown statistical method {method}")
    if method == "t0_stnorm":
        return STNormLatent(width, groups, channels, patches)
    return METHODS[method](width, groups)
