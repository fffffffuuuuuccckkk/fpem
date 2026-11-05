"""Eight lightweight H-space shared/partial factorization experiments.

S1/S2/S3/S7/S8 are fitted from TRAIN hidden summaries and stored as buffers.
S4/S5/S6 are differentiable encoders. None reads future targets in forward.
The named literature inspired the designs; these are FPEM adaptations, not
line-by-line implementations of COBE/AJIVE/RPCA/sJIVE/DSN.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .covariance_utils import soft_projectors, spectral_energy_statistics, symmetric


def _diagnostics(hidden, inv, var, rare=None):
    reconstruct = inv + var + (rare if rare is not None else 0)
    return {
        "zinv_rms": inv.square().mean().sqrt().detach(),
        "zvar_rms": var.square().mean().sqrt().detach(),
        "zrare_rms": (rare.square().mean().sqrt().detach() if rare is not None
                       else hidden.new_zeros(())),
        "reconstruction_mae": (hidden - reconstruct).abs().mean().detach(),
        "variation_energy_ratio": (var.square().sum() /
                                   (hidden.square().sum() + 1e-8)).detach(),
    }


def _output(hidden, inv, var, *, rare=None, reconstruction=None,
            aux_losses=None, extra=None):
    diagnostics = _diagnostics(hidden, inv, var, rare)
    if extra:
        diagnostics.update(extra)
    return {
        "z_inv_tokens": inv, "z_var_tokens": var,
        "z_var_stats": {"pooled": var.mean((1, 2))},
        "z_rare_tokens": rare,
        "reconstruction": reconstruction if reconstruction is not None else
                          inv + var + (rare if rare is not None else 0),
        "aux_losses": aux_losses or {}, "diagnostics": diagnostics,
    }


class SharedFactorDecomposer(nn.Module):
    method = "base"
    needs_fit = False

    def fit(self, statistics, **kwargs):
        return {}

    def forward(self, hidden, *, sample_indices=None, train_env_q=None,
                return_diagnostics=False):
        raise NotImplementedError


class OperatorDecomposer(SharedFactorDecomposer):
    needs_fit = True

    def __init__(self, width):
        super().__init__()
        eye = torch.eye(width)
        self.register_buffer("inv_operator", eye.clone())
        self.register_buffer("var_operator", torch.zeros_like(eye))
        self.register_buffer("rare_operator", torch.zeros_like(eye))
        self.register_buffer("fitted", torch.tensor(False))
        self.fit_info = {}

    def _set(self, inv, var, rare=None, **info):
        for name, matrix in (("inv_operator", inv), ("var_operator", var),
                             ("rare_operator", rare if rare is not None else torch.zeros_like(inv))):
            getattr(self, name).copy_(symmetric(matrix).to(getattr(self, name)))
        self.fitted.fill_(True)
        self.fit_info = {k: float(v) if isinstance(v, (int, float)) else v for k, v in info.items()}
        return self.fit_info

    def forward(self, hidden, *, sample_indices=None, train_env_q=None,
                return_diagnostics=False):
        inv = hidden @ self.inv_operator
        var = hidden @ self.var_operator
        rare = hidden @ self.rare_operator if self.rare_operator.abs().max() > 0 else None
        extra = {"operator_reconstruction_error":
                 (self.inv_operator + self.var_operator + self.rare_operator -
                  torch.eye(hidden.shape[-1], device=hidden.device)).abs().mean().detach()}
        return _output(hidden, inv, var, rare=rare, extra=extra)


class CommonSubspace(OperatorDecomposer):
    method = "common_subspace"

    def fit(self, statistics, **kwargs):
        projectors, ridge = soft_projectors(statistics)
        k = len(projectors)
        if k < 2:
            common = projectors[0]
            variation = torch.zeros_like(common)
        else:
            common = sum(projectors[i] @ projectors[j] @ projectors[i]
                         for i in range(k) for j in range(k) if i != j) / (k * (k - 1))
            weight = statistics.environment_mass / statistics.environment_mass.sum()
            mean = torch.einsum("k,kde->de", weight, projectors)
            variation = torch.einsum("k,kde,kef->df", weight,
                                     projectors - mean, projectors - mean)
        return self._set(common, symmetric(variation), ridge=ridge, environment_count=k)


class SharedSpectral(OperatorDecomposer):
    method = "shared_spectral"

    def fit(self, statistics, **kwargs):
        projectors, ridge = soft_projectors(statistics)
        weights = statistics.environment_mass / statistics.environment_mass.sum()
        mean = torch.einsum("k,kde->de", weights, projectors)
        diff = projectors - mean
        variation = torch.einsum("k,kde,kef->df", weights, diff, diff)
        _, basis = torch.linalg.eigh(symmetric(mean + variation))
        shared = torch.einsum("di,de,ei->i", basis, mean, basis).clamp_min(0)
        variable = torch.einsum("di,de,ei->i", basis, variation, basis).clamp_min(0)
        mixture = (shared / (shared + variable + 1e-8)).clamp(0, 1)
        inv = (basis * mixture) @ basis.T
        return self._set(inv, torch.eye(inv.shape[0], dtype=inv.dtype) - inv,
                         ridge=ridge, shared_weight_mean=float(mixture.mean()),
                         low_energy_direction_fraction=float((shared + variable < 1e-4).double().mean()))


class RobustLowrankSparse(OperatorDecomposer):
    method = "robust_lowrank_sparse"

    def __init__(self, width):
        super().__init__(width)
        self.register_buffer("prox_threshold", torch.tensor(0.0))
        self.register_buffer("selected_rank", torch.tensor(0, dtype=torch.long))

    def fit(self, statistics, **kwargs):
        matrix = statistics.sample_rows - statistics.sample_rows.mean(0)
        sparse = torch.zeros_like(matrix)
        rank = max(1, min(matrix.shape[1] // 2, matrix.shape[0] - 1))
        # Bounded 4096 x D TRAIN matrix; three alternating spectral/row-prox steps.
        for _ in range(3):
            _, singular, vh = torch.linalg.svd(matrix - sparse, full_matrices=False)
            energy = singular.square().cumsum(0) / singular.square().sum().clamp_min(1e-8)
            chosen = max(1, min(rank, int(torch.searchsorted(energy, 0.9).item()) + 1))
            lowrank = (matrix - sparse) @ vh[:chosen].T @ vh[:chosen]
            residual = matrix - lowrank
            threshold = residual.norm(dim=1).median().clamp_min(1e-8) * 0.5
            sparse = residual * (1 - threshold / residual.norm(dim=1, keepdim=True).clamp_min(1e-8)).clamp_min(0)
        basis = vh[:chosen].T
        projection = basis @ basis.T
        self.prox_threshold.copy_(threshold.to(self.prox_threshold))
        self.selected_rank.fill_(chosen)
        return self._set(projection, torch.zeros_like(projection),
                         effective_rank=chosen, prox_threshold=float(threshold),
                         train_sparse_row_fraction=float((sparse.norm(dim=1) > 0).double().mean()))

    def forward(self, hidden, *, sample_indices=None, train_env_q=None,
                return_diagnostics=False):
        inv = hidden @ self.inv_operator
        residual = hidden - inv
        # Sparse group prox on each hidden token; never re-fit on TEST population.
        norms = residual.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        var = residual * (1 - self.prox_threshold / norms).clamp_min(0)
        rare = residual - var
        return _output(hidden, inv, var, rare=rare,
                       extra={"sparse_token_fraction": (var.norm(dim=-1) > 0).float().mean().detach(),
                              "effective_rank": hidden.new_tensor(float(self.selected_rank))})


def _cross_covariance(inv, var):
    left, right = inv.flatten(0, 2), var.flatten(0, 2)
    left = left - left.mean(0)
    right = right - right.mean(0)
    return (left.T @ right).square().mean() / max(left.shape[0], 1) ** 2


class SupervisedJointPrivate(SharedFactorDecomposer):
    method = "supervised_joint_private"

    def __init__(self, width, lambda_rec=0.01, lambda_decorr=0.001):
        super().__init__()
        self.shared = nn.Linear(width, width, bias=False)
        self.private = nn.Linear(width, width, bias=False)
        nn.init.eye_(self.shared.weight)
        nn.init.normal_(self.private.weight, std=1e-3)
        self.lambda_rec, self.lambda_decorr = float(lambda_rec), float(lambda_decorr)

    def forward(self, hidden, *, sample_indices=None, train_env_q=None,
                return_diagnostics=False):
        inv, var = self.shared(hidden), self.private(hidden)
        recon = F.linear(inv, self.shared.weight.T) + F.linear(var, self.private.weight.T)
        aux = {"reconstruction": self.lambda_rec * (recon - hidden.detach()).square().mean(),
               "decorrelation": self.lambda_decorr * _cross_covariance(inv, var)}
        return _output(hidden, inv, var, reconstruction=recon, aux_losses=aux,
                       extra={"reconstruction_mae": (recon - hidden).abs().mean().detach()})


class SharedPrivateLite(SharedFactorDecomposer):
    method = "shared_private_lite"

    def __init__(self, width, bottleneck=32, lambda_decorr=0.001):
        super().__init__()
        self.private = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, bottleneck),
                                     nn.GELU(), nn.Linear(bottleneck, width))
        nn.init.normal_(self.private[-1].weight, std=1e-4)
        nn.init.zeros_(self.private[-1].bias)
        self.lambda_decorr = float(lambda_decorr)

    def forward(self, hidden, *, sample_indices=None, train_env_q=None,
                return_diagnostics=False):
        var = self.private(hidden)
        inv = hidden - var
        aux = {"decorrelation": self.lambda_decorr * _cross_covariance(inv, var)}
        return _output(hidden, inv, var, aux_losses=aux)


class PairwisePartialShared(SharedPrivateLite):
    method = "pairwise_partial_shared"

    def __init__(self, width, bottleneck=32, lambda_decorr=0.001,
                 lambda_pair=0.01, affinity="env_affinity", seq_len=96):
        super().__init__(width, bottleneck, lambda_decorr)
        if affinity not in {"env_affinity", "hybrid_affinity"}:
            raise ValueError(affinity)
        self.lambda_pair, self.affinity = float(lambda_pair), affinity
        self.seq_len = int(seq_len)

    def forward(self, hidden, *, sample_indices=None, train_env_q=None,
                return_diagnostics=False):
        result = super().forward(hidden)
        if not self.training or train_env_q is None or hidden.shape[0] < 3:
            return result
        q = train_env_q.detach().to(hidden)
        pooled = result["z_var_tokens"].mean((1, 2))
        similarity = q @ q.T
        if self.affinity == "hybrid_affinity":
            h = hidden.detach().mean((1, 2))
            dist = torch.cdist(h, h).square()
            similarity = similarity * torch.exp(-dist / dist.median().clamp_min(1e-8))
        valid = ~torch.eye(len(q), dtype=torch.bool, device=q.device)
        if sample_indices is not None:
            ids = torch.as_tensor(sample_indices, device=q.device)
            valid &= (ids[:, None] - ids[None, :]).abs() >= self.seq_len
        weights = similarity * valid
        if weights.sum() > 1e-8:
            distance = torch.cdist(pooled, pooled).square()
            pair = (weights * distance).sum() / weights.sum()
            result["aux_losses"]["partial_shared"] = self.lambda_pair * pair
            # Scale-aware variance floor prevents the all-zero private shortcut.
            target = 0.01 * hidden.detach().mean((1, 2)).var(0, unbiased=False).mean()
            actual = pooled.var(0, unbiased=False).mean()
            result["aux_losses"]["private_variance_floor"] = self.lambda_pair * F.relu(target - actual)
            result["diagnostics"]["valid_pair_fraction"] = valid.float().mean().detach()
        return result


class SamplePrevalence(OperatorDecomposer):
    method = "sample_prevalence"

    def fit(self, statistics, *, model, loader, device, q, **kwargs):
        _, basis = torch.linalg.eigh(symmetric(statistics.covariance))
        prevalence, env_energy, mean_energy = spectral_energy_statistics(model, loader, device, basis, q)
        inv = (basis * prevalence) @ basis.T
        env_cv = env_energy.std(0, unbiased=False) / env_energy.mean(0).clamp_min(1e-8)
        return self._set(inv, torch.eye(inv.shape[0], dtype=inv.dtype) - inv,
                         participation_mean=float(prevalence.mean()),
                         environment_energy_cv_mean=float(env_cv.mean()),
                         direction_energy_mean=float(mean_energy.mean()))


class SubspaceRecurrence(OperatorDecomposer):
    method = "subspace_recurrence"

    def __init__(self, width, polynomial=False):
        super().__init__(width)
        self.polynomial = bool(polynomial)

    def fit(self, statistics, **kwargs):
        projectors, ridge = soft_projectors(statistics)
        weights = statistics.environment_mass / statistics.environment_mass.sum()
        average = torch.einsum("k,kde->de", weights, projectors)
        _, basis = torch.linalg.eigh(symmetric(average))
        responses = torch.einsum("di,kde,ei->ki", basis, projectors, basis).clamp(0, 1)
        arithmetic = torch.einsum("k,ki->i", weights, responses).clamp(0, 1)
        if self.polynomial:
            global_w = arithmetic.square()
            partial_w = 2 * arithmetic * (1 - arithmetic)
            rare_w = (1 - arithmetic).square()
            geometric = torch.zeros_like(arithmetic)
        else:
            geometric = torch.exp(torch.einsum("k,ki->i", weights,
                                               responses.clamp_min(1e-12).log()))
            global_w = torch.minimum(geometric, arithmetic)
            partial_w = (arithmetic - global_w).clamp_min(0)
            rare_w = (1 - arithmetic).clamp_min(0)
        global_op = (basis * global_w) @ basis.T
        partial_op = (basis * partial_w) @ basis.T
        rare_op = (basis * rare_w) @ basis.T
        return self._set(global_op, partial_op, rare_op, ridge=ridge,
                         arithmetic_mean=float(arithmetic.mean()),
                         geometric_mean=float(geometric.mean()),
                         global_weight_mean=float(global_w.mean()),
                         partial_weight_mean=float(partial_w.mean()),
                         rare_weight_mean=float(rare_w.mean()),
                         response_env_spread=float(responses.std(0, unbiased=False).mean()))


class FunctionPreservingWrapper(SharedFactorDecomposer):
    """Optional identity-path control; eta=0 exactly preserves the A0 function."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.eta = nn.Parameter(torch.zeros(()))
        self.method = inner.method
        self.needs_fit = inner.needs_fit

    def fit(self, statistics, **kwargs):
        return self.inner.fit(statistics, **kwargs)

    def forward(self, hidden, *, sample_indices=None, train_env_q=None,
                return_diagnostics=False):
        raw = self.inner(hidden, sample_indices=sample_indices,
                         train_env_q=train_env_q, return_diagnostics=return_diagnostics)
        eta = self.eta
        inv = hidden + eta * (raw["z_inv_tokens"] - hidden)
        var = eta * raw["z_var_tokens"]
        rare = eta * raw["z_rare_tokens"] if raw["z_rare_tokens"] is not None else None
        return _output(hidden, inv, var, rare=rare,
                       aux_losses={key: eta.abs() * val for key, val in raw["aux_losses"].items()},
                       extra={"eta": eta.detach()})


METHODS = {
    "common_subspace": CommonSubspace,
    "shared_spectral": SharedSpectral,
    "robust_lowrank_sparse": RobustLowrankSparse,
    "supervised_joint_private": SupervisedJointPrivate,
    "shared_private_lite": SharedPrivateLite,
    "pairwise_partial_shared": PairwisePartialShared,
    "sample_prevalence": SamplePrevalence,
    "subspace_recurrence": SubspaceRecurrence,
    "subspace_recurrence_poly": SubspaceRecurrence,
}


def make_shared_factor_decomposer(method, width, *, function_preserving_init=False,
                                  seq_len=96, affinity="env_affinity", **kwargs):
    if method not in METHODS:
        raise ValueError(f"unknown shared factorization method: {method}")
    if method == "subspace_recurrence_poly":
        decomposer = SubspaceRecurrence(width, polynomial=True)
    elif method == "pairwise_partial_shared":
        decomposer = PairwisePartialShared(width, seq_len=seq_len, affinity=affinity,
                                           **kwargs)
    else:
        decomposer = METHODS[method](width, **kwargs)
    return FunctionPreservingWrapper(decomposer) if function_preserving_init else decomposer
