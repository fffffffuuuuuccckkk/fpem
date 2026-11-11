"""OCD, TRAIN-supervised PSD, and PSD-initialized OCD for FPEM.

No environment labels or future targets are accepted by forward(). The
supervised spectrum is fitted once from TRAIN sufficient statistics only.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _orthonormal(raw):
    # Do not perform QR in AMP half precision. QR remains differentiable.
    with torch.amp.autocast(device_type=raw.device.type, enabled=False):
        q, r = torch.linalg.qr(raw.float(), mode="reduced")
        sign = torch.where(torch.diagonal(r) >= 0, 1.0, -1.0)
        return q * sign.unsqueeze(0)


def _participation(energy, dim=0):
    count = energy.shape[dim]
    return energy.sum(dim).square() / (count * energy.square().sum(dim) + 1e-12)


def _output(hidden, inv, var, aux, extra):
    reconstruction = inv + var
    diagnostics = {
        "reconstruction_mae": (reconstruction-hidden).abs().mean().detach(),
        "inv_energy_ratio": (inv.float().square().mean() /
                             hidden.float().square().mean().clamp_min(1e-12)).detach(),
        "var_energy_ratio": (var.float().square().mean() /
                             hidden.float().square().mean().clamp_min(1e-12)).detach(),
        "vector_orthogonality": (inv.float()*var.float()).sum(-1).abs().mean().detach(),
        "inv_participation": _participation(inv.float().square().mean((1, 2, 3))).detach(),
    }
    diagnostics.update(extra)
    return {"z_inv_tokens": inv, "z_var_tokens": var,
            "z_var_stats": {"pooled": var.mean((1, 2))},
            "z_rare_tokens": None, "reconstruction": reconstruction,
            "aux_losses": aux, "diagnostics": diagnostics}


class OrthogonalComplementaryDecomposer(nn.Module):
    """Learn an exact rank-r variation projector, never a full D-by-D matrix."""
    def __init__(self, width, variation_rank, *, coverage=False,
                 lambda_shared=0.01, lambda_partial=0.01, lambda_variance=0.001,
                 shared_target=0.75, partial_low=0.05, partial_high=0.8,
                 std_fraction=0.05):
        super().__init__()
        if not 1 <= variation_rank < width:
            raise ValueError("variation rank must be in [1, D-1]")
        self.width, self.variation_rank = int(width), int(variation_rank)
        self.raw_basis = nn.Parameter(torch.randn(width, variation_rank) / math.sqrt(width))
        self.coverage = bool(coverage)
        self.lambda_shared = float(lambda_shared)
        self.lambda_partial = float(lambda_partial)
        self.lambda_variance = float(lambda_variance)
        self.shared_target, self.partial_low = float(shared_target), float(partial_low)
        self.partial_high, self.std_fraction = float(partial_high), float(std_fraction)

    def basis(self):
        return _orthonormal(self.raw_basis)

    @torch.no_grad()
    def initialize_variation_basis(self, basis):
        if basis.shape != self.raw_basis.shape:
            raise ValueError("PSD complement shape does not match OCD rank")
        gram = basis.T @ basis
        if not torch.allclose(gram, torch.eye(self.variation_rank, device=basis.device), atol=1e-4):
            raise ValueError("PSD complement must be orthonormal")
        self.raw_basis.copy_(basis.to(self.raw_basis))

    def forward(self, hidden, *, sample_indices=None, train_env_q=None,
                return_diagnostics=False):
        if hidden.ndim != 4 or hidden.shape[-1] != self.width:
            raise ValueError("hidden must be [B,C,P,D] with expected D")
        basis = self.basis()
        # FP32 projection protects complementarity under mixed precision.
        h = hidden.float()
        coefficients = h @ basis
        var = (coefficients @ basis.T).to(hidden.dtype)
        inv = hidden - var
        participation = _participation(coefficients.square().mean((1, 2)))
        aux = {}
        if self.training and self.coverage:
            inv_part = _participation(inv.float().square().mean((1, 2, 3)))
            shared = F.relu(self.shared_target-inv_part).square()
            partial = (F.relu(self.partial_low-participation).square() +
                       F.relu(participation-self.partial_high).square()).mean()
            variation_std = coefficients.flatten(0, 2).std(dim=0, unbiased=False)
            std_target = self.std_fraction*h.detach().std(unbiased=False)
            variance = F.relu(std_target-variation_std).square().mean()
            aux = {"shared_coverage": self.lambda_shared*shared,
                   "partial_coverage": self.lambda_partial*partial,
                   "variation_variance": self.lambda_variance*variance}
        extra = {"basis_orthogonal_error": (basis.T@basis - torch.eye(
                     self.variation_rank, device=basis.device)).norm().detach(),
                 "var_participation_mean": participation.mean().detach(),
                 "var_participation_min": participation.min().detach(),
                 "variation_rank": hidden.new_tensor(float(self.variation_rank))}
        return _output(hidden, inv, var, aux, extra)


class StreamingPredictiveSpectrum:
    """FP64 pooled-H/future-Y moments; each row is one TRAIN sample-variable."""
    def __init__(self, width, horizon):
        self.width, self.horizon = int(width), int(horizon)
        d, t = self.width, self.horizon
        self.count = 0
        self.sum_x = torch.zeros(d, dtype=torch.float64)
        self.sum_y = torch.zeros(t, dtype=torch.float64)
        self.xtx = torch.zeros(d, d, dtype=torch.float64)
        self.xty = torch.zeros(d, t, dtype=torch.float64)
        self.y2 = torch.zeros(t, dtype=torch.float64)
        self.variable_count = None

    @torch.no_grad()
    def add(self, hidden, future, *, source_split):
        if source_split != "train":
            raise ValueError("PSD statistics may use TRAIN only")
        if hidden.ndim != 4 or future.ndim != 3:
            raise ValueError("hidden [B,C,P,D], future [B,T,C] required")
        b, c, _, d = hidden.shape
        if future.shape != (b, self.horizon, c) or d != self.width:
            raise ValueError("Backbone variable axis and future target do not align")
        if self.variable_count is not None and self.variable_count != c:
            raise ValueError("inconsistent variable count")
        self.variable_count = c
        x = hidden.detach().mean(2).reshape(-1, d).to(device="cpu", dtype=torch.float64)
        y = future.detach().permute(0, 2, 1).reshape(-1, self.horizon).to(
            device="cpu", dtype=torch.float64)
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise FloatingPointError("nonfinite TRAIN spectral inputs")
        self.count += len(x)
        self.sum_x += x.sum(0)
        self.sum_y += y.sum(0)
        self.xtx += x.T@x
        self.xty += x.T@y
        self.y2 += y.square().sum(0)

    @torch.no_grad()
    def fit(self, shared_rank, ridge_fraction=1e-4):
        if self.count < 2 or not 1 <= shared_rank < self.width:
            raise ValueError("insufficient TRAIN rows or invalid rank")
        mean_x, mean_y = self.sum_x/self.count, self.sum_y/self.count
        cxx = self.xtx/self.count - torch.outer(mean_x, mean_x)
        cxy = self.xty/self.count - torch.outer(mean_x, mean_y)
        y_var = (self.y2/self.count-mean_y.square()).clamp_min(1e-12)
        y_std = y_var.sqrt()
        cxy = cxy/y_std.unsqueeze(0)
        cxx = (cxx+cxx.T)*0.5
        ridge = float(ridge_fraction*cxx.trace()/self.width)
        ridge = max(ridge, 1e-10)
        vals, vecs = torch.linalg.eigh(cxx+ridge*torch.eye(self.width, dtype=torch.float64))
        cutoff = max(float(vals.max())*1e-10, 1e-12)
        inverse_sqrt = (vecs*vals.clamp_min(cutoff).rsqrt())@vecs.T
        whitened = inverse_sqrt@cxy
        spectral = whitened@whitened.T
        spectral = (spectral+spectral.T)*0.5
        eigenvalues, eigenvectors = torch.linalg.eigh(spectral)
        order = torch.argsort(eigenvalues, descending=True)
        spectrum = eigenvalues[order].clamp_min(0)
        numerical_rank = int((spectrum > max(float(spectrum.max())*1e-8, 1e-12)).sum())
        signal_rank = min(shared_rank, numerical_rank)
        if signal_rank:
            signal_raw = inverse_sqrt@eigenvectors[:, order[:signal_rank]]
            q_signal, _ = torch.linalg.qr(signal_raw, mode="reduced")
        else:
            q_signal = torch.zeros(self.width, 0, dtype=torch.float64)
        if signal_rank < shared_rank:
            # The nearly-zero eigenspace is degenerate. Select its mapped
            # directions by TRAIN H variance, rather than arbitrary eigh
            # ordering, while retaining the exact requested projection rank.
            null_raw = inverse_sqrt@eigenvectors[:, order[signal_rank:]]
            null_raw = null_raw-q_signal@(q_signal.T@null_raw)
            q_null, _ = torch.linalg.qr(null_raw, mode="reduced")
            null_cov = q_null.T@cxx@q_null
            null_values, null_vectors = torch.linalg.eigh((null_cov+null_cov.T)*0.5)
            fill = shared_rank-signal_rank
            q_fill = q_null@null_vectors[:, torch.argsort(null_values, descending=True)[:fill]]
            q, _ = torch.linalg.qr(torch.cat((q_signal, q_fill), dim=1), mode="reduced")
        else:
            q = q_signal
        q = q.float()
        total = float(spectrum.sum())
        info = {"train_rows": self.count, "train_variables": self.variable_count,
                "pooled_patch_representation": True, "target_horizon": self.horizon,
                "ridge": ridge, "cxx_condition": float(vals.max()/vals.min().clamp_min(cutoff)),
                "numerical_predictive_rank": numerical_rank,
                "train_covariance_null_tiebreak_count": shared_rank-signal_rank,
                "selected_shared_rank": shared_rank,
                "selected_spectral_energy_fraction": float(spectrum[:shared_rank].sum())/max(total, 1e-12),
                "spectrum": spectrum.tolist(), "target_std": y_std.tolist()}
        return q, info


def orthogonal_complement(shared_basis):
    """Return the Euclidean nullspace of Q.T with exactly D-r columns."""
    with torch.no_grad():
        _, _, vh = torch.linalg.svd(shared_basis.double().T, full_matrices=True)
        complement = vh.T[:, shared_basis.shape[1]:].float()
        return complement


class PredictiveSpectralDecomposer(nn.Module):
    def __init__(self, width, shared_rank):
        super().__init__()
        if not 1 <= shared_rank < width:
            raise ValueError("shared rank must lie in [1, D-1]")
        self.width, self.shared_rank = int(width), int(shared_rank)
        self.register_buffer("shared_basis", torch.zeros(width, shared_rank))
        self.register_buffer("fitted", torch.tensor(False))
        self.fit_info = {}

    @torch.no_grad()
    def fit(self, q, info, *, source_split):
        if source_split != "train":
            raise ValueError("PSD fit is TRAIN-only")
        if q.shape != self.shared_basis.shape:
            raise ValueError("predictive spectral basis shape mismatch")
        self.shared_basis.copy_(q.to(self.shared_basis))
        self.fitted.fill_(True)
        self.fit_info = info

    def forward(self, hidden, *, sample_indices=None, train_env_q=None,
                return_diagnostics=False):
        if not bool(self.fitted):
            raise RuntimeError("TRAIN predictive spectrum not fitted")
        if hidden.ndim != 4 or hidden.shape[-1] != self.width:
            raise ValueError("hidden must be [B,C,P,D] with expected D")
        q = self.shared_basis.float()
        inv = ((hidden.float()@q)@q.T).to(hidden.dtype)
        var = hidden-inv
        extra = {"basis_orthogonal_error": (q.T@q-torch.eye(
                     self.shared_rank, device=q.device)).norm().detach(),
                 "variation_rank": hidden.new_tensor(float(self.width-self.shared_rank))}
        return _output(hidden, inv, var, {}, extra)
