"""TRAIN-fitted, environment-conditioned predictive mapping decomposition.

The stored spectral factor represents the *exact* operator
``B**.5 @ pinv(A+B) @ B**.5`` on the joint column space of the environment
forecast mappings.  No rank, ridge, or stability trade-off is tuned here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.linalg
import torch
from torch import nn


def flatten_forecast_order(hidden: torch.Tensor) -> torch.Tensor:
    """Match ``head_linear(Flatten(hidden.permute(0, 1, 3, 2)))``."""
    if hidden.ndim != 4:
        raise ValueError("hidden must have shape [B,C,P,D]")
    return hidden.permute(0, 1, 3, 2).reshape(*hidden.shape[:2], -1)


def restore_forecast_order(flat: torch.Tensor, patches: int, width: int) -> torch.Tensor:
    return flat.reshape(*flat.shape[:2], width, patches).permute(0, 1, 3, 2)


def _spectral_cutoff(values: np.ndarray, dimension: int) -> float:
    return float(np.finfo(np.float64).eps * max(1, dimension) *
                 max(float(np.max(np.abs(values), initial=0.0)), 1e-300))


def _psd_summary(matrix: np.ndarray) -> dict[str, float]:
    eigenvalues = scipy.linalg.eigvalsh(matrix, check_finite=False)
    positive = eigenvalues[eigenvalues > _spectral_cutoff(eigenvalues, len(eigenvalues))]
    trace = float(np.trace(matrix))
    frobenius = float(np.linalg.norm(matrix, ord="fro"))
    return {
        "trace": trace,
        "frobenius": frobenius,
        "effective_rank": float(trace * trace / max(frobenius * frobenius, 1e-300)),
        "numerical_rank": int(len(positive)),
        "condition": float(positive.max() / positive.min()) if len(positive) else math.inf,
    }


def mapping_operator(mappings: np.ndarray, masses: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return an exact positive spectral factor ``(Q, eigenvalues)`` for G.

    All columns of A and B lie in span([W_1,...,W_K]); reducing to this
    span avoids a prohibitively large F x F eigendecomposition for PatchTST.
    The returned Q has orthonormal columns and G = Q diag(eigenvalues) Q.T.
    """
    mappings = np.asarray(mappings, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    if mappings.ndim != 3 or masses.shape != (mappings.shape[0],):
        raise ValueError("mappings must be [K,F,T] and masses [K]")
    if not np.isfinite(mappings).all() or not np.isfinite(masses).all() or masses.sum() <= 0:
        raise ValueError("invalid environment mapping statistics")
    pi = masses / masses.sum()
    common = np.einsum("k,kft->ft", pi, mappings)
    differences = mappings - common[None]
    f = mappings.shape[1]
    numerical_mapping_zero = (
        np.finfo(np.float64).eps * max(mappings.shape)
        * float(np.max(np.abs(mappings), initial=0.0))
    )
    if float(np.max(np.abs(differences), initial=0.0)) <= numerical_mapping_zero:
        return np.zeros((f, 0)), np.zeros(0), {
            "A": _psd_summary(common.T @ common), "B": {"trace": 0.0, "frobenius": 0.0,
            "effective_rank": 0.0, "numerical_rank": 0, "condition": math.inf},
            "G_trace": 0.0, "G_rank": 0, "G_eigenvalues": [],
        }
    # S=A+B=sum_e pi_e W_e W_e.T.  Economy QR changes coordinates only.
    columns = np.concatenate([math.sqrt(pi[e]) * mappings[e] for e in range(len(pi))], axis=1)
    q, _ = scipy.linalg.qr(columns, mode="economic", check_finite=False)
    common_r = q.T @ common
    variation_r = np.concatenate(
        [math.sqrt(pi[e]) * (q.T @ differences[e]) for e in range(len(pi))], axis=1
    )
    a = common_r @ common_r.T
    b = variation_r @ variation_r.T
    a = (a + a.T) * 0.5
    b = (b + b.T) * 0.5
    s = a + b
    b_values, b_vectors = scipy.linalg.eigh(b, check_finite=False)
    b_values = np.maximum(b_values, 0.0)
    if b_values.max(initial=0.0) == 0:
        return np.zeros((f, 0)), np.zeros(0), {
            "A": _psd_summary(a), "B": _psd_summary(b),
            "G_trace": 0.0, "G_rank": 0, "G_eigenvalues": [],
        }
    sqrt_b = (b_vectors * np.sqrt(b_values)) @ b_vectors.T
    s_values, s_vectors = scipy.linalg.eigh(s, check_finite=False)
    cutoff = _spectral_cutoff(s_values, len(s_values))
    inverse = np.where(s_values > cutoff, 1.0 / np.maximum(s_values, cutoff), 0.0)
    s_pinv = (s_vectors * inverse) @ s_vectors.T
    g = sqrt_b @ s_pinv @ sqrt_b
    g = (g + g.T) * 0.5
    values, vectors = scipy.linalg.eigh(g, check_finite=False)
    violation = max(float(-values.min(initial=0)), float(values.max(initial=0) - 1), 0.0)
    if violation > 1e-5:
        raise FloatingPointError(f"EPMD operator spectrum violates [0,1] by {violation:g}")
    values = np.clip(values, 0.0, 1.0)
    keep = values > _spectral_cutoff(values, len(values))
    basis = q @ vectors[:, keep]
    eigenvalues = values[keep]
    diagnostics = {
        "A": _psd_summary(a), "B": _psd_summary(b),
        "G_trace": float(values.sum()), "G_rank": int(keep.sum()),
        "G_eigenvalues": np.quantile(values, [0, .1, .5, .9, 1]).tolist(),
        "G_spectral_violation": violation,
        "joint_mapping_dimension": int(q.shape[1]),
    }
    return basis, eigenvalues, diagnostics


@dataclass
class _WeightedStatistics:
    mass: float
    h: np.ndarray
    y: np.ndarray
    hh: np.ndarray
    hy: np.ndarray
    weight_square: float
    sample_mass: float
    sample_weight_square: float


class EnvironmentMappingEstimator:
    """Accumulate all TRAIN samples/channels without retaining raw windows."""

    def __init__(self, features: int, horizon: int, environments: int):
        self.features, self.horizon = int(features), int(horizon)
        self.statistics = [
            _WeightedStatistics(0.0, np.zeros(features), np.zeros(horizon),
                                np.zeros((features, features)), np.zeros((features, horizon)),
                                0.0, 0.0, 0.0)
            for _ in range(environments)
        ]
        self.total_h = np.zeros(features)
        self.total_count = 0

    def add(self, hidden: torch.Tensor, target_normalized: torch.Tensor, assignment: torch.Tensor):
        if target_normalized.shape != (hidden.shape[0], hidden.shape[1], self.horizon):
            raise ValueError("target must be normalized [B,C,T]")
        flat = flatten_forecast_order(hidden).detach().to(torch.float64).cpu().numpy()
        x = flat.reshape(-1, self.features)
        y = target_normalized.detach().to(torch.float64).cpu().numpy().reshape(-1, self.horizon)
        sample_q = np.asarray(assignment.detach().cpu(), dtype=np.float64)
        q = np.repeat(sample_q, hidden.shape[1], axis=0)
        if q.shape != (len(x), len(self.statistics)):
            raise ValueError("assignment must have shape [B,K]")
        self.total_h += x.sum(axis=0)
        self.total_count += len(x)
        for env, stats in enumerate(self.statistics):
            weight = q[:, env]
            weighted_x = x * weight[:, None]
            stats.mass += float(weight.sum())
            stats.weight_square += float(np.square(weight).sum())
            stats.sample_mass += float(sample_q[:, env].sum())
            stats.sample_weight_square += float(np.square(sample_q[:, env]).sum())
            stats.h += weighted_x.sum(axis=0)
            stats.y += weight @ y
            stats.hh += x.T @ weighted_x
            stats.hy += weighted_x.T @ y

    def finish(self, reference_weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        w0 = np.asarray(reference_weight, dtype=np.float64)
        if w0.shape != (self.features, self.horizon):
            raise ValueError("A0 head weight must be [F,T]")
        if self.total_count <= 0:
            raise ValueError("empty TRAIN statistics")
        mappings = []
        masses = []
        fits = []
        for stats in self.statistics:
            if stats.mass <= 0:
                raise ValueError("environment has zero TRAIN mass")
            mean_h, mean_y = stats.h / stats.mass, stats.y / stats.mass
            hh = stats.hh - np.outer(stats.h, stats.h) / stats.mass
            hy = stats.hy - np.outer(stats.h, stats.y) / stats.mass
            hh = (hh + hh.T) * 0.5
            values, vectors = scipy.linalg.eigh(hh, check_finite=False)
            cutoff = _spectral_cutoff(values, max(self.total_count, self.features))
            valid = values > cutoff
            residual = hy - hh @ w0
            # Minimum Euclidean deviation from frozen A0 in unidentified directions.
            change = vectors[:, valid] @ ((vectors[:, valid].T @ residual) / values[valid, None])
            mapping = w0 + change
            mappings.append(mapping)
            masses.append(stats.mass)
            fits.append({
                "observation_mass": stats.mass,
                "mass": stats.sample_mass,
                "effective_sample_mass": stats.sample_mass ** 2 / max(stats.sample_weight_square, 1e-300),
                "rank": int(valid.sum()),
                "condition": float(values[valid].max() / values[valid].min()) if valid.any() else math.inf,
                "bias_norm": float(np.linalg.norm(mean_y - mean_h @ mapping)),
            })
        basis, eigenvalues, spectral = mapping_operator(np.stack(mappings), np.asarray(masses))
        return basis, eigenvalues, self.total_h / self.total_count, np.asarray(masses), {
            "environments": fits, "spectral": spectral,
        }


class EnvironmentMappingDecomposer(nn.Module):
    def __init__(self, features: int, *, init_alpha_logit: float = -4.0, alpha_mode: str = "learned"):
        super().__init__()
        if alpha_mode not in ("learned", "fixed_one"):
            raise ValueError("alpha_mode must be learned or fixed_one")
        self.features = int(features)
        self.alpha_mode = alpha_mode
        self.alpha_logit = nn.Parameter(torch.tensor(float(init_alpha_logit)),
                                        requires_grad=alpha_mode == "learned")
        self.register_buffer("mean_h", torch.zeros(features))
        self.register_buffer("basis", torch.empty(features, 0))
        self.register_buffer("eigenvalues", torch.empty(0))
        self.last_fit_diagnostics: dict = {}

    @property
    def alpha(self) -> torch.Tensor:
        return self.alpha_logit.sigmoid() if self.alpha_mode == "learned" else self.alpha_logit.new_ones(())

    @torch.no_grad()
    def update_mapping(self, basis: np.ndarray, eigenvalues: np.ndarray, mean_h: np.ndarray, diagnostics: dict):
        device, dtype = self.mean_h.device, self.mean_h.dtype
        if basis.shape != (self.features, len(eigenvalues)):
            raise ValueError("spectral factor shape mismatch")
        self.mean_h = torch.as_tensor(mean_h, device=device, dtype=dtype)
        self.basis = torch.as_tensor(basis, device=device, dtype=dtype)
        self.eigenvalues = torch.as_tensor(eigenvalues, device=device, dtype=dtype)
        self.last_fit_diagnostics = diagnostics

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # The stage-fitted spectral rank is data-dependent, so resize the two
        # buffers before PyTorch validates checkpoint tensor shapes.
        for name in ("basis", "eigenvalues"):
            key = prefix + name
            if key in state_dict:
                incoming = state_dict[key]
                setattr(self, name, torch.empty(incoming.shape, device=self.mean_h.device,
                                                 dtype=self.mean_h.dtype))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, hidden: torch.Tensor):
        patches, width = hidden.shape[-2:]
        flat = flatten_forecast_order(hidden)
        if flat.shape[-1] != self.features:
            raise ValueError("hidden feature dimension does not match EPMD mapping")
        if self.basis.shape[1]:
            projection = (flat - self.mean_h) @ self.basis
            variant = (projection * self.eigenvalues) @ self.basis.T
            variant = self.alpha * variant
        else:
            variant = torch.zeros_like(flat)
        invariant = flat - variant
        return (restore_forecast_order(invariant, patches, width),
                restore_forecast_order(variant, patches, width),
                torch.zeros_like(hidden))
