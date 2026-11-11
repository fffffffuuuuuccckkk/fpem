"""TRAIN-memory-only dual-consensus retrieval for frozen forecast backbones.

The two local heads are weighted, closed-form ridge solutions, not trainable
neural modules. All targets and predictive-evidence scores are written only
after their source future has become causally observable.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


def entmax15(scores: torch.Tensor, valid: torch.Tensor, steps: int = 32):
    """Exact-support alpha=1.5 entmax by a monotone threshold solve.

    p_i = [max(s_i - tau, 0)/2]^2. An all-invalid row returns zeros.
    """
    if scores.shape != valid.shape:
        raise ValueError("scores/valid shape mismatch")
    safe = scores.float().masked_fill(~valid, -1e30)
    any_valid = valid.any(-1, keepdim=True)
    shifted = safe - safe.max(-1, keepdim=True).values
    shifted = shifted.masked_fill(~valid, -1e30)
    lower = torch.full_like(shifted[..., :1], -2.)
    upper = torch.zeros_like(lower)
    for _ in range(steps):
        middle = (lower + upper) * .5
        mass = (.5 * (shifted - middle).clamp_min(0)).square().sum(-1, keepdim=True)
        lower = torch.where(mass > 1, middle, lower)
        upper = torch.where(mass > 1, upper, middle)
    weight = (.5 * (shifted - upper).clamp_min(0)).square() * valid.float()
    return torch.where(any_valid, weight / weight.sum(-1, keepdim=True).clamp_min(1e-12),
                       torch.zeros_like(weight))


def weighted_median(values: torch.Tensor, weights: torch.Tensor):
    """Values [N,T], nonnegative normalized weights [N]."""
    ordered, position = values.sort(dim=0)
    cumulative = weights[position].cumsum(0)
    first = (cumulative >= .5).float().argmax(0)
    return ordered.gather(0, first[None]).squeeze(0)


@dataclass
class RidgeEvidence:
    candidate: torch.Tensor
    common: torch.Tensor
    disagreement: torch.Tensor
    gain: torch.Tensor
    map_error: torch.Tensor
    reference_error: torch.Tensor
    effective_count: torch.Tensor
    supported: bool


def weighted_ridge_evidence(query: torch.Tensor, features: torch.Tensor,
                            targets: torch.Tensor, weights: torch.Tensor,
                            ridge: float, constant: bool = False):
    """Cholesky ridge and exact weighted leave-one-out for one query/channel.

    Analytic LOO uses the weighted ridge leverage h_i. Query LOO forecasts
    are obtained without N additional linear solves. Float64 is intentional.
    """
    chosen = weights > 0
    n, horizon = targets.shape
    zero = targets.new_zeros(horizon)
    if int(chosen.sum()) < 3:
        return RidgeEvidence(zero, zero, zero, zero, zero, zero,
                             zero.new_tensor(0.), False)
    x = features[chosen].double()
    y = targets[chosen].double()
    w = weights[chosen].double()
    w = w / w.sum().clamp_min(1e-12)
    effective = 1. / w.square().sum()
    if float(effective) < 2.:
        return RidgeEvidence(zero, zero, zero, zero, zero, zero,
                             effective.float(), False)
    if constant:
        candidate = (w[:, None] * y).sum(0)
        remaining = (candidate[None] - w[:, None] * y) / (1 - w[:, None]).clamp_min(1e-8)
        leave_query = remaining
        loo_error = y - remaining
    else:
        root = w.sqrt()[:, None]
        scaled = x * root
        gram = scaled.T @ scaled
        identity = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
        chol = torch.linalg.cholesky(gram + max(float(ridge), 1e-8) * identity)
        inv = torch.cholesky_inverse(chol)
        coefficient = torch.cholesky_solve(x.T @ (w[:, None] * y), chol)
        candidate = query.double() @ coefficient
        fitted_error = y - x @ coefficient
        leverage = w * (x @ inv * x).sum(-1)
        loo_error = fitted_error / (1 - leverage[:, None]).clamp_min(1e-8)
        influence = w * (x @ inv @ query.double())
        leave_query = candidate[None] - influence[:, None] * loo_error
    common = weighted_median(leave_query, w)
    disagreement = weighted_median((leave_query - common[None]).abs(), w)
    map_error = (w[:, None] * loo_error.square()).sum(0)
    zero_error = (w[:, None] * y.square()).sum(0)
    center = (w[:, None] * y).sum(0)
    constant_loo = (center[None] - w[:, None] * y) / (1 - w[:, None]).clamp_min(1e-8)
    constant_error = (w[:, None] * (y - constant_loo).square()).sum(0)
    reference = torch.minimum(zero_error, constant_error)
    return RidgeEvidence(candidate.float(), common.float(), disagreement.float(),
                         (reference - map_error).float(), map_error.float(),
                         reference.float(), effective.float(), True)


def selection_from_memory(key: torch.Tensor, memory_key: torch.Tensor,
                          eligible: torch.Tensor, temperature: float,
                          random: bool = False, fixed_topk: int = 0,
                          random_seed: int = 2021):
    """All-candidate raw-history retrieval; top-K only for its explicit ablation."""
    distance = (memory_key - key[None]).square().mean(-1)
    score = -distance / max(float(temperature), 1e-6)
    if random:
        # Dedicated generator: negative control never advances training RNG.
        gen = torch.Generator(device=score.device).manual_seed(int(random_seed))
        score = torch.rand(score.shape, generator=gen, device=score.device)
    if fixed_topk:
        take = min(fixed_topk, int(eligible.sum()))
        mask = torch.zeros_like(eligible)
        if take:
            index = score.masked_fill(~eligible, -1e30).topk(take).indices
            mask[index] = True
        eligible = mask
    return entmax15(score, eligible), distance


def two_stage_weights(raw: torch.Tensor, query_key: torch.Tensor,
                      memory_key: torch.Tensor, memory_common: torch.Tensor,
                      memory_variation: torch.Tensor, stable_quality: torch.Tensor,
                      variation_quality: torch.Tensor, temperature_stable: float,
                      temperature_variation: float, beta_stable: float,
                      beta_variation: float, null_score: float,
                      use_null: bool = True, use_quality: bool = True):
    initial_support = raw > 0
    common = (raw[:, None] * memory_key).sum(0)
    variation = query_key - common
    stable_score = -(memory_common - common[None]).square().mean(-1) / max(temperature_stable, 1e-6)
    variant_score = -(memory_variation - variation[None]).square().mean(-1) / max(temperature_variation, 1e-6)
    if use_quality:
        stable_score = stable_score + beta_stable * stable_quality
        variant_score = variant_score + beta_variation * variation_quality
    stable = entmax15(stable_score, initial_support)
    if use_null:
        score = torch.cat((variant_score, variant_score.new_tensor([null_score])))
        mask = torch.cat((initial_support, initial_support.new_ones(1)))
        with_null = entmax15(score, mask)
        variant, null_weight = with_null[:-1], with_null[-1]
    else:
        variant = entmax15(variant_score, initial_support)
        null_weight = variant.new_tensor(1.) if not bool(initial_support.any()) else variant.new_tensor(0.)
    return common, variation, stable, variant, null_weight


def reliable_correction(evidence: RidgeEvidence, gain_scale: float,
                        consensus: bool, reliability: bool):
    if not evidence.supported:
        return (evidence.common, evidence.common.new_zeros(evidence.common.shape),
                evidence.common.new_zeros(evidence.common.shape))
    common = evidence.common if consensus else evidence.candidate
    agreement = (common.abs() / (common.abs() + evidence.disagreement + 1e-8)
                 if consensus else torch.ones_like(common))
    map_confidence = (torch.sigmoid(evidence.gain / max(gain_scale, 1e-6))
                      if reliability else torch.ones_like(common))
    return common, agreement, map_confidence
