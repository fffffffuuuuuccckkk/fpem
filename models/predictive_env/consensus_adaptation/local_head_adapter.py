"""Small-sample dual ridge for a shared query-local forecast-head update.

The finite retrieval set produces ONE linear operator delta, not a mixture of
source futures. All outputs use source-normalized labels and query denormalization.
"""
from __future__ import annotations

import torch


def unified_operator_adapt(query_hidden: torch.Tensor, source_hidden: torch.Tensor,
                           source_residual: torch.Tensor, valid: torch.Tensor,
                           p_well: torch.Tensor, consensus_weight: torch.Tensor,
                           lambda_ridge: float = 1000., lambda_huber: float = 1.,
                           mode: str = "both_square", use_loss_weight: bool = False,
                           use_output_weight: bool = False, shuffle_future: bool = False,
                           irls_steps: int = 3, huber_delta: float = 1.):
    """One query-local operator, trained on unified Top-K retrieved pairs.

    Modes: ``legacy_protect``, ``both_square``, ``well_square_poor_huber``,
    ``all_huber``. The source future/residual is never the query target.
    """
    if mode not in {"legacy_protect", "both_square", "well_square_poor_huber", "all_huber"}:
        raise ValueError(mode)
    if query_hidden.ndim != 4 or source_hidden.ndim != 5:
        raise ValueError("query [B,C,P,D], sources [B,C,K,P,D] required")
    b, c, k = valid.shape
    horizon = source_residual.shape[-1]
    if (source_hidden.shape[:3] != (b, c, k) or
            source_residual.shape != (b, c, k, horizon) or
            p_well.shape != valid.shape or consensus_weight.shape != (b, c, horizon)):
        raise ValueError("source, posterior and consensus layouts differ")
    if k == 0:
        return query_hidden.new_zeros(b, c, horizon), {"valid_fraction": 0.}
    source = source_hidden.permute(0, 1, 2, 4, 3).flatten(-2).float()
    query = query_hidden.permute(0, 1, 3, 2).flatten(-2).float()
    source = torch.cat((source, torch.ones_like(source[..., :1])), -1)
    query = torch.cat((query, torch.ones_like(query[..., :1])), -1)
    target = source_residual.float().clone()
    if shuffle_future:
        # Permute only the already eligible/selected rows of each query.
        for bi in range(b):
            for ci in range(c):
                chosen = valid[bi, ci].nonzero(as_tuple=True)[0]
                if len(chosen) > 1:
                    target[bi, ci, chosen] = target[bi, ci, chosen.roll(1)].clone()
    well = valid & (p_well >= .5)
    poor = valid & ~well
    if mode == "legacy_protect":
        target = target * (~well)[..., None]
        base_weight = (well.float() * p_well + poor.float() * (1 - p_well))
    elif mode == "all_huber":
        base_weight = valid.float()
    else:
        base_weight = well.float() * p_well + poor.float() * (1 - p_well) * lambda_huber
    base_weight = base_weight.clamp_min(0.)
    if use_loss_weight:
        base_weight = base_weight[..., None] * consensus_weight[:, :, None].float()
    else:
        base_weight = base_weight[..., None].expand(b, c, k, horizon)
    weight = base_weight.clone()
    delta = target.new_zeros(b, c, horizon)
    iterations = irls_steps if mode in {"well_square_poor_huber", "all_huber"} else 1
    source_gram = source @ source.transpose(-1, -2)
    query_source = (query.unsqueeze(-2) @ source.transpose(-1, -2)).squeeze(-2)
    for _ in range(max(1, iterations)):
        # Per-horizon KxK dual ridge without materializing [B,C,H,K,D*P].
        root = weight.permute(0, 1, 3, 2).clamp_min(0.).sqrt()
        gram = source_gram[:, :, None] * root[..., :, None] * root[..., None, :]
        eye = torch.eye(k, dtype=gram.dtype, device=gram.device)
        rhs = target.permute(0, 1, 3, 2) * root
        dual = torch.linalg.solve(gram + max(float(lambda_ridge), 1e-5) * eye,
                                  rhs[..., None]).squeeze(-1)
        coefficient = root * dual
        delta = (query_source[:, :, None] * coefficient).sum(-1)
        if iterations > 1:
            fitted = torch.einsum("bcij,bchj->bchi", source_gram, coefficient)
            error = (target.permute(0, 1, 3, 2) - fitted).abs().clamp_min(1e-6)
            huber_weight = (float(huber_delta) / error).clamp(max=1.)
            robust = valid if mode == "all_huber" else poor
            factor = torch.where(robust[:, :, None], huber_weight, 1.)
            weight = base_weight * factor.permute(0, 1, 3, 2)
    if use_output_weight:
        delta = delta * consensus_weight
    delta = torch.where(valid.any(-1)[..., None], delta, torch.zeros_like(delta))
    if not torch.isfinite(delta).all():
        raise FloatingPointError("non-finite IRLS/Ridge update")
    return delta.to(query_hidden.dtype), {
        "valid_fraction": float(valid.float().mean()),
        "well_count": float(well.float().sum(-1).mean()),
        "poor_count": float(poor.float().sum(-1).mean()),
        "correction_abs_mean": float(delta.abs().mean()),
        "consensus_mean": float(consensus_weight.mean()),
    }


def local_operator_adapt(query_hidden: torch.Tensor, source_hidden: torch.Tensor,
                         source_residual: torch.Tensor, valid: torch.Tensor,
                         stable_probability: torch.Tensor,
                         under_probability: torch.Tensor, stable_k: int,
                         lambda_stable: float = 1., lambda_ridge: float = 10.,
                         protect: bool = True, only_under: bool = False,
                         all_supervised: bool = False, shuffle_future: bool = False):
    """Return normalized Δforecast [B,C,H], diagnostics, no target query access.

    ``source_residual`` is source (real Y - frozen A0 prediction) divided by
    that source's historical standard deviation. Bias is part of the lifted
    feature vector, so the same operator updates slope and intercept.
    """
    if query_hidden.ndim != 4 or source_hidden.ndim != 5:
        raise ValueError("query [B,C,P,D], sources [B,C,K,P,D] required")
    b, c, k = source_hidden.shape[:3]
    horizon = source_residual.shape[-1]
    if source_residual.shape[:3] != (b, c, k) or valid.shape != (b, c, k):
        raise ValueError("retrieval values do not align")
    if k == 0:
        return query_hidden.new_zeros(b, c, horizon), {"update_norm": 0., "valid_fraction": 0.}
    source = source_hidden.permute(0, 1, 2, 4, 3).flatten(-2).float()
    query = query_hidden.permute(0, 1, 3, 2).flatten(-2).float()
    # Bias must be weighted in the same lifted-space objective.
    source = torch.cat((source, torch.ones_like(source[..., :1])), -1)
    query = torch.cat((query, torch.ones_like(query[..., :1])), -1)
    targets = source_residual.float()
    if shuffle_future and k > stable_k:
        # Shuffle only within the already-causally-eligible source set.
        # A global bank permutation could attach a not-yet-observed future to
        # an older source and silently leak the current query's future.
        under = targets[:, :, stable_k:].roll(1, dims=2)
        targets = torch.cat((targets[:, :, :stable_k], under), 2)
    if all_supervised:
        weight = valid.float()
        residual = targets
    else:
        slot = torch.arange(k, device=valid.device).view(1, 1, k)
        is_stable = slot < stable_k
        u_weight = valid.float() * under_probability * (~is_stable)
        s_weight = valid.float() * stable_probability * is_stable if protect else torch.zeros_like(u_weight)
        weight = u_weight + lambda_stable * s_weight
        residual = targets * (u_weight / weight.clamp_min(1e-12))[..., None]
    scaled = source * weight.clamp_min(0).sqrt()[..., None]
    response = residual * weight.clamp_min(0).sqrt()[..., None]
    gram = scaled @ scaled.transpose(-1, -2)
    eye = torch.eye(k, device=gram.device, dtype=gram.dtype)
    matrix = gram + max(float(lambda_ridge), 1e-5) * eye
    dual = torch.linalg.solve(matrix, response)
    update = scaled.transpose(-1, -2) @ dual
    delta = (query.unsqueeze(-2) @ update).squeeze(-2)
    active = valid.any(-1)
    delta = torch.where(active[..., None], delta, torch.zeros_like(delta))
    source_delta = source @ update
    under_mask = valid[:, :, stable_k:] if not all_supervised else valid
    under_before = source_residual[:, :, stable_k:] if not all_supervised else source_residual
    under_after = under_before - (source_delta[:, :, stable_k:] if not all_supervised else source_delta)
    stable_mask = valid[:, :, :stable_k]
    def masked_mse(values, mask):
        return float((values.square() * mask[..., None]).sum().detach() /
                     (mask.sum() * horizon).clamp_min(1))
    return delta.to(query_hidden.dtype), {
        "update_norm": float(update.norm(dim=(-2, -1)).mean().detach()),
        "valid_fraction": float(valid.float().mean()),
        "under_count": float((valid[:, :, stable_k:]).float().sum(-1).mean()) if not all_supervised else float(valid.float().sum(-1).mean()),
        "stable_count": float(valid[:, :, :stable_k].float().sum(-1).mean()),
        "under_before_mse": masked_mse(under_before, under_mask),
        "under_after_mse": masked_mse(under_after, under_mask),
        "stable_prediction_shift_mse": masked_mse(source_delta[:, :, :stable_k], stable_mask),
    }
