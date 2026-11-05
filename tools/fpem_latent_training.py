"""TRAIN-only statistics and opt-in losses for latent decomposition pilots."""

import torch
import numpy as np

from models.predictive_env.latent_disentanglement_losses import orthogonality_loss
from models.predictive_env.environment_mapping_decomposition import (
    EnvironmentMappingEstimator,
)


@torch.no_grad()
def refresh_environment_mapping(model, ordered_train, assignment, device):
    """Fit EPMD only from the complete ordered TRAIN split at an EIIL update."""
    if model.decomposition_type != "environment_mapping":
        return {}
    was_training = model.training
    model.eval()
    q = torch.as_tensor(np.asarray(assignment), dtype=torch.float64)
    if q.ndim != 2 or q.shape[0] != len(ordered_train.dataset):
        raise ValueError("EPMD requires one soft assignment for every TRAIN sample")
    if not torch.allclose(q.sum(-1), torch.ones(len(q), dtype=q.dtype), atol=1e-5):
        raise ValueError("environment rows must sum to one")
    estimator = EnvironmentMappingEstimator(
        model.decomposer.features, model.pred_len, q.shape[1]
    )
    seen = torch.zeros(len(q), dtype=torch.bool)
    try:
        for x, y, sample_id, cycle_index in ordered_train:
            sample_id = sample_id.long()
            if seen[sample_id].any():
                raise RuntimeError("duplicate TRAIN sample ID in EPMD fit")
            seen[sample_id] = True
            hidden, mean, std = model.encode(x.float().to(device), cycle_index.to(device))
            future = y[:, -model.pred_len:].float().to(device)
            target_norm = ((future - mean[:, :1]) / std[:, :1]).permute(0, 2, 1)
            cycle = model.normalized_cycle_forecast(cycle_index.to(device), x.shape[0], device)
            if cycle is not None:
                target_norm = target_norm - cycle
            estimator.add(hidden, target_norm, q[sample_id])
        if not seen.all():
            raise RuntimeError("EPMD fit did not cover the complete TRAIN split")
        reference = model.h_forecast_head.weight.detach().T.double().cpu().numpy()
        basis, values, mean_h, masses, diagnostics = estimator.finish(reference)
        model.decomposer.update_mapping(basis, values, mean_h, diagnostics)
        return {
            "epmd/environment_observation_mass": masses.tolist(),
            "epmd/environment_sample_mass": [
                item["mass"] for item in diagnostics["environments"]
            ],
            "epmd/alpha": float(model.decomposer.alpha.detach()),
            "epmd/mapping_fit": diagnostics,
        }
    finally:
        model.train(was_training)


@torch.no_grad()
def refresh_latent_statistics(model, ordered_train, assignment, device, ema=0.0):
    if model.decomposition_type not in ("env_guided_subspace", "reference_guided"):
        return {}
    was_training = model.training
    model.eval()
    pooled = []
    ids = []
    for x, _, sample_id, cycle_index in ordered_train:
        hidden, _, _ = model.encode(x.float().to(device), cycle_index.to(device))
        pooled.append(hidden.mean((1, 2)).detach().float().cpu())
        ids.append(sample_id.cpu())
    model.train(was_training)
    ids = torch.cat(ids).long()
    h = torch.cat(pooled)
    if len(ids) != len(assignment) or not torch.equal(ids.sort().values, torch.arange(len(ids))):
        raise RuntimeError("TRAIN sample IDs must cover the environment assignment exactly")
    q = torch.as_tensor(assignment, dtype=torch.float32)[ids]
    h, q = h.to(device), q.to(device)
    if model.decomposition_type == "env_guided_subspace":
        model.decomposer.update_environment_statistics(h, q, ema)
        return {"subspace/stage_explained_ratio":
                float(-model.decomposer.environment_subspace_loss().detach())}
    model.decomposer.update_references(h, q, ema)
    return {"reference/stage_environment_mass": q.mean(0).cpu().tolist()}


def latent_auxiliary_loss(model, output, q, args):
    """Return only explicitly enabled losses; no extra graph for lambda=0."""
    terms = {}
    pieces = []
    if model.decomposition_type == "env_guided_subspace" and args.lambda_sub != 0:
        value = model.decomposer.environment_subspace_loss()
        pieces.append(args.lambda_sub * value)
        terms["loss/subspace"] = value.detach()
    if model.decomposition_type == "reference_guided" and args.reference_mode == "environment" and args.lambda_reference_q != 0:
        value = model.decomposer.q_supervision_loss(output["hidden_tokens"], q)
        pieces.append(args.lambda_reference_q * value)
        terms["loss/reference_q"] = value.detach()
    if args.lambda_orth != 0:
        value = orthogonality_loss(
            output["z_inv_tokens"], output["z_var_tokens"], args.orth_type
        )
        pieces.append(args.lambda_orth * value)
        terms["loss/orth"] = value.detach()
    if not pieces:
        return None, terms
    return sum(pieces), terms
