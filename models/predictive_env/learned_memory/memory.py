"""TRAIN-only learnability labels on real historical/future memory pairs.

The prediction error of a reference fitted on all TRAIN is deliberately named
``train_fitted_prediction_error``; it is not an out-of-fold estimate. A strict
OOF reference must be supplied separately for confirmatory probability work.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from torch import nn
from torch.nn import functional as F

from models.predictive_env.paired_memory import PairedMemoryBank


class LinearHistoryDecoder(nn.Module):
    def __init__(self, token_count: int, dim: int, seq_len: int):
        super().__init__()
        self.linear = nn.Linear(token_count * dim, seq_len)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim not in (3, 4):
            raise ValueError("hidden must be [N,P,D] or [B,C,P,D]")
        return self.linear(hidden.flatten(-2))


def _robust(values: np.ndarray):
    center = np.median(values, axis=0)
    mad = np.median(np.abs(values - center), axis=0)
    scale = np.maximum(1.4826 * mad, 1e-3)
    return np.clip((values - center) / scale, -20, 20), center, scale


def _mixture(features: np.ndarray, ranking_error: np.ndarray, seed: int,
             ranking_name: str):
    one = GaussianMixture(1, reg_covar=1e-4, random_state=seed).fit(features)
    two = GaussianMixture(2, reg_covar=1e-4, n_init=5,
                          random_state=seed).fit(features)
    posterior = two.predict_proba(features)
    error_by_group = (posterior * ranking_error[:, None]).sum(0) / posterior.sum(0).clip(1e-8)
    stable = int(error_by_group.argmin())
    return posterior[:, stable].astype(np.float32), {
        "bic_one": float(one.bic(features)), "bic_two": float(two.bic(features)),
        "two_components_bic_supported": bool(two.bic(features) < one.bic(features)),
        "stable_component": stable,
        f"component_{ranking_name}_error": error_by_group.tolist(),
    }


def fit_learnability(records: dict, seed: int = 2021, epochs: int = 20,
                     batch_size: int = 256, device: str | torch.device = "cpu") -> dict:
    """Fit only a linear detached-H decoder, then TRAIN-only GMMs."""
    hidden = records["hidden"].detach().float().cpu()
    x = records["x"].detach().float().cpu().transpose(1, 2)
    y = records["target"].detach().float().cpu().transpose(1, 2)
    a0 = records["a0_pred"].detach().float().cpu().transpose(1, 2)
    if hidden.shape[:2] != x.shape[:2] or len(y) != len(x):
        raise ValueError("TRAIN anchor/hidden/future records are not paired")
    raw_std = x.var(-1, keepdim=True, unbiased=False).add(1e-5).sqrt()
    # Entirely TRAIN-history-derived floor, shared with local supervised
    # residual normalization. No future-scale statistics are used.
    floor = (.25 * raw_std.median(dim=0).values.median(dim=0).values).clamp_min(.01)
    std = raw_std.clamp_min(floor)
    normalized_x = ((x - x.mean(-1, keepdim=True)) / std).clamp(-50, 50)
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    decoder = LinearHistoryDecoder(hidden.shape[2], hidden.shape[3], x.shape[-1]).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    flat_h = hidden.flatten(0, 1)
    flat_x = normalized_x.flatten(0, 1)
    for _ in range(epochs):
        for ids in torch.randperm(len(flat_h), generator=generator).split(batch_size):
            prediction = decoder(flat_h[ids].to(device))
            loss = F.mse_loss(prediction, flat_x[ids].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        reconstruction = torch.cat([
            decoder(part.to(device)).cpu() for part in flat_h.split(batch_size)
        ]).reshape_as(x)
    rec_error = (reconstruction - normalized_x).square().mean((1, 2)).clamp_min(1e-8)
    pred_error = ((y - a0) / std).square().mean((1, 2)).clamp_min(1e-8)
    rec_only = rec_error.log().numpy()[:, None]
    both = torch.stack((rec_error.log(), pred_error.log()), -1).numpy()
    standardized_rec, center_rec, scale_rec = _robust(rec_only)
    standardized_both, center_both, scale_both = _robust(both)
    p_rec, audit_rec = _mixture(standardized_rec, rec_error.numpy(), seed,
                                "reconstruction")
    p_both, audit_both = _mixture(standardized_both, pred_error.numpy(), seed,
                                  "prediction")
    result = {
        "decoder": {k: v.detach().cpu() for k, v in decoder.state_dict().items()},
        "reconstruction_error": rec_error,
        "train_fitted_prediction_error": pred_error,
        "stable_probability_rec": torch.from_numpy(p_rec),
        "p_well": torch.from_numpy(p_rec),
        "p_poor": 1 - torch.from_numpy(p_rec),
        "stable_probability_joint": torch.from_numpy(p_both),
        "underlearned_probability_joint": 1 - torch.from_numpy(p_both),
        "gmm_rec": audit_rec, "gmm_joint": audit_both,
        "robust_normalization_rec": {"center": center_rec.tolist(), "scale": scale_rec.tolist()},
        "robust_normalization_joint": {"center": center_both.tolist(), "scale": scale_both.tolist()},
        "prediction_error_provenance": "TRAIN-fitted frozen A0, NOT strict A0-OOF",
        "history_std_floor": float(floor),
    }
    return result


class LearnabilityMemory(PairedMemoryBank):
    """Reuses causal whole-window retrieval and changes only group semantics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("density_common", torch.empty(0, dtype=torch.bool))

    @torch.no_grad()
    def fit(self, records):
        report = super().fit(records)
        self.density_common = self.common.clone()
        return report

    def set_group(self, fitted: dict, method: str, seed: int = 2021):
        if method not in {"density", "reconstruction", "joint", "random",
                          "random_reconstruction", "none"}:
            raise ValueError(method)
        n, c = self.common.shape
        if method == "density":
            if self.density_common.shape != self.common.shape:
                raise ValueError("density grouping was not preserved in memory cache")
            probability = self.density_common.float()
        elif method == "none":
            probability = torch.full((n, c), .5)
        else:
            key = ("stable_probability_rec" if method in
                   {"reconstruction", "random_reconstruction"} else
                   "stable_probability_joint")
            value = fitted[key].float()
            if len(value) != n:
                raise ValueError("GMM and TRAIN memory anchor counts differ")
            if method in {"random", "random_reconstruction"}:
                value = value[torch.randperm(n, generator=torch.Generator().manual_seed(seed))]
            probability = value[:, None].expand(n, c).clone()
        self.stable_probability = probability.to(self.keys.device)
        self.underlearned_probability = 1 - self.stable_probability
        self.common = self.stable_probability >= .5
        self.group_method = method

    def _select(self, distance, eligible, count, mode, sample_ids):
        return super()._select(distance, eligible, count,
                               "random" if getattr(self, "_random_group", False) else mode,
                               sample_ids)

    @torch.no_grad()
    def retrieve(self, *args, **kwargs):
        view = kwargs.get("view", args[1] if len(args) > 1 else None)
        self._random_group = view == "random_group"
        if self._random_group:
            kwargs = dict(kwargs)
            if "view" in kwargs:
                kwargs["view"] = "mixed"
            else:
                args = list(args)
                args[1] = "mixed"
        try:
            result = super().retrieve(*args, **kwargs)
        finally:
            self._random_group = False
        channel = torch.arange(self.keys.shape[1], device=result["indices"].device)[None, :, None]
        result["stable_probability"] = self.stable_probability[result["indices"], channel] * result["valid"]
        result["underlearned_probability"] = self.underlearned_probability[result["indices"], channel] * result["valid"]
        result["p_well"] = result["stable_probability"]
        result["p_poor"] = result["underlearned_probability"]
        result["source_residual_raw"] = self.residual_raw.to(channel.device)[result["indices"], channel] * result["valid"][..., None]
        return result
