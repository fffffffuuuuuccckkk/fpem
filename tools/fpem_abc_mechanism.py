"""Read-only diagnostics for the frozen-A0 / G1 / R3 mechanism study.

All arrays here use the history-only normalisation of the existing G1/R3
runner.  Predictions are converted to original units for reported MSE/MAE.
This module deliberately does not alter either fitted point model.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from models.predictive_env.dual_pattern_memory.forecast_fusion import variation_forecast
from tools.run_fpem_dppm_g1_r3_full import metric, residual_ridge, ridge_query


def write_rows(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sample_error(data, prediction, absolute=False):
    err = (prediction - data["truth"]) * data["scale"].transpose(1, 2)
    return (err.abs() if absolute else err.square()).mean((1, 2)).numpy()


def effect(data, prediction, reference):
    return sample_error(data, reference) - sample_error(data, prediction)


def by_time(values, groups=10):
    values = np.asarray(values)
    return [values[idx] for idx in np.array_split(np.arange(len(values)), groups)]


def rank_rows(rows, objective="gain_mse"):
    return sorted(rows, key=lambda row: float(row.get(objective) or -1e30), reverse=True)


def chronological_blocks(ids, n=4):
    order = torch.argsort(ids)
    return [order[idx] for idx in np.array_split(np.arange(len(order)), n) if len(idx)]


def purged_past(ids, target_index, span):
    return (ids + span <= ids[target_index].min()).nonzero().flatten()


def nearest(source, query, k=8, chunk=256, source_ids=None, query_ids=None,
            span=None):
    """Exact history-key neighbours with optional causal future-end filter."""
    src = source.reshape(-1, source.shape[-1]).float()
    dst = query.reshape(-1, query.shape[-1]).float()
    out_index, out_distance, out_valid = [], [], []
    for begin in range(0, len(dst), chunk):
        end = min(len(dst), begin + chunk)
        dist = torch.cdist(dst[begin:end], src).square() / src.shape[-1]
        if source_ids is not None:
            if query_ids is None or span is None:
                raise ValueError("causal retrieval requires query IDs and span")
            valid = source_ids[None] + span <= query_ids[begin:end, None]
            dist = dist.masked_fill(~valid, float("inf"))
        best = dist.topk(min(k, len(src)), largest=False)
        out_index.append(best.indices)
        out_distance.append(best.values)
        out_valid.append(torch.isfinite(best.values))
    return torch.cat(out_index), torch.cat(out_distance), torch.cat(out_valid)


def simple_key_drift(source, query):
    mu = source.reshape(-1, 64).mean(0)
    scale = source.reshape(-1, 64).std(0).clamp_min(.05)
    return (((query - mu) / scale).square().mean(-1)).mean(-1).numpy()


def residual_correction(data, g1, learner, reliability, offset, chunk=24):
    """Collect compact R3 predictions and expert-disagreement summaries."""
    output = {name: [] for name in (
        "full", "anchor", "candidate_mean", "expert_variance", "null_variance",
        "pair_l1", "pair_l2", "pair_max", "pair_iqr", "null", "route_entropy",
        "effective_support", "active", "reliability", "candidate_l2",
        "seg_disagreement", "seg_effect", "seg_gain", "seg_reliability",
        "frequency_low_disagreement", "frequency_high_disagreement",
        "expert_weight", "candidate_projection")}
    horizon = data["truth"].shape[-1]
    segments = [x for x in torch.tensor_split(torch.arange(horizon), 8) if len(x)]
    for start in range(0, len(data["key"]), chunk):
        end = min(start + chunk, len(data["key"]))
        piece = {name: value[start:end] for name, value in data.items()}
        anchor = piece["base"] + g1["scale"] * ridge_query(piece["key"], g1["weight"])
        query = learner.query(piece["key"], offset)
        candidate, weight = query["candidate"], query["weight"]
        empty = anchor.new_zeros(horizon, 0)
        full, beta = variation_forecast(anchor, candidate, weight, reliability,
                                         empty, True, 8)
        # The common reference is removed before measuring expert disagreement.
        independent = candidate - learner.net.reference[-1].view(1, 1, 1, -1)
        mass = weight.sum(2, keepdim=True)
        norm_w = weight / mass.clamp_min(1e-8)
        center = (norm_w[..., None] * independent).sum(2)
        deviation = independent - center[:, :, None]
        expert_var = (norm_w[..., None] * deviation.square()).sum(2)
        mixture = (weight[..., None]*candidate).sum(2) + \
            query["null_weight"][..., None]*anchor
        null_var = (weight[..., None]*(candidate-mixture[:, :, None]).square()).sum(2) + \
            query["null_weight"][..., None]*(anchor-mixture).square()
        pair = (independent[:, :, :, None] - independent[:, :, None, :]).abs()
        pair_l1 = pair.mean((2, 3))
        pair_l2 = pair.square().mean((2, 3)).sqrt()
        pair_max = pair.amax((2, 3))
        pair_iqr = independent.quantile(.75, dim=2) - independent.quantile(.25, dim=2)
        ent = -(weight.clamp_min(1e-8) * weight.clamp_min(1e-8).log()).sum(2)
        stats = {
            "full": full, "anchor": anchor, "candidate_mean": center,
            "expert_variance": expert_var, "null_variance": null_var,
            "pair_l1": pair_l1, "pair_l2": pair_l2,
            "pair_max": pair_max, "pair_iqr": pair_iqr,
            "null": query["null_weight"], "route_entropy": ent,
            "effective_support": (weight > 1e-6).float().sum(2),
            "active": beta.abs().mean(-1),
            "reliability": (weight[..., None] * reliability[None, None]).sum(2).mean(-1),
            "candidate_l2": (candidate - anchor[:, :, None]).square().mean((2, 3)).sqrt(),
            "expert_weight": weight,
            "candidate_projection": (norm_w[..., None] * candidate).sum(2),
        }
        stats["seg_disagreement"] = torch.stack([expert_var[..., ix].mean(-1) for ix in segments], -1)
        frequency=torch.fft.rfft(deviation,dim=-1).abs().square()
        frequency=(norm_w[..., None]*frequency).sum(2)
        split=max(2,frequency.shape[-1]//4)
        stats["frequency_low_disagreement"]=frequency[...,1:split].mean(-1)
        stats["frequency_high_disagreement"]=frequency[...,split:].mean(-1)
        stats["seg_effect"] = torch.stack([(full-anchor)[..., ix].abs().mean(-1) for ix in segments], -1)
        stats["seg_gain"] = torch.stack([
            ((anchor[..., ix]-piece["truth"][..., ix]).square() -
             (full[..., ix]-piece["truth"][..., ix]).square()).mean(-1)
            for ix in segments], -1)
        stats["seg_reliability"] = torch.stack([
            (weight * reliability[:, min(7, int(s * 8 / len(segments)))][None, None]).sum(2)
            for s in range(len(segments))], -1)
        for name, value in stats.items():
            output[name].append(value.detach().cpu())
    return {name: torch.cat(values) for name, values in output.items()}


def pearson(x, y):
    x, y = np.asarray(x).reshape(-1), np.asarray(y).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5 or min(np.std(x[mask]), np.std(y[mask])) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def gain_row(code, split, gain, **other):
    values = np.asarray(gain).reshape(-1)
    return {"experiment": code, "split": split, "gain_mse": float(values.mean()),
            "positive_ratio": float((values > 0).mean()),
            "median_gain": float(np.median(values)),
            "p10_gain": float(np.quantile(values, .1)),
            "p90_gain": float(np.quantile(values, .9)), **other}


def block_bootstrap_gain(values, blocks=20, repeats=1000, seed=2021):
    """Resample ordered time blocks, never overlapping windows individually."""
    pieces=[x for x in by_time(np.asarray(values),blocks) if len(x)]
    means=np.asarray([x.mean() for x in pieces])
    rng=np.random.default_rng(seed)
    draws=means[rng.integers(0,len(means),(repeats,len(means)))].mean(1)
    return {"block_bootstrap_ci_low":float(np.quantile(draws,.025)),
            "block_bootstrap_ci_high":float(np.quantile(draws,.975)),
            "block_bootstrap_p_gain_nonpositive":float((1+(draws<=0).sum())/(repeats+1)),
            "effective_time_blocks":len(pieces)}


def synthetic_g1(seed=2021):
    gen = torch.Generator().manual_seed(seed)
    n, horizon = 1200, 16
    key = torch.randn(n, 1, 64, generator=gen)
    shared = torch.randn(65, horizon, generator=gen) * .07
    truth = ridge_query(key, shared) + .02 * torch.randn(n, 1, horizon, generator=gen)
    truth[::5] += .4 * key[::5, :, :1]
    ids = torch.arange(n)
    rows = chronological_blocks(ids, 5)
    fit = rows[0]
    hold = rows[-1]
    weight = residual_ridge(key[fit], truth[fit], .01)
    predicted = ridge_query(key[hold], weight)
    mse = float((predicted-truth[hold]).square().mean())
    base_mse = float(truth[hold].square().mean())
    return {"experiment": "A10", "split": "synthetic", "gain_mse": base_mse-mse,
            "shared_map_cosine": float(F.cosine_similarity(weight.flatten(), shared.flatten(), dim=0)),
            "predictive_mse": mse, "base_mse": base_mse,
            "type": "synthetic_not_real_data"}


def synthetic_r3(seed=2021):
    gen = torch.Generator().manual_seed(seed)
    n = 1200
    key = torch.randn(n, 1, 64, generator=gen)
    event = torch.randint(0, 4, (n,), generator=gen)
    key[:, :, :4] += F.one_hot(event, 4).float()[:, None] * 2
    templates = torch.randn(4, 16, generator=gen) * .3
    future = templates[event, None] + .05 * torch.randn(n, 1, 16, generator=gen)
    unknown = torch.arange(n) % 7 == 0
    future[unknown] += torch.randn(int(unknown.sum()), 1, 16, generator=gen)
    train, test = torch.arange(0, 800), torch.arange(800, n)
    centroids = torch.stack([key[train][event[train] == j].mean(0) for j in range(4)])
    proto = torch.stack([future[train][event[train] == j].mean(0) for j in range(4)])
    distance = torch.cdist(key[test].squeeze(1), centroids.squeeze(1))
    route = distance.argmin(-1)
    forecast = proto[route]
    known = ~unknown[test]
    return {"experiment": "B11", "split": "synthetic", "gain_mse": float(
        future[test][known].square().mean() -
        (forecast[known] - future[test][known]).square().mean()),
        "unknown_error": float((forecast[~known]-future[test][~known]).square().mean()),
        "route_accuracy": float((route[known] == event[test][known]).float().mean()),
        "type": "synthetic_not_real_data"}
