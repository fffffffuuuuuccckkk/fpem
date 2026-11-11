#!/usr/bin/env python3
"""Frozen-A0, TRAIN-only eight-prototype extraction and chronological probes."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage, signal
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_fpem_f03_r3_invariance import collect, prepare
from tools.run_fpem_shared_factorization_g1r3 import load_experiment

METHODS = ("Raw", "Hidden", "Frequency", "Trend-Seasonal", "Future-Supervised",
           "Phase-Only", "Raw + Weak Phase", "Raw + Shuffled Phase")
RIDGES = (0.1, 1.0, 10.0, 100.0)
SEED = 2022


def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("empty CSV: " + str(path))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_sq(x, y):
    return float(np.mean((x - y) ** 2))


def mae(x, y):
    return float(np.mean(np.abs(x - y)))


def independent_events(ids, span):
    starts = np.unique(ids)
    chosen = []
    last_end = -10**20
    for start in starts:
        if start >= last_end:
            chosen.append(int(start))
            last_end = int(start) + span
    return len(chosen)


def partition(raw, span):
    ids = raw["ids"].numpy().astype(np.int64)
    end = int(ids.max()) + span
    cut_a, cut_b = int(end * .50), int(end * .75)
    masks = {"A": ids + span <= cut_a,
             "B": (ids >= cut_a) & (ids + span <= cut_b),
             "C": ids >= cut_b}
    for name, mask in masks.items():
        if int(mask.sum()) < 24:
            raise ValueError(f"{name}: insufficient windows after purging ({mask.sum()})")
    if int((ids[masks["A"]] + span).max()) > int(ids[masks["B"]].min()):
        raise AssertionError("A/B label overlap")
    if int((ids[masks["B"]] + span).max()) > int(ids[masks["C"]].min()):
        raise AssertionError("B/C label overlap")
    return masks, {"cut_A": cut_a, "cut_B": cut_b, "span": span,
                   "independent_events": {k: independent_events(ids[v], span)
                                          for k, v in masks.items()}}


def rows(block, mask, phase_period, seq_len):
    n = int(mask.sum())
    c = block["key"].shape[1]
    key = block["key"][mask].numpy().reshape(n * c, 64).astype(np.float32)
    hidden = block["hidden"][mask].numpy().reshape(n * c, -1).astype(np.float32)
    future = block["truth_hist_norm"][mask].numpy().reshape(n * c, -1).astype(np.float32)
    ids = np.repeat(block["ids"][mask].numpy().astype(np.int64), c)
    channels = np.tile(np.arange(c), n)
    phase = np.mod((ids + seq_len) / phase_period, 1.0).astype(np.float32)
    return {"raw": key, "hidden": hidden, "y": future, "ids": ids,
            "channels": channels, "phase": phase}


def estimate_period(series):
    series = np.asarray(series, dtype=np.float64)
    if series.ndim == 1:
        series = series[:, None]
    change = np.diff(series, axis=0)
    frequencies, power = signal.periodogram(signal.detrend(change, axis=0), axis=0)
    score = np.mean(power / np.maximum(power.sum(axis=0, keepdims=True), 1e-12), axis=1)
    period = np.divide(1.0, frequencies, out=np.zeros_like(frequencies), where=frequencies > 0)
    valid = (period >= 8) & (period <= min(512, len(series) // 4))
    if not np.any(valid):
        return 24, "fallback: no identifiable TRAIN frequency in [8,512]"
    index = np.flatnonzero(valid)[np.argmax(score[valid])]
    return max(8, int(round(period[index]))), "TRAIN A first-difference periodogram"


def frequency_features(raw):
    spectrum = np.fft.rfft(raw.astype(np.float64), axis=1)
    energy = np.abs(spectrum[:, 1:]) ** 2 + 1e-8  # 32 non-DC bins
    log_power = np.log1p(energy)
    dominant = np.argsort(energy, axis=1)[:, -8:][:, ::-1]
    loc = dominant / 32.0
    amp = np.take_along_axis(log_power, dominant, axis=1)
    bands = np.stack([np.log1p(chunk.mean(axis=1)) for chunk in np.split(energy, 16, axis=1)], axis=1)
    return np.concatenate((log_power, loc, amp, bands), axis=1).astype(np.float32)


def trend_seasonal_features(raw):
    trend = ndimage.uniform_filter1d(raw, size=9, axis=1, mode="nearest")
    seasonal = raw - trend
    volatility = ndimage.uniform_filter1d(np.abs(seasonal), size=5, axis=1, mode="nearest")
    return np.concatenate((trend.reshape(-1, 32, 2).mean(-1),
                           seasonal.reshape(-1, 16, 4).mean(-1),
                           volatility.reshape(-1, 16, 4).mean(-1)), axis=1).astype(np.float32)


def phase_features(phase):
    output = np.zeros((len(phase), 64), np.float32)
    output[:, 0] = np.sin(2 * np.pi * phase)
    output[:, 1] = np.cos(2 * np.pi * phase)
    return output


def distances(x, centers):
    return np.maximum(0, np.mean(x[:, None, :] ** 2, axis=2)
                      + np.mean(centers[None, :, :] ** 2, axis=2)
                      - 2 * (x @ centers.T) / x.shape[1])


def phase_route(raw_distance, phase, proto_phase):
    angle = 2 * np.pi * phase
    query = np.stack((np.sin(angle), np.cos(angle)), axis=1)
    phase_distance = np.sum((query[:, None, :] - proto_phase[None, :, :]) ** 2, axis=2)
    ordered = np.partition(raw_distance, 1, axis=1)[:, :2]
    ambiguity = np.exp(-(ordered[:, 1] - ordered[:, 0]) /
                       np.maximum(ordered[:, 1], 1e-8))
    weight = .15 * ambiguity  # same fixed rule for real and shuffled phase
    return raw_distance + weight[:, None] * phase_distance, weight


def fit_representation(method, blocks):
    a = blocks["A"]
    transform_params = 0
    if method in ("Raw", "Raw + Weak Phase", "Raw + Shuffled Phase"):
        feature = lambda b: b["raw"]
    elif method == "Hidden":
        width = min(64, len(a["hidden"]) - 1, a["hidden"].shape[1])
        pca = PCA(n_components=width, svd_solver="randomized", random_state=SEED)
        pca.fit(a["hidden"])
        transform_params = int(pca.components_.size + pca.mean_.size)
        feature = lambda b: np.pad(pca.transform(b["hidden"]),
                                   ((0, 0), (0, 64 - width))).astype(np.float32)
    elif method == "Frequency":
        feature = lambda b: frequency_features(b["raw"])
    elif method == "Trend-Seasonal":
        feature = lambda b: trend_seasonal_features(b["raw"])
    elif method == "Future-Supervised":
        standard_x = StandardScaler().fit(a["raw"])
        x = standard_x.transform(a["raw"]).astype(np.float64)
        y = a["y"].astype(np.float64)
        y = (y - y.mean(axis=0)) / np.maximum(y.std(axis=0), 1e-6)
        cross = (x.T @ y) / len(x)
        left, singular, _ = np.linalg.svd(cross, full_matrices=False)
        width = min(16, left.shape[1])
        basis = (left[:, :width] * np.sqrt(singular[:width] / max(singular[0], 1e-8)))
        transform_params = int(basis.size)
        feature = lambda b: np.pad(standard_x.transform(b["raw"]) @ basis,
                                   ((0, 0), (0, 64 - width))).astype(np.float32)
    elif method == "Phase-Only":
        feature = lambda b: phase_features(b["phase"])
    else:
        raise ValueError(method)
    features = {name: feature(block) for name, block in blocks.items()}
    scaler = StandardScaler().fit(features["A"])
    normalized = {name: scaler.transform(value).astype(np.float32)
                  for name, value in features.items()}
    return normalized, transform_params


def fit_prototypes(method, blocks, feature):
    km = MiniBatchKMeans(n_clusters=8, batch_size=256, n_init=3,
                         max_iter=150, random_state=SEED)
    km.fit(feature["A"])
    own_dist = {name: distances(value, km.cluster_centers_) for name, value in feature.items()}
    fit_label = own_dist["A"].argmin(axis=1)
    rng = np.random.default_rng(SEED + 17)
    phase = {name: block["phase"].copy() for name, block in blocks.items()}
    if method == "Raw + Shuffled Phase":
        for name in phase:
            phase[name] = rng.permutation(phase[name])
    proto_phase = np.zeros((8, 2), np.float32)
    for index in range(8):
        selected = phase["A"][fit_label == index]
        if len(selected):
            proto_phase[index] = [np.sin(2 * np.pi * selected).mean(),
                                  np.cos(2 * np.pi * selected).mean()]
    labels, weights = {}, {}
    for name, d in own_dist.items():
        if method in ("Raw + Weak Phase", "Raw + Shuffled Phase"):
            adjusted, weight = phase_route(d, phase[name], proto_phase)
            labels[name] = adjusted.argmin(axis=1)
            weights[name] = float(weight.mean())
        else:
            labels[name] = d.argmin(axis=1)
            weights[name] = 0.0
    # Recalculate TRAIN centers from final routing, including phase-adjusted arms.
    future_center = np.stack([blocks["A"]["y"][labels["A"] == k].mean(axis=0)
                              if (labels["A"] == k).any() else blocks["A"]["y"].mean(axis=0)
                              for k in range(8)])
    raw_center = np.stack([blocks["A"]["raw"][labels["A"] == k].mean(axis=0)
                           if (labels["A"] == k).any() else blocks["A"]["raw"].mean(axis=0)
                           for k in range(8)])
    raw_scaler = StandardScaler().fit(blocks["A"]["raw"])
    raw_c = raw_scaler.transform(raw_center).astype(np.float32)
    raw_coverage = {name: distances(raw_scaler.transform(block["raw"]), raw_c).min(axis=1)
                    for name, block in blocks.items()}
    return labels, own_dist, raw_coverage, future_center, raw_center, weights, proto_phase


def pair_evidence(block, label, span, seed=SEED):
    n = len(label)
    rng = np.random.default_rng(seed)
    draw = max(100000, min(500000, n * 80))
    left, right = rng.integers(n, size=draw), rng.integers(n, size=draw)
    valid = (block["channels"][left] == block["channels"][right]) & (
        np.abs(block["ids"][left] - block["ids"][right]) >= span)
    left, right = left[valid][:8000], right[valid][:8000]
    if len(left) < 50:
        return {"lift": np.nan, "reason": "fewer than 50 non-overlapping same-variable pairs",
                "pairs": len(left), "within_pairs": 0}, None
    hx, fy = [], []
    for lo in range(0, len(left), 512):
        i, j = left[lo:lo + 512], right[lo:lo + 512]
        hx.append(np.mean((block["raw"][i] - block["raw"][j]) ** 2, axis=1))
        fy.append(np.mean((block["y"][i] - block["y"][j]) ** 2, axis=1))
    hx, fy = np.concatenate(hx), np.concatenate(fy)
    same = label[left] == label[right]
    # Matched random control: different-prototype pairs in identical RAW-distance bins.
    edges = np.unique(np.quantile(hx, np.linspace(0, 1, 11)))
    matched_same, matched_control, matched_bins = [], [], 0
    if len(edges) >= 4:
        bins = np.clip(np.digitize(hx, edges[1:-1]), 0, len(edges) - 2)
        for b in range(len(edges) - 1):
            si = np.flatnonzero((bins == b) & same)
            di = np.flatnonzero((bins == b) & ~same)
            if min(len(si), len(di)) < 10:
                continue
            size = min(len(si), len(di), 200)
            matched_same.extend(fy[rng.choice(si, size, replace=False)])
            matched_control.extend(fy[rng.choice(di, size, replace=False)])
            matched_bins += 1
    reason = "ok" if matched_bins >= 3 and len(matched_same) >= 50 else "insufficient RAW-distance-matched random controls"
    lift = 1 - np.mean(matched_same) / max(np.mean(matched_control), 1e-12) if reason == "ok" else np.nan
    frame = {"history_similarity": np.exp(-hx), "future_similarity": np.exp(-fy),
             "future_mse": fy, "same": same, "left_label": label[left]}
    return {"lift": float(lift), "reason": reason, "pairs": len(left),
            "within_pairs": int(same.sum()), "matched_pairs": len(matched_same),
            "matched_bins": matched_bins,
            "within_future_similarity_mean": float(np.exp(-fy[same]).mean()) if same.any() else np.nan,
            "matched_random_future_mse": float(np.mean(matched_control)) if matched_control else np.nan}, frame


def cluster_metrics(raw_a, labels_a):
    rng = np.random.default_rng(SEED)
    take = rng.choice(len(raw_a), min(1200, len(raw_a)), replace=False)
    x, lab = raw_a[take], labels_a[take]
    if len(np.unique(lab)) < 2 or len(np.unique(lab)) >= len(lab):
        return {"silhouette_raw": np.nan, "davies_bouldin_raw": np.nan,
                "calinski_harabasz_raw": np.nan}
    return {"silhouette_raw": float(silhouette_score(x, lab, sample_size=min(1000, len(x)), random_state=SEED)),
            "davies_bouldin_raw": float(davies_bouldin_score(x, lab)),
            "calinski_harabasz_raw": float(calinski_harabasz_score(x, lab))}


def probe_predict(kind, alpha, x_a, y_a, label_a, x_eval, label_eval):
    if kind == "Shared Mapping Probe":
        counts = np.bincount(label_a, minlength=8).clip(min=1)
        weight = len(label_a) / (8 * counts[label_a])
        model = Ridge(alpha=alpha).fit(x_a, y_a, sample_weight=weight)
        return model.predict(x_eval)
    global_model = Ridge(alpha=alpha).fit(x_a, y_a)
    if kind == "Global Ridge":
        return global_model.predict(x_eval)
    pred = global_model.predict(x_eval)
    for k in range(8):
        train = label_a == k
        query = label_eval == k
        if train.sum() >= max(16, x_a.shape[1] // 2) and query.any():
            pred[query] = Ridge(alpha=alpha).fit(x_a[train], y_a[train]).predict(x_eval[query])
    return pred


def mapping_probes(method, blocks, labels, common):
    x = {name: common[name] for name in ("A", "B", "C")}
    y = {name: blocks[name]["y"] for name in ("A", "B", "C")}
    rng = np.random.default_rng(SEED + 501)
    random_labels = {name: rng.integers(8, size=len(block["y"]))
                     for name, block in blocks.items()}
    rows, candidates = [], []
    for kind in ("Global Ridge", "Prototype Ridge", "Shared Mapping Probe", "Random Group Ridge"):
        lab = random_labels if kind == "Random Group Ridge" else labels
        choices = []
        for alpha in RIDGES:
            pred_b = probe_predict(kind, alpha, x["A"], y["A"], lab["A"], x["B"], lab["B"])
            value = mean_sq(pred_b, y["B"])
            choices.append((value, alpha))
            candidates.append({"method": method, "probe": kind, "ridge_alpha": alpha,
                               "selection_B_mse": value})
        b_mse, selected = min(choices)
        pred_c = probe_predict(kind, selected, x["A"], y["A"], lab["A"], x["C"], lab["C"])
        rows.append({"method": method, "probe": kind, "selected_alpha_on_B": selected,
                     "B_mse": b_mse, "C_mse": mean_sq(pred_c, y["C"]),
                     "C_mae": mae(pred_c, y["C"]),
                     "C_independent_events": independent_events(blocks["C"]["ids"],
                                                                int(blocks["C"]["span"]))})
    baseline = next(row["C_mse"] for row in rows if row["probe"] == "Global Ridge")
    for row in rows:
        row["gain_vs_global_C"] = (baseline - row["C_mse"]) / max(baseline, 1e-12)
    return rows, candidates


def draw_method(out, method, per_proto, block_names):
    safe = method.lower().replace(" ", "_").replace("+", "plus").replace("-", "_")
    matrix = np.array([[next((row["within_future_mse"] for row in per_proto
                             if row["prototype"] == k and row["block"] == b), np.nan)
                        for b in block_names] for k in range(8)])
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set(xticks=range(len(block_names)), xticklabels=block_names,
           yticks=range(8), yticklabels=range(8), xlabel="time block", ylabel="prototype",
           title=method + ": future within-MSE")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / (safe + "_prototype_time_heatmap.png"), dpi=170)
    plt.close(fig)
    phase = np.array([next(row["phase_histogram"] for row in per_proto
                           if row["prototype"] == k and row["block"] == "A") for k in range(8)])
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(phase, aspect="auto", cmap="magma")
    ax.set(xlabel="phase bin", ylabel="prototype", xticks=range(8), yticks=range(8),
           title=method + ": TRAIN phase distributions")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / (safe + "_phase_distribution.png"), dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_run_config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--allow_legacy_reference_signature", action="store_true")
    cli = parser.parse_args()
    out = Path(cli.output)
    if (out / "complete").exists():
        print("ALREADY_COMPLETE", out, flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    saved = json.loads(Path(cli.base_run_config).read_text(encoding="utf-8"))
    cli.seed = saved["seed"]
    started = time.perf_counter()
    args, model, loaders, device, a0_sha = load_experiment(cli)
    if any(p.requires_grad for p in model.parameters()):
        raise AssertionError("A0 must be frozen")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    original = collect(model, loaders["train"].dataset.dataset, args, device, cli.stride)
    peak_gpu = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    floor_std = original["x"].float().var(1, unbiased=False).add(1e-5).sqrt()
    floor = (floor_std.median(0).values * .25).clamp_min(.01)
    prepared = prepare(original, floor)
    masks, split_audit = partition(prepared, args.seq_len + args.pred_len)
    fit_cut = split_audit["cut_A"]
    series = loaders["train"].dataset.dataset.data_x[:fit_cut]
    period, period_source = estimate_period(series)
    blocks = {name: rows(prepared, mask, period, args.seq_len) for name, mask in masks.items()}
    for block in blocks.values():
        block["span"] = args.seq_len + args.pred_len
    raw_scaler = StandardScaler().fit(blocks["A"]["raw"])
    common = {name: raw_scaler.transform(block["raw"]).astype(np.float32)
              for name, block in blocks.items()}
    reference = blocks["A"]["y"].mean(axis=0)
    base_var = float(np.mean((blocks["C"]["y"] - reference) ** 2))
    summary, per_proto, per_block, probe_rows, candidate_rows, bin_rows = [], [], [], [], [], []
    for method in METHODS:
        clock = time.perf_counter()
        feature, transform_params = fit_representation(method, blocks)
        feature_seconds = time.perf_counter() - clock
        clock = time.perf_counter()
        label, own_distance, raw_coverage, center_y, center_raw, phase_weight, proto_phase = (
            fit_prototypes(method, blocks, feature))
        proto_seconds = time.perf_counter() - clock
        counts = np.bincount(label["A"], minlength=8)
        mass = counts / max(counts.sum(), 1)
        k_eff = float(np.exp(-np.sum(mass[mass > 0] * np.log(mass[mass > 0]))))
        global_a = blocks["A"]["y"].mean(axis=0)
        between = float(np.sum(mass[:, None] * (center_y - global_a) ** 2) / center_y.shape[1])
        total = float(np.mean((blocks["A"]["y"] - global_a) ** 2))
        center_distance = float(np.mean([np.mean((center_y[i] - center_y[j]) ** 2) ** .5
                                         for i in range(8) for j in range(i + 1, 8)]))
        pair_by_block, proto_benefit = {}, {}
        time_drift = []
        for block_name, block in blocks.items():
            lab = label[block_name]
            pred = center_y[lab]
            y = block["y"]
            var_ref = float(np.mean((y - global_a) ** 2))
            within = mean_sq(pred, y)
            own_min = own_distance[block_name].min(axis=1)
            raw_min = raw_coverage[block_name]
            pair, pair_frame = pair_evidence(block, lab, args.seq_len + args.pred_len,
                                             seed=SEED + (0 if block_name == "A" else 1 if block_name == "B" else 2))
            pair_by_block[block_name] = (pair, pair_frame)
            present = []
            benefit = np.full(8, np.nan)
            for k in range(8):
                chosen = lab == k
                ids_k = block["ids"][chosen]
                n_k = int(chosen.sum())
                n_event = independent_events(ids_k, args.seq_len + args.pred_len)
                future_k = y[chosen].mean(axis=0) if n_k else np.full_like(center_y[k], np.nan)
                drift = mean_sq(future_k, center_y[k]) if n_k else np.nan
                within_k = mean_sq(y[chosen], center_y[k]) if n_k else np.nan
                if n_k:
                    present.append((n_k, drift))
                    benefit[k] = mean_sq(y[chosen], global_a) - within_k
                current_phase = block["phase"][chosen]
                hist = np.histogram(current_phase, bins=np.linspace(0, 1, 9))[0].astype(float)
                hist = (hist / hist.sum()).tolist() if hist.sum() else [0.0] * 8
                phase_complex = np.exp(2j * np.pi * current_phase)
                concentration = float(abs(phase_complex.mean())) if n_k else np.nan
                same_pair = pair_frame["same"] & (pair_frame["left_label"] == k) if pair_frame is not None else None
                similarity = pair_frame["future_similarity"][same_pair] if same_pair is not None else np.array([])
                per_proto.append({"method": method, "prototype": k, "block": block_name,
                                  "support_windows": n_k, "support_fraction": n_k / max(len(lab), 1),
                                  "independent_event_support": n_event,
                                  "within_future_mse": within_k, "center_drift_mse_from_A": drift,
                                  "future_similarity_p10": float(np.quantile(similarity, .1)) if len(similarity) >= 10 else np.nan,
                                  "future_similarity_median": float(np.median(similarity)) if len(similarity) >= 10 else np.nan,
                                  "future_similarity_p90": float(np.quantile(similarity, .9)) if len(similarity) >= 10 else np.nan,
                                  "phase_concentration": concentration, "phase_histogram": hist,
                                  "similarity_pair_support": len(similarity)})
            proto_benefit[block_name] = benefit
            drift_weighted = float(sum(n * d for n, d in present) / sum(n for n, _ in present)) if present else np.nan
            if block_name != "A":
                time_drift.append(drift_weighted)
            per_block.append({"method": method, "block": block_name,
                              "window_variable_rows": len(lab),
                              "independent_events": independent_events(block["ids"], args.seq_len + args.pred_len),
                              "future_mse": within, "fve": 1 - within / max(var_ref, 1e-12),
                              "future_lift": pair["lift"], "future_lift_reason": pair["reason"],
                              "future_lift_matched_pairs": pair.get("matched_pairs", 0),
                              "mean_own_feature_distance": float(own_min.mean()),
                              "mean_common_raw_distance": float(raw_min.mean()),
                              "prototype_center_drift_mse": drift_weighted,
                              "k_eff": float(np.exp(-np.sum((np.bincount(lab, minlength=8)/len(lab) + 1e-12) *
                                                        np.log(np.bincount(lab, minlength=8)/len(lab) + 1e-12))))})
            if pair_frame is not None:
                sim = pair_frame["history_similarity"]
                edges = np.unique(np.quantile(sim, np.linspace(0, 1, 21)))
                if len(edges) >= 3:
                    bins = np.digitize(sim, edges[1:-1])
                    for i in range(len(edges) - 1):
                        selected = bins == i
                        if not selected.any():
                            continue
                        bin_rows.append({"method": method, "block": block_name, "bin": i,
                                         "count": int(selected.sum()),
                                         "history_similarity_mean": float(sim[selected].mean()),
                                         "future_similarity_mean_all": float(pair_frame["future_similarity"][selected].mean()),
                                         "future_similarity_mean_same_prototype": float(pair_frame["future_similarity"][selected & pair_frame["same"]].mean())
                                         if (selected & pair_frame["same"]).sum() >= 5 else np.nan})
        valid = np.isfinite(proto_benefit["B"]) & np.isfinite(proto_benefit["C"])
        consistency = float(np.corrcoef(proto_benefit["B"][valid], proto_benefit["C"][valid])[0, 1]) if valid.sum() >= 3 else np.nan
        clock = time.perf_counter()
        this_probe, this_candidates = mapping_probes(method, blocks, label, common)
        probe_seconds = time.perf_counter() - clock
        probe_rows.extend(this_probe)
        candidate_rows.extend(this_candidates)
        probe_gain = next(row["gain_vs_global_C"] for row in this_probe if row["probe"] == "Prototype Ridge")
        raw_metrics = cluster_metrics(common["A"], label["A"])
        c_entry = next(row for row in per_block if row["method"] == method and row["block"] == "C")
        b_entry = next(row for row in per_block if row["method"] == method and row["block"] == "B")
        c_cov = raw_coverage["C"]
        own_cov = own_distance["C"].min(axis=1)
        summary.append({"backbone": args.backbone, "dataset": args.dataset_name,
                        "pred_len": args.pred_len, "method": method, "prototype_count": 8,
                        "fit_rows": len(blocks["A"]["y"]), "selection_rows": len(blocks["B"]["y"]),
                        "evaluation_rows": len(blocks["C"]["y"]),
                        "independent_fit_events": independent_events(blocks["A"]["ids"], args.seq_len + args.pred_len),
                        "independent_eval_events": independent_events(blocks["C"]["ids"], args.seq_len + args.pred_len),
                        "FVE_C": c_entry["fve"], "FVE_B": b_entry["fve"],
                        "prototype_future_mse_C": c_entry["future_mse"],
                        "global_future_center_mse_C": base_var,
                        "future_similarity_mean_C": pair_by_block["C"][0]["within_future_similarity_mean"],
                        "future_lift_C": pair_by_block["C"][0]["lift"],
                        "future_lift_reason_C": pair_by_block["C"][0]["reason"],
                        "future_lift_matched_pairs_C": pair_by_block["C"][0].get("matched_pairs", 0),
                        "future_center_distance_rmse_A": center_distance,
                        "between_prototype_variance_A": between,
                        "between_prototype_variance_ratio_A": between / max(total, 1e-12),
                        "mean_future_center_drift_BC": float(np.nanmean(time_drift)),
                        "mapping_benefit_consistency_BC": consistency,
                        "coverage_own_mean_C": float(own_cov.mean()),
                        "coverage_own_p90_C": float(np.quantile(own_cov, .9)),
                        "coverage_own_p95_C": float(np.quantile(own_cov, .95)),
                        "coverage_own_worst10_mean_C": float(own_cov[own_cov >= np.quantile(own_cov, .9)].mean()),
                        "coverage_common_raw_mean_C": float(c_cov.mean()),
                        "coverage_common_raw_p90_C": float(np.quantile(c_cov, .9)),
                        "coverage_common_raw_p95_C": float(np.quantile(c_cov, .95)),
                        "coverage_common_raw_worst10_mean_C": float(c_cov[c_cov >= np.quantile(c_cov, .9)].mean()),
                        "k_eff_A": k_eff, "prototype_mass_A": counts.tolist(),
                        "collapsed": bool((counts < max(5, len(label["A"]) * .02)).any()),
                        "phase_concentration_A_mean": float(np.nanmean([row["phase_concentration"]
                            for row in per_proto if row["method"] == method and row["block"] == "A"])),
                        "mean_phase_hint_weight_C": phase_weight["C"],
                        "prototype_mapping_gain_vs_global_C": probe_gain,
                        "train_feature_seconds": feature_seconds,
                        "train_prototype_seconds": proto_seconds,
                        "train_probe_seconds": probe_seconds,
                        "train_total_seconds": feature_seconds + proto_seconds + probe_seconds,
                        "peak_gpu_bytes_extract_A0": peak_gpu,
                        "representation_parameters": transform_params,
                        "prototype_memory_floats": int(8 * (64 + args.pred_len)),
                        **raw_metrics})
        draw_method(out, method, per_proto, ("A", "B", "C"))
        save_csv(out / "summary.csv", summary)
        save_csv(out / "per_prototype.csv", per_proto)
        save_csv(out / "per_time_block.csv", per_block)
        save_csv(out / "mapping_probe.csv", probe_rows)
        save_csv(out / "mapping_probe_candidates.csv", candidate_rows)
        save_csv(out / "similarity_bins.csv", bin_rows)
        print("METHOD_COMPLETE", args.backbone, args.dataset_name, args.pred_len, method,
              "FVE_C", c_entry["fve"], "probe_gain_C", probe_gain, flush=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in summary:
        ax.scatter(row["prototype_mapping_gain_vs_global_C"], row["FVE_C"],
                   s=30 + 2 * row["train_total_seconds"], alpha=.8)
        ax.annotate(row["method"], (row["prototype_mapping_gain_vs_global_C"], row["FVE_C"]), fontsize=7)
    ax.axhline(0, color="gray", linewidth=.7)
    ax.axvline(0, color="gray", linewidth=.7)
    ax.set(xlabel="Prototype Ridge gain vs Global Ridge on C (higher better)",
           ylabel="Out-of-fit future variance explained on C (higher better)",
           title=f"{args.backbone}/{args.dataset_name}-{args.pred_len}: quality Pareto")
    fig.tight_layout(); fig.savefig(out / "quality_pareto.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in METHODS:
        selected = [row for row in bin_rows if row["method"] == method and row["block"] == "C"]
        ax.plot([row["history_similarity_mean"] for row in selected],
                [row["future_similarity_mean_same_prototype"] for row in selected],
                marker=".", linewidth=1, label=method)
    ax.set(xlabel="raw history similarity exp(-MSE)",
           ylabel="future similarity exp(-MSE), same prototype",
           title="Historical similarity bins vs future consistency (C)")
    ax.legend(fontsize=6); fig.tight_layout()
    fig.savefig(out / "history_similarity_future_consistency.png", dpi=170); plt.close(fig)
    audit = {"a0_sha256": a0_sha, "base_run_config": cli.base_run_config,
             "backbone": args.backbone, "dataset": args.dataset_name, "pred_len": args.pred_len,
             "seed": SEED, "stride": cli.stride, "phase_period": period,
             "phase_period_source": period_source, "split": split_audit,
             "source": "TRAIN only; A fits representation, prototype and Ridge; B selects alpha; C evaluates",
             "warning": "A0 is pretrained on full TRAIN. Window-level metrics are descriptive; independent events are conservative non-overlapping counts.",
             "elapsed_seconds": time.perf_counter() - started}
    (out / "run_config.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (out / "complete").write_text("complete\n")
    print("COMPLETE", out, flush=True)


if __name__ == "__main__":
    main()
