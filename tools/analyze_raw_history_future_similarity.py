#!/usr/bin/env python3
"""History/future similarity from raw TRAIN series, without any predictor.

Windows are disjoint within each dataset/horizon. Both X and Y of a window
use its historical X mean and scale; Y is never independently normalized.
Pair-dependence and local temporal autocorrelation are reflected by resampling
short consecutive blocks of source windows for the plotted 95% intervals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist


DATASETS = {
    "ETTm2": ("dataset/ETT-small/ETTm2.csv", "ettm_train_12_months"),
    "Electricity": ("dataset/electricity/electricity.csv", "first_70_percent"),
    "Weather": ("dataset/weather/weather.csv", "first_70_percent"),
}
COLORS = {"ETTm2": "#2c6fbb", "Electricity": "#d07732", "Weather": "#368a63"}


def load_train(root: Path, name: str):
    relative, policy = DATASETS[name]
    source = root / relative
    frame = pd.read_csv(source)
    train_end = 12 * 30 * 24 * 4 if name == "ETTm2" else int(len(frame) * .7)
    if train_end > len(frame):
        raise ValueError(f"TRAIN boundary exceeds {name} length")
    columns = [column for column in frame if column != "date"]
    values = frame.loc[:train_end-1, columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    valid = np.isfinite(values).all(0) & (values.std(0) > 1e-8)
    if not valid.any():
        raise ValueError(f"no finite, nonconstant TRAIN channels in {name}")
    payload = {
        "source": str(source.resolve()), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_rows": len(frame), "train_end_exclusive": train_end, "split_policy": policy,
        "channels_total": len(columns), "channels_used": int(valid.sum()),
        "channels_dropped_nonfinite_or_constant": [str(col) for col, keep in zip(columns, valid) if not keep],
    }
    return values[:, valid].astype(np.float32), payload


def windows(train: np.ndarray, history: int, horizon: int):
    span = history + horizon
    starts = np.arange(0, len(train)-span+1, span, dtype=np.int64)
    if len(starts) < 8:
        raise ValueError(f"only {len(starts)} nonoverlapping windows for horizon {horizon}")
    x = np.stack([train[start:start+history] for start in starts])
    y = np.stack([train[start+history:start+span] for start in starts])
    mu = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    # One TRAIN-history-derived floor per channel avoids exploding distances
    # for nearly constant historical windows without normalizing future Y.
    scale_floor = np.maximum(.1 * np.median(std, axis=0, keepdims=True), 1e-4)
    scale = np.maximum(std, scale_floor)
    x_norm = (x-mu)/scale
    y_norm = (y-mu)/scale
    channel_energy = np.mean(np.square(y_norm.astype(np.float64)), axis=(0, 1))
    audit = {
        "history_len": history, "horizon": horizon, "stride": span,
        "window_count": len(starts), "pair_count": len(starts)*(len(starts)-1)//2,
        "history_scale_floor_fraction": float((std < scale_floor).mean()),
        "history_scale_floor_rule": "max(0.1*TRAIN-window median history std per channel,1e-4)",
        "normalization": "each X and its own Y share that X's per-channel mean and scale",
        "window_overlap": "none: stride equals history+horizon",
        "largest_channel_share_of_normalized_future_energy": float(channel_energy.max() / channel_energy.sum()),
        "largest_channel_index_after_filtering": int(channel_energy.argmax()),
    }
    return starts, x_norm, y_norm, audit


def pair_table(starts, x_norm, y_norm):
    n = len(starts)
    left, right = np.triu_indices(n, 1)
    # Converting each channel's MSE to similarity *before* averaging prevents
    # a single near-constant historical channel with a later spike from
    # overwhelming the multivariate relation. Retain pooled MSE as an audit.
    count = len(left)
    channels = x_norm.shape[-1]
    d_x = np.zeros(count)
    d_y = np.zeros(count)
    s_x = np.zeros(count)
    s_y = np.zeros(count)
    for channel in range(channels):
        hx = pdist(np.ascontiguousarray(x_norm[:, :, channel]), "sqeuclidean") / x_norm.shape[1]
        fy = pdist(np.ascontiguousarray(y_norm[:, :, channel]), "sqeuclidean") / y_norm.shape[1]
        d_x += hx
        d_y += fy
        s_x += 1 / (1 + hx)
        s_y += 1 / (1 + fy)
    d_x /= channels
    d_y /= channels
    s_x /= channels
    s_y /= channels
    if not (np.isfinite(d_x).all() and np.isfinite(d_y).all()):
        raise FloatingPointError("nonfinite normalized pair distance")
    return pd.DataFrame({
        "window_i": left, "window_j": right,
        "start_i": starts[left], "start_j": starts[right],
        "start_gap": starts[right]-starts[left],
        "history_mse_normalized": d_x, "future_mse_history_normalized": d_y,
        "S_X": s_x, "S_Y": s_y,
        "S_X_from_pooled_MSE_sensitivity": 1/(1+d_x),
        "S_Y_from_pooled_MSE_sensitivity": 1/(1+d_y),
    })


def moving_block_counts(rng, n: int, block: int):
    draws = int(np.ceil(n / block))
    starts = rng.integers(0, n-block+1, draws)
    sample = np.concatenate([np.arange(start, start+block) for start in starts])[:n]
    return np.bincount(sample, minlength=n)


def finite_row_mean(values):
    count = np.isfinite(values).sum(1)
    return np.divide(np.nansum(values, axis=1), count,
                     out=np.full(len(values), np.nan), where=count > 0)


def bin_and_bootstrap(pairs, window_count, *, bins=20, repetitions=600, seed=2026):
    sx, sy = pairs["S_X"].to_numpy(), pairs["S_Y"].to_numpy()
    order = np.argsort(sx, kind="mergesort")
    groups = np.array_split(order, bins)
    if min(map(len, groups)) < 2:
        raise ValueError("not enough independent-window pairs for 20 bins")
    bin_index = np.empty(len(pairs), dtype=np.int16)
    for index, members in enumerate(groups):
        bin_index[members] = index
    pairs["similarity_bin"] = bin_index + 1
    rows = []
    for index, members in enumerate(groups):
        rows.append({"bin": index+1, "n_pairs": len(members),
                     "S_X_min": float(sx[members].min()), "S_X_max": float(sx[members].max()),
                     "S_X_mean": float(sx[members].mean()),
                     "S_Y_mean": float(sy[members].mean())})
    stats = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    block = max(2, min(window_count // 3, round(window_count ** (1/3))))
    left = pairs["window_i"].to_numpy()
    right = pairs["window_j"].to_numpy()
    boot = np.full((repetitions, bins), np.nan)
    for repeat in range(repetitions):
        counts = moving_block_counts(rng, window_count, block)
        weights = counts[left] * counts[right]
        for index, members in enumerate(groups):
            w = weights[members]
            if w.sum() > 0:
                boot[repeat, index] = np.average(sy[members], weights=w)
    stats["S_Y_ci95_low"] = np.nanquantile(boot, .025, axis=0)
    stats["S_Y_ci95_high"] = np.nanquantile(boot, .975, axis=0)
    stats["bootstrap_valid_replicates"] = np.isfinite(boot).sum(0)
    valid_peak = np.isfinite(boot).all(1)
    peak_boot = boot[valid_peak].argmax(1) + 1
    peak = int(stats["S_Y_mean"].to_numpy().argmax())
    # Fixed quartile contrasts avoid optimistic inference from a selected peak.
    low = finite_row_mean(boot[:, 0:4])
    middle = finite_row_mean(boot[:, 8:14])
    tail = finite_row_mean(boot[:, 17:20])
    rise = middle-low
    fall = tail-middle
    result = {
        "bootstrap_unit": "consecutive disjoint source-window blocks, not independent pairs",
        "bootstrap_repetitions": repetitions, "bootstrap_block_windows": block,
        "peak_bin_exploratory": peak+1,
        "peak_S_X_min": float(stats.iloc[peak]["S_X_min"]),
        "peak_S_X_max": float(stats.iloc[peak]["S_X_max"]),
        "peak_bin_bootstrap_95_range": [int(np.quantile(peak_boot, .025)),
                                         int(np.quantile(peak_boot, .975))] if len(peak_boot) else None,
        "middle_minus_low": float(stats.iloc[8:14]["S_Y_mean"].mean() - stats.iloc[:4]["S_Y_mean"].mean()),
        "middle_minus_low_ci95": np.nanquantile(rise, [.025, .975]).tolist(),
        "tail_minus_middle": float(stats.iloc[17:20]["S_Y_mean"].mean() - stats.iloc[8:14]["S_Y_mean"].mean()),
        "tail_minus_middle_ci95": np.nanquantile(fall, [.025, .975]).tolist(),
        "inverted_u_supported_by_fixed_contrasts": bool(np.nanquantile(rise, .025) > 0 and
                                                         np.nanquantile(fall, .975) < 0),
        "caveat": "exploratory dyadic/time-series analysis; selected peak is not a causal or independent-test optimum",
    }
    return stats, result


def plot_one(pairs, stats, dataset, horizon, audit, destination, seed):
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    sample = pairs.sample(min(len(pairs), 12000), random_state=seed)
    ax.scatter(sample["S_X"], sample["S_Y"], s=5, alpha=.12, color=COLORS[dataset],
               rasterized=True, label=f"Window pairs (shown {len(sample):,}/{len(pairs):,})")
    ax.fill_between(stats["S_X_mean"], stats["S_Y_ci95_low"], stats["S_Y_ci95_high"],
                    color=COLORS[dataset], alpha=.22, label="95% window-block bootstrap CI")
    ax.plot(stats["S_X_mean"], stats["S_Y_mean"], "o-", ms=4, lw=2,
            color=COLORS[dataset], label="Mean in 20 equal-count bins")
    ax.set(xlabel=r"Historical similarity $S_X=\mathrm{mean}_c[1/(1+\mathrm{MSE}_t(X_{ic},X_{jc}))]$",
           ylabel=r"True-future similarity $S_Y=\mathrm{mean}_c[1/(1+\mathrm{MSE}_t(Y_{ic},Y_{jc}))]$",
           title=f"{dataset}  |  history 96, future {horizon}")
    ax.set_xlim(0, 1.01); ax.set_ylim(0, 1.01)
    ax.grid(alpha=.18)
    ax.legend(loc="best", frameon=False, fontsize=8)
    fig.text(.12, .01, f"TRAIN only · {audit['window_count']} disjoint windows · future uses history scale · no model", fontsize=8)
    fig.tight_layout(rect=(0, .03, 1, 1))
    fig.savefig(destination.with_suffix(".png"), dpi=180)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)


def plot_overview(all_stats, output):
    fig, axes = plt.subplots(3, 4, figsize=(19, 12), sharex=True, sharey=True)
    for row, dataset in enumerate(DATASETS):
        for col, horizon in enumerate((96, 192, 336, 720)):
            stats = all_stats[(dataset, horizon)]
            ax = axes[row, col]
            ax.fill_between(stats["S_X_mean"], stats["S_Y_ci95_low"],
                            stats["S_Y_ci95_high"], color=COLORS[dataset], alpha=.24)
            ax.plot(stats["S_X_mean"], stats["S_Y_mean"], "o-", ms=3, lw=1.5, color=COLORS[dataset])
            ax.set_title(f"{dataset} · horizon {horizon}")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.grid(alpha=.15)
            if col == 0:
                ax.set_ylabel("True-future similarity")
            if row == 2:
                ax.set_xlabel("Historical similarity")
    fig.suptitle("Raw TRAIN series: 20 equal-count bins, window-block bootstrap 95% CI")
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history", type=int, default=96)
    parser.add_argument("--bootstrap", type=int, default=600)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary, all_stats = [], {}
    for dataset in DATASETS:
        train, source = load_train(args.repo, dataset)
        for horizon in (96, 192, 336, 720):
            starts, x, y, audit = windows(train, args.history, horizon)
            assert np.all(np.diff(starts) >= args.history + horizon)
            pairs = pair_table(starts, x, y)
            assert pairs[["S_X", "S_Y"]].to_numpy().min() >= 0
            assert pairs[["S_X", "S_Y"]].to_numpy().max() <= 1
            stats, relation = bin_and_bootstrap(pairs, len(starts),
                                                repetitions=args.bootstrap, seed=args.seed)
            assert stats["n_pairs"].max() - stats["n_pairs"].min() <= 1
            target = args.output / dataset / f"horizon_{horizon}"
            target.mkdir(parents=True, exist_ok=True)
            pairs.to_csv(target / "all_window_pairs.csv.gz", index=False, compression="gzip")
            stats.to_csv(target / "binned_statistics.csv", index=False)
            (target / "audit.json").write_text(json.dumps({**source, **audit, **relation}, indent=2) + "\n")
            plot_one(pairs, stats, dataset, horizon, audit, target / "similarity_relation", args.seed)
            all_stats[(dataset, horizon)] = stats
            summary.append({"dataset": dataset, "horizon": horizon,
                            "window_count": audit["window_count"], "pair_count": audit["pair_count"],
                            "peak_bin": relation["peak_bin_exploratory"],
                            "peak_S_X_min": relation["peak_S_X_min"],
                            "peak_S_X_max": relation["peak_S_X_max"],
                            "middle_minus_low": relation["middle_minus_low"],
                            "tail_minus_middle": relation["tail_minus_middle"],
                            "inverted_u_supported": relation["inverted_u_supported_by_fixed_contrasts"]})
            print(f"DONE {dataset} {horizon}: {len(starts)} windows, {len(pairs)} pairs, "
                  f"peak bin {relation['peak_bin_exploratory']}, "
                  f"inverted-U support={relation['inverted_u_supported_by_fixed_contrasts']}", flush=True)
    pd.DataFrame(summary).to_csv(args.output / "summary.csv", index=False)
    plot_overview(all_stats, args.output / "overview_3_datasets_4_horizons")
    (args.output / "method.json").write_text(json.dumps({
        "history": args.history, "horizons": [96, 192, 336, 720],
        "similarity": "mean over channels of 1/(1+history-scale-normalized time-MSE); pooled MSE and its similarity also stored for sensitivity",
        "bins": "20 equal pair-count by S_X",
        "window_sampling": "nonoverlapping raw TRAIN windows, stride=history+horizon",
        "CI": "95% percentile moving-block bootstrap over source windows, fixed bins",
        "bootstrap_repetitions": args.bootstrap, "seed": args.seed,
        "no_prediction_model_or_prototype_library": True,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
