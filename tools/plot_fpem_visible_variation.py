"""Raw-data test of whether visible historical changes distinguish future responses.

No forecasting model, environment label, or prototype bank is read. All fit
statistics and thresholds come from TRAIN; VAL selects logistic regularization.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (balanced_accuracy_score, f1_score, roc_auc_score,
                             average_precision_score, ndcg_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SOURCES = {
    "ETTh1": ("ETT-small/ETTh1.csv", "hour", 12 * 30 * 24, 4 * 30 * 24),
    "ETTm2": ("ETT-small/ETTm2.csv", "15min", 12 * 30 * 24 * 4, 4 * 30 * 24 * 4),
    "Weather": ("weather/weather.csv", "10min", None, None),
}


def boundaries(name: str, n: int) -> list[tuple[int, int]]:
    _, _, train, val = SOURCES[name]
    if train is None:
        train, test = int(n * .7), int(n * .2)
        return [(0, train), (train, n - test), (n - test, n)]
    return [(0, train), (train, train + val), (train + val, min(n, train + 2 * val))]


def pool8(x: np.ndarray) -> np.ndarray:
    return x.reshape(8, x.size // 8).mean(axis=1)


def slope(x: np.ndarray) -> float:
    t = np.linspace(-1., 1., len(x))
    return float(np.dot(x - x.mean(), t) / np.dot(t, t))


def records(values, dates, lo, hi, seq, horizon, stride, floor, channels):
    rows = {k: [] for k in ("u", "z", "v", "start", "channel", "phase", "weekday", "month", "block", "x", "future")}
    starts = np.arange(lo, hi - seq - horizon + 1, stride, dtype=np.int64)
    for ch in channels:
        series = values[:, ch]
        for st in starts:
            x = series[st:st + seq].astype(np.float64)
            y = series[st + seq:st + seq + horizon].astype(np.float64)
            med = np.median(x)
            scale = max(1.4826 * np.median(np.abs(x - med)), float(floor[ch]), 1e-4)
            early, late = x[:seq // 2], x[seq // 2:]
            recent = x[-seq // 4:]
            prior = x[:-seq // 4]
            z = [(slope(late) - slope(early)) / scale,
                 np.log((np.var(late) + floor[ch] ** 2) / (np.var(early) + floor[ch] ** 2)),
                 (late.mean() - early.mean()) / scale,
                 (recent.mean() - prior.mean()) / scale]
            normalized = (x - x.mean()) / scale
            future = (y - x[-1]) / scale
            rows["u"].append(pool8(normalized))
            rows["z"].append(z)
            rows["v"].append(np.array([a.mean() for a in np.array_split(future, 8)]))
            rows["start"].append(int(st))
            rows["channel"].append(int(ch))
            rows["phase"].append(int(dates[st + seq - 1].hour))
            rows["weekday"].append(int(dates[st + seq - 1].dayofweek))
            rows["month"].append(int(dates[st + seq - 1].month))
            rows["block"].append(min(2, int(3 * (st - lo) / max(1, hi - lo))))
            rows["x"].append((x - x[-1]) / scale)
            rows["future"].append(future)
    return {k: np.asarray(v, dtype=np.float32 if k in ("u", "z", "v", "x", "future") else np.int64)
            for k, v in rows.items()}


def make_pairs(r, span, maximum=10):
    """Select by historical u only; same channel/hour; disjoint events."""
    left, right, ud = [], [], []
    for ch in np.unique(r["channel"]):
        for phase in np.unique(r["phase"]):
            ids = np.flatnonzero((r["channel"] == ch) & (r["phase"] == phase))
            if len(ids) < 12:
                continue
            fit = NearestNeighbors(n_neighbors=min(len(ids), 160)).fit(r["u"][ids])
            dist, neigh = fit.kneighbors(r["u"][ids])
            for row, anchor in enumerate(ids):
                count = 0
                for d, j in zip(dist[row], ids[neigh[row]]):
                    if abs(int(r["start"][anchor]) - int(r["start"][j])) < span:
                        continue
                    left.append(anchor); right.append(j); ud.append(d)
                    count += 1
                    if count == maximum:
                        break
    if not left:
        raise ValueError("No independent same-channel/phase historical pairs")
    return np.asarray(left), np.asarray(right), np.asarray(ud)


def pair_arrays(r, pairs, max_ud, low=None, high=None, z_override=None):
    i, j, ud = pairs
    keep = ud <= max_ud
    i, j, ud = i[keep], j[keep], ud[keep]
    z = r["z"] if z_override is None else z_override
    u = np.abs(r["u"][i] - r["u"][j])
    dz = np.abs(z[i] - z[j])
    prod = z[i] * z[j]
    # Calendar values are known before the forecast origin; global sample
    # indices and train/val/test block IDs are never model inputs.
    phase = np.stack([np.minimum(np.abs(r["weekday"][i] - r["weekday"][j]),
                                7 - np.abs(r["weekday"][i] - r["weekday"][j])) / 3.5,
                      np.minimum(np.abs(r["month"][i] - r["month"][j]),
                                 12 - np.abs(r["month"][i] - r["month"][j])) / 6.], axis=1)
    d = np.sqrt(np.mean((r["v"][i] - r["v"][j]) ** 2, axis=1))
    label = None if low is None else np.where(d <= low, 0, np.where(d >= high, 1, -1))
    return dict(i=i, j=j, ud=ud, d=d, u=u, uz=np.concatenate((u, dz, prod), axis=1),
                up=np.concatenate((u, phase), axis=1), dz=np.linalg.norm(dz, axis=1),
                label=label)


def shuffle_z(r, rng):
    z = r["z"].copy()
    for ch in np.unique(r["channel"]):
        for phase in np.unique(r["phase"]):
            for block in range(3):
                ids = np.flatnonzero((r["channel"] == ch) & (r["phase"] == phase)
                                     & (r["block"] == block))
                if len(ids) > 1:
                    z[ids] = z[rng.permutation(ids)]
    return z


def choose_binary(train, val, field):
    tr, va = train["label"] >= 0, val["label"] >= 0
    if len(np.unique(train["label"][tr])) < 2 or len(np.unique(val["label"][va])) < 2:
        return None
    candidates = []
    for c in (.1, 1., 10.):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=300))
        model.fit(train[field][tr], train["label"][tr])
        score = roc_auc_score(val["label"][va], model.predict_proba(val[field][va])[:, 1])
        candidates.append((score, c, model))
    return max(candidates, key=lambda x: x[0])[2]


def binary_metrics(y, p):
    if len(y) == 0 or len(np.unique(y)) < 2:
        return dict(auc=np.nan, ap=np.nan, bacc=np.nan)
    return dict(auc=roc_auc_score(y, p), ap=average_precision_score(y, p),
                bacc=balanced_accuracy_score(y, p >= .5))


def rank_metrics(data, prediction, seed):
    rng = np.random.default_rng(seed)
    anchors = np.unique(data["i"])
    rows = []
    for anchor in anchors:
        ids = np.flatnonzero(data["i"] == anchor)
        if len(ids) < 5:
            continue
        truth = data["d"][ids]
        pred = prediction[ids]
        rho = spearmanr(truth, pred).statistic if np.std(pred) > 1e-10 else 0.
        rows.append((int(anchor), float(ndcg_score(truth[None], pred[None], k=5)),
                     float(rho) if np.isfinite(rho) else 0.,
                     float(len(set(np.argsort(truth)[-3:]) & set(np.argsort(pred)[-3:])) / 3)))
    return np.asarray(rows, dtype=float).reshape(-1, 4)


def event_ci(rows, r, span, seed, reps=250):
    if len(rows) < 5:
        return (np.nan, np.nan)
    events = r["start"][rows[:, 0].astype(int)] // span
    blocks = [rows[events == e, 1].mean() for e in np.unique(events)]
    if len(blocks) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    arr = np.asarray(blocks)
    boot = [arr[rng.integers(len(arr), size=len(arr))].mean() for _ in range(reps)]
    return tuple(np.percentile(boot, [2.5, 97.5]))


def auc_event_ci(data, valid, prob, r, span, seed, reps=250):
    ids = np.flatnonzero(valid)
    event = r["start"][data["i"][ids]] // span
    unique = np.unique(event)
    if len(unique) < 4:
        return (np.nan, np.nan)
    groups = [ids[event == e] for e in unique]
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(reps):
        chosen = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
        if len(np.unique(data["label"][chosen])) == 2:
            estimates.append(roc_auc_score(data["label"][chosen], prob[chosen]))
    return tuple(np.percentile(estimates, [2.5, 97.5])) if estimates else (np.nan, np.nan)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def run(args):
    root = Path(args.data_root)
    out = Path(args.output) / f"{args.dataset}_{args.horizon}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    rel, freq, _, _ = SOURCES[args.dataset]
    df = pd.read_csv(root / rel)
    dates = pd.DatetimeIndex(pd.to_datetime(df.iloc[:, 0]))
    values = df.iloc[:, 1:].to_numpy(dtype=np.float64)
    cuts = boundaries(args.dataset, len(df))
    scaler = StandardScaler().fit(values[cuts[0][0]:cuts[0][1]])
    values = scaler.transform(values)
    floor = np.maximum(np.std(values[cuts[0][0]:cuts[0][1]], axis=0) * .05, .01)
    channels = np.arange(values.shape[1]) if args.channels == "all" else np.asarray([int(x) for x in args.channels.split(",")])
    stride = args.stride or {"ETTh1": 12, "ETTm2": 24, "Weather": 24}[args.dataset]
    rec = {name: records(values, dates, lo, hi, args.seq, args.horizon, stride, floor, channels)
           for name, (lo, hi) in zip(("train", "val", "test"), cuts)}
    span = args.seq + args.horizon
    raw_pairs = {name: make_pairs(r, span) for name, r in rec.items()}
    max_ud = float(np.quantile(raw_pairs["train"][2], .90))
    pre = {name: pair_arrays(rec[name], p, max_ud) for name, p in raw_pairs.items()}
    low, high = np.quantile(pre["train"]["d"], [.25, .75])
    pairs = {name: pair_arrays(rec[name], p, max_ud, low, high) for name, p in raw_pairs.items()}
    rng = np.random.default_rng(args.seed)
    shuffled = {name: pair_arrays(rec[name], raw_pairs[name], max_ud, low, high,
                                  shuffle_z(rec[name], rng)) for name in rec}
    metrics, rank_rows = [], {}
    models = {}
    for label, field, source in (("pair_u", "u", pairs), ("pair_u_z", "uz", pairs),
                                 ("pair_u_shuffled_z", "uz", shuffled),
                                 ("pair_u_time", "up", pairs)):
        model = choose_binary(source["train"], source["val"], field)
        models[label] = model
        for split in ("val", "test"):
            dat = source[split]; valid = dat["label"] >= 0
            prob = np.full(len(dat["d"]), .5) if model is None else model.predict_proba(dat[field])[:, 1]
            result = binary_metrics(dat["label"][valid], prob[valid])
            ci = auc_event_ci(dat, valid, prob, rec[split], span, args.seed) if split == "test" else (np.nan, np.nan)
            metrics.append(dict(dataset=args.dataset, horizon=args.horizon, task="B_pair_class",
                                split=split, method=label, n=int(valid.sum()),
                                auc_ci_low=ci[0], auc_ci_high=ci[1], **result))
            if split == "test":
                rank_rows[label] = rank_metrics(dat, prob, args.seed)
    dat = pairs["test"]
    for label, pred in (("u_distance", dat["ud"]), ("z_distance", dat["dz"]),
                        ("random", rng.random(len(dat["d"])))):
        rank_rows[label] = rank_metrics(dat, pred, args.seed)
    for label, rows in rank_rows.items():
        ci = event_ci(rows, rec["test"], span, args.seed)
        metrics.append(dict(dataset=args.dataset, horizon=args.horizon, task="C_rank", split="test",
                            method=label, n=len(rows), ndcg5=float(rows[:, 1].mean()) if len(rows) else np.nan,
                            ndcg5_ci_low=ci[0], ndcg5_ci_high=ci[1],
                            spearman=float(rows[:, 2].mean()) if len(rows) else np.nan,
                            top3_recall=float(rows[:, 3].mean()) if len(rows) else np.nan))
    for block in range(3):
        chosen = dat["label"] >= 0
        chosen &= rec["test"]["block"][dat["i"]] == block
        if chosen.sum() >= 10 and len(np.unique(dat["label"][chosen])) == 2:
            for label, field, source in (("pair_u", "u", pairs), ("pair_u_z", "uz", pairs),
                                         ("pair_u_shuffled_z", "uz", shuffled)):
                block_data = source["test"]
                p = models[label].predict_proba(block_data[field])[:, 1] if models[label] else np.full(len(block_data["d"]), .5)
                use = (block_data["label"] >= 0) & (rec["test"]["block"][block_data["i"]] == block)
                result = binary_metrics(block_data["label"][use], p[use])
                metrics.append(dict(dataset=args.dataset, horizon=args.horizon, task="B_pair_class_by_time_block",
                                    split="test", method=label, block=block, n=int(use.sum()), **result))
    # A: future-response centers are fitted exclusively on TRAIN.
    cluster = KMeans(n_clusters=3, random_state=args.seed, n_init=10).fit(rec["train"]["v"])
    labels = {s: cluster.predict(rec[s]["v"]) for s in rec}
    z_shuff = {s: shuffle_z(rec[s], rng) for s in rec}
    for method in ("prior", "u", "u_z", "u_shuffled_z"):
        if method == "prior":
            guess = np.bincount(labels["train"]).argmax()
            predictor = None
        else:
            def features(s):
                return rec[s]["u"] if method == "u" else np.concatenate(
                    (rec[s]["u"], z_shuff[s] if method == "u_shuffled_z" else rec[s]["z"]), axis=1)
            choices = []
            for c in (.1, 1., 10.):
                m = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=300))
                m.fit(features("train"), labels["train"])
                choices.append((f1_score(labels["val"], m.predict(features("val")), average="macro"), m))
            predictor = max(choices, key=lambda t: t[0])[1]
        for split in ("val", "test"):
            pred = np.full(len(labels[split]), guess) if predictor is None else predictor.predict(features(split))
            metrics.append(dict(dataset=args.dataset, horizon=args.horizon, task="A_response_class",
                                split=split, method=method, n=len(pred),
                                macro_f1=f1_score(labels[split], pred, average="macro"),
                                bacc=balanced_accuracy_score(labels[split], pred)))
    # D: two purged TRAIN periods, same-capacity ridge, overlap-filtered VAL.
    train = rec["train"]; cut = int(np.median(train["start"]))
    b1 = train["start"] + span < cut; b2 = train["start"] > cut + span
    for method, field in (("u", train["u"]), ("u_z", np.concatenate((train["u"], train["z"]), axis=1))):
        if min(b1.sum(), b2.sum()) < 20:
            continue
        m1 = Ridge(alpha=10.).fit(field[b1], train["v"][b1])
        m2 = Ridge(alpha=10.).fit(field[b2], train["v"][b2])
        vfield = rec["val"]["u"] if method == "u" else np.concatenate((rec["val"]["u"], rec["val"]["z"]), axis=1)
        # Common support: distances to both fit blocks below TRAIN-only P90.
        n1 = NearestNeighbors(n_neighbors=1).fit(field[b1]); n2 = NearestNeighbors(n_neighbors=1).fit(field[b2])
        t1 = np.quantile(n1.kneighbors(field[b2], return_distance=True)[0], .90)
        t2 = np.quantile(n2.kneighbors(field[b1], return_distance=True)[0], .90)
        support = ((n1.kneighbors(vfield, return_distance=True)[0][:, 0] <= t1) &
                   (n2.kneighbors(vfield, return_distance=True)[0][:, 0] <= t2))
        if support.sum() < 20:
            continue
        p1, p2 = m1.predict(vfield[support]), m2.predict(vfield[support])
        metrics.append(dict(dataset=args.dataset, horizon=args.horizon, task="D_mapping",
                            split="val_common_support", method=method, n=int(support.sum()),
                            cross_period_disagreement=float(np.sqrt(np.mean((p1-p2)**2))),
                            later_mapping_mse=float(np.mean((p2-rec["val"]["v"][support])**2))))
    write_csv(out / "metrics.csv", metrics)
    np.savez_compressed(out / "train_future_centers.npz", centers=cluster.cluster_centers_)
    pair_table = [dict(i=int(i), j=int(j), start_i=int(rec["test"]["start"][i]),
                       start_j=int(rec["test"]["start"][j]), channel=int(rec["test"]["channel"][i]),
                       u_distance=float(u), future_divergence=float(d), label=int(y))
                  for i, j, u, d, y in zip(dat["i"], dat["j"], dat["ud"], dat["d"], dat["label"])]
    write_csv(out / "test_pairs.csv", pair_table)
    config = dict(dataset=args.dataset, horizon=args.horizon, seq=args.seq, seed=args.seed,
                  split_bounds=cuts, stride=stride, channels=channels.tolist(), span=span,
                  train_u_distance_p90=max_ud, train_future_divergence_p25=float(low),
                  train_future_divergence_p75=float(high),
                  sample_counts={s:len(rec[s]["u"]) for s in rec},
                  pair_counts={s:len(pairs[s]["d"]) for s in rec},
                  no_forecast_model=True, selection="VAL only", test_role="report only")
    (out / "config.json").write_text(json.dumps(config, indent=2))
    draw(out, args, rec["test"], dat, metrics, rank_rows)
    return metrics


def draw(out, args, test, pairs, metrics, rank_rows):
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    # Fixed, disclosed selection: median future divergence in a pre-defined
    # high-u-similarity / high-z-difference stratum, not the most dramatic pair.
    subset = (pairs["ud"] <= np.quantile(pairs["ud"], .25)) & (pairs["dz"] >= np.quantile(pairs["dz"], .75))
    available = np.flatnonzero(subset)
    if len(available):
        pick = available[np.argmin(np.abs(pairs["d"][available] - np.median(pairs["d"][available])))]
        i, j = int(pairs["i"][pick]), int(pairs["j"][pick])
        x1, x2 = test["x"][i], test["x"][j]
        y1, y2 = test["future"][i], test["future"][j]
        fig, ax = plt.subplots(figsize=(7.0, 2.5))
        t1, t2 = np.arange(-len(x1), 0), np.arange(len(y1))
        ax.plot(t1, x1, color="#707782", alpha=.8, lw=1.1)
        ax.plot(t1, x2, color="#a6acb4", alpha=.8, lw=1.1)
        ax.plot(t2, y1, color="#1a8a62", lw=1.2)
        ax.plot(t2, y2, color="#e49328", lw=1.2)
        ax.axvline(0, color="black", ls="--", lw=.8)
        ax.set(xlabel="Time relative to forecast origin", ylabel="History-scaled change",
               title=f"{args.dataset}, horizon {args.horizon}: pre-specified median-divergence example")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out / f"historical_pair.{ext}", dpi=300)
        plt.close(fig)
        (out / "selected_pair.json").write_text(json.dumps({
            "i": i, "j": j, "start_i": int(test["start"][i]), "start_j": int(test["start"][j]),
            "channel": int(test["channel"][i]), "u_distance": float(pairs["ud"][pick]),
            "future_divergence": float(pairs["d"][pick]),
            "rule": "median future divergence within bottom-quartile u distance and top-quartile z difference"}, indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 2.8))
    names = ("pair_u", "pair_u_z", "pair_u_shuffled_z", "pair_u_time")
    vals = [next((m.get("auc", np.nan) for m in metrics if m["task"] == "B_pair_class" and m["split"] == "test" and m["method"] == n), np.nan) for n in names]
    axes[0].bar(range(4), vals, color=["#7d8790", "#1a8a62", "#e49328", "#9aa5b2"])
    axes[0].set(xticks=range(4), xticklabels=["u", "u+z", "shuffle z", "time"], ylabel="TEST ROC-AUC", ylim=(0, 1))
    axes[0].tick_params(axis="x", rotation=25)
    rank_names = ("random", "u_distance", "z_distance", "pair_u_z", "pair_u_shuffled_z")
    axes[1].bar(range(5), [rank_rows[n][:, 1].mean() if len(rank_rows[n]) else np.nan for n in rank_names],
                color=["#a6acb4", "#7d8790", "#669ec4", "#1a8a62", "#e49328"])
    axes[1].set(xticks=range(5), xticklabels=["random", "u dist", "z dist", "u+z", "shuffle"],
                ylabel="TEST NDCG@5", ylim=(0, 1))
    axes[1].tick_params(axis="x", rotation=25)
    bins = np.quantile(pairs["dz"], np.linspace(0, 1, 9))
    x, y = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        use = (pairs["dz"] >= lo) & (pairs["dz"] <= hi)
        if use.sum() >= 5:
            x.append(float(np.median(pairs["dz"][use]))); y.append(float(np.median(pairs["d"][use])))
    axes[2].plot(x, y, marker="o", color="#1a8a62")
    axes[2].set(xlabel="Visible-change distance |zᵢ-zⱼ|", ylabel="Median future-response divergence")
    fig.suptitle(f"{args.dataset}-{args.horizon}: history-visible variation test", y=1.03)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"classification_ranking_summary.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(SOURCES), required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--seq", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--channels", default="all")
    parser.add_argument("--stride", type=int, default=0)
    parser.add_argument("--data-root", default="dataset/all_datasets")
    parser.add_argument("--output", default="results/fpem_visible_variation")
    run(parser.parse_args())
