#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def make_window_features(values, seq_len):
    feats = []
    for start in range(0, len(values) - seq_len + 1):
        x = values[start:start + seq_len]
        t = np.linspace(-1.0, 1.0, seq_len, dtype=np.float64)[:, None]
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        last = x[-1]
        trend = ((x - mean) * t).mean(axis=0)
        feats.append(np.r_[mean.mean(), std.mean(), last.mean(), trend.mean(), mean.std(), std.std()])
    return np.asarray(feats, dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root_path", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--target", default="OT")
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--env_num", type=int, default=6)
    p.add_argument("--out_dir", default="foil_env")
    p.add_argument("--seed", type=int, default=2024)
    args = p.parse_args()

    root = Path(args.root_path)
    if (root / "train_data.npy").exists():
        series = {s: np.load(root / f"{s}_data.npy").astype("float32") for s in ("train", "val", "test")}
        cols = [f"var{i}" for i in range(series["train"].shape[1])]
        feats = {k: make_window_features(v, args.seq_len) for k, v in series.items()}
        source_name = str(root)
    else:
        csv_path = root / args.data_path
        df = pd.read_csv(csv_path)
        cols = [c for c in df.columns if c not in {"date"}]
        if args.target in cols:
            cols = [c for c in cols if c != args.target] + [args.target]
        values = df[cols].astype("float32").values
        n = len(df)
        n_train = int(n * 0.7)
        n_test = int(n * 0.2)
        n_val = n - n_train - n_test
        borders = {
            "train": (0, n_train),
            "val": (n_train - args.seq_len, n_train + n_val),
            "test": (n - n_test - args.seq_len, n),
        }
        feats = {k: make_window_features(values[a:b], args.seq_len) for k, (a, b) in borders.items()}
        source_name = str(csv_path)
    scaler = StandardScaler().fit(feats["train"])
    x_train = scaler.transform(feats["train"])
    km = KMeans(n_clusters=args.env_num, random_state=args.seed, n_init=20).fit(x_train)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"centers": km.cluster_centers_.astype("float32"), "columns": np.asarray(cols)}
    stem = Path(args.data_path).stem if args.data_path else Path(args.root_path).name
    summary = {"source": source_name, "seq_len": args.seq_len, "env_num": args.env_num, "splits": {}}
    for split, x in feats.items():
        z = scaler.transform(x)
        dist = ((z[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(-1)
        labels = dist.argmin(axis=1).astype("int64")
        prob = np.exp(-dist / (dist.std() + 1e-6))
        prob = prob / prob.sum(axis=1, keepdims=True)
        payload[f"{split}_labels"] = labels
        payload[f"{split}_soft"] = prob.astype("float32")
        payload[f"{split}_features"] = x.astype("float32")
        summary["splits"][split] = {"num_windows": int(len(labels)), "counts": np.bincount(labels, minlength=args.env_num).tolist()}
    np.savez_compressed(out / f"{stem}_foil_env_k{args.env_num}.npz", **payload)
    with open(out / f"{stem}_foil_env_k{args.env_num}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
