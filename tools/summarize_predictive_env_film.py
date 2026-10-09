#!/usr/bin/env python3
"""Summarize feature-wise direct-gated versus FiLM experiments."""

import argparse
import csv
import json
from pathlib import Path


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Electricity", "ExchangeRate", "Weather", "Traffic")
BACKBONES = ("patchtst", "cyclenet")
FUSIONS = ("direct_gated", "film")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for backbone in BACKBONES:
        for dataset in DATASETS:
            loaded = {}
            for fusion in FUSIONS:
                path = args.root / backbone / dataset / fusion / "A2" / "metrics_and_diagnostics.json"
                loaded[fusion] = json.loads(path.read_text())
            baseline = loaded["direct_gated"]
            for fusion in FUSIONS:
                item = loaded[fusion]
                rows.append({
                    "backbone": backbone,
                    "dataset": dataset,
                    "fusion": fusion,
                    "MSE": item["MSE"],
                    "MAE": item["MAE"],
                    "delta_MSE_vs_direct": item["MSE"] - baseline["MSE"],
                    "delta_MAE_vs_direct": item["MAE"] - baseline["MAE"],
                    "full_MSE": item["full_MSE"],
                    "inv_MSE": item["inv_MSE"],
                    "mean_gain": item["mean_gain"],
                    "positive_gain_ratio": item["positive_gain_ratio"],
                    "DeltaZ/Zinv": item["DeltaZ/Zinv"],
                    "gamma_mean": item.get("gamma_mean"),
                    "gamma_std": item.get("gamma_std"),
                    "gamma_abs_mean": item.get("gamma_abs_mean"),
                    "beta_mean": item.get("beta_mean"),
                    "beta_std": item.get("beta_std"),
                    "beta_abs_mean": item.get("beta_abs_mean"),
                })
    fields = list(rows[0])
    with (args.root / "film_vs_direct_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [" ".join(fields)]
    lines.extend(" ".join(str(row[field]) for field in fields) for row in rows)
    (args.root / "film_vs_direct_summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

