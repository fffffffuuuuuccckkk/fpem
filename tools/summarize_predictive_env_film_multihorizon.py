#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "pred_len", "dataset", "MSE", "MAE", "inv_MSE", "mean_gain",
    "positive_gain_ratio", "gamma_mean", "gamma_std", "gamma_abs_mean",
    "beta_mean", "beta_std", "beta_abs_mean", "DeltaZ/Zinv",
]


def first(row, *names):
    for name in names:
        if name in row:
            return row[name]
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for path in sorted(root.glob("pred_*/*/film/A2/metrics_and_diagnostics.json")):
        metrics = json.loads(path.read_text())
        rows.append({
            "pred_len": int(path.parts[-5].split("_", 1)[1]),
            "dataset": path.parts[-4],
            "MSE": first(metrics, "MSE", "mse"),
            "MAE": first(metrics, "MAE", "mae"),
            "inv_MSE": first(metrics, "inv_MSE", "inv_mse"),
            "mean_gain": first(metrics, "mean_gain"),
            "positive_gain_ratio": first(metrics, "positive_gain_ratio"),
            "gamma_mean": first(metrics, "gamma_mean"),
            "gamma_std": first(metrics, "gamma_std"),
            "gamma_abs_mean": first(metrics, "gamma_abs_mean"),
            "beta_mean": first(metrics, "beta_mean"),
            "beta_std": first(metrics, "beta_std"),
            "beta_abs_mean": first(metrics, "beta_abs_mean"),
            "DeltaZ/Zinv": first(metrics, "DeltaZ/Zinv", "delta_z_to_z_inv"),
        })
    rows.sort(key=lambda row: (row["pred_len"], row["dataset"]))
    csv_path = root / "film_multihorizon_summary.csv"
    txt_path = root / "film_multihorizon_summary.txt"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["\t".join(FIELDS)]
    lines.extend("\t".join(str(row[field]) for field in FIELDS) for row in rows)
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(rows)} rows to {txt_path}")


if __name__ == "__main__":
    main()
