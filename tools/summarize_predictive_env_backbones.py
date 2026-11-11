#!/usr/bin/env python
"""Summarize backbone baselines versus the shared PredictiveEnvIV plugin."""

import argparse
import json
from pathlib import Path


DATASETS = (
    "ETTh1", "ETTh2", "ETTm1", "ETTm2", "Electricity", "ExchangeRate",
    "Weather", "Traffic",
)
BACKBONES = ("cyclenet", "itransformer")
VARIANTS = (
    ("A0_backbone", "A0"),
    ("A2_predictive_env", "A2"),
)


def load_rows(root):
    rows = []
    for backbone in BACKBONES:
        for dataset in DATASETS:
            baseline = None
            for label, experiment in VARIANTS:
                path = root / backbone / dataset / label / experiment / "metrics_and_diagnostics.json"
                if not path.is_file():
                    raise FileNotFoundError(path)
                row = json.loads(path.read_text())
                row.update(backbone=backbone, dataset=dataset, variant=label)
                if label == "A0_backbone":
                    baseline = row
                rows.append(row)
            if baseline is None:
                raise RuntimeError(f"missing baseline for {backbone}/{dataset}")
    return rows


def summarize(root):
    rows = load_rows(root)
    lines = [
        "CycleNet/iTransformer PredictiveEnvIV comparison",
        "data: dataset/all_datasets.zip; seed=2021; K=3; pred_len=96",
        "architecture: A2 classification + complementary_gate + direct_gated + conditional_gain=0.2",
        "",
        "backbone dataset variant MSE MAE delta_MSE_vs_A0 relative_MSE_improvement_percent "
        "delta_MAE_vs_A0 relative_MAE_improvement_percent var_acc inv_acc corr_var_env corr_inv_env",
    ]
    by_key = {(r["backbone"], r["dataset"], r["variant"]): r for r in rows}
    gains = {backbone: [] for backbone in BACKBONES}
    for backbone in BACKBONES:
        for dataset in DATASETS:
            base = by_key[(backbone, dataset, "A0_backbone")]
            for label, _ in VARIANTS:
                row = by_key[(backbone, dataset, label)]
                dmse = row["MSE"] - base["MSE"]
                dmae = row["MAE"] - base["MAE"]
                rel_mse = -100.0 * dmse / base["MSE"]
                rel_mae = -100.0 * dmae / base["MAE"]
                if label == "A2_predictive_env":
                    gains[backbone].append((rel_mse, rel_mae))
                def value(name):
                    item = row.get(name, float("nan"))
                    return float(item) if item is not None else float("nan")
                lines.append(
                    f"{backbone} {dataset} {label} {row['MSE']:.9f} {row['MAE']:.9f} "
                    f"{dmse:+.9f} {rel_mse:+.6f} {dmae:+.9f} {rel_mae:+.6f} "
                    f"{value('var_acc'):.6f} {value('inv_acc'):.6f} "
                    f"{value('corr_var_env'):.6f} {value('corr_inv_env'):.6f}"
                )
    lines.extend(("", "overall_architecture_gain"))
    for backbone, values in gains.items():
        mse = [item[0] for item in values]
        mae = [item[1] for item in values]
        lines.append(
            f"{backbone} MSE_wins={sum(v > 0 for v in mse)}/8 "
            f"mean_relative_MSE_improvement={sum(mse)/len(mse):+.6f}% "
            f"MAE_wins={sum(v > 0 for v in mae)}/8 "
            f"mean_relative_MAE_improvement={sum(mae)/len(mae):+.6f}%"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(summarize(args.root))
    print(args.output)


if __name__ == "__main__":
    main()
