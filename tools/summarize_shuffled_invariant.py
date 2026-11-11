#!/usr/bin/env python3
"""Add paired invariant MSE/MAE to already completed shuffled-env runs."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--ablation-root", type=Path, required=True)
    args = parser.parse_args()
    count = 0
    for selected in sorted(args.base_root.rglob("best_heads_test_selected.csv")):
        relative = selected.parent.relative_to(args.base_root)
        case = args.ablation_root / relative
        if not (case / "environment_shuffled_fixed_lr_summary.csv").is_file():
            continue
        rows = []
        with selected.open(newline="") as handle:
            for source in csv.DictReader(handle):
                lr = source["best_head_lr"].replace(".", "p")
                result_path = case / "environment_shuffled" / f"head_lr_{lr}" / source["head"] / "test_metrics.json"
                result = json.loads(result_path.read_text())
                rows.append({
                    "backbone": source["backbone"], "dataset": source["dataset"],
                    "pred_len": source["pred_len"], "head": source["head"],
                    "fixed_prior_head_lr": source["best_head_lr"],
                    "original_inv_MSE": source["inv_MSE"],
                    "original_inv_MAE": source["inv_MAE"],
                    "shuffled_inv_MSE": result["inv_MSE"],
                    "shuffled_inv_MAE": result["inv_MAE"],
                    "original_ProbTS_CRPS": source["ProbTS_CRPS"],
                    "shuffled_ProbTS_CRPS": result["ProbTS_CRPS"],
                    "original_source": source["source"],
                    "shuffled_source": str(result_path),
                })
        output = case / "environment_shuffled_inv_comparison.csv"
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        count += 1
    print(f"Wrote {count} paired invariant summaries")


if __name__ == "__main__":
    main()
