#!/usr/bin/env python3
"""Aggregate paired 24/48 PatchTST warmup-5 probability ablations."""

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(
    "/data/OuXiaoyu/Time-Series-Library-FPEM/results/fpem_probabilistic/"
    "grl_warmup5_patchtst_crps_ablations_20261015"
)
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "ExchangeRate")
HORIZONS = (24, 48)
HEADS = (
    "affine_deterministic", "affine_gaussian", "affine_mdn", "affine_flow",
    "residual_gaussian", "residual_lowrank", "residual_flow", "legacy_full",
)
ARMS = ("predictive", "environment_shuffled", "no_domain")
METRICS = ("ProbTS_CRPS", "CRPS_empirical")


def main():
    rows = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            file = (ROOT / "patchtst" / dataset / f"pred_{horizon}" /
                    "crps_ablation_comparison.csv")
            if not file.is_file():
                raise SystemExit(f"incomplete matrix; missing {file}")
            with file.open(newline="") as handle:
                selected = list(csv.DictReader(handle))
            if {row["head"] for row in selected} != set(HEADS):
                raise SystemExit(f"incomplete head set: {file}")
            rows.extend(selected)
    with (ROOT / "all_head_crps_ablations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for head in HEADS:
        subset = [row for row in rows if row["head"] == head]
        record = {"head": head, "configs": len(subset)}
        for metric in METRICS:
            for arm in ARMS:
                record[f"{arm}_{metric}_mean"] = statistics.mean(
                    float(row[f"{arm}_{metric}"]) for row in subset
                )
            for arm in ARMS[1:]:
                diffs = [
                    float(row[f"{arm}_{metric}"]) -
                    float(row[f"predictive_{metric}"])
                    for row in subset
                ]
                record[f"{arm}_minus_predictive_{metric}_mean"] = (
                    statistics.mean(diffs))
                record[f"{arm}_worse_{metric}_count"] = sum(
                    delta > 1e-6 for delta in diffs)
                record[f"{arm}_better_{metric}_count"] = sum(
                    delta < -1e-6 for delta in diffs)
        summary.append(record)
    with (ROOT / "summary_by_head.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (ROOT / "summary_by_head.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    (ROOT / "selection_note.txt").write_text(
        "The prior LR for each probability head was selected on TEST in an "
        "earlier experiment, then held fixed across all three warmup-5 "
        "anchors. All three anchors are frozen epoch-10 checkpoints; each "
        "head bank trains for 10 epochs and evaluates 100 samples on the "
        "full TEST split. Results are exploratory because the prior LR "
        "selection used TEST. ProbTS_CRPS and empirical CRPS use different "
        "definitions and must not be mixed.\n"
    )
    print("COMPLETE", len(rows), "paired head rows", ROOT)


if __name__ == "__main__":
    main()
