#!/usr/bin/env python3
"""Summarize A-current versus H-reference across all imported datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = (
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "Electricity",
    "ExchangeRate",
    "Weather",
    "Traffic",
)
VARIANTS = (
    ("A_current", "current"),
    ("B_h_reference", "h_reference"),
)


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    lines = [
        "PredictiveEnvIV H-reference multi-dataset comparison",
        "fixed: A2 K=3 seed=2021 classification complementary_gate direct_gated",
        "fixed: conditional_gain=0.2, total_epochs=10, stage_epochs=2",
        "Each dataset uses its own shared warm-up reference; A/B share its hash.",
        "",
        "dataset variant MSE MAE inv_MSE inv_minus_full main_positive "
        "conditional_positive corr_conditional_main env_separation min_env_mass "
        "env_assignment_similarity fusion_rms_ratio g_mean var_acc inv_acc "
        "corr_var_env corr_inv_env reference_sha256",
    ]
    wins = 0
    deltas = []
    for dataset in DATASETS:
        rows = []
        for label, mode in VARIANTS:
            path = args.root / dataset / label / "A2" / "metrics_and_diagnostics.json"
            row = load_metrics(path)
            expected = {
                "dataset_name": dataset,
                "predictive_env_refactor_mode": mode,
                "representation_constraint": "classification",
                "decomposition_type": "complementary_gate",
                "variant_fusion_mode": "direct_gated",
                "environment_count": 3,
                "seed": 2021,
                "optimization_epochs": 10,
                "stage_epochs": 2,
                "lambda_var_conditional_gain": 0.2,
            }
            for key, value in expected.items():
                if row.get(key) != value:
                    raise RuntimeError(
                        f"{path}: expected {key}={value!r}, got {row.get(key)!r}"
                    )
            rows.append((label, row))
            lines.append(
                f"{dataset} {label} {row['MSE']:.9f} {row['MAE']:.9f} "
                f"{row['inv_MSE']:.9f} {row['inv_minus_full']:.9f} "
                f"{row['positive_gain_ratio']:.6f} "
                f"{row['conditional_positive_ratio']:.6f} "
                f"{row['corr_conditional_gain_main_gain']:.6f} "
                f"{row['gradient_separation']:.6f} "
                f"{row['min_environment_mass']:.6f} "
                f"{row['env_assignment_similarity_to_previous']:.6f} "
                f"{row['fusion_rms_ratio_mean']:.6f} "
                f"{row['g_mean']:.6f} {row['var_acc']:.6f} "
                f"{row['inv_acc']:.6f} {row['corr_var_env']:.6f} "
                f"{row['corr_inv_env']:.6f} "
                f"{row['reference_checkpoint_sha256']}"
            )
        hashes = {row["reference_checkpoint_sha256"] for _, row in rows}
        if len(hashes) != 1 or any(row["reference_source"] != "loaded" for _, row in rows):
            raise RuntimeError(f"{dataset}: A/B did not load the same reference")
        delta = rows[1][1]["MSE"] - rows[0][1]["MSE"]
        deltas.append(delta)
        wins += int(delta < 0)
        lines.append(f"{dataset} B_minus_A_MSE {delta:+.9f}")

    lines.extend(
        [
            "",
            f"B_h_reference_wins: {wins}/{len(DATASETS)}",
            f"mean_B_minus_A_MSE: {sum(deltas) / len(deltas):+.9f}",
        ]
    )
    destination = args.root / "all_datasets_summary.txt"
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
