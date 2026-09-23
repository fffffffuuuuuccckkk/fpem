#!/usr/bin/env python3
"""Summarize the fair five-way PredictiveEnvIV multi-dataset comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


DATASETS = (
    "ETTh1", "ETTh2", "ETTm1", "ETTm2",
    "Electricity", "ExchangeRate", "Weather", "Traffic",
)

# label, experiment, refactor mode, RMS mode, fusion, conditional gain, isolated
VARIANTS = (
    ("A0_patchtst", "A0", "current", "none", "off", 0.0, False),
    ("A_current", "A2", "current", "none", "direct_gated", 0.2, False),
    ("B_h_reference", "A2", "h_reference", "none", "direct_gated", 0.2, False),
    ("C_gradient_isolated", "A2", "h_reference_grad_isolated", "none", "direct_gated", 0.2, True),
    ("D_isolated_rms", "A2", "h_reference_grad_isolated", "rms", "direct_gated", 0.2, True),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric(row: dict, key: str) -> float:
    value = row.get(key, float("nan"))
    return float(value) if value is not None else float("nan")


def format_metric(value: float, digits: int = 6) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.{digits}f}"


def load_and_validate(root: Path, dataset: str, variant: tuple, reference_hash: str) -> dict:
    label, experiment, mode, scale, fusion, conditional_gain, isolated = variant
    path = root / dataset / label / experiment / "metrics_and_diagnostics.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "dataset_name": dataset,
        "experiment": experiment,
        "predictive_env_refactor_mode": mode,
        "fusion_scale_calibration": scale,
        "variant_fusion_mode": fusion,
        "lambda_var_conditional_gain": conditional_gain,
        "z_specific_encoder_gradient_isolated": isolated,
        "representation_constraint": "classification",
        "decomposition_type": "complementary_gate",
        "environment_count": 3,
        "seed": 2021,
        "optimization_epochs": 10,
        "stage_epochs": 2,
        "lambda_h_anchor": 0.0,
        "reference_checkpoint_sha256": reference_hash,
        "reference_source": "loaded",
    }
    mismatches = {
        key: (value, row.get(key))
        for key, value in expected.items()
        if row.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"{path}: configuration mismatch: {mismatches}")
    if experiment == "A0" and row.get("environment_updates"):
        raise RuntimeError(f"{path}: A0 unexpectedly performed environment updates")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    lines = [
        "PredictiveEnvIV five-way multi-dataset comparison",
        "fixed: dataset-specific shared reference, seed=2021, K=3, 10 optimization epochs",
        "FPEM fixed: A2 classification complementary_gate direct_gated conditional_gain=0.2",
        "A0 fixed: raw PatchTST path, fusion off, no environment/decomposition constraints",
        "",
        "dataset variant MSE MAE inv_MSE inv_minus_full main_positive "
        "conditional_positive corr_conditional_main env_separation min_env_mass "
        "env_assignment_similarity fusion_rms_ratio g_mean var_acc inv_acc "
        "corr_var_env corr_inv_env delta_MSE_vs_A0 "
        "relative_MSE_improvement_vs_A0_percent delta_MAE_vs_A0 "
        "relative_MAE_improvement_vs_A0_percent reference_sha256",
    ]

    all_rows: dict[str, dict[str, dict]] = {}
    relative_mse = {variant[0]: [] for variant in VARIANTS[1:]}
    relative_mae = {variant[0]: [] for variant in VARIANTS[1:]}

    for dataset in DATASETS:
        reference_hash = file_sha256(args.root / dataset / "shared_reference.pt")
        rows = {
            variant[0]: load_and_validate(args.root, dataset, variant, reference_hash)
            for variant in VARIANTS
        }
        all_rows[dataset] = rows
        baseline = rows["A0_patchtst"]
        for label, *_ in VARIANTS:
            row = rows[label]
            mse, mae = metric(row, "MSE"), metric(row, "MAE")
            delta_mse = mse - metric(baseline, "MSE")
            delta_mae = mae - metric(baseline, "MAE")
            rel_mse = -delta_mse / metric(baseline, "MSE") * 100.0
            rel_mae = -delta_mae / metric(baseline, "MAE") * 100.0
            if label != "A0_patchtst":
                relative_mse[label].append(rel_mse)
                relative_mae[label].append(rel_mae)
            values = (
                mse, mae, metric(row, "inv_MSE"), metric(row, "inv_minus_full"),
                metric(row, "positive_gain_ratio"), metric(row, "conditional_positive_ratio"),
                metric(row, "corr_conditional_gain_main_gain"), metric(row, "gradient_separation"),
                metric(row, "min_environment_mass"), metric(row, "env_assignment_similarity_to_previous"),
                metric(row, "fusion_rms_ratio_mean"), metric(row, "g_mean"),
                metric(row, "var_acc"), metric(row, "inv_acc"),
                metric(row, "corr_var_env"), metric(row, "corr_inv_env"),
            )
            lines.append(
                f"{dataset} {label} "
                + " ".join(format_metric(value, 9) for value in values[:4])
                + " " + " ".join(format_metric(value) for value in values[4:])
                + f" {delta_mse:+.9f} {rel_mse:+.6f}"
                + f" {delta_mae:+.9f} {rel_mae:+.6f} {reference_hash}"
            )

    lines.extend(["", "overall_statistics_relative_to_A0"])
    for label in relative_mse:
        mse_values, mae_values = relative_mse[label], relative_mae[label]
        lines.append(
            f"{label} MSE_wins={sum(value > 0 for value in mse_values)}/8 "
            f"MAE_wins={sum(value > 0 for value in mae_values)}/8 "
            f"mean_relative_MSE_improvement={statistics.fmean(mse_values):+.6f}% "
            f"median_relative_MSE_improvement={statistics.median(mse_values):+.6f}% "
            f"mean_relative_MAE_improvement={statistics.fmean(mae_values):+.6f}% "
            f"median_relative_MAE_improvement={statistics.median(mae_values):+.6f}%"
        )

    lines.extend(["", "best_variant_per_dataset"])
    for dataset, rows in all_rows.items():
        best = min(rows, key=lambda label: metric(rows[label], "MSE"))
        lines.append(f"{dataset} {best}")
    overall_best = max(relative_mse, key=lambda label: statistics.fmean(relative_mse[label]))
    lines.extend(["", f"overall_best_variant_by_mean_relative_MSE: {overall_best}"])

    destination = args.root / "all_datasets_summary.txt"
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
