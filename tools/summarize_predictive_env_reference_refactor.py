#!/usr/bin/env python
"""Summarize H-reference, gradient-isolation, and RMS-fusion ablations."""

import argparse
import json
from pathlib import Path


VARIANTS = (
    ("A_current", "current", "none", False),
    ("B_h_reference", "h_reference", "none", False),
    ("C_gradient_isolated", "h_reference_grad_isolated", "none", True),
    ("D_isolated_rms", "h_reference_grad_isolated", "rms", True),
)


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for label, mode, scale, isolated in VARIANTS:
        row = load(root / label / "A2" / "metrics_and_diagnostics.json")
        if (
            row["predictive_env_refactor_mode"] != mode
            or row["fusion_scale_calibration"] != scale
            or row["z_specific_encoder_gradient_isolated"] != isolated
            or row["lambda_var_conditional_gain"] != 0.2
            or row["lambda_future_var"] != 0.0
            or row["lambda_var_predictive"] != 0.0
            or row["lambda_var_utility"] != 0.0
            or row["lambda_h_anchor"] != 0.0
            or row["optimization_epochs"] != 10
            or row["stage_epochs"] != 2
        ):
            raise RuntimeError(f"Unexpected configuration for {label}")
        rows.append((label, row))
    hashes = {row["reference_checkpoint_sha256"] for _, row in rows}
    if len(hashes) != 1 or any(row["reference_source"] != "loaded" for _, row in rows):
        raise RuntimeError("All variants must load the same shared reference")

    lines = [
        "PredictiveEnvIV H-reference refactor ablation",
        f"shared_reference_sha256: {next(iter(hashes))}",
        "fixed: A2 K=3 seed=2021 classification complementary_gate direct_gated",
        "fixed: conditional_gain=0.2, total_epochs=10, stage_epochs=2",
        "fixed: future_var=0, absolute_var_predictive=0, utility=0, H_anchor=0",
        "",
        "variant MSE MAE inv_MSE inv_minus_full main_positive conditional_positive "
        "corr_conditional_main h_MSE h_MAE h_drift h_corr H_norm H_norm_ratio "
        "env_separation min_env_mass env_assignment_similarity fusion_rms_ratio "
        "g_mean var_acc inv_acc corr_var_env corr_inv_env",
    ]
    for label, row in rows:
        lines.append(
            f"{label} {row['MSE']:.9f} {row['MAE']:.9f} "
            f"{row['inv_MSE']:.9f} {row['inv_minus_full']:.9f} "
            f"{row['positive_gain_ratio']:.6f} "
            f"{row['conditional_positive_ratio']:.6f} "
            f"{row['corr_conditional_gain_main_gain']:.6f} "
            f"{row['h_prediction_MSE']:.9f} "
            f"{row['h_prediction_MAE']:.9f} "
            f"{row['h_prediction_drift_from_pretrained']:.9f} "
            f"{row['h_prediction_corr_with_pretrained']:.6f} "
            f"{row['H_norm']:.6f} {row['H_norm_ratio_to_pretrained']:.6f} "
            f"{row['gradient_separation']:.6f} "
            f"{row['min_environment_mass']:.6f} "
            f"{row['env_assignment_similarity_to_previous']:.6f} "
            f"{row['fusion_rms_ratio_mean']:.6f} "
            f"{row['g_mean']:.6f} {row['var_acc']:.6f} "
            f"{row['inv_acc']:.6f} {row['corr_var_env']:.6f} "
            f"{row['corr_inv_env']:.6f}"
        )
    best_label, best = min(rows, key=lambda item: item[1]["MSE"])
    baseline = rows[0][1]
    lines.extend(
        [
            "",
            f"best_variant: {best_label}",
            f"best_MSE: {best['MSE']:.9f}",
            f"best_MAE: {best['MAE']:.9f}",
            f"best_MSE_delta_vs_A: {best['MSE'] - baseline['MSE']:+.9f}",
        ]
    )
    (root / "reference_refactor_summary.txt").write_text(
        "\n".join(lines) + "\n"
    )


if __name__ == "__main__":
    main()
