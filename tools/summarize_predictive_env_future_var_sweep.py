#!/usr/bin/env python
"""Summarize the complementary/classification future-Zvar sweep."""

import argparse
import json
from pathlib import Path


VALUES = ((0.0, "0"), (0.01, "0p01"), (0.05, "0p05"), (0.1, "0p1"), (0.2, "0p2"))


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--previous_root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    rows = [
        load(root / f"lambda_{tag}" / "A2" / "metrics_and_diagnostics.json")
        for _, tag in VALUES
    ]
    hashes = {row["reference_checkpoint_sha256"] for row in rows}
    if len(hashes) != 1 or any(row["reference_source"] != "loaded" for row in rows):
        raise RuntimeError("Every run must load exactly one shared reference")
    for (expected, _), row in zip(VALUES, rows):
        if (
            row["lambda_future_var"] != expected
            or row["environment_count"] != 3
            or row["seed"] != 2021
            or row["decomposition_type"] != "complementary_gate"
            or row["representation_constraint"] != "classification"
            or row["variant_fusion_mode"] != "direct_gated"
            or row["lambda_invpred"] != 1.0
        ):
            raise RuntimeError("Unexpected future-Zvar sweep configuration")

    previous = load(
        Path(args.previous_root)
        / "lambda_1p0"
        / "direct_gated"
        / "A2"
        / "metrics_and_diagnostics.json"
    )
    lines = [
        "Past Zvar -> future Zvar consistency sweep",
        f"shared_reference_sha256: {next(iter(hashes))}",
        "fixed: A2 K=3 seed=2021 classification complementary_gate",
        "fixed: lambda_invpred=1.0, variant_fusion_mode=direct_gated",
        "future teacher: TRAIN-only earliest seq_len window, shared frozen-target pass",
        "inference: no future teacher input or branch",
        "historical signed_gate conclusion: var_acc exceeded projection, but MSE worsened "
        "about 9%-13%; the suspected cause was information loss from ReLU splitting.",
        f"previous_direct_gated: MSE={previous['MSE']:.9f} MAE={previous['MAE']:.9f}",
        "",
        "lambda MSE MAE inv_MSE gain_mean future_loss future_cos future_l2 "
        "pred_norm target_norm corr_future_similarity_gain var_acc inv_acc "
        "corr_var_env corr_inv_env env_separation min_env_mass final_env_sim_corr",
    ]
    for row in rows:
        lines.append(
            f"{row['lambda_future_var']:.2f} {row['MSE']:.9f} {row['MAE']:.9f} "
            f"{row['inv_only_MSE']:.9f} {row['inv_only_minus_full_MSE']:.9f} "
            f"{row['future_var_loss']:.6f} {row['future_var_cosine']:.6f} "
            f"{row['future_var_l2']:.6f} {row['future_var_pred_norm']:.6f} "
            f"{row['future_var_target_norm']:.6f} "
            f"{row['corr_future_var_similarity_with_sample_gain']:.6f} "
            f"{row['var_env_accuracy_argmax']:.6f} "
            f"{row['inv_env_accuracy_argmax']:.6f} "
            f"{row['corr_var_env']:.6f} {row['corr_inv_env']:.6f} "
            f"{row['gradient_separation']:.6f} "
            f"{row['min_environment_mass']:.6f} "
            f"{row['final_environment_similarity_correlation']:.6f}"
        )
    best = min(rows, key=lambda row: row["MSE"])
    baseline = rows[0]
    lines.extend(
        [
            "",
            f"best_lambda_future_var: {best['lambda_future_var']}",
            f"best_MSE: {best['MSE']:.9f}",
            f"best_MAE: {best['MAE']:.9f}",
            f"best_MSE_delta_vs_lambda0: {best['MSE'] - baseline['MSE']:+.9f}",
            f"best_gain_delta_vs_lambda0: "
            f"{best['inv_only_minus_full_MSE'] - baseline['inv_only_minus_full_MSE']:+.9f}",
        ]
    )
    (root / "future_var_summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
