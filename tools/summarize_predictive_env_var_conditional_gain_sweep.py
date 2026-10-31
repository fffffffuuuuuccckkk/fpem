#!/usr/bin/env python
"""Summarize paired-versus-shuffled conditional Zvar ranking experiments."""

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
            row["lambda_var_conditional_gain"] != expected
            or row["var_conditional_margin"] != 0.0
            or row["lambda_var_predictive"] != 0.0
            or row["lambda_var_utility"] != 0.0
            or row["lambda_future_var"] != 0.0
            or row["lambda_invpred"] != 1.0
            or row["environment_count"] != 3
            or row["seed"] != 2021
            or row["decomposition_type"] != "complementary_gate"
            or row["representation_constraint"] != "classification"
            or row["variant_fusion_mode"] != "direct_gated"
        ):
            raise RuntimeError("Unexpected conditional-gain sweep configuration")

    previous = load(
        Path(args.previous_root) / "lambda_0" / "A2" / "metrics_and_diagnostics.json"
    )
    lines = [
        "Paired-vs-shuffled conditional Zvar gain sweep",
        f"shared_reference_sha256: {next(iter(hashes))}",
        "fixed: A2 K=3 seed=2021 classification complementary_gate direct_gated",
        "fixed: lambda_invpred=1, margin=0, all other Zvar auxiliary losses=0",
        "negative: deterministic cyclic derangement; shuffled loss stop-gradient",
        "main fusion: Z_fused = Z_inv + r * Z_var",
        f"previous_lambda0: MSE={previous['MSE']:.9f} MAE={previous['MAE']:.9f}",
        "",
        "lambda MSE MAE inv_MSE inv_minus_full mean_gain median_gain "
        "positive_gain pair_MSE shuffle_MSE conditional_gain conditional_median "
        "conditional_positive conditional_p10 conditional_p50 conditional_p90 "
        "corr_conditional_main_gain corr_r_gain g_mean g_p10 g_p50 g_p90 "
        "var_acc inv_acc corr_var_env corr_inv_env",
    ]
    for row in rows:
        lines.append(
            f"{row['lambda_var_conditional_gain']:.2f} {row['MSE']:.9f} "
            f"{row['MAE']:.9f} {row['inv_MSE']:.9f} "
            f"{row['inv_minus_full']:.9f} {row['mean_gain']:.9f} "
            f"{row['median_gain']:.9f} {row['positive_gain_ratio']:.6f} "
            f"{row['conditional_pair_MSE']:.9f} "
            f"{row['conditional_shuffle_MSE']:.9f} "
            f"{row['conditional_gain_mean']:.9f} "
            f"{row['conditional_gain_median']:.9f} "
            f"{row['conditional_positive_ratio']:.6f} "
            f"{row['conditional_gain_p10']:.9f} "
            f"{row['conditional_gain_p50']:.9f} "
            f"{row['conditional_gain_p90']:.9f} "
            f"{row['corr_conditional_gain_main_gain']:.6f} "
            f"{row['corr_reliability_gain']:.6f} {row['g_mean']:.6f} "
            f"{row['g_p10']:.6f} {row['g_p50']:.6f} {row['g_p90']:.6f} "
            f"{row['var_acc']:.6f} {row['inv_acc']:.6f} "
            f"{row['corr_var_env']:.6f} {row['corr_inv_env']:.6f}"
        )
    baseline = rows[0]
    best = min(rows, key=lambda row: row["MSE"])
    lines.extend(
        [
            "",
            f"best_lambda_var_conditional_gain: "
            f"{best['lambda_var_conditional_gain']}",
            f"best_MSE: {best['MSE']:.9f}",
            f"best_MAE: {best['MAE']:.9f}",
            f"best_MSE_delta_vs_lambda0: {best['MSE'] - baseline['MSE']:+.9f}",
            f"best_conditional_positive_delta_vs_lambda0: "
            f"{best['conditional_positive_ratio'] - baseline['conditional_positive_ratio']:+.6f}",
            f"best_main_positive_delta_vs_lambda0: "
            f"{best['positive_gain_ratio'] - baseline['positive_gain_ratio']:+.6f}",
            f"lambda0_MSE_delta_vs_previous: "
            f"{baseline['MSE'] - previous['MSE']:+.9f}",
        ]
    )
    (root / "var_conditional_gain_summary.txt").write_text(
        "\n".join(lines) + "\n"
    )


if __name__ == "__main__":
    main()
