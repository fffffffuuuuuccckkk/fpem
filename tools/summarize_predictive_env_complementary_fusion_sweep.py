#!/usr/bin/env python
"""Summarize complementary-gate lambda and direct-fusion experiments."""

import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--previous_root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for value, tag in ((0.5, "0p5"), (1.0, "1p0"), (2.0, "2p0")):
        for fusion in ("off", "direct_gated"):
            row = load(
                root / f"lambda_{tag}" / fusion / "A2" / "metrics_and_diagnostics.json"
            )
            if row["lambda_invpred"] != value:
                raise RuntimeError("lambda_invpred output does not match directory")
            rows.append(row)
    hashes = {row["reference_checkpoint_sha256"] for row in rows}
    paths = {row["reference_checkpoint"] for row in rows}
    if len(hashes) != 1 or len(paths) != 1:
        raise RuntimeError("All runs must use one shared reference")
    if any(row["reference_source"] != "loaded" for row in rows):
        raise RuntimeError("All runs must load the existing reference")
    if any(
        row["environment_count"] != 3
        or row["seed"] != 2021
        or row["decomposition_type"] != "complementary_gate"
        or row["representation_constraint"] != "classification"
        for row in rows
    ):
        raise RuntimeError("Unexpected non-complementary experiment configuration")

    previous_root = Path(args.previous_root)
    baseline = load(previous_root / "baseline" / "A0" / "metrics_and_diagnostics.json")
    previous_complementary = load(
        previous_root
        / "classification"
        / "complementary_gate"
        / "A2"
        / "metrics_and_diagnostics.json"
    )
    lines = [
        "Complementary-gate lambda_invpred and direct Z_var fusion sweep",
        f"shared_reference_sha256: {next(iter(hashes))}",
        f"shared_reference_path: {next(iter(paths))}",
        "fixed: K=3 seed=2021 classification complementary_gate",
        "complementary_activity_regularization: 0.0",
        f"previous_A0: MSE={baseline['MSE']:.9f} MAE={baseline['MAE']:.9f}",
        "previous_complementary_A2_classification: "
        f"MSE={previous_complementary['MSE']:.9f} "
        f"MAE={previous_complementary['MAE']:.9f}",
        "",
        "lambda_invpred fusion MSE MAE inv_MSE inv_minus_full g_mean g_p10 "
        "g_p50 g_p90 DeltaZ/Zinv corr_inv_env corr_var_env var_soft_ce "
        "inv_soft_ce var_acc inv_acc gate_pos gate_neg gate_abs gate_near_zero "
        "reconstruction_error inv_energy var_energy env_separation min_env_mass "
        "final_env_sim_corr",
    ]
    for row in rows:
        lines.append(
            f"{row['lambda_invpred']:.1f} {row['variant_fusion_mode']} "
            f"{row['MSE']:.9f} {row['MAE']:.9f} "
            f"{row['inv_only_MSE']:.9f} "
            f"{row['inv_only_minus_full_MSE']:.9f} "
            f"{row['g_z_mean']:.6f} {row['g_z_p10']:.6f} "
            f"{row['g_z_p50']:.6f} {row['g_z_p90']:.6f} "
            f"{row['feature_variation_to_Zinv']:.6f} "
            f"{row['corr_inv_env']:.6f} {row['corr_var_env']:.6f} "
            f"{row['var_env_soft_ce']:.6f} {row['inv_env_soft_ce']:.6f} "
            f"{row['var_env_accuracy_argmax']:.6f} "
            f"{row['inv_env_accuracy_argmax']:.6f} "
            f"{row['gate/positive_ratio']:.6f} "
            f"{row['gate/negative_ratio']:.6f} "
            f"{row['gate/abs_mean']:.6f} "
            f"{row['gate/near_zero_ratio']:.6f} "
            f"{row['reconstruction_error']:.9g} "
            f"{row['inv_energy_ratio']:.6f} "
            f"{row['var_energy_ratio']:.6f} "
            f"{row['gradient_separation']:.6f} "
            f"{row['min_environment_mass']:.6f} "
            f"{row['final_environment_similarity_correlation']:.6f}"
        )
    for fusion in ("off", "direct_gated"):
        candidates = [row for row in rows if row["variant_fusion_mode"] == fusion]
        best = min(candidates, key=lambda row: row["MSE"])
        lines.append(
            f"best_{fusion}: lambda_invpred={best['lambda_invpred']:.1f} "
            f"MSE={best['MSE']:.9f} MAE={best['MAE']:.9f}"
        )
    overall = min(rows, key=lambda row: row["MSE"])
    lines.append(
        f"best_overall: lambda_invpred={overall['lambda_invpred']:.1f} "
        f"fusion={overall['variant_fusion_mode']} "
        f"MSE={overall['MSE']:.9f} MAE={overall['MAE']:.9f}"
    )
    (root / "complementary_fusion_summary.txt").write_text(
        "\n".join(lines) + "\n"
    )


if __name__ == "__main__":
    main()
