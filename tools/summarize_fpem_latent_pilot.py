"""Collect limited decomposition pilot results; rank exclusively by VAL MSE."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def label(config):
    if config.get("experiments") == "A0":
        return "A0"
    method = config["decomposition_type"]
    if method == "complementary_gate":
        return "D1_function_preserving" if config.get("function_preserving_init") else "D0_gate"
    if method == "low_rank_subspace":
        return "D2_low_rank"
    if method == "env_guided_subspace":
        if config.get("lambda_orth", 0):
            return "D4_env_orth"
        return "D3_env_subspace"
    if method == "reference_guided":
        return "D5_reference_" + config.get("reference_mode", "environment")
    return method


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    rows = []
    for config_path in sorted(root.glob("*/run_config.json")):
        case = config_path.parent
        config = json.loads(config_path.read_text())
        experiment = config.get("experiments", "A2")
        metrics_path = case / experiment / "metrics_and_diagnostics.json"
        complete = (case / "run_complete").exists() and metrics_path.exists()
        metrics = json.loads(metrics_path.read_text()) if complete else {}
        identity = metrics.get("identity", {})
        updates = metrics.get("environment_updates", [])
        training = metrics.get("training_gate_history", [])
        last_update = updates[-1] if updates else {}
        row = {
            "case": case.name, "backbone": config.get("backbone"),
            "dataset": config.get("dataset_name"),
            "pred_len": config.get("pred_len"),
            "method": label(config),
            "environment_supervision": config.get("environment_supervision"),
            "rank": config.get("subspace_rank"),
            "lambda_sub": config.get("lambda_sub"),
            "lambda_orth": config.get("lambda_orth"),
            "reference_mode": config.get("reference_mode"),
            "status": "complete" if complete else "incomplete",
            "selection_split": metrics.get("selection_split"),
            "selected_epoch": metrics.get("selected_epoch"),
            "val_MSE": metrics.get("val/inv_MSE"),
            "val_MAE": metrics.get("val/inv_MAE"),
            "test_MSE": metrics.get("inv_MSE"),
            "test_MAE": metrics.get("inv_only_MAE", metrics.get("MAE")),
            "a0_initial_prediction_mse": identity.get("a0_initial_prediction_mse"),
            "reconstruction_error": metrics.get("reconstruction_error"),
            "inv_energy_ratio": metrics.get("inv_energy_ratio"),
            "var_energy_ratio": metrics.get("var_energy_ratio"),
            "corr_inv_env": metrics.get("corr_inv_env"),
            "corr_var_env": metrics.get("corr_var_env"),
            "orthogonal_error": metrics.get("subspace/orthogonal_error"),
            "stage_explained_ratio": last_update.get("subspace/stage_explained_ratio"),
            "min_environment_mass": last_update.get("min_environment_mass"),
            "training_epochs": len(training),
            "parameter_count": metrics.get("parameter_count"),
        }
        rows.append(row)
    fields = list(rows[0]) if rows else ["case", "status"]
    write_csv(root / "experiment_manifest.csv", rows, fields)
    write_csv(root / "all_results.csv", rows, fields)
    complete_rows = [r for r in rows if r["status"] == "complete"]
    ranked = sorted((r for r in complete_rows if r["val_MSE"] is not None),
                    key=lambda r: (r["backbone"], r["dataset"], r["val_MSE"]))
    write_csv(root / "validation_selection.csv", ranked, fields)
    write_csv(root / "test_results.csv", complete_rows, fields)
    write_csv(root / "representation_diagnostics.csv", complete_rows, fields)
    write_csv(root / "ablation_environment.csv",
              [r for r in complete_rows if r["method"].startswith(("D3", "D5"))], fields)
    write_csv(root / "probabilistic_results.csv", [], ["case", "protocol", "CRPS", "CRPS_sum", "PICP_95", "MPIW_95"])
    groups = defaultdict(list)
    for row in ranked:
        groups[(row["backbone"], row["dataset"])].append(row)
    lines = ["# FPEM latent-decomposition limited pilot",
             "", "Selection: one checkpoint per run by VAL invariant MSE; TEST is reported only after selection.",
             "These are pilot results, not a 90-configuration sweep. Probability-head tests have not run.",
             f"Completed: {len(complete_rows)}/{len(rows)} discovered cases.", ""]
    for (backbone, dataset), group in sorted(groups.items()):
        lines += [f"## {backbone} / {dataset}", "",
                  "| VAL rank | Method | Environment | VAL MSE | TEST MSE | TEST MAE |",
                  "|---:|---|---|---:|---:|---:|"]
        for rank, row in enumerate(group, 1):
            lines.append(f"| {rank} | {row['method']} | {row['environment_supervision']} | "
                         f"{row['val_MSE']:.6f} | {row['test_MSE']:.6f} | {row['test_MAE']:.6f} |")
        lines.append("")
    if len(complete_rows) < len(rows):
        lines.append("Incomplete cases remain; do not infer a final winner yet.")
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
