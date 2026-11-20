#!/usr/bin/env python3
"""Non-destructive comparison of original and exogenous EIIL3 variants."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHODS = ("A0", "F03", "R3", "R3(trick)", "F03+R3")


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def old_metric(case, method, old_root, trick_root):
    if method == "R3(trick)":
        source = read_json(trick_root/case/"metrics.json")
        if source is None:
            return None
        return source.get("r3_trick", {}).get("scores", {}).get("test")
    source = read_json(old_root/case/"metrics.json")
    if source is None:
        return None
    return source.get("versions", {}).get(method, {}).get("test")


def run(args):
    records = []
    for q_mode in ("rediscover", "fixed_original"):
        arm = args.result_root/q_mode
        for done in sorted(arm.glob("**/exog_complete")):
            case_dir = done.parent
            case = case_dir.relative_to(arm).as_posix()
            result = read_json(case_dir/"metrics.json")
            exog = read_json(args.result_root/"rediscover"/case/"exog_search.json")
            if result is None or exog is None:
                continue
            for method in METHODS:
                new = result.get("versions", {}).get(method, {}).get("test")
                if new is None:
                    continue
                old = old_metric(case, method, args.old_root, args.trick_root)
                record = {"case": case, "backbone": case.split("/")[0],
                          "dataset": case.split("/")[1], "q_mode": q_mode,
                          "method": method, "old_test_mse": old["mse"] if old else "",
                          "old_test_mae": old["mae"] if old else "",
                          "exog_test_mse": new["mse"], "exog_test_mae": new["mae"],
                          "mse_improvement_percent":
                          100*(old["mse"]-new["mse"])/old["mse"] if old else "",
                          "exog_lr": exog["selected"]["lr"],
                          "exog_weight_decay": exog["selected"]["weight_decay"],
                          "exog_epoch": exog["selected"]["epoch"],
                          "exog_added_parameters": exog["added_parameters"],
                          "exog_train_seconds": exog["training_seconds"],
                          "exog_peak_gpu_bytes": exog["peak_gpu_bytes"],
                          "f03_alpha": result["head_selected"]["alpha"],
                          "f03_lambda_inv": result["head_selected"]["lambda_inv"],
                          "f03_lr": result["head_selected"]["lr"],
                          "a0_parent_sha256": exog["parent_A0_sha256"],
                          "a0_exog_sha256": result["A0_sha256"],
                          "selection_split": result["selection_split"]}
                records.append(record)
    args.result_root.mkdir(parents=True, exist_ok=True)
    with (args.result_root/"all_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys() if records else
                                ("case", "backbone", "dataset", "q_mode", "method",
                                 "old_test_mse", "old_test_mae", "exog_test_mse",
                                 "exog_test_mae", "mse_improvement_percent"))
        writer.writeheader()
        writer.writerows(records)
    comparative = []
    for case in sorted({row["case"] for row in records}):
        for mode in ("rediscover", "fixed_original"):
            part = {row["method"]: row for row in records
                    if row["case"] == case and row["q_mode"] == mode}
            if not all(method in part for method in METHODS):
                continue
            r3 = float(part["R3"]["exog_test_mse"])
            joined = float(part["F03+R3"]["exog_test_mse"])
            trick = float(part["R3(trick)"]["exog_test_mse"])
            comparative.append({"case": case, "q_mode": mode,
                                "A0_exog_mse": part["A0"]["exog_test_mse"],
                                "F03_exog_mse": part["F03"]["exog_test_mse"],
                                "R3_exog_mse": r3, "R3_trick_exog_mse": trick,
                                "F03_plus_R3_exog_mse": joined,
                                "delta_F03_plus_R3_minus_R3": joined-r3,
                                "F03_plus_R3_better_than_R3": joined < r3-1e-6,
                                "F03_plus_R3_better_than_trick": joined < trick-1e-6,
                                "F03_alpha": part["F03"]["f03_alpha"],
                                "F03_lambda_inv": part["F03"]["f03_lambda_inv"]})
    with (args.result_root/"paired_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=comparative[0].keys() if comparative else
                                ("case", "q_mode", "A0_exog_mse", "F03_exog_mse",
                                 "R3_exog_mse", "R3_trick_exog_mse",
                                 "F03_plus_R3_exog_mse", "delta_F03_plus_R3_minus_R3"))
        writer.writeheader()
        writer.writerows(comparative)
    summary = ["# EIIL3 history-calendar additive embedding", "",
               f"Completed matched arms: {len(comparative)}", "",
               "All old checkpoints and result directories remain untouched.",
               "A0-exog is a new checkpoint; all five new branches use that same identity.",
               "TEST-based selection (when configured) is exploratory and not independent evidence.",
               "R3(trick) trains against Y-F03-exog but deploys on A0-exog.", ""]
    for mode in ("rediscover", "fixed_original"):
        rows = [row for row in comparative if row["q_mode"] == mode]
        summary.append(f"{mode}: {len(rows)} cases; F03+R3 beats R3 in "
                       f"{sum(row['F03_plus_R3_better_than_R3'] for row in rows)}; "
                       f"beats R3(trick) in "
                       f"{sum(row['F03_plus_R3_better_than_trick'] for row in rows)}.")
    (args.result_root/"summary.md").write_text("\n".join(summary)+"\n")
    print(json.dumps({"completed_case_arms": len(comparative),
                      "result_csv": str(args.result_root/"all_results.csv")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, default=Path(
        "results/fpem_eiil3_fulltrain_testselect_20261113/cases"))
    parser.add_argument("--trick-root", type=Path, default=Path(
        "results/fpem_eiil3_r3_f03_full_dual_20261116/cases"))
    run(parser.parse_args())
