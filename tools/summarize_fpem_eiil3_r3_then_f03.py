#!/usr/bin/env python3
"""Summarize fixed-R3 reverse-stage F03 pilot without loading checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.output_root.glob("**/metrics.json")):
        item = json.loads(path.read_text())
        if item.get("status") != "complete":
            continue
        pure = item["pure_R3"]["test"]
        old = item["old_F03_plus_fixed_R3"]["test"]
        blended = item["old_F03_reblended_on_R3"]["scores"]["test"]
        new = item["new_F03_after_fixed_R3"]["test"]
        trick = item.get("r3_trick", {})
        trick_test = trick.get("scores", {}).get("test", {})
        selected = item["selected"]
        rows.append({"case": item["case"],
                     "a0_test_mse": item["A0"]["test"]["mse"],
                     "pure_r3_test_mse": pure["mse"],
                     "pure_r3_test_mae": pure["mae"],
                     "old_f03_plus_fixed_r3_test_mse": old["mse"],
                     "old_f03_reblended_test_mse": blended["mse"],
                     "old_f03_reblend_alpha": item["old_F03_reblended_on_R3"]["alpha"],
                     "new_r3_aware_f03_test_mse": new["mse"],
                     "new_r3_aware_f03_test_mae": new["mae"],
                     "test_mse_gain_vs_pure_r3": pure["mse"] - new["mse"],
                     "selected_lr": selected["lr"],
                     "selected_lambda_inv": selected["lambda_inv"],
                     "selected_epoch": selected["epoch"],
                     "selected_alpha": selected["alpha"],
                     "r3_trick_test_mse": trick_test.get("mse", ""),
                     "r3_trick_test_mae": trick_test.get("mae", ""),
                     "r3_trick_selected_lr": trick.get("selected", {}).get("lr", ""),
                     "r3_trick_selected_epoch": trick.get("selected", {}).get("best_epoch", ""),
                     "r3_trick_f03_increment_rms": trick.get("train_f03_increment_rms", "")})
    columns = ("case", "a0_test_mse", "pure_r3_test_mse", "pure_r3_test_mae",
               "old_f03_plus_fixed_r3_test_mse", "old_f03_reblended_test_mse",
               "old_f03_reblend_alpha", "new_r3_aware_f03_test_mse",
               "new_r3_aware_f03_test_mae", "test_mse_gain_vs_pure_r3",
               "selected_lr", "selected_lambda_inv", "selected_epoch",
               "selected_alpha", "r3_trick_test_mse", "r3_trick_test_mae",
               "r3_trick_selected_lr", "r3_trick_selected_epoch",
               "r3_trick_f03_increment_rms")
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary cases={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
