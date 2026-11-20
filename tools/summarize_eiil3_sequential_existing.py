#!/usr/bin/env python3
"""Read-only summary of completed EIIL3 sequential-search point results."""
import argparse
import collections
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--csv", type=Path, required=True)
    args = p.parse_args()
    rows = []
    for path in args.root.rglob("point_metrics.json"):
        if not (path.parent / "point_complete").exists():
            continue
        value = json.loads(path.read_text())
        if value.get("status") != "complete" or "I3_R3_linear" not in value.get("scores", {}).get("test", {}):
            continue
        parts = value["case"].split("/")
        scores = value["scores"]["test"]
        a0, inv, final = (scores[name] for name in ("A0", "I3", "I3_R3_linear"))
        row = {"case": value["case"], "backbone": parts[0], "dataset": parts[1],
               "pred_len": int(parts[2].split("_")[1]),
               "a0_mse": a0["mse"], "a0_mae": a0["mae"],
               "inv_mse": inv["mse"], "inv_mae": inv["mae"],
               "final_mse": final["mse"], "final_mae": final["mae"],
               "inv_gain_pct": 100 * (a0["mse"] - inv["mse"]) / a0["mse"],
               "final_gain_pct": 100 * (a0["mse"] - final["mse"]) / a0["mse"],
               "r3_gain_pct": 100 * (inv["mse"] - final["mse"]) / inv["mse"],
               "head_lr": value["head_lr_selected"],
               "variation_lr": value["variation_lr_selected"],
               "head_candidates": value["head_candidates_evaluated"],
               "variation_candidates": value["variation_candidates_evaluated"],
               "A0_sha256": value["A0_sha256"],
               "q_sha256": value["q_sha256"]}
        rows.append(row)
    rows.sort(key=lambda row: (row["backbone"], row["dataset"], row["pred_len"]))
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["case"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for backbone in ["ALL", *sorted({row["backbone"] for row in rows})]:
        group = rows if backbone == "ALL" else [row for row in rows if row["backbone"] == backbone]
        if not group:
            continue
        summary[backbone] = {"complete": len(group),
            "inv_better_a0_mse": sum(row["inv_mse"] < row["a0_mse"] - 1e-6 for row in group),
            "final_better_a0_mse": sum(row["final_mse"] < row["a0_mse"] - 1e-6 for row in group),
            "final_better_inv_mse": sum(row["final_mse"] < row["inv_mse"] - 1e-6 for row in group),
            "final_better_a0_mae": sum(row["final_mae"] < row["a0_mae"] - 1e-6 for row in group),
            "mean_a0_mse": sum(row["a0_mse"] for row in group) / len(group),
            "mean_inv_mse": sum(row["inv_mse"] for row in group) / len(group),
            "mean_final_mse": sum(row["final_mse"] for row in group) / len(group),
            "mean_final_gain_pct": sum(row["final_gain_pct"] for row in group) / len(group),
            "mean_r3_gain_pct": sum(row["r3_gain_pct"] for row in group) / len(group),
            "head_early_stop": sum(row["head_candidates"] < 4 for row in group),
            "variation_early_stop": sum(row["variation_candidates"] < 4 for row in group)}
    print(json.dumps({"rows": len(rows), "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
