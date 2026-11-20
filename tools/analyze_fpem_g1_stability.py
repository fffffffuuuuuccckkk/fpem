"""Aggregate completed local G1 stability pilots without inventing missing runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", type=Path, default=Path("results/fpem_g1_stability_search_20261102"))
    args = parser.parse_args()
    root = args.result_root
    rows = []
    for result in sorted(root.glob("*/**/result.json")):
        if not (result.parent/"complete").exists():
            continue
        data = json.loads(result.read_text(encoding="utf-8"))
        winner = data["winner"]
        rows.append({"backbone": result.parts[-4], "dataset": result.parts[-3],
                     "pred_len": result.parts[-2].removeprefix("pred_"),
                     "a0_mse": data["a0"]["mse"], "old_g1_mse": data["old_g1"]["mse"],
                     "best_method": winner["trial"], "family": winner["family"],
                     "best_test_mse": winner["test_mse"], "best_test_mae": winner["test_mae"],
                     "test_selected_exploratory": True,
                     "case_dir": str(result.parent)})
    if rows:
        with (root/"pilot_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# G1 stability search: current local completion", "",
             f"Completed pilots: {len(rows)}. No missing case is reported as completed.", "",
             "All best TEST numbers below are **TEST-selected exploratory development results**, not independent generalization evidence.", ""]
    for row in rows:
        lines.append(f"- {row['backbone']} / {row['dataset']}-{row['pred_len']}: "
                     f"A0 {float(row['a0_mse']):.6f}; original G1 {float(row['old_g1_mse']):.6f}; "
                     f"{row['best_method']} {float(row['best_test_mse']):.6f} ({row['family']}).")
    lines += ["", "No strict environment-invariance claim follows from this pilot. "
              "R3 and Flow are untouched, and their cascade/OOF probability follow-ups remain separate stages.", ""]
    (root/"report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"aggregated {len(rows)} completed pilots", flush=True)


if __name__ == "__main__":
    main()
