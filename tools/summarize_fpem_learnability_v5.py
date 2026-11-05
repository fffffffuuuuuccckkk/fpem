"""Write a compact, honest per-case report after the unattended pilot."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    parser.add_argument("--root", default="results/fpem_learnability/unified_consensus_v5_20261026")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / args.root
    point = root / "point" / args.case
    probability = root / "probability" / args.case
    with (point / "all_point_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = {row["mode"]: row for row in csv.DictReader(handle)}
    chosen = json.loads((point / "point_selection.json").read_text(encoding="utf-8"))
    mode = chosen["selected_mode"]
    lines = [f"# {args.case}: unified retrieval / consensus pilot", "",
             "All point hyperparameters and the single reported checkpoint/config "
             "were chosen by the first half of VAL, not TEST.",
             "C0–C7 are exploratory probability ablations unless real paired "
             "memory also beat A0 and both negative controls on VAL.", "",
             "| Mode | VAL MSE | TEST MSE | TEST MAE |",
             "|---|---:|---:|---:|"]
    for name, row in rows.items():
        lines.append(f"| {name} | {float(row['val_mse']):.6f} | "
                     f"{float(row['test_mse']):.6f} | {float(row['test_mae']):.6f} |")
    lines.extend(["", f"VAL-selected point mode: **{mode}**. "
                  f"A0 TEST MSE {float(rows['A0']['test_mse']):.6f}; "
                  f"selected TEST MSE {float(rows[mode]['test_mse']):.6f}.",
                  f"Matched shuffled-pair B6 VAL MSE: {float(rows['B6']['val_mse']):.6f}; "
                  f"random retrieval B7: {float(rows['B7']['val_mse']):.6f}.", ""])
    audit = json.loads((point / "learnability_audit.json").read_text(encoding="utf-8"))
    supported = audit["gmm_rec"]["two_components_bic_supported"]
    lines.append(f"Reconstruction-only two-component GMM supported by BIC: {supported}. "
                 "When false, grouped interpretations are exploratory; A3 is the "
                 "no-group control.")
    interval = probability / "all_interval_results.csv"
    if interval.exists():
        lines.extend(["", "| C mode | VAL CRPS | TEST CRPS | TEST PICP95 | TEST MPIW95 |",
                      "|---|---:|---:|---:|---:|"])
        with interval.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                lines.append(f"| {row['mode']} | {float(row['val_crps']):.6f} | "
                             f"{float(row['test_crps']):.6f} | "
                             f"{float(row['PICP95_marginal']):.6f} | "
                             f"{float(row['MPIW95']):.6f} |")
    else:
        lines.extend(["", "Probability phase has not completed; no CRPS claim."])
    output = root / args.case / "summary.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
