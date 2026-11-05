"""Build the three-backbone summary after the independent pilots complete."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"results/fpem_dual_pattern_memory/g1_r3_optimization_pilot"
CASES=("patchtst_ETTh1_720","itransformer_ETTm1_720","moderntcn_ETTm2_720")

def main():
    if not all((OUTPUT/case/"complete").exists() for case in CASES):
        raise SystemExit("Some pilots are still running; summary not written")
    lines=["# G1 + R3 optimizer pilot", "",
           "TEST is report-only; G1 and R3 settings were chosen on purged VAL.", "",
           "| Case | A0 MSE | G1 MSE | Original G1+R3 MSE | Optimized G1+R3 MSE | MAE | Gain vs G1 | Selected |",
           "|---|---:|---:|---:|---:|---:|---:|---|"]
    audit=[]
    for case in CASES:
        with (OUTPUT/case/"all_results.csv").open(newline="",encoding="utf-8") as handle:
            row=list(csv.DictReader(handle))[-1]
        lines.append("| {case} | {a0_mse:.6f} | {g1_mse:.6f} | {old:.6f} | "
                     "{new:.6f} | {mae:.6f} | {gain:+.6f} | {selected_config} |".format(
                         case=case,a0_mse=float(row["a0_mse"]),g1_mse=float(row["g1_mse"]),
                         old=float(row["original_g1_r3_mse"]),new=float(row["optimized_g1_r3_mse"]),
                         mae=float(row["optimized_g1_r3_mae"]),
                         gain=float(row["incremental_gain_vs_g1"]),selected_config=row["selected_config"]))
        with (OUTPUT/case/"lr_group_results.csv").open(newline="",encoding="utf-8") as handle:
            grid=list(csv.DictReader(handle))
        productive=sum(float(item["selection_candidate_gain"])>0 and
                       float(item["val_gain_vs_g1"])>0 and
                       float(item["reliability_mean"])>0 for item in grid)
        audit.append(f"- {case}: {len(grid)} LR settings; {productive} pass candidate/VAL/reliability checks. "
                     f"Optimizer/schedule stage {'ran' if row['advanced_search_started']=='True' else 'stopped'}.")
    lines.extend(["",*audit,"", "Original G1+R3 used the previous G1 ridge grid; the optimized "
                  "G1 anchor is reselected. Its row is therefore not an R3-only causal comparison.","",
                  "The A0 TRAIN cache is in-sample. Early VAL is R3-out-of-sample but "
                  "G1-in-sample; do not interpret that candidate metric as full-system OOF. "
                  "The later purged VAL and TEST remain future-disjoint from G1 fitting."])
    (OUTPUT/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(OUTPUT/"summary.md")

if __name__=="__main__":
    main()
