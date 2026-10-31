"""Rebuild aggregate R0–R7 CSVs from resumable case directories."""
from __future__ import annotations

import argparse
import csv
import fcntl
from pathlib import Path


FILES = ("all_results.csv", "rare_activation_statistics.csv",
         "rare_mapping_oof.csv", "rare_pairing_controls.csv",
         "common_rare_contributions.csv", "efficiency.csv")


def collect(root, filename):
    rows=[]
    for path in sorted(root.glob(f"*/{filename}")):
        with path.open(newline="",encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def write(root, filename, rows):
    if not rows:
        return
    columns=[]
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    temp=root/(filename+".tmp")
    with temp.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=columns,extrasaction="ignore")
        writer.writeheader();writer.writerows(rows)
    temp.replace(root/filename)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="results/fpem_dual_pattern_memory/rare_learning_pilot")
    args=parser.parse_args()
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    with (root/".aggregate.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for filename in FILES:
            write(root,filename,collect(root,filename))
        results=collect(root,"all_results.csv")
        controls=collect(root,"rare_pairing_controls.csv")
        lines=["# DPPM rare-pattern R0–R7 pilot", "",
               "TRAIN-only prototype learning; early purged VAL calibrates reliability; "
               "later purged VAL selects NULL offset; TEST is report-only.", "",
               "The frozen A0 TRAIN cache is in-sample, so TRAIN mapping error is not "
               "a full-system OOF estimate. Early VAL mapping error is A0-out-of-sample.", "",
               f"Completed method/case pairs: {len(results)}/24; pairing controls: {len(controls)}.", "",
               "| Case | Method | A0 MSE | Stable MSE | G1 MSE | Full MSE | Early VAL candidate gain |",
               "|---|---|---:|---:|---:|---:|---:|"]
        for row in results:
            mapping_path=root/row["case"]/row["method"]/"mapping_oos.json"
            import json
            mapping=json.loads(mapping_path.read_text(encoding="utf-8"))["metrics"]
            gain=mapping["early_val_stable_mse"]-mapping["early_val_rare_candidate_mse"]
            lines.append(f"| {row['case']} | {row['method']} | {float(row['test_a0_mse']):.6f} | "
                         f"{float(row['test_stable_mse']):.6f} | {float(row['test_g1_mse']):.6f} | "
                         f"{float(row['test_full_mse']):.6f} | {gain:+.6f} |")
        lines.extend(["", "Only if a mode has positive early VAL candidate gain, nonzero "
                      "OOF segment reliability and true-pair superiority to shuffled-pair "
                      "may it be considered evidence of a reusable rare future map. "
                      "Otherwise NULL/stable fallback is a safety result, not discovery.", ""])
        (root/"summary.md").write_text("\n".join(lines),encoding="utf-8")


if __name__=="__main__":
    main()
