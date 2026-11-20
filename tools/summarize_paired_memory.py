"""Combine completed staged pilots without selecting parameters on TEST."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

CASES=("patchtst_ETTh1_720","itransformer_ETTm1_720","moderntcn_ETTm2_720")


def read_csv(path):
    with path.open(encoding="utf-8",newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",default="results/fpem_paired_memory/pilot_20261026")
    p.add_argument("--wait",action="store_true")
    p.add_argument("--timeout_hours",type=float,default=72.)
    cli=p.parse_args()
    root=Path(cli.root)
    if cli.wait:
        deadline=time.monotonic()+cli.timeout_hours*3600
        while True:
            ready=all((root/c/"point_pilot_complete").is_file() and
                      ((root/c/"risk_intervals/run_complete").is_file() or
                       (root/c/"risk_evidence_gate_skipped.txt").is_file() or
                       (root/c/"risk_intervals/failed.txt").is_file()) for c in CASES)
            if ready:break
            if time.monotonic()>deadline:
                (root/"summary_timeout.txt").write_text("Timed out waiting for staged queues\n")
                return
            time.sleep(120)
    point=[];prob=[];lines=["# Paired historical/future memory staged Pilot","",
        "Checkpoint and method comparison use the first half of VAL; the latter half is independent interval calibration. TEST is exploratory only.",""]
    for case in CASES:
        case_path=root/case
        if not (case_path/"all_results.csv").is_file():continue
        rows=read_csv(case_path/"all_results.csv")
        for row in rows:
            point.append({"case":case,**row})
        a0=next(r for r in rows if r["mode"]=="A0")
        best_a=min((r for r in rows if r["phase"]=="A" and r["mode"]!="A0"),
                   key=lambda r:float(r["val_selection_mse"]))
        best_b=min((r for r in rows if r["phase"]=="B" and r["mode"]!="B0"),
                   key=lambda r:float(r["val_selection_mse"]))
        b1=next(r for r in rows if r["mode"]=="B1")
        b3=next(r for r in rows if r["mode"]=="B3")
        lines += [f"## {case}","",
            f"- A0 TEST MSE/MAE: {float(a0['test_mse']):.6f}/{float(a0['test_mae']):.6f}",
            f"- VAL-selected H concat: {best_a['mode']} → TEST {float(best_a['test_mse']):.6f}",
            f"- VAL-selected future correction: {best_b['mode']} → TEST {float(best_b['test_mse']):.6f}",
            f"- Paired real future B1 vs shuffled B3 VAL MSE: {float(b1['val_selection_mse']):.6f} vs {float(b3['val_selection_mse']):.6f}",""]
        risk=case_path/"risk_intervals/interval_results.csv"
        if risk.is_file():
            for row in read_csv(risk):prob.append({"case":case,**row})
            fixed=next(r for r in read_csv(risk) if r["model"]=="fixed")
            full=next(r for r in read_csv(risk) if r["model"]=="full")
            lines.append(f"- Risk intervals: fixed/full trajectory PICP95 {float(fixed['picp95_trajectory']):.4f}/{float(full['picp95_trajectory']):.4f}; width {float(fixed['mpiw95']):.4f}/{float(full['mpiw95']):.4f}")
        elif (case_path/"risk_evidence_gate_skipped.txt").is_file():
            lines.append("- Risk stage skipped by predeclared VAL evidence gate.")
        elif (case_path/"risk_intervals/failed.txt").is_file():
            lines.append("- Risk stage failed; inspect its log.")
        lines.append("")
    if point:
        with (root/"all_point_results.csv").open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=point[0].keys());writer.writeheader();writer.writerows(point)
    if prob:
        with (root/"all_interval_results.csv").open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=prob[0].keys());writer.writeheader();writer.writerows(prob)
    lines += ["## Interpretation boundary","",
        "Common/Rare denotes TRAIN historical density, not invariant/variant ground truth. The risk score estimates mapping-transfer/error scale, not a calibrated probability of a genuinely unseen environment. Arbitrary distribution shift voids ordinary split-conformal coverage guarantees.",
        "P1-P5 retrieval variants and multi-seed/full-matrix expansion are deferred until the real-future versus shuffled-future mechanism is validated. No result is removed for poor performance.",""]
    (root/"summary.md").write_text("\n".join(lines),encoding="utf-8")


if __name__=="__main__":main()
