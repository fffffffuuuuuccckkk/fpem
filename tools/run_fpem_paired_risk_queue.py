"""Wait quietly for point Pilot, then conditionally run OOF interval stage."""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"results/fpem_latent_decomposition_search/pilot_20261015"
CASES=("patchtst_ETTh1_720","itransformer_ETTm1_720","moderntcn_ETTm2_720")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("case",choices=CASES)
    p.add_argument("--gpu",type=int,required=True)
    p.add_argument("--root",default="results/fpem_paired_memory/pilot_20261026")
    p.add_argument("--poll_seconds",type=int,default=90)
    p.add_argument("--timeout_hours",type=float,default=72.)
    p.add_argument("--oof_epochs",type=int,default=10)
    cli=p.parse_args()
    case=ROOT/cli.root/cli.case
    deadline=time.monotonic()+cli.timeout_hours*3600
    while not (case/"point_pilot_complete").is_file():
        if time.monotonic()>deadline:
            (case/"risk_timeout.txt").write_text("Point queue did not complete before timeout\n")
            return
        time.sleep(cli.poll_seconds)
    with (case/"all_results.csv").open(encoding="utf-8",newline="") as handle:
        rows=list(csv.DictReader(handle))
    a0=next(row for row in rows if row["mode"]=="A0")
    b1=next(row for row in rows if row["mode"]=="B1")
    b3=next(row for row in rows if row["mode"]=="B3")
    candidates=[row for row in rows if row["phase"]=="B" and row["mode"]!="B0"]
    best=min(candidates,key=lambda row:float(row["val_selection_mse"]))
    if (float(best["val_selection_mse"])>=float(a0["val_selection_mse"])
            or float(b1["val_selection_mse"])>=float(b3["val_selection_mse"])):
        (case/"risk_evidence_gate_skipped.txt").write_text(
            "Risk stage not launched: no VAL point gain over A0 or real retrieved "
            "future not better than shuffled future on VAL. This is not a "
            "positive result.\n",encoding="utf-8")
        return
    mode="B1" if best["mode"]=="B6" and best["reused_identical_run"]=="True" else best["mode"]
    point=case/f"{mode}_{best['hidden_view']}"
    if not (point/"run_complete").is_file():
        raise RuntimeError(f"selected point checkpoint missing: {point}")
    base=BASE/(f"{cli.case}_A0_complementary_gate_predictive_"
               "sub0_orth0_refnone_fp0_rank8_bias-4/run_config.json")
    out=case/"risk_intervals"
    out.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,str(ROOT/"tools/run_fpem_paired_risk.py"),
             "--base_run_config",str(base),"--point_run",str(point),
             "--output",str(out),"--anchor_cache",
             str(ROOT/"results/fpem_predictive_memory/pilot_20261024"/cli.case/"phase0_cache.pt"),
             "--memory_cache",str(case/"paired_memory_v2.pt"),
             "--gpu",str(cli.gpu),"--oof_epochs",str(cli.oof_epochs)]
    if cli.case.startswith("moderntcn"):
        command.append("--allow_legacy_reference_signature")
    env=os.environ.copy()
    env.update({"OPENBLAS_NUM_THREADS":"4","OMP_NUM_THREADS":"4",
                "MKL_NUM_THREADS":"4","NUMEXPR_NUM_THREADS":"4"})
    for attempt in (1,2):
        with (out/"run.log").open("a",encoding="utf-8") as handle:
            handle.write(f"\nATTEMPT {attempt}: {' '.join(command)}\n")
            handle.flush()
            status=subprocess.run(command,cwd=ROOT,env=env,
                                  stdout=handle,stderr=subprocess.STDOUT).returncode
        if status==0 and (out/"run_complete").is_file():
            return
        if attempt==1:time.sleep(60)
    (out/"failed.txt").write_text("Two risk attempts failed; inspect run.log\n",encoding="utf-8")
    raise RuntimeError("risk stage failed twice")


if __name__=="__main__":main()
