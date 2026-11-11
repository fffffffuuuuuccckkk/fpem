"""Resumable A0/A1-A5/B0-B7 pilot; choose checkpoints on VAL only."""
from __future__ import annotations

import argparse
import csv
import json
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
    p.add_argument("--output",default="results/fpem_paired_memory/pilot_20261026")
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--max_batches",type=int,default=0)
    cli=p.parse_args()
    base=BASE/(f"{cli.case}_A0_complementary_gate_predictive_"
               "sub0_orth0_refnone_fp0_rank8_bias-4/run_config.json")
    if not base.is_file():raise FileNotFoundError(base)
    root=ROOT/cli.output/cli.case;root.mkdir(parents=True,exist_ok=True)
    anchor=ROOT/"results/fpem_predictive_memory/pilot_20261024"/cli.case/"phase0_cache.pt"
    memory=root/"paired_memory_v2.pt"
    legacy=["--allow_legacy_reference_signature"] if cli.case.startswith("moderntcn") else []
    env=os.environ.copy()
    env.update({"OPENBLAS_NUM_THREADS":"4","OMP_NUM_THREADS":"4",
                "MKL_NUM_THREADS":"4","NUMEXPR_NUM_THREADS":"4"})

    def run(mode,view="mixed"):
        path=root/f"{mode}_{view}"
        path.mkdir(parents=True,exist_ok=True)
        if (path/"run_complete").is_file():
            return json.loads((path/"result.json").read_text(encoding="utf-8"))
        command=[sys.executable,str(ROOT/"tools/run_fpem_paired_memory_pilot.py"),
                 "--base_run_config",str(base),"--output",str(path),
                 "--mode",mode,"--hidden_view",view,"--gpu",str(cli.gpu),
                 "--epochs",str(cli.epochs),"--memory_cache",str(memory),
                 "--anchor_cache",str(anchor),"--max_batches",str(cli.max_batches)]+legacy
        for attempt in (1,2):
            with (path/"run.log").open("a",encoding="utf-8") as handle:
                handle.write(f"\nATTEMPT {attempt}: {' '.join(command)}\n")
                handle.flush()
                status=subprocess.run(command,cwd=ROOT,env=env,
                                      stdout=handle,stderr=subprocess.STDOUT).returncode
            if status==0 and (path/"run_complete").is_file():
                return json.loads((path/"result.json").read_text(encoding="utf-8"))
            if attempt==1:time.sleep(60)
        (path/"failed.txt").write_text("Two attempts failed; see run.log\n",encoding="utf-8")
        raise RuntimeError(f"{cli.case} {mode} failed twice")

    rows=[]
    def add(phase,mode,view,result,reused=False):
        selected=result["val_selection"]
        test=result["test"]
        rows.append({"phase":phase,"mode":mode,"hidden_view":view,
                     "best_epoch":result.get("best_epoch",0),
                     "val_selection_mse":selected["mse"],
                     "val_selection_mae":selected["mae"],
                     "test_mse":test["mse"],"test_mae":test["mae"],
                     "a0_test_mse":test["a0_mse"],"a0_test_mae":test["a0_mae"],
                     "point_mse":test["point_mse"],
                     "correction_abs_mean":test["correction_abs_mean"],
                     "reused_identical_run":reused})

    a0=run("A0");add("A","A0","none",a0)
    a_rows=[]
    for mode in ("A1","A2","A3","A4","A5"):
        result=run(mode)
        add("A",mode,{"A1":"random","A2":"all","A3":"common",
                      "A4":"rare","A5":"mixed"}[mode],result)
        a_rows.append((result["val_selection"]["mse"],mode,result))
    _,winner_mode,winner=min(a_rows,key=lambda item:item[0])
    view={"A1":"random","A2":"all","A3":"common",
          "A4":"rare","A5":"mixed"}[winner_mode]
    add("B","B0",view,winner,reused=True)
    for mode in ("B1","B2","B3","B4","B5","B6","B7"):
        if mode=="B6" and view=="mixed":
            result=json.loads((root/f"B1_{view}/result.json").read_text(encoding="utf-8"))
            add("B",mode,view,result,reused=True)
        else:
            result=run(mode,view)
            add("B",mode,view,result)
    with (root/"all_results.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    best_test=min(rows,key=lambda item:item["test_mse"])
    evidence=(next(r for r in rows if r["mode"]=="B1")["val_selection_mse"]
              <next(r for r in rows if r["mode"]=="B3")["val_selection_mse"])
    (root/"summary.md").write_text(
        f"# {cli.case} paired-memory Pilot\n\n"
        f"A0 TEST MSE: {a0['test']['mse']:.6f}; best H concat by VAL: {winner_mode} ({view}).\n\n"
        f"B1 real future better than B3 shuffled on VAL: {evidence}.\n"
        f"Exploratory TEST-best (not a valid selection rule): {best_test['mode']} "
        f"{best_test['test_mse']:.6f}.\n\n"
        "All checkpoints selected on the first half of VAL; the second half is held for interval calibration. "
        "No test-based checkpoint selection. TRAIN query memory only uses earlier, fully observed futures. "
        "Risk/interval stage is separate and must use point-model OOF residuals.\n",
        encoding="utf-8")
    (root/"point_pilot_complete").touch()


if __name__=="__main__":main()
