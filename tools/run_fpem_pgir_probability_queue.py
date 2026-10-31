"""Resumable P4 probability pilots, each waiting for its selected P2 point model."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

CASES=("patchtst_ETTh1_720","itransformer_ETTm1_720","moderntcn_ETTm2_720")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--selected",default="results/fpem_orthogonal_spectral_full_testlr_20261020/selected_by_test.csv")
    p.add_argument("--output",default="results/fpem_pgir/pilot_20261022")
    p.add_argument("--gpu",type=int,default=3)
    p.add_argument("--max_batches",type=int,default=0)
    p.add_argument("--core_epochs",type=int,default=10)
    p.add_argument("--mask_epochs",type=int,default=10)
    p.add_argument("--prior_epochs",type=int,default=10)
    p.add_argument("--flow_epochs",type=int,default=10)
    p.add_argument("--wait_seconds",type=int,default=60)
    cli=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=root/cli.output;output.mkdir(parents=True,exist_ok=True)
    selected={(r["case"],r["method"]):r for r in
              csv.DictReader((root/cli.selected).open(newline=""))}
    for case in CASES:
        p2=output/case/"P2"
        trial=output/case/"P4";trial.mkdir(parents=True,exist_ok=True)
        if (trial/"run_complete").exists():
            print(f"SKIP complete {case}/P4",flush=True);continue
        while not (p2/"run_complete").exists():
            if (p2/"run_failed").exists():
                (trial/"run_failed").write_text("P2 point model failed\n")
                break
            time.sleep(cli.wait_seconds)
        if not (p2/"run_complete").exists():continue
        source=selected[(case,"O3")]
        saved=json.loads((root/source["source"]/"run_config.json").read_text())["cli"]
        command=[sys.executable,str(root/"tools/run_fpem_pgir_probabilistic.py"),
                 "--base_run_config",saved["base_run_config"],
                 "--point_checkpoint",str(p2/"best.pt"),
                 "--output",str(trial),"--gpu",str(cli.gpu),
                 "--seed",str(saved["seed"]),
                 "--core_epochs",str(cli.core_epochs),"--mask_epochs",str(cli.mask_epochs),
                 "--prior_epochs",str(cli.prior_epochs),"--flow_epochs",str(cli.flow_epochs)]
        if saved.get("allow_legacy_reference_signature"):
            command.append("--allow_legacy_reference_signature")
        if cli.max_batches:command += ["--max_batches",str(cli.max_batches)]
        for attempt in (1,2):
            print(f"START {case}/P4 attempt={attempt}",flush=True)
            with (trial/"run.log").open("a") as log:
                log.write("COMMAND "+" ".join(command)+"\n");log.flush()
                status=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
            if status.returncode==0 and (trial/"run_complete").exists():break
            print(f"FAILED {case}/P4 rc={status.returncode}",flush=True)
            if attempt==1:time.sleep(60)
        else:
            (trial/"run_failed").write_text("two failures; inspect run.log\n")
    print("P4 WORKER COMPLETE",flush=True)


if __name__=="__main__":main()
