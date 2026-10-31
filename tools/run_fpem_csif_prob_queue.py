"""Wait for VAL-selected B3 checkpoints and run strict OOF B4 + Flow B6/B5.

Resumable; never substitutes full-TRAIN A0 for a core-only reference.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

CASES=("patchtst_ETTh1_720","itransformer_ETTm1_720","moderntcn_ETTm2_720")
FIELDS=("case","mode","a0_mse","a0_mae","point_mse","point_mae",
        "cfip_val_nll","CRPS","PICP_95","MPIW_95","WIS","mean_MSE",
        "mean_MAE","inv_MSE","inv_MAE","best_epoch","source")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--point_root",default="results/csif_factor_innovation/pilot_20261022")
    p.add_argument("--output",default="results/csif_factor_innovation/probabilistic_20261022")
    p.add_argument("--gpu",type=int,default=3)
    p.add_argument("--samples",type=int,default=32)
    p.add_argument("--max_batches",type=int,default=0)
    p.add_argument("--wait_seconds",type=int,default=48*3600)
    opt=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    point_root=root/opt.point_root
    output=root/opt.output;output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for case in CASES:
        point=point_root/case/"B3"
        deadline=time.monotonic()+opt.wait_seconds
        while not (point/"run_complete").exists():
            if (point/"run_failed").exists():
                print(f"BLOCKED {case}: B3 failed; see point log",flush=True)
                break
            if time.monotonic()>deadline:
                print(f"TIMEOUT {case}: B3 not ready",flush=True)
                break
            time.sleep(60)
        else:
            config=json.loads((point/"run_config.json").read_text())["cli"]
            trial=output/case;trial.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,str(root/"tools/run_fpem_csif_probabilistic.py"),
                     "--base_run_config",config["base_run_config"],
                     "--point_checkpoint",str(point/"best.pt"),
                     "--output",str(trial),"--gpu",str(opt.gpu),
                     "--seed",str(config["seed"]),"--lr_head",str(config["lr_head"]),
                     "--samples",str(opt.samples)]
            if config.get("allow_legacy_reference_signature"):
                command.append("--allow_legacy_reference_signature")
            if opt.max_batches:
                command.extend(("--max_batches",str(opt.max_batches)))
            if not (trial/"run_complete").exists():
                for attempt in (1,2):
                    print(f"START strict OOF + Flow {case} attempt={attempt}",flush=True)
                    with (trial/"run.log").open("a") as log:
                        log.write("COMMAND "+" ".join(command)+"\n");log.flush()
                        result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
                    if result.returncode==0 and (trial/"run_complete").exists():break
                    if attempt==1:time.sleep(60)
                else:
                    (trial/"run_failed").write_text("two failures; inspect run.log\n")
                    print(f"FAILED {case}",flush=True)
                    continue
            comparison=json.loads((trial/"comparison.json").read_text())
            point_test=json.loads((point/"result.json").read_text())["test"]
            for mode in ("B6","B5"):
                result=json.loads((trial/mode/"result.json").read_text())
                test=result["test"]
                rows.append({"case":case,"mode":mode,
                    "a0_mse":point_test["a0_mse"],"a0_mae":point_test["a0_mae"],
                    "point_mse":point_test["mse"],"point_mae":point_test["mae"],
                    "cfip_val_nll":comparison["cfip"]["val_nll"],
                    **{key:test[key] for key in ("CRPS","PICP_95","MPIW_95",
                          "WIS","mean_MSE","mean_MAE","inv_MSE","inv_MAE")},
                    "best_epoch":result["best_epoch"],"source":str(trial/mode)})
            print(f"DONE {case}",flush=True)
            with (output/"probability_results.csv").open("w",newline="") as handle:
                writer=csv.DictWriter(handle,fieldnames=FIELDS)
                writer.writeheader();writer.writerows(rows)
    (output/"queue_complete").touch()
    print("PROBABILITY QUEUE COMPLETE",flush=True)


if __name__=="__main__":main()
