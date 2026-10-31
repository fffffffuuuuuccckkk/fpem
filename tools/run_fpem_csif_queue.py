"""Resumable 3-backbone B1--B3 CSIF pilot, reusing verified A0 checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

CASES = ("patchtst_ETTh1_720", "itransformer_ETTm1_720", "moderntcn_ETTm2_720")
MODES = ("B1", "B2", "B3")
FIELDS = ("case", "mode", "a0_mse", "a0_mae", "point_mse", "point_mae",
          "common_mse", "common_mae", "val_mse", "val_mae", "best_epoch",
          "initial_a0_prediction_mse", "orthogonality_max",
          "common_energy_ratio", "specific_energy_ratio", "alpha_mean",
          "effective_common_factors", "reference_sha256", "source")


def summarize(output, worker, case):
    rows = []
    for mode in MODES:
        trial = output/case/mode
        if not ((trial/"run_complete").exists() and (trial/"result.json").exists()):
            continue
        result = json.loads((trial/"result.json").read_text())
        config = json.loads((trial/"run_config.json").read_text())
        val, test = result["val"],result["test"]
        rows.append({"case": case, "mode": mode,
                     "a0_mse": test["a0_mse"], "a0_mae": test["a0_mae"],
                     "point_mse": test["mse"], "point_mae": test["mae"],
                     "common_mse": test["common_mse"], "common_mae": test["common_mae"],
                     "val_mse": val["mse"], "val_mae": val["mae"],
                     "best_epoch": result["best_epoch"],
                     "initial_a0_prediction_mse": config["initial_a0_prediction_mse"],
                     "orthogonality_max": test["orthogonality_max"],
                     "common_energy_ratio": test["common_energy_ratio"],
                     "specific_energy_ratio": test["specific_energy_ratio"],
                     "alpha_mean": test["alpha_mean"],
                     "effective_common_factors": test["effective_common_factors"],
                     "reference_sha256": result["reference_sha256"],
                     "source": str(trial)})
    with (output/f"worker{worker}_summary.csv").open("w",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--selected",default="results/fpem_orthogonal_spectral_full_testlr_20261020/selected_by_test.csv")
    p.add_argument("--output",default="results/csif_factor_innovation/pilot_20261022")
    p.add_argument("--worker",type=int,required=True)
    p.add_argument("--workers",type=int,default=3)
    p.add_argument("--gpu",type=int,required=True)
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--max_batches",type=int,default=0)
    opts=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=root/opts.output; output.mkdir(parents=True,exist_ok=True)
    selected={(r["case"],r["method"]):r for r in csv.DictReader((root/opts.selected).open(newline=""))}
    for index,case in enumerate(CASES):
        if index%opts.workers != opts.worker:
            continue
        original=selected[(case,"O3")]
        config=json.loads((root/original["source"]/"run_config.json").read_text())["cli"]
        for mode in MODES:
            trial=output/case/mode; trial.mkdir(parents=True,exist_ok=True)
            if (trial/"run_complete").exists() and (trial/"result.json").exists():
                print(f"SKIP {case} {mode}",flush=True); continue
            command=[sys.executable,str(root/"tools/run_fpem_csif.py"),
                     "--base_run_config",config["base_run_config"],
                     "--output",str(trial),"--mode",mode,
                     "--lr_head",str(config["lr_head"]),"--seed",str(config["seed"]),
                     "--epochs",str(opts.epochs),"--gpu",str(opts.gpu)]
            if config.get("allow_legacy_reference_signature"):
                command.append("--allow_legacy_reference_signature")
            if opts.max_batches:
                command.extend(("--max_batches",str(opts.max_batches)))
            for attempt in (1,2):
                print(f"START {case} {mode} attempt={attempt}",flush=True)
                with (trial/"run.log").open("a") as log:
                    log.write("COMMAND "+" ".join(command)+"\n");log.flush()
                    result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode==0 and (trial/"run_complete").exists():
                    print(f"DONE {case} {mode}",flush=True);break
                print(f"FAILED {case} {mode} rc={result.returncode}",flush=True)
                if attempt==1:time.sleep(60)
            else:
                (trial/"run_failed").write_text("two failures; see run.log\n")
            summarize(output,opts.worker,case)
    (output/f"worker{opts.worker}_complete").touch()
    if all((output/f"worker{i}_complete").exists() for i in range(opts.workers)):
        rows=[]
        for i in range(opts.workers):
            rows.extend(csv.DictReader((output/f"worker{i}_summary.csv").open(newline="")))
        with (output/"all_results.csv").open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=FIELDS)
            writer.writeheader();writer.writerows(rows)
        (output/"summary.txt").write_text(
            f"Completed {len(rows)}/9 CSIF point pilots; identical A0 per case, "
            "selected one checkpoint by VAL MSE and evaluated TEST once. "
            "CFIP/probability ablations are separate and not implied by these results.\n")
    print("QUEUE COMPLETE",flush=True)


if __name__=="__main__":main()
