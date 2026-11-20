"""One-case-per-GPU T1-T5 queue; T4/T5 inherit VAL-selected flags."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

CASES=("patchtst_ETTh1_720","itransformer_ETTm1_720","moderntcn_ETTm2_720")
MODES=("T1","T2","T3","T4","T5")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--selected",default="results/fpem_orthogonal_spectral_full_testlr_20261020/selected_by_test.csv")
    p.add_argument("--baseline",default="results/fpem_pgir/pilot_20261022")
    p.add_argument("--output",default="results/fpem_pgir/tricks_20261023")
    p.add_argument("--worker",type=int,required=True)
    p.add_argument("--workers",type=int,default=3)
    p.add_argument("--gpu",type=int,required=True)
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--max_batches",type=int,default=0)
    cli=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=root/cli.output;output.mkdir(parents=True,exist_ok=True)
    baseline=root/cli.baseline
    selected={(r["case"],r["method"]):r for r in
              csv.DictReader((root/cli.selected).open(newline=""))}
    for index,case in enumerate(CASES):
        if index%cli.workers!=cli.worker:continue
        source=selected[(case,"O3")]
        saved=json.loads((root/source["source"]/"run_config.json").read_text())["cli"]
        bank=baseline/case/"prototype_bank.pt"
        core=baseline/case/"P4"
        if not (core/"run_complete").exists():
            raise RuntimeError(f"strict OOF core prerequisite missing: {core}")
        if not (baseline/case/"P2"/"run_complete").exists():
            raise RuntimeError(f"P2 baseline missing: {case}")
        for mode in MODES:
            trial=output/case/mode;trial.mkdir(parents=True,exist_ok=True)
            if (trial/"run_complete").exists():
                print(f"SKIP {case}/{mode}",flush=True);continue
            if mode in ("T1","T2","T3"):
                options={"local":mode=="T2","gain_supervision":mode=="T3",
                         "boundary":False,"recent":False}
                origin=mode
            else:
                predecessors=("T1","T2","T3") if mode=="T4" else ("T1","T2","T3","T4")
                candidates=[]
                for previous in predecessors:
                    path=output/case/previous
                    if (path/"run_complete").exists():
                        result=json.loads((path/"result.json").read_text())
                        candidates.append((result["val"]["mse"],previous))
                if len(candidates)!=len(predecessors):
                    raise RuntimeError(f"prior VAL results missing for {case}/{mode}")
                origin=min(candidates)[1]
                config=json.loads((output/case/origin/"run_config.json").read_text())["cli"]
                options={key:bool(config[key]) for key in
                         ("local","gain_supervision","boundary","recent")}
                options["boundary"]=options["boundary"] or mode=="T4"
                options["recent"]=options["recent"] or mode=="T5"
            command=[sys.executable,str(root/"tools/run_fpem_pgir_tricks.py"),
                     "--base_run_config",saved["base_run_config"],
                     "--bank_checkpoint",str(bank),"--core_p4_dir",str(core),
                     "--gain_cache",str(output/case/"oof_gain_labels.pt"),
                     "--output",str(trial),"--mode",mode,
                     "--seed",str(saved["seed"]),"--gpu",str(cli.gpu),
                     "--epochs",str(cli.epochs)]
            for flag,on in options.items():
                if on:command.append("--"+flag)
            if saved.get("allow_legacy_reference_signature"):
                command.append("--allow_legacy_reference_signature")
            if cli.max_batches:command+=["--max_batches",str(cli.max_batches)]
            (trial/"inheritance.txt").write_text(
                f"{mode} inherits VAL-selected {origin}; options={options}\n")
            for attempt in (1,2):
                print(f"START {case}/{mode} origin={origin} attempt={attempt}",flush=True)
                with (trial/"run.log").open("a") as log:
                    log.write("COMMAND "+" ".join(command)+"\n");log.flush()
                    result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode==0 and (trial/"run_complete").exists():break
                print(f"FAILED {case}/{mode} rc={result.returncode}",flush=True)
                if attempt==1:time.sleep(60)
            else:
                (trial/"run_failed").write_text("two failures; inspect run.log\n")
                break
            summarize(output,baseline)
    print("TRICK WORKER COMPLETE",flush=True)


def summarize(output,baseline):
    rows=[]
    for case in CASES:
        b=json.loads((baseline/case/"P2"/"result.json").read_text())
        rows.append({"case":case,"mode":"P2","source":"original baseline",
                     "a0_mse":b["test"]["a0_mse"],"val_mse":b["val"]["mse"],
                     "test_mse":b["test"]["mse"],"test_mae":b["test"]["mae"],
                     "best_epoch":b["best_epoch"],
                     "mean_replaced_patches_per_series":"",
                     "fraction_series_intervened":""})
        for mode in MODES:
            trial=output/case/mode
            if not (trial/"run_complete").exists():continue
            result=json.loads((trial/"result.json").read_text())
            test=result["test"]
            rows.append({"case":case,"mode":mode,
                         "source":(trial/"inheritance.txt").read_text().strip(),
                         "a0_mse":test["a0_mse"],"val_mse":result["val"]["mse"],
                         "test_mse":test["mse"],"test_mae":test["mae"],
                         "best_epoch":result["best_epoch"],
                         "mean_replaced_patches_per_series":test["mean_replaced_patches_per_series"],
                         "fraction_series_intervened":test["fraction_series_intervened"]})
    fields=("case","mode","source","a0_mse","val_mse","test_mse","test_mae",
            "best_epoch","mean_replaced_patches_per_series",
            "fraction_series_intervened")
    with (output/"all_results.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)
    lines=["PGIR patch-trick pilot: A0 frozen; one checkpoint per run chosen by VAL MSE.",
           "T1 caps evidence-positive replacement at two segments; T2 adds local context retrieval;",
           "T3 trains replacement scores with independent TRAIN chronological OOF labels;",
           "T4/T5 inherit the lowest-VAL preceding option, not the lowest-TEST option.",
           "T3 labels use non-overlapping calibration anchors and no VAL/TEST targets.",
           ""]
    for row in rows:
        lines.append(f'{row["case"]} {row["mode"]}: A0={float(row["a0_mse"]):.6f}, '
                     f'VAL={float(row["val_mse"]):.6f}, TEST={float(row["test_mse"]):.6f}, '
                     f'MAE={float(row["test_mae"]):.6f}')
    (output/"summary.txt").write_text("\n".join(lines)+"\n")
    if len(rows)==len(CASES)*(len(MODES)+1):
        (output/"queue_complete").touch()


if __name__=="__main__":main()
