"""Resumable, one-case-per-GPU PGIR pilot using existing A0 checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

CASES = ("patchtst_ETTh1_720", "itransformer_ETTm1_720", "moderntcn_ETTm2_720")
MODES = ("A0", "P1", "P2", "P3")
FIELDS = ("case", "mode", "a0_mse", "a0_mae", "val_mse", "val_mae",
          "test_mse", "test_mae", "prototype_only_mse", "prototype_only_mae",
          "best_epoch", "replacement_fraction", "variation_keep_fraction",
          "training_seconds", "random_control_mse", "smooth_control_mse", "source")


def summarize(output):
    rows = []
    for case in CASES:
        for mode in MODES:
            trial = output/case/mode
            if not (trial/"run_complete").exists():
                continue
            result = json.loads((trial/"result.json").read_text())
            val, test = result["val"], result["test"]
            controls = result.get("controls", {})
            rows.append({"case": case, "mode": mode, "a0_mse": test["a0_mse"],
                         "a0_mae": test["a0_mae"], "val_mse": val["mse"],
                         "val_mae": val["mae"], "test_mse": test["mse"],
                         "test_mae": test["mae"],
                         "prototype_only_mse": test["prototype_mse"],
                         "prototype_only_mae": test["prototype_mae"],
                         "best_epoch": result["best_epoch"],
                         "replacement_fraction": test["replacement_fraction"],
                         "variation_keep_fraction": test["variation_keep_fraction"],
                         "training_seconds": result["training_seconds"],
                         "random_control_mse": controls.get("random", {}).get("mse", ""),
                         "smooth_control_mse": controls.get("smooth", {}).get("mse", ""),
                         "source": str(trial)})
    with (output/"all_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    lines = ["# PGIR point-forecast pilot", "",
             "Prototype bank and invariant threshold: TRAIN anchors only. A0 frozen.",
             "One checkpoint per trial selected by VAL MSE; TEST evaluated after selection.",
             "Random and smoothing controls use the selected P2 mask and are post-hoc TEST diagnostics, not independently trained.",
             "P1 retains all observed history and must exactly equal A0.", "",
             "| Case | Mode | A0 MSE | VAL MSE | TEST MSE | TEST MAE | Replaced |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f'| {row["case"]} | {row["mode"]} | {float(row["a0_mse"]):.6f} | '
                     f'{float(row["val_mse"]):.6f} | {float(row["test_mse"]):.6f} | '
                     f'{float(row["test_mae"]):.6f} | {float(row["replacement_fraction"]):.3f} |')
    (output/"ablation_summary.md").write_text("\n".join(lines)+"\n")
    if len(rows) == len(CASES)*len(MODES):
        (output/"point_queue_complete").touch()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selected", default="results/fpem_orthogonal_spectral_full_testlr_20261020/selected_by_test.csv")
    p.add_argument("--output", default="results/fpem_pgir/pilot_20261022")
    p.add_argument("--worker", type=int, required=True)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--max_batches", type=int, default=0)
    cli = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root/cli.output; output.mkdir(parents=True, exist_ok=True)
    selected = {(r["case"], r["method"]): r for r in
                csv.DictReader((root/cli.selected).open(newline=""))}
    for index, case in enumerate(CASES):
        if index % cli.workers != cli.worker: continue
        source = selected[(case, "O3")]
        saved = json.loads((root/source["source"]/"run_config.json").read_text())["cli"]
        bank = output/case/"prototype_bank.pt"
        for mode in MODES:
            trial = output/case/mode; trial.mkdir(parents=True, exist_ok=True)
            if (trial/"run_complete").exists():
                print(f"SKIP {case}/{mode}", flush=True); continue
            command = [sys.executable, str(root/"tools/run_fpem_pgir.py"),
                       "--base_run_config", saved["base_run_config"],
                       "--output", str(trial), "--bank_checkpoint", str(bank),
                       "--mode", mode, "--seed", str(saved["seed"]),
                       "--gpu", str(cli.gpu), "--epochs", str(cli.epochs)]
            if saved.get("allow_legacy_reference_signature"):
                command.append("--allow_legacy_reference_signature")
            if cli.max_batches:
                command += ["--max_batches", str(cli.max_batches)]
            for attempt in (1, 2):
                print(f"START {case}/{mode}, attempt {attempt}", flush=True)
                with (trial/"run.log").open("a") as log:
                    log.write("COMMAND "+" ".join(command)+"\n"); log.flush()
                    status = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
                if status.returncode == 0 and (trial/"run_complete").exists(): break
                print(f"FAILED {case}/{mode} rc={status.returncode}", flush=True)
                if attempt == 1: time.sleep(60)
            else:
                (trial/"run_failed").write_text("two attempts failed; inspect run.log\n")
                break
            summarize(output)
    (output/f"worker{cli.worker}_complete").touch()
    summarize(output)
    print("WORKER COMPLETE", flush=True)


if __name__ == "__main__":
    main()
