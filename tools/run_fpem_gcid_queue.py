"""Resumable three-backbone GCID A1-A6 pilot; A0 is directly re-evaluated."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


CASES = ("patchtst_ETTh1_720", "itransformer_ETTm1_720", "moderntcn_ETTm2_720")
MODES = ("A1", "A2", "A3", "A4", "A5", "A6")
FIELDS = ("case", "mode", "a0_mse", "a0_mae", "inv_mse", "inv_mae", "full_mse",
          "full_mae", "val_mse", "val_mae", "CRPS", "PICP_95", "MPIW_95",
          "sharedness_mean", "dynamic_energy_ratio", "alpha_mean", "best_epoch",
          "reference_sha256", "source")


def summarize(root, output, worker, case):
    rows = []
    for mode in MODES:
        trial = output / case / mode
        if not ((trial / "run_complete").is_file() and (trial / "result.json").is_file()):
            continue
        result = json.loads((trial / "result.json").read_text())
        test, val = result["test"], result["val"]
        rows.append({"case": case, "mode": mode, "a0_mse": test["a0_mse"],
                     "a0_mae": test["a0_mae"], "inv_mse": test["inv_mse"],
                     "inv_mae": test["inv_mae"], "full_mse": test["mse"],
                     "full_mae": test["mae"], "val_mse": val["mse"],
                     "val_mae": val["mae"], "CRPS": test.get("CRPS", ""),
                     "PICP_95": test["picp_95_exact"],
                     "MPIW_95": test["mpiw_95_exact"],
                     "sharedness_mean": test["sharedness_mean"],
                     "dynamic_energy_ratio": test["dynamic_energy_ratio"],
                     "alpha_mean": test["alpha_mean"],
                     "best_epoch": result["best_epoch"],
                     "reference_sha256": result["reference_sha256"],
                     "source": str(trial)})
    target = output / f"worker{worker}_summary.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selected", default="results/fpem_orthogonal_spectral_full_testlr_20261020/selected_by_test.csv")
    p.add_argument("--output", default="results/fpem_gcid_pilot_20261021")
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    opts = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / opts.output
    output.mkdir(parents=True, exist_ok=True)
    selected = {(r["case"], r["method"]): r
                for r in csv.DictReader((root / opts.selected).open(newline="", encoding="utf-8"))}
    for index, case in enumerate(CASES):
        if index % opts.workers != opts.worker:
            continue
        original = selected[(case, "O3")]
        config = json.loads((root / original["source"] / "run_config.json").read_text())["cli"]
        for mode in MODES:
            trial = output / case / mode
            trial.mkdir(parents=True, exist_ok=True)
            if (trial / "run_complete").is_file() and (trial / "result.json").is_file():
                print(f"SKIP {case} {mode}", flush=True)
                continue
            command = [sys.executable, str(root / "tools/run_fpem_gcid.py"),
                       "--base_run_config", config["base_run_config"],
                       "--output", str(trial), "--mode", mode,
                       "--lr_head", str(config["lr_head"]),
                       "--seed", str(config["seed"]), "--epochs", str(opts.epochs),
                       "--gpu", str(opts.gpu)]
            if config.get("allow_legacy_reference_signature"):
                command.append("--allow_legacy_reference_signature")
            if opts.max_batches:
                command += ["--max_batches", str(opts.max_batches)]
            for attempt in (1, 2):
                print(f"START {case} {mode} attempt={attempt}", flush=True)
                with (trial / "run.log").open("a", encoding="utf-8") as log:
                    log.write("COMMAND " + " ".join(command) + "\n")
                    log.flush()
                    result = subprocess.run(command, cwd=root, stdout=log,
                                            stderr=subprocess.STDOUT)
                if result.returncode == 0 and (trial / "run_complete").is_file():
                    print(f"DONE {case} {mode}", flush=True)
                    break
                print(f"FAILED {case} {mode} rc={result.returncode}", flush=True)
                if attempt == 1:
                    time.sleep(60)
            else:
                (trial / "run_failed").write_text("two failed attempts; see run.log\n")
            summarize(root, output, opts.worker, case)
        summarize(root, output, opts.worker, case)
    (output / f"worker{opts.worker}_complete").touch()
    if all((output / f"worker{i}_complete").is_file() for i in range(opts.workers)):
        rows = []
        for i in range(opts.workers):
            with (output / f"worker{i}_summary.csv").open(newline="", encoding="utf-8") as handle:
                rows.extend(csv.DictReader(handle))
        with (output / "all_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader(); writer.writerows(rows)
        (output / "summary.txt").write_text(
            f"Completed {len(rows)}/18 GCID pilot runs. Same A0 checkpoint per case; "
            "no A0 retraining. Each mode selects one checkpoint by VAL MSE and "
            "then evaluates TEST. A0 MSE/MAE, invariant, full and Gaussian "
            "residual interval metrics accompany each row.\n", encoding="utf-8")
    print("QUEUE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
