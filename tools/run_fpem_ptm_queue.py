"""Resumable three-GPU PTM pilot; never trains a new A0."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

CASES = ("patchtst_ETTh1_720", "itransformer_ETTm1_720", "moderntcn_ETTm2_720")
MODES = ("A0", "T0", "T1", "T2", "T3", "T4")
FIELDS = ("case", "mode", "a0_mse", "a0_mae", "val_mse", "val_mae",
          "test_mse", "test_mae", "prototype_only_mse", "prototype_only_mae",
          "best_epoch", "gate_mean", "gate_abs_mean", "match_max_weight",
          "match_entropy", "match_min_distance", "input_or_output_change",
          "trainable_parameters", "bank_float_count", "training_seconds",
          "inference_milliseconds_per_batch", "checkpoint", "source")


def summarize_case(output, worker, case):
    rows = []
    for mode in MODES:
        trial = output/case/mode
        if not ((trial/"run_complete").exists() and (trial/"result.json").exists()):
            continue
        result = json.loads((trial/"result.json").read_text())
        cfg = json.loads((trial/"run_config.json").read_text())
        test, val = result["test"], result["val"]
        rows.append({
            "case": case, "mode": mode, "a0_mse": test["a0_mse"],
            "a0_mae": test["a0_mae"], "val_mse": val["mse"],
            "val_mae": val["mae"], "test_mse": test["mse"],
            "test_mae": test["mae"], "prototype_only_mse": test["prototype_mse"],
            "prototype_only_mae": test["prototype_mae"],
            "best_epoch": result["best_epoch"], "gate_mean": test["gate_mean"],
            "gate_abs_mean": test["gate_abs_mean"],
            "match_max_weight": test["match_max_weight"],
            "match_entropy": test["match_entropy"],
            "match_min_distance": test["match_min_distance"],
            "input_or_output_change": test["input_or_output_change"],
            "trainable_parameters": cfg["trainable_parameters"],
            "bank_float_count": cfg["bank_float_count"],
            "training_seconds": result["training_seconds"],
            "inference_milliseconds_per_batch": test["milliseconds_per_batch"],
            "checkpoint": result["checkpoint"], "source": str(trial)})
    path = output/f"worker{worker}_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)


def finalize(output, workers):
    rows = []
    for worker in range(workers):
        rows.extend(csv.DictReader((output/f"worker{worker}_summary.csv").open(newline="")))
    with (output/"all_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    diagnostics = []
    for row in rows:
        path = Path(row["source"])/"result.json"
        result = json.loads(path.read_text())
        diagnostics.append({
            "case": row["case"], "mode": row["mode"],
            "match_max_weight": row["match_max_weight"],
            "match_entropy": row["match_entropy"],
            "match_min_distance": row["match_min_distance"],
            "gate_mean": row["gate_mean"],
            "input_or_output_change": row["input_or_output_change"],
            "usage_fraction": json.dumps(result["test"]["prototype_usage_fraction"]),
            "coverage_counts": json.dumps(result["test"]["prototype_coverage_counts"])})
    with (output/"prototype_diagnostics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0]) if diagnostics else
                                ["case","mode"])
        writer.writeheader(); writer.writerows(diagnostics)
    lines = ["# PTM pilot", "",
             "VAL MSE selects one checkpoint per run; TEST is evaluated once.",
             "These three cases and one seed are exploratory, not a broad superiority claim.", "",
             "| Case | Mode | A0 TEST MSE | VAL MSE | TEST MSE | TEST MAE |",
             "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f'| {row["case"]} | {row["mode"]} | {float(row["a0_mse"]):.6f} | '
                     f'{float(row["val_mse"]):.6f} | {float(row["test_mse"]):.6f} | '
                     f'{float(row["test_mae"]):.6f} |')
    (output/"ablation_summary.md").write_text("\n".join(lines)+"\n")
    failures = ["# Failure and negative-result analysis", ""]
    for case in CASES:
        baseline = next((r for r in rows if r["case"] == case and r["mode"] == "A0"), None)
        if baseline is None:
            failures.append(f"- {case}: A0 missing; inspect worker log.")
            continue
        a0 = float(baseline["test_mse"])
        for mode in MODES[1:]:
            row = next((r for r in rows if r["case"] == case and r["mode"] == mode), None)
            if row is None:
                failures.append(f"- {case}/{mode}: missing or failed; inspect run.log.")
            elif float(row["test_mse"]) >= a0:
                failures.append(
                    f'- {case}/{mode}: TEST MSE {float(row["test_mse"]):.6f} >= '
                    f'A0 {a0:.6f}; prototype-only {float(row["prototype_only_mse"]):.6f}, '
                    f'gate {float(row["gate_mean"]):.4f}. '
                    "Check mismatch/over-smoothing before attributing causality.")
    failures.append("")
    failures.append("Prototype-only accuracy and match diagnostics are descriptive, not proof of why an adapter failed.")
    (output/"failure_analysis.md").write_text("\n".join(failures)+"\n")
    (output/"queue_complete").touch()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selected", default="results/fpem_orthogonal_spectral_full_testlr_20261020/selected_by_test.csv")
    p.add_argument("--output", default="results/fpem_ptm/pilot_20261022")
    p.add_argument("--worker", type=int, required=True)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--max_batches", type=int, default=0)
    cli = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root/cli.output; output.mkdir(parents=True, exist_ok=True)
    selected = {(row["case"], row["method"]): row for row in
                csv.DictReader((root/cli.selected).open(newline=""))}
    for index, case in enumerate(CASES):
        if index % cli.workers != cli.worker:
            continue
        original = selected[(case, "O3")]
        saved = json.loads((root/original["source"]/"run_config.json").read_text())["cli"]
        bank = output/case/"prototype_bank.pt"
        first_stage_ok = True
        for mode in MODES:
            if mode in ("T3", "T4") and not first_stage_ok:
                print(f"SKIP {case} {mode}: first stage incomplete", flush=True)
                continue
            trial = output/case/mode; trial.mkdir(parents=True, exist_ok=True)
            if (trial/"run_complete").exists() and (trial/"result.json").exists():
                print(f"SKIP complete {case} {mode}", flush=True); continue
            command = [sys.executable, str(root/"tools/run_fpem_ptm.py"),
                       "--base_run_config", saved["base_run_config"],
                       "--output", str(trial), "--bank_checkpoint", str(bank),
                       "--mode", mode, "--seed", str(saved["seed"]),
                       "--lr_head", str(saved["lr_head"]),
                       "--epochs", str(cli.epochs), "--gpu", str(cli.gpu)]
            if saved.get("allow_legacy_reference_signature"):
                command.append("--allow_legacy_reference_signature")
            if cli.max_batches:
                command.extend(("--max_batches", str(cli.max_batches)))
            ok = False
            for attempt in (1, 2):
                print(f"START {case} {mode} attempt={attempt}", flush=True)
                with (trial/"run.log").open("a") as log:
                    log.write("COMMAND "+" ".join(command)+"\n"); log.flush()
                    result = subprocess.run(command, cwd=root, stdout=log,
                                            stderr=subprocess.STDOUT)
                if result.returncode == 0 and (trial/"run_complete").exists():
                    ok = True; print(f"DONE {case} {mode}", flush=True); break
                print(f"FAILED {case} {mode}: rc={result.returncode}", flush=True)
                if attempt == 1:
                    time.sleep(60)
            if not ok:
                (trial/"run_failed").write_text("two failures; inspect run.log\n")
                if mode in ("A0", "T0", "T1", "T2"):
                    first_stage_ok = False
            summarize_case(output, cli.worker, case)
    (output/f"worker{cli.worker}_complete").touch()
    if all((output/f"worker{worker}_complete").exists() for worker in range(cli.workers)):
        finalize(output, cli.workers)
    print("WORKER COMPLETE", flush=True)


if __name__ == "__main__":
    main()
