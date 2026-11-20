"""Restartable G0/G1/G2 comparison using existing E0 as G0."""
from __future__ import annotations

import concurrent.futures
import csv
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.run_fpem_s4_losswise_queue import CASES, RESULTS, selected_prior


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = RESULTS / "gradient_surgery_20261020"
LOCK = threading.Lock()


def summarize():
    with LOCK:
        rows = []
        for case, gpu in CASES:
            for mode in ("G0", "G1", "G2"):
                path = (RESULTS / case / "E0") if mode == "G0" else (OUTPUT / case / mode)
                result_path = path / "result.json"
                if not result_path.is_file() or not (path / "run_complete").is_file():
                    continue
                result = json.loads(result_path.read_text(encoding="utf-8"))
                rows.append({"case": case, "mode": mode, "gpu": gpu,
                             "surgery_fraction": {"G0": 0, "G1": 1, "G2": 0.5}[mode],
                             "reference_sha256": result["reference_sha256"],
                             "best_epoch_by_val": result["best_epoch"],
                             "val_mse": result["val"]["mse"],
                             "val_mae": result["val"]["mae"],
                             "test_mse": result["test"]["mse"],
                             "test_mae": result["test"]["mae"],
                             "training_seconds": result["training_seconds"],
                             "peak_gpu_memory_bytes": result["peak_gpu_memory_bytes"]})
        if rows:
            with (OUTPUT / "pilot_results.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        lines = ["# S4 prediction-priority gradient surgery", "",
                 "G0 reuses the completed E0 run; G1 removes 100% and G2 50% of the",
                 "conflicting auxiliary gradient component on Ws only. All other S4",
                 "parameters and the backbone retain their original Adam update.", "",
                 "VAL selects each run's checkpoint; TEST ranks the three tricks as",
                 "exploratory development results, not unbiased final evaluation.", "",
                 f"Completed: {len(rows)}/{len(CASES)*3}", ""]
        for case, _ in CASES:
            subset = {row["mode"]: row for row in rows if row["case"] == case}
            if not subset:
                continue
            hashes = {row["reference_sha256"] for row in subset.values()}
            if len(hashes) > 1:
                raise RuntimeError(f"A0 reference differs across surgery modes: {case}")
            base = subset.get("G0")
            for mode in ("G0", "G1", "G2"):
                if mode not in subset:
                    continue
                row = subset[mode]
                gain = 100*(base["test_mse"]-row["test_mse"])/base["test_mse"] if base else float("nan")
                lines.append(f"- {case} {mode}: VAL MSE={row['val_mse']:.6f}, "
                             f"TEST MSE={row['test_mse']:.6f}, MAE={row['test_mae']:.6f}, "
                             f"vs G0={gain:+.3f}%")
        (OUTPUT / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run_case(case, gpu):
    _, prior_path, _, previous = selected_prior(case)
    cli = previous["cli"]
    g0 = RESULTS / case / "E0"
    if not (g0 / "run_complete").is_file():
        raise FileNotFoundError(f"G0/E0 missing for {case}: {g0}")
    for mode in ("G1", "G2"):
        output = OUTPUT / case / mode
        if (output / "run_complete").is_file() and (output / "result.json").is_file():
            print(f"SKIP {case} {mode}", flush=True)
            continue
        output.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "tools/run_fpem_s4_losswise.py",
               "--base_run_config", cli["base_run_config"], "--output", str(output),
               "--mode", mode, "--seed", str(cli["seed"]), "--epochs", "10",
               "--lr_head", str(cli["lr_head"]),
               "--lr_decomposer", str(cli["lr_decomposer"]),
               "--lr_backbone", str(cli["lr_backbone"]), "--gpu", "0"]
        if cli.get("train_backbone"):
            cmd.append("--train_backbone")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        for attempt in range(1, 4):
            print(f"START {case} {mode} GPU {gpu} attempt {attempt}", flush=True)
            with (output / "run.log").open("a", encoding="utf-8") as log:
                log.write("\nCOMMAND " + " ".join(cmd) + "\n")
                code = subprocess.call(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if code == 0 and (output / "run_complete").is_file():
                print(f"DONE {case} {mode}", flush=True)
                summarize()
                break
            print(f"FAIL {case} {mode}: exit {code}; waiting 60 seconds", flush=True)
            time.sleep(60)
        else:
            (output / "failed_after_retries").touch()
            summarize()
            return


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for case, gpu in CASES:
        _, path, _, config = selected_prior(case)
        manifest.append({"case": case, "gpu": gpu, "prior_pilot": str(path),
                         "G0_reused": str(RESULTS / case / "E0"),
                         "base_run_config": config["cli"]["base_run_config"],
                         "head_lr": config["cli"]["lr_head"],
                         "decomposer_lr": config["cli"]["lr_decomposer"]})
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    summarize()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for future in concurrent.futures.as_completed(
                [pool.submit(run_case, case, gpu) for case, gpu in CASES]):
            future.result()
    summarize()
    if all((OUTPUT / case / mode / "run_complete").is_file()
           for case, _ in CASES for mode in ("G1", "G2")):
        (OUTPUT / "queue_complete").touch()
        print("QUEUE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
