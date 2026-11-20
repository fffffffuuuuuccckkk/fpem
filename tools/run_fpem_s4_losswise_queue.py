"""Four-GPU E0-E5 pilot launcher; restartable without retraining A0."""
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


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/fpem_s4_losswise_optimizer"
PREVIOUS = ROOT / "results/fpem_shared_factorization/pilot_lr_search_20261019"
CASES = (
    ("patchtst_ETTh1_96", 0),
    ("patchtst_ETTh1_720", 1),
    ("moderntcn_ETTm2_720", 2),
    ("itransformer_ETTm1_720", 3),
)
MODES = ("E0", "E1", "E2", "E3", "E4", "E5")
SUMMARY_LOCK = threading.Lock()


def selected_prior(case):
    candidates = []
    for path in (PREVIOUS / case / "supervised_joint_private").glob("lr_*/result.json"):
        result = json.loads(path.read_text())
        config = json.loads((path.parent / "run_config.json").read_text())
        candidates.append((result["val"]["mse"], path.parent, result, config))
    if not candidates:
        raise FileNotFoundError(f"no preexisting S4 pilot for {case}")
    return min(candidates, key=lambda item: item[0])


def csv_write(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize():
    with SUMMARY_LOCK:
        _summarize_locked()


def _summarize_locked():
    rows, grads, lrs = [], [], []
    for case, gpu in CASES:
        base_val, base_path, base, config = selected_prior(case)
        for mode in MODES:
            output = RESULTS / case / mode
            path = output / "result.json"
            if not path.is_file() or not (output / "run_complete").is_file():
                continue
            data = json.loads(path.read_text())
            rows.append({"case": case, "mode": mode, "gpu": gpu,
                         "reference_sha256": data["reference_sha256"],
                         "base_head_lr": config["cli"]["lr_head"],
                         "base_decomposer_lr": config["cli"]["lr_decomposer"],
                         "prior_s4_val_mse": base["val"]["mse"],
                         "prior_s4_test_mse": base["test"]["mse"],
                         "val_mse": data["val"]["mse"], "val_mae": data["val"]["mae"],
                         "test_mse": data["test"]["mse"], "test_mae": data["test"]["mae"],
                         "best_epoch_val_selected": data["best_epoch"],
                         "training_seconds": data["training_seconds"],
                         "peak_gpu_memory_bytes": data["peak_gpu_memory_bytes"]})
            grads.extend({"case": case, "mode": mode, **row} for row in data["gradient_history"])
            lrs.extend({"case": case, "mode": mode, **row} for row in data["lr_history"])
    csv_write(RESULTS / "pilot_results.csv", rows)
    csv_write(RESULTS / "gradient_diagnostics.csv", grads)
    csv_write(RESULTS / "lr_trace.csv", lrs)
    ranking = []
    for case, _ in CASES:
        relevant = sorted((row for row in rows if row["case"] == case), key=lambda row: row["test_mse"])
        e0 = next((row for row in relevant if row["mode"] == "E0"), None)
        for position, row in enumerate(relevant, 1):
            ranking.append({"case": case, "rank_by_test_mse": position,
                            "mode": row["mode"], "test_mse": row["test_mse"],
                            "test_mae": row["test_mae"], "val_mse": row["val_mse"],
                            "improvement_vs_e0_percent":
                            100 * (e0["test_mse"]-row["test_mse"])/e0["test_mse"] if e0 else ""})
    csv_write(RESULTS / "test_ranking.csv", ranking)
    lines = ["# S4 loss-wise optimizer exploratory pilots", "",
             "Trick ranking uses TEST MSE as requested. These four TEST splits are development data;", 
             "do not report the selected winner as an unbiased final TEST result.", "",
             f"Completed: {len(rows)}/{len(CASES)*len(MODES)}", ""]
    for case, _ in CASES:
        subset = [row for row in ranking if row["case"] == case]
        if subset:
            best = subset[0]
            lines.append(f"- {case}: best {best['mode']} TEST MSE={best['test_mse']:.6f}, "
                         f"MAE={best['test_mae']:.6f}, E0 improvement={best['improvement_vs_e0_percent']}%")
    if len(rows) == len(CASES)*len(MODES):
        conflicts = [row["ws_conflict_fraction"] for row in grads if row["mode"] == "E0"]
        conflict = sum(conflicts)/len(conflicts) if conflicts else float("nan")
        lines += ["", f"E0 mean Ws gradient conflict fraction: {conflict:.4f}",
                  "Gradient surgery is a separate follow-up; skipped if conflict is rare."]
    (RESULTS / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run_case(case, gpu):
    base_val, base_path, base, config = selected_prior(case)
    previous_cli = config["cli"]
    for mode in MODES:
        output = RESULTS / case / mode
        if (output / "run_complete").is_file() and (output / "result.json").is_file():
            print(f"SKIP {case} {mode}", flush=True)
            continue
        output.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "tools/run_fpem_s4_losswise.py",
               "--base_run_config", previous_cli["base_run_config"],
               "--output", str(output), "--mode", mode,
               "--seed", str(previous_cli["seed"]), "--epochs", "10",
               "--lr_head", str(previous_cli["lr_head"]),
               "--lr_decomposer", str(previous_cli["lr_decomposer"]),
               "--lr_backbone", str(previous_cli["lr_backbone"]), "--gpu", "0"]
        if previous_cli.get("train_backbone"):
            cmd.append("--train_backbone")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        for attempt in range(1, 4):
            print(f"START {case} {mode} gpu={gpu} attempt={attempt}", flush=True)
            with (output / "run.log").open("a", encoding="utf-8") as log:
                log.write("\nCOMMAND " + " ".join(cmd) + "\n")
                code = subprocess.call(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if code == 0 and (output / "run_complete").is_file():
                print(f"DONE {case} {mode}", flush=True)
                summarize()
                break
            print(f"FAIL {case} {mode} exit={code}; sleep 60s", flush=True)
            time.sleep(60)
        else:
            (output / "failed_after_retries").touch()
            summarize()
            return


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    audit = ("# Optimizer audit\n\n"
             "Existing S4 pilot: supervised_joint_private Ws/Wv, MSE prediction + "
             "0.01 reconstruction + 0.001 cross-covariance, Adam(0.9,0.999), "
             "weight_decay=0, no AMP/clipping/scheduler. A0 backbone is frozen "
             "in the four selected pilots; TRAIN-only EIIL q is refreshed at existing stages.\n\n"
             "E1-E5 use separate Adam first/second moments and independent counters "
             "per prediction/auxiliary loss on S4 parameters; directions combine before "
             "a single parameter update. This adapts AdaTask but is not an exact "
             "reproduction of its official shared/task-specific optimizer. "
             "The original A0 checkpoint and S4 initialization/seed are reused. "
             "Head LR is the pre-existing VAL-selected pilot LR for each case; "
             "decomposer LR stays 1e-4. Multipliers apply loss-wise to each group. "
             "VAL selects epoch, TEST ranks tricks only as exploratory development.\n")
    (RESULTS / "optimizer_audit.md").write_text(audit, encoding="utf-8")
    manifest = []
    for case, gpu in CASES:
        _, path, _, config = selected_prior(case)
        manifest.append({"case": case, "gpu": gpu,
                         "prior_run": str(path), "cli": config["cli"]})
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    summarize()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_case, case, gpu) for case, gpu in CASES]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    summarize()
    if all((RESULTS / case / mode / "run_complete").is_file()
           for case, _ in CASES for mode in MODES):
        (RESULTS / "queue_complete").touch()
        print("QUEUE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
