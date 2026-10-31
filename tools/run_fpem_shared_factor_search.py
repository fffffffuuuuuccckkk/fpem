"""Balanced three-server, restartable VAL-only LR search for S1-S8."""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


METHODS = (
    "common_subspace", "shared_spectral", "robust_lowrank_sparse",
    "supervised_joint_private", "shared_private_lite", "pairwise_partial_shared",
    "sample_prevalence", "subspace_recurrence", "subspace_recurrence_poly",
)
HEAD_LRS = ("2e-5", "5e-5", "1e-4")
CASES = {
    "patchtst_ETTh1_96": (
        "results/fpem_no_future_anchor_patchtst_search/patchtst/ETTh1/pred_96/A0/run_config.json",
        "8704815217f295d6295b861fdfa17ec693997e0c1570a44573a142a00b641664", 30),
    "patchtst_ETTh1_720": (
        "results/fpem_latent_decomposition_search/pilot_20261015/patchtst_ETTh1_720_A0_complementary_gate_predictive_sub0_orth0_refnone_fp0_rank8_bias-4/run_config.json",
        "aea74f86ffb18bf35d2787a527262db04b88c1c4e4b35ba1e2fb641d0002f76e", 60),
    "moderntcn_ETTm2_720": (
        "results/fpem_latent_decomposition_search/pilot_20261015/moderntcn_ETTm2_720_A0_complementary_gate_predictive_sub0_orth0_refnone_fp0_rank8_bias-4/run_config.json",
        "e6ad2c723815043fb6e296902a09eba793ce4409e6a929fe59d3b3f8412445ae", 450),
    "itransformer_ETTm1_720": (
        "results/fpem_latent_decomposition_search/pilot_20261015/itransformer_ETTm1_720_A0_complementary_gate_predictive_sub0_orth0_refnone_fp0_rank8_bias-4/run_config.json",
        "03417e14c3e44a01aaadf4bd7e048d3787472414a60a7eb1cc831f6d21a01ae1", 100),
}
FIELDS = ("case", "method", "base_run_config", "reference_sha256", "estimated_seconds",
          "lane", "server_index", "gpu")


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plan(args):
    jobs = []
    for case, (config, digest, cost) in CASES.items():
        for method_index, method in enumerate(METHODS):
            jobs.append({"case": case, "method": method, "base_run_config": config,
                         "reference_sha256": digest, "estimated_seconds": cost * len(HEAD_LRS),
                         "server_index": method_index % 3})
    load, count = [0] * 12, [0] * 12
    # Each server receives exactly three methods for each of four cases.
    # Longest-job-first scheduling then balances that server's four GPUs.
    for server in range(3):
        assigned = sorted((x for x in jobs if x["server_index"] == server),
                          key=lambda row: -row["estimated_seconds"])
        for job in assigned:
            lane = min((4 * server + i for i in range(4) if count[4 * server + i] < 3),
                       key=lambda i: (load[i], count[i], i))
            job.update({"lane": lane, "gpu": lane % 4})
            load[lane] += job["estimated_seconds"]
            count[lane] += 1
    write_csv(args.manifest, sorted(jobs, key=lambda row: (row["lane"], row["case"])), FIELDS)
    print(json.dumps({"specs": len(jobs), "trials": len(jobs) * len(HEAD_LRS),
                      "methods_per_lane": count, "estimated_seconds_per_lane": load}, indent=2))


def trial_root(root, row, lr):
    return root / row["case"] / row["method"] / f"lr_{lr.replace('.', 'p')}"


def valid_trial(path, row, lr):
    if not (path / "run_complete").is_file():
        return False
    config = json.loads((path / "run_config.json").read_text())
    result = json.loads((path / "result.json").read_text())
    cli = config["cli"]
    if not (config["reference_sha256"] == row["reference_sha256"]
            and result["reference_sha256"] == row["reference_sha256"]
            and result["method"] == row["method"]
            and result["environment_mode"] == "predictive"
            and math.isclose(float(cli["lr_head"]), float(lr))
            and math.isclose(float(cli["lr_decomposer"]), 1e-4)
            and not cli["train_backbone"] and not cli["function_preserving_init"]
            and cli["epochs"] == 10 and len(result["train_history"]) == 10):
        raise ValueError(f"completed trial protocol mismatch: {path}")
    return True


def locate_trial(root, row, lr):
    base = trial_root(root, row, lr)
    possibilities = [base, *(Path(str(base) + f"_retry{i}") for i in range(1, 20))]
    for path in possibilities:
        if valid_trial(path, row, lr):
            return path, True
    for path in possibilities:
        if not path.exists():
            return path, False
    raise RuntimeError(f"all retry slots occupied: {base}")


def summarize(root, rows):
    trial_rows, best_rows, incomplete = [], [], []
    for row in rows:
        candidates = []
        for lr in HEAD_LRS:
            base = trial_root(root, row, lr)
            paths = [base, *(Path(str(base) + f"_retry{i}") for i in range(1, 20))]
            good = next((p for p in paths if valid_trial(p, row, lr)), None)
            if good is None:
                incomplete.append({"case": row["case"], "method": row["method"], "lr_head": lr})
                continue
            result = json.loads((good / "result.json").read_text())
            candidate = {"case": row["case"], "method": row["method"],
                         "lr_head": lr, "environment_mode": "predictive",
                         "train_backbone": False, "best_epoch": result["best_epoch"],
                         "val_mse": result["val"]["mse"], "val_mae": result["val"]["mae"],
                         "test_mse": result["test"]["mse"], "test_mae": result["test"]["mae"],
                         "trainable_parameters": result["trainable_parameters"],
                         "buffer_bytes": result["buffer_bytes"],
                         "training_seconds": result["training_seconds"],
                         "decomposition_ms": result["decomposition_milliseconds_per_batch"],
                         "cpu_peak_rss_bytes": result["cpu_peak_rss_bytes"],
                         "zvar_mse": result["test"]["zvar_mse"],
                         "reconstruction_mae": result["test"]["reconstruction_mae"],
                         "peak_gpu_memory_bytes": result["peak_gpu_memory_bytes"],
                         "reference_sha256": result["reference_sha256"], "path": str(good),
                         "server_index": row["server_index"], "gpu": row["gpu"]}
            candidates.append(candidate)
            trial_rows.append(candidate)
        if len(candidates) == len(HEAD_LRS):
            best = min(candidates, key=lambda candidate: (candidate["val_mse"], float(candidate["lr_head"])))
            best_rows.append({**best, "selection_split": "val"})
    if trial_rows:
        write_csv(root / "all_results.csv", trial_rows, list(trial_rows[0]))
    if best_rows:
        write_csv(root / "validation_selection.csv", best_rows, list(best_rows[0]))
    (root / "status.json").write_text(json.dumps({"assigned_specs": len(rows),
        "completed_trials": len(trial_rows), "selected_specs": len(best_rows),
        "incomplete": incomplete, "selection_rule": "VAL MSE, TEST only after checkpoint selection"},
        indent=2) + "\n")
    return incomplete


def run(args):
    project = Path(args.project_dir).resolve()
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = [row for row in read_csv(args.manifest) if int(row["server_index"]) == args.server_index]
    if len(rows) != 12 or sorted(set(int(row["gpu"]) for row in rows)) != [0, 1, 2, 3]:
        raise ValueError(f"invalid server shard {args.server_index}: {len(rows)} specs")
    for row in rows:
        if not (project / row["base_run_config"]).is_file():
            raise FileNotFoundError(project / row["base_run_config"])
    lock = threading.Lock()
    failures = []

    def log(message):
        with lock:
            print(message, flush=True)
            with (root / "queue.log").open("a") as handle:
                handle.write(message + "\n")

    def worker(gpu):
        for row in (row for row in rows if int(row["gpu"]) == gpu):
            for lr in HEAD_LRS:
                try:
                    output, complete = locate_trial(root, row, lr)
                    if complete:
                        log(f"REUSE gpu={gpu} {row['case']} {row['method']} lr={lr}")
                        continue
                    output.mkdir(parents=True)
                    command = [sys.executable, "tools/run_fpem_shared_factorization.py",
                        "--base_run_config", row["base_run_config"], "--output", str(output),
                        "--method", row["method"], "--epochs", "10", "--seed", "2021",
                        "--gpu", "0", "--lr_head", lr, "--lr_decomposer", "1e-4",
                        "--lr_backbone", "2e-5", "--environment_mode", "predictive"]
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                    log(f"START gpu={gpu} {row['case']} {row['method']} lr={lr}")
                    with (output / "run.log").open("w") as handle:
                        status = subprocess.run(command, cwd=project, env=env,
                                                stdout=handle, stderr=subprocess.STDOUT,
                                                check=False).returncode
                    if status:
                        raise RuntimeError(f"exit={status} log={output / 'run.log'}")
                    valid_trial(output, row, lr)
                    log(f"DONE gpu={gpu} {row['case']} {row['method']} lr={lr}")
                except Exception as exc:
                    issue = f"FAIL gpu={gpu} {row['case']} {row['method']} lr={lr}: {exc}"
                    failures.append(issue)
                    log(issue)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(worker, range(4)))
    incomplete = summarize(root, rows)
    log(f"SEARCH_END server={args.server_index} failures={len(failures)} incomplete={len(incomplete)}")
    if failures or incomplete:
        raise SystemExit(1)
    (root / "search_complete").touch()


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    planner = sub.add_parser("plan")
    planner.add_argument("--manifest", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--manifest", required=True)
    runner.add_argument("--server-index", type=int, choices=(0, 1, 2), required=True)
    runner.add_argument("--project-dir", required=True)
    runner.add_argument("--output-root", required=True)
    args = p.parse_args()
    if args.command == "plan":
        plan(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
