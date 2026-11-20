"""Sharded, restartable invariant-head LR search for the statistical pilot.

The 5e-5 trial is the completed pilot. New trials change only --lr_head;
each method/case is assigned to one of twelve GPU lanes. Selection uses VAL
MSE and reports TEST only from that selected checkpoint.
"""

from __future__ import annotations

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


HEAD_LRS = ("2e-5", "5e-5", "1e-4")
CASES = {
    "patchtst_ETTh1_96": "results/fpem_no_future_anchor_patchtst_search/patchtst/ETTh1/pred_96/A0/run_config.json",
    "patchtst_ETTh1_720": "results/fpem_latent_decomposition_search/pilot_20261015/patchtst_ETTh1_720_A0_complementary_gate_predictive_sub0_orth0_refnone_fp0_rank8_bias-4/run_config.json",
    "moderntcn_ETTm2_720": "results/fpem_latent_decomposition_search/pilot_20261015/moderntcn_ETTm2_720_A0_complementary_gate_predictive_sub0_orth0_refnone_fp0_rank8_bias-4/run_config.json",
    "itransformer_ETTm1_720": "results/fpem_latent_decomposition_search/pilot_20261015/itransformer_ETTm1_720_A0_complementary_gate_predictive_sub0_orth0_refnone_fp0_rank8_bias-4/run_config.json",
}
FIELDS = ("case", "method", "groups", "protocol", "base_run_config",
          "reference_sha256", "pilot_val_mse", "pilot_test_mse", "pilot_test_mae",
          "pilot_training_seconds", "lane", "server_index", "gpu")


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def spec_name(row):
    return f"{row['protocol']}_{row['method']}_g{row['groups']}"


def plan(args):
    pilot_root = Path(args.pilot_csv).parent
    pilot = read_csv(Path(args.pilot_csv))
    jobs = []
    seen = set()
    for row in pilot:
        case, method = row["case"], row["method"]
        if case not in CASES:
            raise ValueError(f"Unexpected case: {case}")
        if method == "a0":
            continue  # A0 has no trainable invariant head.
        protocol = "joint" if method != "a1" and row["train_backbone"] == "True" else "frozen"
        key = (case, protocol, method, int(row["groups"]))
        if key in seen:
            raise ValueError(f"Duplicate pilot candidate: {key}")
        seen.add(key)
        result_dir = pilot_root / case / f"{protocol}_{method}_g{row['groups']}"
        config_path = result_dir / "run_config.json"
        if args.verify_pilot:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            cli = config["cli"]
            if not (result_dir / "run_complete").is_file():
                raise ValueError(f"Pilot is incomplete: {result_dir}")
            if not (math.isclose(float(cli["lr_head"]), 5e-5)
                    and math.isclose(float(cli["lr_stat"]), 1e-4)
                    and math.isclose(float(cli["lr_backbone"]), 2e-5)
                    and int(cli["epochs"]) == 10
                    and config["reference_sha256"] == row["reference_sha256"]):
                raise ValueError(f"Pilot protocol mismatch: {config_path}")
        jobs.append({
            "case": case, "method": method, "groups": str(int(row["groups"])),
            "protocol": protocol, "base_run_config": CASES[case],
            "reference_sha256": row["reference_sha256"],
            "pilot_val_mse": row["val_mse"], "pilot_test_mse": row["test_mse"],
            "pilot_test_mae": row["test_mae"],
            "pilot_training_seconds": row["training_seconds"],
        })
    if len(jobs) != 72 or len(pilot) != 76:
        raise ValueError(f"Expected 72 trainable candidates + 4 A0, got {len(jobs)} + {len(pilot)-len(jobs)}")
    # Greedy longest-job-first packing. Each of 12 lanes gets exactly six
    # method/case pairs, hence twelve new runs and a near-balanced load.
    load = [0.0] * 12
    count = [0] * 12
    for job in sorted(jobs, key=lambda x: -float(x["pilot_training_seconds"])):
        lane = min((i for i in range(12) if count[i] < 6), key=lambda i: (load[i], count[i], i))
        job.update(lane=str(lane), server_index=str(lane // 4), gpu=str(lane % 4))
        load[lane] += 2 * float(job["pilot_training_seconds"])
        count[lane] += 1
    write_csv(Path(args.manifest), sorted(jobs, key=lambda x: (int(x["lane"]), x["case"], spec_name(x))), FIELDS)
    print(json.dumps({"candidates": len(jobs), "new_trials": 2 * len(jobs),
                      "lanes": count, "estimated_seconds_per_lane": [round(x) for x in load]}, indent=2))


def trial_root(root: Path, row, lr):
    return root / row["case"] / spec_name(row) / f"lr_{lr.replace('.', 'p')}"


def valid_trial(path: Path, row, lr):
    if not (path / "run_complete").is_file():
        return False
    cfg = json.loads((path / "run_config.json").read_text(encoding="utf-8"))
    result = json.loads((path / "result.json").read_text(encoding="utf-8"))
    cli = cfg["cli"]
    expected_backbone = row["protocol"] == "joint" or row["method"] == "a1"
    if not (cfg["reference_sha256"] == row["reference_sha256"]
            and result["reference_sha256"] == row["reference_sha256"]
            and result["method"] == row["method"]
            and int(result["groups"]) == int(row["groups"])
            and bool(result["train_backbone"]) == expected_backbone
            and math.isclose(float(cli["lr_head"]), float(lr), rel_tol=1e-9)
            and math.isclose(float(cli["lr_stat"]), 1e-4, rel_tol=1e-9)
            and math.isclose(float(cli["lr_backbone"]), 2e-5, rel_tol=1e-9)
            and int(cli["epochs"]) == 10 and len(result["train_history"]) == 10):
        raise ValueError(f"Completed trial protocol mismatch: {path}")
    return True


def locate_or_create_trial(root: Path, row, lr):
    base = trial_root(root, row, lr)
    candidates = [base, *(Path(str(base) + f"_retry{i}") for i in range(1, 101))]
    for path in candidates:
        if valid_trial(path, row, lr):
            return path, True
    for path in candidates:
        if not path.exists():
            return path, False
    raise RuntimeError(f"Too many incomplete attempts: {base}")


def summarize(args, rows):
    root = Path(args.output_root)
    choices, best_rows, incomplete = [], [], []
    for row in rows:
        candidates = [{"lr_head": "5e-5", "val_mse": float(row["pilot_val_mse"]),
                       "test_mse": float(row["pilot_test_mse"]),
                       "test_mae": float(row["pilot_test_mae"]),
                       "source": "reused_pilot", "path": ""}]
        for lr in ("2e-5", "1e-4"):
            path = trial_root(root, row, lr)
            paths = [path, *(Path(str(path) + f"_retry{i}") for i in range(1, 101))]
            good = next((p for p in paths if valid_trial(p, row, lr)), None)
            if good is None:
                incomplete.append({"case": row["case"], "spec": spec_name(row), "lr_head": lr})
                continue
            result = json.loads((good / "result.json").read_text(encoding="utf-8"))
            candidates.append({"lr_head": lr, "val_mse": float(result["val"]["mse"]),
                               "test_mse": float(result["test"]["mse"]),
                               "test_mae": float(result["test"]["mae"]),
                               "source": "new_trial", "path": str(good)})
        for candidate in candidates:
            choices.append({**{k: row[k] for k in ("case", "method", "groups", "protocol", "lane")}, **candidate})
        if len(candidates) == 3:
            best = min(candidates, key=lambda x: (x["val_mse"], float(x["lr_head"])))
            best_rows.append({**{k: row[k] for k in ("case", "method", "groups", "protocol", "lane")}, **best,
                              "selection_split": "val", "pilot_val_mse": row["pilot_val_mse"],
                              "pilot_test_mse": row["pilot_test_mse"]})
    if choices:
        fields = ["case", "method", "groups", "protocol", "lane", "lr_head", "val_mse", "test_mse", "test_mae", "source", "path"]
        write_csv(root / "candidate_metrics.csv", choices, fields)
    if best_rows:
        write_csv(root / "best_by_method.csv", best_rows,
                  list(best_rows[0]))
    (root / "status.json").write_text(json.dumps({"assigned_methods": len(rows),
        "selected_methods": len(best_rows), "incomplete": incomplete,
        "selection_rule": "minimum validation MSE, TEST only reported after selection",
        "head_lrs": HEAD_LRS}, indent=2) + "\n", encoding="utf-8")
    return incomplete


def run(args):
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    rows = [x for x in read_csv(Path(args.manifest)) if int(x["server_index"]) == args.server_index]
    if len(rows) != 24 or sorted(set(int(x["gpu"]) for x in rows)) != [0, 1, 2, 3]:
        raise ValueError(f"Invalid shard {args.server_index}: {len(rows)} methods")
    project = Path(args.project_dir).resolve()
    for row in rows:
        if not (project / row["base_run_config"]).is_file():
            raise FileNotFoundError(project / row["base_run_config"])
    lock = threading.Lock()
    failures = []

    def log(message):
        with lock:
            print(message, flush=True)
            with (root / "queue.log").open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")

    def lane_worker(gpu):
        for row in (x for x in rows if int(x["gpu"]) == gpu):
            for lr in ("2e-5", "1e-4"):
                try:
                    output, complete = locate_or_create_trial(root, row, lr)
                    if complete:
                        log(f"REUSE gpu={gpu} {row['case']} {spec_name(row)} lr={lr}")
                        continue
                    output.mkdir(parents=True)
                    command = [sys.executable, "tools/run_fpem_statistical.py",
                        "--base_run_config", row["base_run_config"], "--output", str(output),
                        "--decomposition_type", "latent_statistics", "--stat_method", row["method"],
                        "--stat_groups", row["groups"], "--epochs", "10", "--seed", "2021",
                        "--gpu", "0", "--lr_head", lr, "--lr_backbone", "2e-5", "--lr_stat", "1e-4"]
                    if row["protocol"] == "joint":
                        command.append("--train_backbone")
                    if row["method"] == "t2_orthogonal":
                        command.append("--stat_use_orthogonal_basis")
                    if row["method"] in ("c1_covariance", "c3_dual"):
                        command.extend(("--stat_whitening", "full"))
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                    log(f"START gpu={gpu} {row['case']} {spec_name(row)} lr={lr} output={output}")
                    with (output / "run.log").open("w", encoding="utf-8") as handle:
                        status = subprocess.run(command, cwd=project, env=env, stdout=handle,
                                                stderr=subprocess.STDOUT, check=False).returncode
                    if status:
                        raise RuntimeError(f"exit={status}, log={output / 'run.log'}")
                    valid_trial(output, row, lr)
                    log(f"DONE gpu={gpu} {row['case']} {spec_name(row)} lr={lr}")
                except Exception as error:
                    issue = f"FAIL gpu={gpu} {row['case']} {spec_name(row)} lr={lr}: {error}"
                    failures.append(issue)
                    log(issue)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lane_worker, range(4)))
    incomplete = summarize(args, rows)
    log(f"SEARCH_END shard={args.server_index} failures={len(failures)} incomplete={len(incomplete)}")
    if failures or incomplete:
        raise SystemExit(1)
    (root / "search_complete").touch()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    planner = sub.add_parser("plan")
    planner.add_argument("--pilot-csv", required=True)
    planner.add_argument("--manifest", required=True)
    planner.add_argument("--verify-pilot", action="store_true")
    runner = sub.add_parser("run")
    runner.add_argument("--manifest", required=True)
    runner.add_argument("--server-index", required=True, type=int, choices=(0, 1, 2))
    runner.add_argument("--project-dir", required=True)
    runner.add_argument("--output-root", required=True)
    args = parser.parse_args()
    if args.command == "plan":
        plan(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
