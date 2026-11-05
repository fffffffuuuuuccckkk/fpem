#!/usr/bin/env python3
"""One persistent, restartable dispatcher for explicitly approved FPEM jobs.

Run on server 1.  The JSON manifest is data, not executable code; adding a job
there makes it available at the next five-minute scan.  A job is never guessed
from a result directory, since some FPEM sweeps are intentionally gated off.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path("/data/OuXiaoyu/Time-Series-Library-FPEM")
ROOT = PROJECT / "results/fpem_performance_optimization_20261103/dispatcher"
MANIFEST = ROOT / "manifest.json"
STATE = ROOT / "state.json"
LOG = ROOT / "dispatcher.log"
REPORT = ROOT / "live_summary"
HOSTS = {
    "server1": {"address": None, "python": "/data/OuXiaoyu/miniconda3/envs/basicts/bin/python", "max_jobs": 3},
    "server2": {"address": "OuXiaoyu@211.71.76.187", "python": "/data/OuXiaoyu/miniconda3/envs/basicts/bin/python", "max_jobs": 4},
    "server3": {"address": "OuXiaoyu@211.71.76.44", "python": "/data/OuXiaoyu/miniconda3/envs/tslib/bin/python", "max_jobs": 3},
}
SSH_KEY = "/data/OuXiaoyu/.ssh/fpem_scheduler_rsa"
INTERVAL_SECONDS = 300
MIN_AVAILABLE_RAM_KIB = 16 * 1024 * 1024
MAX_IDLE_MEMORY_MIB = 350
MAX_IDLE_UTIL = 5


def log(message: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {message}"
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def atomic_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def run(host: str, command: str, timeout: int = 20) -> subprocess.CompletedProcess[str] | None:
    cfg = HOSTS[host]
    if cfg["address"] is None:
        argv = ["bash", "-lc", command]
    else:
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                "-o", "StrictHostKeyChecking=yes", "-i", SSH_KEY,
                cfg["address"], command]
    try:
        return subprocess.run(argv, text=True, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None


def validated_tasks() -> list[dict]:
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tasks = raw["tasks"]
    seen = set()
    for task in tasks:
        ident = task["id"]
        if not re.fullmatch(r"[A-Za-z0-9_]{1,64}", ident) or ident in seen:
            raise ValueError(f"invalid or duplicate task id: {ident}")
        seen.add(ident)
        script = task["script"]
        if not script.startswith("scripts/") or ".." in script or not script.endswith(".sh"):
            raise ValueError(f"unsafe script: {script}")
        for key in ("complete_rel", "required_rel"):
            value = task[key]
            if value.startswith("/") or ".." in Path(value).parts:
                raise ValueError(f"unsafe {key}: {value}")
        if not task["allowed_hosts"] or any(h not in HOSTS for h in task["allowed_hosts"]):
            raise ValueError(f"invalid hosts for {ident}")
        if not isinstance(task["args"], list) or task["args"].count("{gpu}") != 1:
            raise ValueError(f"one GPU placeholder required for {ident}")
    return sorted(tasks, key=lambda task: (-int(task.get("priority", 0)), task["id"]))


def host_snapshot(host: str) -> dict | None:
    command = (
        "nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu "
        "--format=csv,noheader,nounits; "
        "printf 'APPS\\n'; "
        "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits; "
        "printf 'RAM\\n'; grep MemAvailable /proc/meminfo; "
        "printf 'SCREENS\\n'; screen -ls 2>/dev/null || true"
    )
    result = run(host, command)
    if result is None or result.returncode != 0:
        return None
    output = result.stdout
    if not all(marker in output for marker in ("APPS\n", "RAM\n", "SCREENS\n")):
        return None
    gpu_text, remainder = output.split("APPS\n", 1)
    apps_text, remainder = remainder.split("RAM\n", 1)
    ram_text, screen_text = remainder.split("SCREENS\n", 1)
    busy_uuids = {line.split(",", 1)[0].strip() for line in apps_text.splitlines() if "," in line}
    match = re.search(r"MemAvailable:\s*(\d+)\s+kB", ram_text)
    if not match:
        return None
    gpus = []
    for line in gpu_text.splitlines():
        columns = [part.strip() for part in line.split(",")]
        if len(columns) != 4:
            continue
        try:
            index, uuid, memory, utilization = int(columns[0]), columns[1], int(columns[2]), int(columns[3])
        except ValueError:
            continue
        gpus.append({"index": index, "uuid": uuid, "memory": memory, "util": utilization,
                     "idle": uuid not in busy_uuids and memory <= MAX_IDLE_MEMORY_MIB
                     and utilization <= MAX_IDLE_UTIL})
    if not gpus:
        return None
    return {"gpus": gpus, "ram_kib": int(match.group(1)), "screens": screen_text}


def remote_exists(host: str, relative: str) -> bool | None:
    target = PROJECT / relative
    result = run(host, "test -f " + shlex.quote(str(target)))
    return None if result is None or result.returncode == 255 else result.returncode == 0


def screen_name(ident: str) -> str:
    return "fpem_disp_" + ident


def reconcile(tasks: list[dict], state: dict, snapshots: dict) -> bool:
    changed = False
    for task in tasks:
        ident = task["id"]
        current = state.setdefault(ident, {"status": "pending", "attempts": 0})
        if current["status"] == "complete":
            continue
        if current["status"] == "running":
            host = current["host"]
            if snapshots.get(host) is None:
                continue  # Never infer failure from an SSH outage.
            marker = remote_exists(host, task["complete_rel"])
            if marker:
                current.update(status="complete", finished_at=time.time())
                log(f"COMPLETE {ident} on {host}")
                changed = True
            elif marker is False and screen_name(ident) not in snapshots[host]["screens"]:
                current.update(status="pending", retry_after=time.time() + 60)
                log(f"RETRY_PENDING {ident} on {host}: screen ended without completion marker")
                changed = True
            continue
        for host in task["allowed_hosts"]:
            if snapshots.get(host) is not None and remote_exists(host, task["complete_rel"]):
                current.update(status="complete", host=host, finished_at=time.time())
                log(f"FOUND_COMPLETE {ident} on {host}")
                changed = True
                break
    return changed


def dispatch(tasks: list[dict], state: dict, snapshots: dict, dry_run: bool) -> bool:
    changed = False
    reserved = {host: set() for host in HOSTS}
    running = {host: 0 for host in HOSTS}
    for current in state.values():
        if current.get("status") == "running" and current.get("host") in HOSTS:
            host = current["host"]
            running[host] += 1
            reserved[host].add(current["gpu"])
    for task in tasks:
        ident = task["id"]
        current = state[ident]
        if current["status"] != "pending" or current.get("retry_after", 0) > time.time():
            continue
        if current["attempts"] >= int(task.get("max_attempts", 3)):
            current["status"] = "failed"
            log(f"FAILED {ident}: exhausted launches")
            changed = True
            continue
        # Manifest order is the preference order; only hosts with inputs qualify.
        for host in task["allowed_hosts"]:
            snapshot = snapshots.get(host)
            if snapshot is None or snapshot["ram_kib"] < MIN_AVAILABLE_RAM_KIB:
                continue
            if running[host] >= HOSTS[host]["max_jobs"]:
                continue
            candidates = [gpu["index"] for gpu in snapshot["gpus"]
                          if gpu["idle"] and gpu["index"] not in reserved[host]]
            if not candidates:
                continue
            if not remote_exists(host, task["required_rel"]):
                continue
            if not remote_exists(host, task["script"]):
                continue
            gpu = min(candidates)
            arguments = [str(gpu) if arg == "{gpu}" else str(arg) for arg in task["args"]]
            command = "cd " + shlex.quote(str(PROJECT)) + " && screen -dmS " + shlex.quote(screen_name(ident))
            command += " env PYTHON_BIN=" + shlex.quote(HOSTS[host]["python"])
            command += " bash " + shlex.quote(task["script"])
            command += " " + " ".join(shlex.quote(arg) for arg in arguments)
            if dry_run:
                log(f"DRY_RUN {ident} -> {host} GPU {gpu}")
            else:
                result = run(host, command)
                if result is None or result.returncode != 0:
                    log(f"LAUNCH_ERROR {ident} -> {host} GPU {gpu}: {'' if result is None else result.stderr.strip()}")
                    continue
                current.update(status="running", host=host, gpu=gpu, started_at=time.time(),
                               attempts=current["attempts"] + 1)
                log(f"START {ident} -> {host} GPU {gpu}")
                changed = True
            reserved[host].add(gpu)
            running[host] += 1
            break
    return changed


def sync_and_report(tasks: list[dict], state: dict, snapshots: dict) -> None:
    """Mirror non-checkpoint artifacts and write an always-readable live index."""
    REPORT.mkdir(parents=True, exist_ok=True)
    source_root = str(PROJECT / "results/fpem_performance_optimization_20261103")
    sync_state = {}
    for host in ("server2", "server3"):
        destination = PROJECT / f"results/fpem_performance_optimization_20261103/{host}_imports"
        destination.mkdir(parents=True, exist_ok=True)
        cfg = HOSTS[host]
        ssh_cmd = f"ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=yes -i {SSH_KEY}"
        argv = ["rsync", "-a", "--partial", "--exclude=*.pt",
                "--exclude=dispatcher/", "--exclude=server*_imports/", "-e", ssh_cmd,
                f"{cfg['address']}:{source_root}/", str(destination) + "/"]
        try:
            result = subprocess.run(argv, text=True, capture_output=True, timeout=180, check=False)
            sync_state[host] = {"ok": result.returncode == 0,
                                "error": result.stderr.strip()[-400:] if result.returncode else ""}
        except (OSError, subprocess.TimeoutExpired) as exc:
            sync_state[host] = {"ok": False, "error": str(exc)[-400:]}

    # The resumed prototype pilot predates this optimization root. Keep a
    # checkpoint-free mirror so its stage evidence is visible on server 1.
    prototype_remote = str(PROJECT / "results/fpem_prototype_guided_invariant_20261103/moderntcn/ETTm2/pred_96")
    prototype_local = PROJECT / "results/fpem_performance_optimization_20261103/server3_imports/prototype_guided_invariant/moderntcn/ETTm2/pred_96"
    prototype_local.mkdir(parents=True, exist_ok=True)
    proto_ssh = f"ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=yes -i {SSH_KEY}"
    try:
        proc = subprocess.run(["rsync", "-a", "--partial", "--exclude=*.pt",
                               "--exclude=*.pth", "--exclude=*.ckpt", "-e", proto_ssh,
                               f"{HOSTS['server3']['address']}:{prototype_remote}/",
                               str(prototype_local) + "/"],
                              text=True, capture_output=True, timeout=180, check=False)
        sync_state["prototype_server3"] = {"ok": proc.returncode == 0,
                                            "error": proc.stderr.strip()[-400:] if proc.returncode else ""}
    except (OSError, subprocess.TimeoutExpired) as exc:
        sync_state["prototype_server3"] = {"ok": False, "error": str(exc)[-400:]}

    status_rows = []
    for task in tasks:
        current = state.get(task["id"], {})
        host = current.get("host", "")
        base = (PROJECT / task["complete_rel"]).parent
        if host in ("server2", "server3"):
            if task["id"] == "resume_prototype_moderntcn_ETTm2_96":
                # This legacy pilot lives outside the optimization root and is
                # mirrored separately above. relative_to() would raise here.
                base = prototype_local
            else:
                relative = base.relative_to(PROJECT / "results/fpem_performance_optimization_20261103")
                base = PROJECT / f"results/fpem_performance_optimization_20261103/{host}_imports" / relative
        tail = ""
        queue_log = base / "queue.log"
        if queue_log.exists():
            try:
                lines = queue_log.read_text(encoding="utf-8", errors="replace").splitlines()
                tail = lines[-1][-250:] if lines else ""
            except OSError:
                pass
        status_rows.append({"task": task["id"], "status": current.get("status", "pending"),
                            "host": host, "gpu": current.get("gpu", ""),
                            "attempts": current.get("attempts", 0), "latest_log": tail})
    atomic_csv(REPORT / "task_status.csv",
               ["task", "status", "host", "gpu", "attempts", "latest_log"], status_rows)

    metric_rows = []
    locations = []
    for scope in ("direct_shared_pilot", "direct_expansion"):
        locations.append(("server1", scope, PROJECT / f"results/fpem_performance_optimization_20261103/{scope}"))
        locations += [(host, scope, PROJECT / f"results/fpem_performance_optimization_20261103/{host}_imports/{scope}")
                      for host in ("server2", "server3")]
    for host, scope, base in locations:
        if not base.exists():
            continue
        for file in sorted(base.glob("*/*/pred_*/*/metrics.json")):
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                metric_rows.append({"host": host, "experiment": scope,
                                    "backbone": data.get("backbone", ""),
                                    "dataset": data.get("dataset", ""),
                                    "pred_len": data.get("pred_len", ""),
                                    "method": data.get("method", file.parent.name),
                                    "val_mse": data.get("val", {}).get("mse", ""),
                                    "val_mae": data.get("val", {}).get("mae", ""),
                                    "test_mse": data.get("test", {}).get("mse", ""),
                                    "test_mae": data.get("test", {}).get("mae", ""),
                                    "best_epoch": data.get("best_epoch", ""),
                                    "reference_sha256": data.get("reference_sha256", ""),
                                    "source_file": str(file)})
            except (OSError, ValueError, TypeError):
                continue  # A writer may be replacing a file during this scan.
    atomic_csv(REPORT / "method_metrics.csv",
               ["host", "experiment", "backbone", "dataset", "pred_len", "method", "val_mse", "val_mae",
                "test_mse", "test_mae", "best_epoch", "reference_sha256", "source_file"], metric_rows)
    ridge_rows = []
    ridge_locations = [("server1", PROJECT / "results/fpem_performance_optimization_20261103/ridge_audit")]
    ridge_locations += [(host, PROJECT / f"results/fpem_performance_optimization_20261103/{host}_imports/ridge_audit")
                        for host in ("server2", "server3")]
    for host, base in ridge_locations:
        for file in sorted(base.glob("*/*/pred_*/selection.json")):
            try:
                item = json.loads(file.read_text(encoding="utf-8"))
                choice = item["best"]
                ridge_rows.append({"host": host, "backbone": item["backbone"],
                                   "dataset": item["dataset"], "pred_len": item["pred_len"],
                                   "mode": choice["mode"], "ridge": choice["ridge"],
                                   "oof_mse": choice["train_forward_oof_mse"],
                                   "val_mse": choice["val_mse"],
                                   "test_mse": item["selected_test"]["mse"],
                                   "a0_test_mse": item["a0_test"]["mse"],
                                   "source_file": str(file)})
            except (OSError, ValueError, KeyError, TypeError):
                continue
    atomic_csv(REPORT / "ridge_audit_metrics.csv",
               ["host", "backbone", "dataset", "pred_len", "mode", "ridge",
                "oof_mse", "val_mse", "test_mse", "a0_test_mse", "source_file"], ridge_rows)
    f03_rows = []
    f03_locations = [("server1", PROJECT / "results/fpem_performance_optimization_20261103/f03_lr_search")]
    f03_locations += [(host, PROJECT / f"results/fpem_performance_optimization_20261103/{host}_imports/f03_lr_search")
                      for host in ("server2", "server3")]
    for host, base in f03_locations:
        for file in sorted(base.glob("*/*/pred_*/selection.json")):
            try:
                item = json.loads(file.read_text(encoding="utf-8"))
                selected = item["selected"]
                f03_rows.append({"host": host, "backbone": item["backbone"],
                                 "dataset": item["dataset"], "pred_len": item["pred_len"],
                                 "selected_lr": selected["lr"],
                                 "selected_epoch": selected["best_epoch"],
                                 "val_mse": selected["val"]["mse"],
                                 "test_mse": selected["test"]["mse"],
                                 "a0_test_mse": item["a0_test"]["mse"],
                                 "source_file": str(file)})
            except (OSError, ValueError, KeyError, TypeError):
                continue
    atomic_csv(REPORT / "f03_lr_metrics.csv",
               ["host", "backbone", "dataset", "pred_len", "selected_lr",
                "selected_epoch", "val_mse", "test_mse", "a0_test_mse", "source_file"], f03_rows)
    anchor_rows = []
    anchor_locations = [("server1", PROJECT / "results/fpem_performance_optimization_20261103/f03_anchor_search")]
    anchor_locations += [(host, PROJECT / f"results/fpem_performance_optimization_20261103/{host}_imports/f03_anchor_search")
                         for host in ("server2", "server3")]
    for host, base in anchor_locations:
        for file in sorted(base.glob("*/*/pred_*/selection.json")):
            try:
                item = json.loads(file.read_text(encoding="utf-8"))
                selected = item["selected"]
                anchor_rows.append({"host": host, "backbone": item["backbone"],
                                    "dataset": item["dataset"], "pred_len": item["pred_len"],
                                    "selected_anchor": selected.get("anchor_lambda", 0.0),
                                    "selected_epoch": selected["best_epoch"],
                                    "val_mse": selected["val"]["mse"],
                                    "test_mse": selected["test"]["mse"],
                                    "a0_test_mse": item["a0_test"]["mse"],
                                    "source_file": str(file)})
            except (OSError, ValueError, KeyError, TypeError):
                continue
    atomic_csv(REPORT / "f03_anchor_metrics.csv",
               ["host", "backbone", "dataset", "pred_len", "selected_anchor",
                "selected_epoch", "val_mse", "test_mse", "a0_test_mse", "source_file"], anchor_rows)
    robustness_rows = []
    robustness_locations = [("server1", PROJECT / "results/fpem_performance_optimization_20261103/f03_robustness")]
    robustness_locations += [(host, PROJECT / f"results/fpem_performance_optimization_20261103/{host}_imports/f03_robustness")
                             for host in ("server2", "server3")]
    for host, base in robustness_locations:
        for file in sorted(base.glob("*/*/pred_*/summary.json")):
            try:
                item = json.loads(file.read_text(encoding="utf-8"))
                robustness_rows.append({"host": host, "backbone": item["backbone"],
                                        "dataset": item["dataset"], "pred_len": item["pred_len"],
                                        "a0_test_mse": item["a0_test"]["mse"],
                                        "test_mse_mean": item["test_mse_mean"],
                                        "test_mse_std": item["test_mse_std"],
                                        "test_win_count": item["test_win_count"],
                                        "seed_count": len(item["selected"]),
                                        "source_file": str(file)})
            except (OSError, ValueError, KeyError, TypeError):
                continue
    atomic_csv(REPORT / "f03_robustness_metrics.csv",
               ["host", "backbone", "dataset", "pred_len", "a0_test_mse",
                "test_mse_mean", "test_mse_std", "test_win_count", "seed_count",
                "source_file"], robustness_rows)
    fusion_rows = []
    fusion_locations = [("server1", PROJECT / "results/fpem_performance_optimization_20261103/f03_r3_fusion")]
    fusion_locations += [(host, PROJECT / f"results/fpem_performance_optimization_20261103/{host}_imports/f03_r3_fusion")
                         for host in ("server2", "server3")]
    for host, base in fusion_locations:
        for file in sorted(base.glob("*/*/pred_*/seed_*/metrics.json")):
            try:
                item = json.loads(file.read_text(encoding="utf-8"))
                fusion_rows.append({"host": host, "backbone": item["backbone"],
                                    "dataset": item["dataset"], "pred_len": item["pred_len"],
                                    "seed": item["seed"], "a0_test_mse": item["a0"]["mse"],
                                    "f03_test_mse": item["f03"]["mse"],
                                    "f03_r3_test_mse": item["f03_r3"]["mse"],
                                    "f03_r3_test_mae": item["f03_r3"]["mae"],
                                    "increment_mse": item["f03_r3"]["mse"] - item["f03"]["mse"],
                                    "null_mean": item["null_mean"],
                                    "active_sample_ratio": item["active_sample_ratio"],
                                    "source_file": str(file)})
            except (OSError, ValueError, KeyError, TypeError):
                continue
    atomic_csv(REPORT / "f03_r3_metrics.csv",
               ["host", "backbone", "dataset", "pred_len", "seed", "a0_test_mse",
                "f03_test_mse", "f03_r3_test_mse", "f03_r3_test_mae", "increment_mse",
                "null_mean", "active_sample_ratio", "source_file"], fusion_rows)
    continuation_rows = []
    continuation_locations = [("server1", PROJECT / "results/fpem_performance_optimization_20261103/f03_backbone_continuation")]
    continuation_locations += [(host, PROJECT / f"results/fpem_performance_optimization_20261103/{host}_imports/f03_backbone_continuation")
                               for host in ("server2", "server3")]
    for host, base in continuation_locations:
        for file in sorted(base.glob("*/*/pred_*/seed_*/*/metrics.json")):
            try:
                item = json.loads(file.read_text(encoding="utf-8"))
                continuation_rows.append({"host": host, "dataset": item["dataset"],
                                          "pred_len": item["pred_len"], "seed": item["seed"],
                                          "mode": item["mode"], "best_extra_epoch": item["best_extra_epoch"],
                                          "f03_test_mse": item["initial_test"]["mse"],
                                          "selected_val_mse": item["selected_val"]["mse"],
                                          "selected_test_mse": item["selected_test"]["mse"],
                                          "selected_test_mae": item["selected_test"]["mae"],
                                          "gain_vs_f03": item["initial_test"]["mse"] - item["selected_test"]["mse"],
                                          "source_file": str(file)})
            except (OSError, ValueError, KeyError, TypeError):
                continue
    atomic_csv(REPORT / "f03_backbone_continuation_metrics.csv",
               ["host", "dataset", "pred_len", "seed", "mode", "best_extra_epoch",
                "f03_test_mse", "selected_val_mse", "selected_test_mse", "selected_test_mae",
                "gain_vs_f03", "source_file"], continuation_rows)
    prototype_status = {name: (prototype_local / name).is_file()
                        for name in ("shared_queue_complete", "finetune_queue_complete",
                                     "forward_queue_complete", "fusion_queue_complete",
                                     "analysis_queue_complete", "all_stages_complete")}
    atomic_json(REPORT / "prototype_moderntcn_status.json", prototype_status)
    summary = {"updated_at_utc": datetime.now(timezone.utc).isoformat(),
               "poll_interval_seconds": INTERVAL_SECONDS, "tasks": status_rows,
               "completed_method_count": len(metric_rows),
               "completed_ridge_audits": len(ridge_rows),
               "completed_f03_searches": len(f03_rows),
               "completed_f03_anchor_searches": len(anchor_rows),
               "completed_f03_robustness_cases": len(robustness_rows),
               "completed_f03_r3_seed_trials": len(fusion_rows), "mirror_sync": sync_state,
               "completed_f03_continuation_arms": len(continuation_rows),
               "gpu_snapshot": {host: (None if value is None else value["gpus"])
                                for host, value in snapshots.items()}}
    atomic_json(REPORT / "status.json", summary)
    log("REPORT_UPDATED methods=" + str(len(metric_rows)) + " sync=" +
        ",".join(f"{host}:{'ok' if item['ok'] else 'error'}" for host, item in sync_state.items()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "dispatcher.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
        while True:
            tasks = validated_tasks()
            snapshots = {host: host_snapshot(host) for host in HOSTS}
            reconcile(tasks, state, snapshots)
            dispatch(tasks, state, snapshots, args.dry_run)
            if not args.dry_run:
                atomic_json(STATE, state)
                try:
                    sync_and_report(tasks, state, snapshots)
                except Exception as exc:  # Reporting must never stop live training.
                    log(f"REPORT_ERROR {type(exc).__name__}: {exc}")
            statuses = {task["id"]: state[task["id"]]["status"] for task in tasks}
            log("SNAPSHOT " + json.dumps(statuses, sort_keys=True))
            if args.once or args.dry_run:
                return 0
            if tasks and all(status == "complete" for status in statuses.values()):
                # Stay resident: a future planning run may append approved tasks.
                if not getattr(main, "all_complete_logged", False):
                    log("ALL_COMPLETE; dispatcher remains idle for new manifest tasks")
                    main.all_complete_logged = True
            else:
                main.all_complete_logged = False
            if any(status == "failed" for status in statuses.values()):
                log("UNRECOVERABLE_FAILURE; dispatcher exits for inspection")
                return 1
            time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
