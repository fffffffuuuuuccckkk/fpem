#!/usr/bin/env python3
"""Recoverable, per-GPU EIIL3 sequential LR-search queue."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def log(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--queue-id", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument("--prune-old-features", action="store_true")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    rows = {json.loads(line)["case_key"]: json.loads(line)
            for line in args.manifest.read_text().splitlines()}
    plan = json.loads(args.plan.read_text())
    cases = plan[args.queue_id]
    state = args.output_root / "queue_logs" / (args.queue_id + ".jsonl")
    for case in cases:
        row = rows[case]
        old_case = args.old_root / case
        cache_ok = (args.reuse_features and (old_case / "features.npz").exists()
                    and (old_case / "features.json").exists())
        a0_ok = bool(row.get("A0_ckpt") and row.get("base_run_config") and
                     Path(row["A0_ckpt"]).exists() and Path(row["base_run_config"]).exists())
        if args.audit:
            print(json.dumps({"case": case, "cache": cache_ok, "A0": a0_ok,
                              "manifest_status": row["status"]}))
            continue
        if (args.output_root / case / "point_complete").exists():
            log(state, {"case": case, "status": "SKIP_COMPLETE"})
            continue
        if not (cache_ok or a0_ok) or row["status"] == "MISSING_A0" and not cache_ok:
            log(state, {"case": case, "status": "BLOCKED_INPUT",
                        "cache": cache_ok, "A0": a0_ok})
            continue
        command = [sys.executable, str(Path(__file__).with_name("run_fpem_eiil3_sequential.py")),
                   "--manifest", str(args.manifest), "--case-key", case,
                   "--output-root", str(args.output_root), "--old-root", str(args.old_root),
                   "--gpu", str(args.gpu), "--generate-missing-q",
                   "--stop-on-gain", "--resume"]
        if cache_ok:
            command.append("--reuse-features")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        # Child sees one physical GPU as cuda:0.
        command[command.index("--gpu") + 1] = "0"
        log(state, {"case": case, "status": "START", "time": time.time(), "command": command})
        with (args.output_root / "queue_logs" / (args.queue_id + ".log")).open("a") as output:
            output.write("\nSTART " + case + "\n")
            output.flush()
            code = subprocess.call(command, stdout=output, stderr=subprocess.STDOUT, env=env)
        log(state, {"case": case, "status": "COMPLETE" if code == 0 and
                    (args.output_root / case / "point_complete").exists() else "FAILED",
                    "exit_code": code, "time": time.time()})
        if args.prune_old_features and cache_ok and code == 0:
            new_case = (args.output_root / case).resolve()
            old_file = (old_case / "features.npz").resolve()
            old_root = args.old_root.resolve()
            if (new_case / "point_complete").exists() and old_file.is_relative_to(old_root):
                metrics = json.loads((new_case / "point_metrics.json").read_text())
                reused = json.loads((new_case / "input_cache.json").read_text())
                if (metrics["A0_sha256"] == row["A0_sha256"] and reused["path"] == str(old_case)
                        and reused["q_sha256"] == metrics["q_sha256"]):
                    size = old_file.stat().st_size
                    old_file.unlink()
                    log(state, {"case": case, "status": "PRUNED_REBUILDABLE_OLD_FEATURES",
                                "bytes": size, "path": str(old_file)})


if __name__ == "__main__":
    main()
