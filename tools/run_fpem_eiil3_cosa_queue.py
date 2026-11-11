#!/usr/bin/env python3
"""Resumable GPU queue; reuse completed Ridge/F03 cases, skip beneficial Ridge cases."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(value) + "\n")


def command(arguments, log):
    with log.open("a") as output:
        output.write("COMMAND " + " ".join(map(str, arguments)) + "\n")
        output.flush()
        return subprocess.run(arguments, stdout=output, stderr=subprocess.STDOUT, check=False).returncode


def gpu_free(gpu):
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                             "--format=csv,noheader,nounits", "-i", str(gpu)],
                            capture_output=True, text=True, check=False)
    if result.returncode:
        return False
    used, utilization = [int(piece.strip()) for piece in result.stdout.splitlines()[0].split(",")]
    return used < 500 and utilization < 20


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, default=Path("results/fpem_eiil3_20261110"))
    p.add_argument("--output-root", type=Path, default=Path("results/fpem_eiil3_cosa_20261110"))
    p.add_argument("--host-group", choices=("server1", "server2"), required=True)
    p.add_argument("--gpu", type=int, default=3)
    p.add_argument("--poll-seconds", type=int, default=300)
    p.add_argument("--max-hours", type=int, default=72)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    rows = [json.loads(line) for line in (args.source_root / "manifest.jsonl").read_text().splitlines()]
    if args.host_group == "server1":
        rows = [row for row in rows if row["backbone"] == "patchtst" or
                (row["backbone"] == "moderntcn" and row["pred_len"] in (96, 192))]
        source_queues = ("patchtst_short_gpu0", "patchtst_long_gpu1", "moderntcn_short_gpu2")
    else:
        rows = [row for row in rows if row["backbone"] == "itransformer" or
                (row["backbone"] == "moderntcn" and row["pred_len"] in (336, 720))]
        source_queues = ("itransformer_short_gpu0", "itransformer_long_gpu1", "moderntcn_long_gpu2")
    if args.dry_run:
        print(json.dumps({"host_group": args.host_group, "cases": len(rows),
                          "source_queues": source_queues})); return
    args.output_root.mkdir(parents=True, exist_ok=True)
    log = args.output_root / f"{args.host_group}_gpu{args.gpu}.log"
    status = args.output_root / f"{args.host_group}_gpu{args.gpu}.jsonl"
    point = Path(__file__).with_name("run_fpem_eiil3_cosa.py")
    flow = Path(__file__).with_name("run_fpem_eiil3_flow.py")
    deadline = time.time() + args.max_hours * 3600
    handled = set()
    while time.time() < deadline:
        ready = [row for row in rows if row["case_key"] not in handled and
                 (args.source_root / row["case_key"] / "point_complete").exists()]
        if ready and gpu_free(args.gpu):
            row = ready[0]
            case = row["case_key"]
            source = args.source_root / case
            dest = args.output_root / case
            start = time.time()
            code = command([sys.executable, str(point), "--source-case", str(source),
                            "--output-case", str(dest), "--gpu", str(args.gpu),
                            "--skip-good-ridge", "--resume"], log)
            append(status, {"case": case, "stage": "linear", "exit_code": code,
                            "seconds": time.time() - start})
            if code == 0 and (dest / "point_complete").exists():
                start = time.time()
                flow_code = command([sys.executable, str(flow), "--case-dir", str(dest),
                                     "--gpu", str(args.gpu), "--resume"], log)
                append(status, {"case": case, "stage": "flow", "exit_code": flow_code,
                                "seconds": time.time() - start})
            handled.add(case)
            continue
        source_finished = all((args.source_root / "queue_logs" / (name + ".jsonl")).exists()
                              and '"phase": "queue_end"' in
                              (args.source_root / "queue_logs" / (name + ".jsonl")).read_text()
                              for name in source_queues)
        if source_finished and not ready:
            append(status, {"stage": "queue_end", "handled": len(handled),
                            "blocked_or_unavailable": len(rows) - len(handled)})
            return
        time.sleep(args.poll_seconds)
    append(status, {"stage": "time_limit", "handled": len(handled),
                    "note": "safe to resume; completed results retained"})


if __name__ == "__main__":
    main()
