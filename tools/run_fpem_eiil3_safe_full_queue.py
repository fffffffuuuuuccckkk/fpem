#!/usr/bin/env python3
"""Sequential, restartable GPU queue; each case creates four point variants."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


def ready_manifest(manifest: Path, case_key: str, repo: Path) -> Optional[Path]:
    candidates = [manifest]
    parts = case_key.split("/")
    if len(parts) == 4:
        candidates.append(repo / "results" / "fpem_eiil3_mse_priority_20261112" /
                          "a0_assets" / parts[0] / parts[1] / parts[2] / parts[3] /
                          "case_manifest.jsonl")
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            for line in candidate.read_text().splitlines():
                item = json.loads(line)
                if item.get("case_key") == case_key and item.get("status") != "MISSING_A0":
                    a0 = Path(item["A0_ckpt"])
                    if a0.is_file():
                        return candidate
        except (OSError, ValueError, KeyError):
            continue
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--retry", type=int, default=2)
    parser.add_argument("--full-train", action="store_true")
    parser.add_argument("--selection-split", choices=("val", "test"), default="val")
    parser.add_argument("--environment-supervision",
                        choices=("predictive", "shuffled", "random", "uniform"),
                        default="predictive")
    parser.add_argument("--exhaustive-r3", action="store_true")
    parser.add_argument("--a0-wait-hours", type=float, default=72.)
    parser.add_argument("--ready-first", action="store_true",
                        help="Process presently ready cases before waiting for missing A0 assets")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    logs = args.output_root / "queue_logs"
    logs.mkdir(exist_ok=True)
    script = Path(__file__).with_name("run_fpem_eiil3_safe_full.py")
    aggregate = Path(__file__).with_name("aggregate_fpem_eiil3_safe_full.py")
    with args.plan.open(newline="") as stream:
        items = list(csv.DictReader(stream))
    repo = Path(__file__).resolve().parents[1]
    if args.ready_first:
        items.sort(key=lambda item: ready_manifest(Path(item["manifest"]), item["case_key"], repo) is None)
    for item in items:
        key = item["case_key"]
        manifest = Path(item["manifest"])
        deadline = time.monotonic() + args.a0_wait_hours * 3600
        chosen = ready_manifest(manifest, key, repo)
        if args.full_train:
            while chosen is None and time.monotonic() < deadline:
                time.sleep(60)
                chosen = ready_manifest(manifest, key, repo)
        if chosen is None:
            print(json.dumps({"case": key, "status": "MISSING_VALID_A0_MANIFEST",
                              "path": str(manifest)}), flush=True)
            continue
        log = logs / (key.replace("/", "_") + ".log")
        cmd = [sys.executable, str(script), "--manifest", str(chosen), "--case-key", key,
               "--output-root", str(args.output_root), "--gpu", str(args.gpu),
               "--resume", "--selection-split", args.selection_split,
               "--environment-supervision", args.environment_supervision]
        if not args.exhaustive_r3:
            cmd.append("--stop-on-gain")
        if args.full_train:
            cmd.append("--full-train")
        for attempt in range(1, args.retry + 1):
            print(json.dumps({"case": key, "status": "START", "attempt": attempt}), flush=True)
            with log.open("a") as handle:
                handle.write(f"START attempt={attempt} command={' '.join(cmd)}\n")
                handle.flush()
                code = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT).returncode
                handle.write(f"EXIT code={code}\n")
            if code == 0:
                print(json.dumps({"case": key, "status": "COMPLETE_OR_REUSED"}), flush=True)
                break
            print(json.dumps({"case": key, "status": "FAILED", "attempt": attempt,
                              "exit_code": code, "log": str(log)}), flush=True)
            if attempt < args.retry:
                time.sleep(60)
        subprocess.run([sys.executable, str(aggregate), "--root", str(args.output_root)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
