#!/usr/bin/env python3
"""Preserve EIIL3 teacher labels and plans before replacing old split-run outputs.

This deliberately does not touch A0 checkpoints or source environment artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", choices=("server1", "server2"), required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--gpus", required=True, help="comma-separated available GPU indices")
    parser.add_argument("--prune-old-cases", action="store_true")
    parser.add_argument("--disable-old-queue", action="store_true")
    args = parser.parse_args()
    old = args.old_root.resolve()
    new = args.new_root.resolve()
    repo = Path(__file__).resolve().parents[1]
    results = (repo / "results").resolve()
    if old.parent != results or new.parent != results or old == new:
        raise ValueError("both roots must be distinct, direct children of repository results")
    if not old.is_dir() or old.name != "fpem_eiil3_safe_full_20261112":
        raise ValueError("unexpected old experiment directory")
    if new.name != "fpem_eiil3_fulltrain_20261113":
        raise ValueError("unexpected new experiment directory")
    gpus = [int(x) for x in args.gpus.split(",")]
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("invalid GPU list")
    for name in ("asset_audit", "audit"):
        source = old / name
        if source.is_dir():
            shutil.copytree(source, new / name, dirs_exist_ok=True)
    plans = new / "plans"
    plans.mkdir(parents=True, exist_ok=True)
    rows = []
    seen = set()
    for gpu in range(4):
        source = old / "plans" / f"{args.host}_gpu{gpu}.csv"
        with source.open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = row["case_key"]
                if key in seen:
                    raise ValueError(f"duplicate case {key}")
                seen.add(key)
                manifest = row["manifest"]
                if old.name in manifest:
                    manifest = manifest.replace(old.name, new.name)
                # A0 builders place exact A0-specific manifests outside the
                # split-run result root. Prefer one once it exists.
                if "asset_audit/manifest.jsonl" in manifest or "/audit/manifest.jsonl" in manifest:
                    parts = key.split("/")
                    candidate = repo / "results" / "fpem_eiil3_mse_priority_20261112" / "a0_assets" / parts[0] / parts[1] / parts[2] / parts[3] / "case_manifest.jsonl"
                    if candidate.is_file():
                        manifest = str(candidate.relative_to(repo))
                rows.append({"case_key": key, "manifest": manifest})
    # Already available A0 manifests first; pending A0 jobs run later.
    rows.sort(key=lambda row: (not (repo / row["manifest"]).is_file(), row["case_key"]))
    buckets = {gpu: [] for gpu in gpus}
    for index, row in enumerate(rows):
        buckets[gpus[index % len(gpus)]].append(row)
        source = old / "cases" / row["case_key"]
        target = new / "cases" / row["case_key"]
        for name in ("teacher_q.npz", "teacher_q.json"):
            if (source / name).is_file():
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / name, target / name)
    for gpu, items in buckets.items():
        with (plans / f"{args.host}_gpu{gpu}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("case_key", "manifest"))
            writer.writeheader()
            writer.writerows(items)
    q_count = len(list((new / "cases").rglob("teacher_q.npz")))
    summary = {"host": args.host, "cases": len(rows), "gpus": gpus,
               "cases_per_gpu": {str(k): len(v) for k, v in buckets.items()},
               "teacher_q_reused": q_count, "source": str(old), "destination": str(new)}
    if args.disable_old_queue:
        # The active wrapper is currently building A0 and opens this plan only
        # afterwards. Keep its A0 job alive but prevent old-protocol trials.
        old_plan = old / "plans" / f"{args.host}_gpu3.csv"
        if old_plan.is_file():
            with old_plan.open("w", newline="") as handle:
                handle.write("case_key,manifest\n")
            summary["old_gpu3_queue_disabled"] = True
    if args.prune_old_cases:
        # The old server2 screen may still be building missing A0 checkpoints;
        # its A0 queue and scripts must remain. Remove only split-run outputs.
        source_cases = old / "cases"
        if source_cases.is_dir():
            shutil.rmtree(source_cases)
        for name in ("all_results.csv", "summary.md"):
            (old / name).unlink(missing_ok=True)
        summary["old_split_case_outputs_removed"] = True
    (new / f"prepare_{args.host}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
