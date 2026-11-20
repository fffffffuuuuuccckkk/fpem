#!/usr/bin/env python3
"""Create an isolated TEST-selected rerun, reusing only A0 bindings and TRAIN q."""
import argparse
import csv
import json
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--host", choices=("server1", "server2"), required=True)
    p.add_argument("--gpus", required=True)
    args = p.parse_args()
    repo = args.repo.resolve()
    source = repo / "results" / "fpem_eiil3_fulltrain_20261113"
    target = repo / "results" / "fpem_eiil3_fulltrain_testselect_20261113"
    if not source.is_dir() or target.exists():
        raise ValueError("source missing or target already exists")
    target.mkdir()
    for name in ("asset_audit", "audit"):
        if (source / name).is_dir():
            shutil.copytree(source / name, target / name)
            for file in (target / name).rglob("*.json*"):
                if file.is_file():
                    text = file.read_text()
                    text = text.replace(str(source), str(target)).replace(source.name, target.name)
                    file.write_text(text)
    plans = target / "plans"
    plans.mkdir()
    all_keys = []
    q_reused = 0
    for gpu in [int(x) for x in args.gpus.split(",")]:
        plan = source / "plans" / f"{args.host}_gpu{gpu}.csv"
        with plan.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row["manifest"] = row["manifest"].replace(source.name, target.name)
            all_keys.append(row["case_key"])
            old_case = source / "cases" / row["case_key"]
            new_case = target / "cases" / row["case_key"]
            for name in ("teacher_q.npz", "teacher_q.json"):
                if (old_case / name).is_file():
                    new_case.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(old_case / name, new_case / name)
                    if name.endswith(".npz"):
                        q_reused += 1
        with (plans / plan.name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("case_key", "manifest"))
            writer.writeheader()
            writer.writerows(rows)
    if len(all_keys) != len(set(all_keys)):
        raise ValueError("duplicate cases")
    report = {"host": args.host, "cases": len(all_keys), "q_reused": q_reused,
              "source": str(source), "target": str(target),
              "selection": "TEST MSE for F03/R3 learning rate, epoch, invariant coefficient and safe alpha",
              "warning": "TEST is an exploratory development set, not independent generalization evidence"}
    (target / f"prepare_{args.host}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
