#!/usr/bin/env python3
"""Remove superseded EIIL3 case results only after their replacement is complete.

Inputs and protected A0/environment assets outside the old result root are not
deleted. A case with no verified replacement is retained.
"""
import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--old-root", type=Path, required=True)
    p.add_argument("--new-root", type=Path, required=True)
    p.add_argument("--wait", action="store_true")
    args = p.parse_args()
    old_root = args.old_root.resolve()
    new_root = args.new_root.resolve()
    if old_root == new_root or old_root.name != "fpem_eiil3_20261110":
        raise ValueError("unexpected cleanup roots")
    if args.wait:
        while True:
            active = subprocess.run(["screen", "-ls"], capture_output=True, text=True).stdout
            if "fpem_eiil3_s1_gpu" not in active and "fpem_eiil3_s2_gpu" not in active \
                    and "fpem_eiil3_memory_retry" not in active:
                break
            time.sleep(60)
    report = []
    for line in args.manifest.read_text().splitlines():
        row = json.loads(line)
        case = row["case_key"]
        source = (old_root / case).resolve()
        replacement = (new_root / case).resolve()
        if not source.exists() or not (replacement / "point_complete").exists():
            continue
        if not source.is_relative_to(old_root) or source == old_root \
                or not replacement.is_relative_to(new_root):
            raise ValueError("unsafe cleanup target " + case)
        if len(source.relative_to(old_root).parts) != 4:
            raise ValueError("unexpected case path " + case)
        for protected in (row.get("A0_ckpt"), row.get("env_artifact")):
            if protected and Path(protected).resolve().is_relative_to(source):
                raise ValueError("protected asset inside cleanup target " + case)
        metrics = json.loads((replacement / "point_metrics.json").read_text())
        if metrics["A0_sha256"] != row["A0_sha256"]:
            raise ValueError("A0 identity mismatch " + case)
        for name in ("teacher_q.npz", "teacher_q.json"):
            if (source / name).exists() and not (replacement / name).exists():
                shutil.copy2(source / name, replacement / name)
        if (source / "teacher_q.npz").exists() and not (replacement / "teacher_q.npz").exists():
            raise ValueError("teacher q not preserved " + case)
        bytes_removed = sum(item.stat().st_size for item in source.rglob("*") if item.is_file())
        shutil.rmtree(source)
        report.append({"case": case, "old_result_deleted": True, "bytes_removed": bytes_removed})
        print("PRUNED", case, bytes_removed, flush=True)
    destination = new_root / "legacy_cleanup_report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
