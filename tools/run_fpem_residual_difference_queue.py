"""Evidence-gated, resumable Flow phase; quiet while point pilot runs."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("case")
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--point_root", default="results/fpem_learnability/pilot_v4_20261026")
    p.add_argument("--output", default="results/fpem_learnability/residual_pilot_20261026")
    p.add_argument("--protocol", choices=("legacy", "unified"), default="legacy")
    args = p.parse_args()
    point = ROOT / args.point_root / args.case
    out = ROOT / args.output / args.case
    out.mkdir(parents=True, exist_ok=True)
    while not (point / "point_pilot_complete").exists():
        if (point / "point_pilot_failed.txt").exists():
            (out / "probability_skipped.txt").write_text(
                "Point pilot failed; no downstream model trained.\n", encoding="utf-8")
            return
        time.sleep(60)
    selected = json.loads((point / "point_selection.json").read_text(encoding="utf-8"))
    with (point / "all_point_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = {row["mode"]: row for row in csv.DictReader(handle)}
    mode = selected["selected_mode"]
    base, shuffled, random = (("A0", "B6", "B7") if args.protocol == "unified"
                              else ("P0", "P7", "P8"))
    criterion = (
        mode not in {"P0", "P4", "P7", "P8", "A0", "A4", "B6", "B7"}
        and float(rows[mode]["val_mse"]) < float(rows[base]["val_mse"])
        and float(rows[mode]["val_mse"]) < float(rows[shuffled]["val_mse"])
        and float(rows[mode]["val_mse"]) < float(rows[random]["val_mse"])
    )
    if not criterion and args.protocol == "legacy":
        (out / "probability_skipped.txt").write_text(
            "Predeclared VAL gate not met: real learned memory must beat A0, "
            "shuffled-pair and random-retrieval controls. "
            f"Selected={mode}; A0={rows[base]['val_mse']}; "
            f"real={rows[mode]['val_mse']}; shuffled={rows[shuffled]['val_mse']}; "
            f"random={rows[random]['val_mse']}.\n",
            encoding="utf-8")
        return
    command = [sys.executable, str(ROOT / "tools/run_fpem_residual_difference.py"),
               args.case, "--gpu", str(args.gpu), "--point_root", args.point_root,
               "--output", args.output, "--protocol", args.protocol]
    if args.protocol == "unified" and not criterion:
        (out / "exploratory_probability.txt").write_text(
            "Point pilot did not pass the predeclared VAL gate; C0-C7 are exploratory "
            "and cannot be described as confirmatory evidence.\n", encoding="utf-8")
    env = os.environ.copy()
    env.update({"OPENBLAS_NUM_THREADS": "4", "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4"})
    for attempt in (1, 2, 3):
        if (out / "probability_pilot_complete").exists():
            return
        with (out / "queue.log").open("a", encoding="utf-8") as log:
            log.write(f"\nATTEMPT {attempt}: {' '.join(command)}\n")
            log.flush()
            process = subprocess.run(command, cwd=ROOT, env=env,
                                     stdout=log, stderr=subprocess.STDOUT)
        if process.returncode == 0 and (out / "probability_pilot_complete").exists():
            return
        if attempt < 3:
            time.sleep(60)
    (out / "probability_failed.txt").write_text(
        "Three attempts failed; inspect queue.log. OOF folds and finished U rows "
        "are preserved for safe continuation.\n", encoding="utf-8")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
