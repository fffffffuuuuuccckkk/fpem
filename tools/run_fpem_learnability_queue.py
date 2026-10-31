"""Resume one case after transient process/memory failure; never retrain A0."""
from __future__ import annotations

import argparse
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
    p.add_argument("--output", default="results/fpem_learnability/pilot_20261026")
    p.add_argument("--protocol", choices=("legacy", "unified"), default="legacy")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--attempts", type=int, default=3)
    args = p.parse_args()
    run = ROOT / args.output / args.case
    run.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ROOT / "tools/run_fpem_learnability_pilot.py"),
               args.case, "--gpu", str(args.gpu), "--output", args.output,
               "--protocol", args.protocol, "--topk", str(args.topk)]
    env = os.environ.copy()
    env.update({"OPENBLAS_NUM_THREADS": "4", "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4"})
    for attempt in range(1, args.attempts + 1):
        if (run / "point_pilot_complete").exists():
            return
        with (run / "queue.log").open("a", encoding="utf-8") as log:
            log.write(f"\nATTEMPT {attempt}: {' '.join(command)}\n")
            log.flush()
            result = subprocess.run(command, cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode == 0 and (run / "point_pilot_complete").exists():
            return
        if attempt < args.attempts:
            time.sleep(60)
    (run / "point_pilot_failed.txt").write_text(
        "Three attempts failed; inspect queue.log. Completed P rows remain reusable.\n",
        encoding="utf-8")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
