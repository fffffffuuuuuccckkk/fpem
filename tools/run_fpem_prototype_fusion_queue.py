"""Resumable Phase A/B/C queue, one independent pilot per GPU."""
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
BASE = ROOT / "results/fpem_latent_decomposition_search/pilot_20261015"
CASES = ("patchtst_ETTh1_720", "itransformer_ETTm1_720", "moderntcn_ETTm2_720")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("case", choices=CASES)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--output", default="results/fpem_prototype_fusion_search/pilot_20261025")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--max_batches", type=int, default=0)
    cli = p.parse_args()
    base = BASE / (f"{cli.case}_A0_complementary_gate_predictive_"
                   "sub0_orth0_refnone_fp0_rank8_bias-4/run_config.json")
    if not base.is_file():
        raise FileNotFoundError(base)
    root = ROOT / cli.output / cli.case
    root.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "results/fpem_predictive_memory/pilot_20261024" / cli.case / "phase0_cache.pt"
    env = os.environ.copy()
    env.update({"OPENBLAS_NUM_THREADS":"4", "OMP_NUM_THREADS":"4", "MKL_NUM_THREADS":"4",
                "NUMEXPR_NUM_THREADS":"4"})
    if cli.case.startswith("moderntcn"):
        legacy = ["--allow_legacy_reference_signature"]
    else:
        legacy = []

    def run(prototype, fusion, mapping="M1"):
        path = root / f"{prototype}_{fusion}_{mapping}"
        path.mkdir(parents=True, exist_ok=True)
        if (path/"run_complete").is_file():
            return json.loads((path/"result.json").read_text(encoding="utf-8"))
        command = [sys.executable, str(ROOT/"tools/run_fpem_prototype_fusion.py"),
                   "--base_run_config", str(base), "--output", str(path),
                   "--prototype", prototype, "--fusion", fusion,
                   "--mapping", mapping, "--gpu", str(cli.gpu),
                   "--epochs", str(cli.epochs), "--phase0_cache", str(cache),
                   "--max_batches", str(cli.max_batches)] + legacy
        for attempt in (1, 2):
            with (path/"run.log").open("a", encoding="utf-8") as handle:
                handle.write(f"\nATTEMPT {attempt}: {' '.join(command)}\n")
                handle.flush()
                result = subprocess.run(command, cwd=ROOT, env=env,
                                        stdout=handle, stderr=subprocess.STDOUT)
            if result.returncode == 0 and (path/"run_complete").is_file():
                return json.loads((path/"result.json").read_text(encoding="utf-8"))
            if attempt == 1:
                time.sleep(60)
        (path/"failed.txt").write_text("Two attempts failed; inspect run.log\n", encoding="utf-8")
        raise RuntimeError(f"{cli.case}: {prototype} {fusion} {mapping} failed")

    rows = []
    for prototype in ("P0", "P2", "P3"):
        result = run(prototype, "phaseA")
        d = result["diagnostics"]
        rows.append({"phase":"A", "prototype":prototype, "fusion":"-", "mapping":"-",
                     "train_oof_response_mse":d["response_mse"],
                     "train_oof_zero_mse":d["zero_response_mse"],
                     "val_mse":"", "test_mse":"", "test_mae":"", "a0_test_mse":""})
    first = []
    for prototype in ("P0", "P2", "P3"):
        result = run(prototype, "F1")
        first.append((prototype, result["val"]["mse"]))
        rows.append({"phase":"B", "prototype":prototype, "fusion":"F1", "mapping":"M1",
                     "train_oof_response_mse":result["diagnostics"]["response_mse"],
                     "train_oof_zero_mse":result["diagnostics"]["zero_response_mse"],
                     "val_mse":result["val"]["mse"], "test_mse":result["test"]["mse"],
                     "test_mae":result["test"]["mae"], "a0_test_mse":result["test"]["a0_mse"]})
    selected = {"P0", min(first, key=lambda row:row[1])[0]}
    for prototype in sorted(selected):
        for fusion in ("F2", "F3"):
            result = run(prototype, fusion)
            rows.append({"phase":"C", "prototype":prototype, "fusion":fusion, "mapping":"M1",
                         "train_oof_response_mse":result["diagnostics"]["response_mse"],
                         "train_oof_zero_mse":result["diagnostics"]["zero_response_mse"],
                         "val_mse":result["val"]["mse"], "test_mse":result["test"]["mse"],
                         "test_mae":result["test"]["mae"], "a0_test_mse":result["test"]["a0_mse"]})
    with (root/"all_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    (root/"summary.md").write_text(
        f"# {cli.case}: prototype × fusion staged Pilot\n\n"
        "VAL selects checkpoints and Phase C prototypes; TEST is exploratory only.\n"
        "Full Cartesian grid and M2 are not run because prior history-only qhat did not pass TRAIN heldout evidence.\n\n"
        + "\n".join(f"- {r['phase']} {r['prototype']} {r['fusion']}: VAL {r['val_mse']}, TEST {r['test_mse']}, A0 {r['a0_test_mse']}" for r in rows)
        + "\n", encoding="utf-8")
    (root/"case_complete").touch()


if __name__ == "__main__":
    main()
