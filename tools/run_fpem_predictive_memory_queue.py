"""Resumable, evidence-gated three-backbone predictive-memory Pilot queue."""
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
BASE = "results/fpem_latent_decomposition_search/pilot_20261015"
CASES = {
    "patchtst_ETTh1_720": ("patchtst", "ETTh1", 720, False),
    "itransformer_ETTm1_720": ("itransformer", "ETTm1", 720, False),
    "moderntcn_ETTm2_720": ("moderntcn", "ETTm2", 720, True),
}


def run_case(name, gpu, outroot):
    backbone, dataset, horizon, legacy = CASES[name]
    base = ROOT / BASE / (f"{name}_A0_complementary_gate_predictive_"
                          "sub0_orth0_refnone_fp0_rank8_bias-4/run_config.json")
    if not base.is_file():
        raise FileNotFoundError(base)
    config = json.loads(base.read_text(encoding="utf-8"))
    case = outroot / name
    case.mkdir(parents=True, exist_ok=True)
    cache = case / "phase0_cache.pt"
    python = sys.executable
    script = ROOT / "tools/run_fpem_predictive_memory.py"
    environment = os.environ.copy()
    environment.update({"OPENBLAS_NUM_THREADS": "4", "OMP_NUM_THREADS": "4",
                        "MKL_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4"})

    def one(mode):
        run = case / mode
        if (run / "run_complete").is_file():
            return json.loads((run / "result.json").read_text(encoding="utf-8"))
        args = [python, str(script), "--base_run_config", str(base),
                "--output", str(run), "--mode", mode, "--gpu", str(gpu),
                "--phase0_cache", str(cache), "--lr_head",
                str(config.get("lr_inv_head", 5e-5))]
        if mode != "phase0":
            args += ["--phase0_json", str(case / "phase0/phase0/phase0.json")]
        if legacy:
            args += ["--allow_legacy_reference_signature"]
        run.mkdir(parents=True, exist_ok=True)
        for attempt in (1, 2):
            with (run / "run.log").open("a", encoding="utf-8") as handle:
                handle.write(f"\nATTEMPT {attempt}: {' '.join(args)}\n")
                handle.flush()
                status = subprocess.run(args, cwd=ROOT, env=environment,
                                        stdout=handle, stderr=subprocess.STDOUT).returncode
            if status == 0 and (run / "run_complete").is_file():
                return json.loads((run / "result.json").read_text(encoding="utf-8"))
            if attempt == 1:
                time.sleep(60)
        (case / f"{mode}_failed.txt").write_text(
            f"Two attempts failed; inspect {run / 'run.log'}\n", encoding="utf-8")
        raise RuntimeError(f"{name} {mode} failed twice")

    phase0 = one("phase0")["phase0"]
    rows = []
    for mode in ("B0", "B1", "P1", "P2"):
        result = one(mode)
        test = result["test"]
        rows.append({"case": name, "mode": mode, "val_mse": result["val"]["mse"],
                     "test_mse": test["mse"], "test_mae": test["mae"],
                     "a0_test_mse": test["a0_mse"], "a0_test_mae": test["a0_mae"],
                     "inv_test_mse": test["inv_mse"]})
    # Phase 0 is explicitly hypothesis checking. No conditionally supervised
    # mapping is trained when history cannot identify the environment.
    if phase0["evidence_for_conditioned_mapping"]:
        for mode in ("P3", "P4"):
            result = one(mode)
            test = result["test"]
            rows.append({"case": name, "mode": mode, "val_mse": result["val"]["mse"],
                         "test_mse": test["mse"], "test_mae": test["mae"],
                         "a0_test_mse": test["a0_mse"], "a0_test_mae": test["a0_mae"],
                         "inv_test_mse": test["inv_mse"]})
    else:
        (case / "conditioned_mapping_skipped.txt").write_text(
            "Phase 0 did not show a history-predictable environment-conditioned "
            "future response. P3/P4 were not trained; this is an evidence gate, "
            "not a positive result.\n", encoding="utf-8")
    with (case / "pilot_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    lines = [f"# {name} predictive-memory Pilot", "",
             f"Phase 0 environment-conditioned evidence: {phase0['evidence_for_conditioned_mapping']}",
             f"TRAIN heldout qhat accuracy: {phase0['qhat_heldout_hard_accuracy']:.5f} "
             f"(majority {phase0['heldout_majority_accuracy']:.5f})", "",
             "Checkpoint/epoch selected by VAL MSE; TEST is exploratory only.", "",
             "| Mode | VAL MSE | TEST MSE | TEST MAE | A0 TEST MSE |",
             "| --- | ---: | ---: | ---: | ---: |"]
    lines.extend(f"| {r['mode']} | {r['val_mse']:.6f} | {r['test_mse']:.6f} | "
                 f"{r['test_mae']:.6f} | {r['a0_test_mse']:.6f} |" for r in rows)
    (case / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    (case / "case_complete").touch()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", default="results/fpem_predictive_memory/pilot_20261024")
    cli = parser.parse_args()
    run_case(cli.case, cli.gpu, ROOT / cli.output)


if __name__ == "__main__":
    main()
