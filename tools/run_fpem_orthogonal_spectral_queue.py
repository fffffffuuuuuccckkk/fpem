"""Restartable two-GPU coordinate search for O1/O2/O3/O4 pilots."""
from __future__ import annotations

import concurrent.futures
import csv
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.run_fpem_s4_losswise_queue import RESULTS as S4_RESULTS, selected_prior


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/"results/fpem_orthogonal_spectral_decomposition"
FRACTIONS = (0.0625, 0.125, 0.25)
# Two independent lanes; launch only on currently available GPUs.
LANES = ((0, ("moderntcn_ETTm2_720", "patchtst_ETTh1_96")),
         (3, ("itransformer_ETTm1_720", "patchtst_ETTh1_720")))
LOCK = threading.Lock()


def tag(number):
    return f"{number:.8g}".replace(".", "p").replace("-", "m")


def output_path(case, method, fraction, head_lr, basis_lr):
    return OUTPUT/case/method/f"r{tag(fraction)}_h{tag(head_lr)}_b{tag(basis_lr)}"


def read_result(path):
    if not (path/"run_complete").is_file() or not (path/"result.json").is_file():
        return None
    return json.loads((path/"result.json").read_text(encoding="utf-8"))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize():
    with LOCK:
        rows = []
        for _, cases in LANES:
            for case in cases:
                _, prior_path, prior, prior_config = selected_prior(case)
                rows.append({"case": case, "method": "O0_S4", "fraction": "",
                             "head_lr": prior_config["cli"]["lr_head"],
                             "basis_lr": prior_config["cli"]["lr_decomposer"],
                             "reference_sha256": prior["reference_sha256"],
                             "val_mse": prior["val"]["mse"],
                             "val_mae": prior["val"]["mae"],
                             "test_mse": prior["test"]["mse"],
                             "test_mae": prior["test"]["mae"],
                             "best_epoch": prior["best_epoch"],
                             "var_energy_ratio": prior["test"]["diagnostics"].get(
                                 "variation_energy_ratio", ""),
                             "inv_participation": "", "var_participation": "",
                             "basis_orthogonal_error": "", "inv_effective_rank": "",
                             "var_effective_rank": "", "train_seconds": prior["training_seconds"],
                             "gpu_memory_bytes": prior["peak_gpu_memory_bytes"],
                             "source": str(prior_path)})
                for method in ("O1", "O2", "O3", "O4"):
                    root = OUTPUT/case/method
                    if not root.is_dir():
                        continue
                    for result_path in root.glob("*/result.json"):
                        path = result_path.parent
                        result = read_result(path)
                        if result is None:
                            continue
                        if result["reference_sha256"] != prior["reference_sha256"]:
                            raise RuntimeError(f"A0 reference mismatch: {case}/{method}")
                        cfg = json.loads((path/"run_config.json").read_text())["cli"]
                        diag = result["test"]["diagnostics"]
                        train_diag = result["train_representation_diagnostics"]
                        rows.append({"case": case, "method": method,
                                     "fraction": cfg["variation_fraction"],
                                     "head_lr": cfg["lr_head"], "basis_lr": cfg["lr_basis"],
                                     "reference_sha256": result["reference_sha256"],
                                     "val_mse": result["val"]["mse"],
                                     "val_mae": result["val"]["mae"],
                                     "test_mse": result["test"]["mse"],
                                     "test_mae": result["test"]["mae"],
                                     "best_epoch": result["best_epoch"],
                                     "var_energy_ratio": diag.get("var_energy_ratio", ""),
                                     "inv_participation": diag.get("inv_participation", ""),
                                     "var_participation": diag.get("var_participation_mean", ""),
                                     "basis_orthogonal_error": diag.get("basis_orthogonal_error", ""),
                                     "inv_effective_rank": train_diag["inv_effective_rank"],
                                     "var_effective_rank": train_diag["var_effective_rank"],
                                     "train_seconds": result["training_seconds"],
                                     "gpu_memory_bytes": result["peak_gpu_memory_bytes"],
                                     "source": str(path)})
        write_csv(OUTPUT/"all_results.csv", rows)
        write_csv(OUTPUT/"method_a_results.csv", [r for r in rows if r["method"] in ("O1","O2")])
        write_csv(OUTPUT/"method_c_results.csv", [r for r in rows if r["method"]=="O3"])
        write_csv(OUTPUT/"spectral_initialization_results.csv", [r for r in rows if r["method"]=="O4"])
        write_csv(OUTPUT/"decomposition_diagnostics.csv", [r for r in rows if r["method"]!="O0_S4"])
        selected = []
        for _, cases in LANES:
            for case in cases:
                for method in ("O0_S4","O1","O2","O3","O4"):
                    subset = [r for r in rows if r["case"]==case and r["method"]==method]
                    if subset:
                        best = min(subset, key=lambda r: float(r["val_mse"]))
                        selected.append(best)
        write_csv(OUTPUT/"validation_selection.csv", selected)
        ranking = []
        for _, cases in LANES:
            for case in cases:
                subset = sorted((r for r in selected if r["case"]==case),
                                key=lambda r: float(r["test_mse"]))
                for position, row in enumerate(subset, 1):
                    ranking.append({"case": case, "test_rank": position, **row})
        write_csv(OUTPUT/"test_ranking.csv", ranking)
        lines = ["# Orthogonal / supervised spectral S4 pilot", "",
                 "O0 is the previous S4 run. O1/O2/O3/O4 reuse its A0 checkpoint and",
                 "10-epoch frozen-backbone pilot protocol. PSD uses TRAIN H/Y only; no q.",
                 "O2 uses full prediction-priority auxiliary-gradient surgery on Uv.",
                 "VAL selects rank/LR/epoch; TEST ranks methods exploratorily only.", "",
                 f"Finished new runs: {len(rows)-4}", ""]
        for case in ("patchtst_ETTh1_96","patchtst_ETTh1_720",
                     "moderntcn_ETTm2_720","itransformer_ETTm1_720"):
            subset = [r for r in selected if r["case"]==case]
            lines.append(f"## {case}")
            for row in sorted(subset, key=lambda r: float(r["test_mse"])):
                energy = row["var_energy_ratio"]
                flag = " [VAR ENERGY COLLAPSE]" if energy != "" and float(energy)<0.001 else ""
                lines.append(f"- {row['method']}: VAL={float(row['val_mse']):.6f}; "
                             f"TEST MSE={float(row['test_mse']):.6f}, MAE={float(row['test_mae']):.6f}; "
                             f"var-energy={energy}{flag}")
            lines.append("")
        (OUTPUT/"summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run_one(case, gpu, base_config, digest, method, fraction, head_lr, basis_lr):
    path = output_path(case, method, fraction, head_lr, basis_lr)
    result = read_result(path)
    if result is not None:
        print(f"SKIP {case} {method} {fraction} {head_lr} {basis_lr}", flush=True)
        return result
    path.mkdir(parents=True, exist_ok=True)
    config = selected_prior(case)[3]["cli"]
    cmd = [sys.executable, "tools/run_fpem_orthogonal_spectral.py",
           "--base_run_config", base_config, "--output", str(path),
           "--method", method, "--variation_fraction", str(fraction),
           "--lr_head", str(head_lr), "--lr_basis", str(basis_lr),
           "--lr_backbone", str(config["lr_backbone"]), "--seed", str(config["seed"]),
           "--epochs", "10", "--gpu", "0", "--gradient_separation", "1.0"]
    if config.get("train_backbone"):
        cmd.append("--train_backbone")
    if method in ("O3", "O4"):
        width = json.loads((OUTPUT/case/"width.json").read_text())["width"]
        variation_rank = max(1, min(width-1, round(width*fraction)))
        cmd += ["--spectrum_cache", str(OUTPUT/case/f"train_spectrum_shared{width-variation_rank}.pt")]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for attempt in range(1, 4):
        print(f"START {case} {method} r={fraction} h={head_lr} b={basis_lr} "
              f"GPU={gpu} attempt={attempt}", flush=True)
        with (path/"run.log").open("a", encoding="utf-8") as log:
            log.write("\nCOMMAND "+" ".join(cmd)+"\n")
            code = subprocess.call(cmd, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT)
        if code==0 and read_result(path) is not None:
            result = read_result(path)
            if result["reference_sha256"] != digest:
                raise RuntimeError(f"reference changed: {case}/{method}")
            summarize()
            print(f"DONE {case} {method} r={fraction} h={head_lr} b={basis_lr}", flush=True)
            return result
        print(f"FAILED {case} {method} exit={code}, waiting 60 seconds", flush=True)
        time.sleep(60)
    (path/"failed_after_retries").touch()
    raise RuntimeError(f"failed after retries: {path}")


def best_val(trials):
    return min(trials, key=lambda trial: trial[0]["val"]["mse"])


def run_case(case, gpu):
    _, prior_path, prior, config = selected_prior(case)
    cli = config["cli"]
    head0 = float(cli["lr_head"])
    basis0 = float(cli["lr_decomposer"])
    base_config = cli["base_run_config"]
    digest = prior["reference_sha256"]
    # Only use existing A0 reference; a forward setup obtains D, not A0 training.
    # D is recorded in the selected S4 model checkpoint's basis shape.
    import torch
    checkpoint = torch.load(prior_path/"best.pt", map_location="cpu", weights_only=False)
    width = checkpoint["decomposer"]["shared.weight"].shape[0]
    (OUTPUT/case).mkdir(parents=True, exist_ok=True)
    (OUTPUT/case/"width.json").write_text(json.dumps({"width": width})+"\n")
    trials = {method: [] for method in ("O1","O2","O3","O4")}
    for method in ("O1","O3"):
        for fraction in FRACTIONS:
            result = run_one(case,gpu,base_config,digest,method,fraction,head0,basis0)
            trials[method].append((result,fraction,head0,basis0))
    for method in ("O1","O3"):
        _, fraction, _, _ = best_val(trials[method])
        for multiplier in (0.5,2.0):
            head = head0*multiplier
            result = run_one(case,gpu,base_config,digest,method,fraction,head,basis0)
            trials[method].append((result,fraction,head,basis0))
    _, o1_fraction, o1_head, _ = best_val(trials["O1"])
    for basis in (basis0*0.5,basis0*2):
        result = run_one(case,gpu,base_config,digest,"O1",o1_fraction,o1_head,basis)
        trials["O1"].append((result,o1_fraction,o1_head,basis))
    _, o1_fraction, o1_head, o1_basis = best_val(trials["O1"])
    _, o3_fraction, o3_head, _ = best_val(trials["O3"])
    for method,fraction,head,basis in (("O2",o1_fraction,o1_head,o1_basis),
                                       ("O4",o3_fraction,o3_head,o1_basis)):
        result = run_one(case,gpu,base_config,digest,method,fraction,head,basis)
        trials[method].append((result,fraction,head,basis))
        for multiplier in (0.5,2.0):
            candidate = head*multiplier
            result = run_one(case,gpu,base_config,digest,method,fraction,candidate,basis)
            trials[method].append((result,fraction,candidate,basis))
        _, fraction, selected_head, _ = best_val(trials[method])
        for multiplier in (0.5,2.0):
            candidate = basis*multiplier
            result = run_one(case,gpu,base_config,digest,method,fraction,selected_head,candidate)
            trials[method].append((result,fraction,selected_head,candidate))
    (OUTPUT/case/"case_complete").touch()
    print(f"CASE COMPLETE {case}", flush=True)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {"methods": ["O0(reused)","O1","O3","O2","O4"],
                "rank_fractions": FRACTIONS,
                "lr_search": "VAL-selected coordinate: head x0.5/x1/x2; basis x0.5/x1/x2",
                "method_selection": "TEST MSE exploratory only",
                "gradient_separation_on_O2": 1.0,
                "lanes": LANES,
                "other_user_gpu_policy": "avoid currently occupied GPU 1 and 2"}
    (OUTPUT/"experiment_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    summarize()
    def run_lane(gpu, cases):
        for case in cases:
            run_case(case, gpu)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_lane, gpu, cases) for gpu, cases in LANES]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    summarize()
    if all((OUTPUT/case/"case_complete").is_file() for _,cases in LANES for case in cases):
        (OUTPUT/"queue_complete").touch()
        print("QUEUE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
