#!/usr/bin/env python3
"""Compare frozen warmup-5 PatchTST anchors with the same probability heads.

Only the established 24/48 probabilistic protocol is used. Each head keeps its
previously recorded LR across predictive, shuffled and no-domain anchors.
Existing complete runs are verified and reused; partial runs are preserved.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "ExchangeRate")
HORIZONS = (24, 48)
ARMS = ("predictive", "environment_shuffled", "no_domain")
HEADS = (
    "affine_deterministic", "affine_gaussian", "affine_mdn", "affine_flow",
    "residual_gaussian", "residual_lowrank", "residual_flow", "legacy_full",
)
DATA_PROJECT = Path("/data/OuXiaoyu/Time-Series-Library-FPEM")
CODE_PROJECT = Path("/data/OuXiaoyu/fpem_best_a0_code_20261013")
ABLATED = DATA_PROJECT / "results/fpem_probabilistic/grl_warmup5_patchtst_ablations_20261015/patchtst"
OUTPUT = DATA_PROJECT / "results/fpem_probabilistic/grl_warmup5_patchtst_crps_ablations_20261015"
PYTHON = Path("/data/OuXiaoyu/miniconda3/envs/basicts/bin/python")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selection(dataset, horizon):
    path = OUTPUT / "prior_head_selection" / dataset / f"pred_{horizon}.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_head = {row["head"]: row for row in rows}
    if set(by_head) != set(HEADS):
        raise ValueError(f"incomplete prior head selection: {path}")
    return by_head, sorted({row["best_head_lr"] for row in rows}, key=float)


def anchor(dataset, horizon, arm):
    if arm == "predictive":
        path = OUTPUT / "source_anchors" / dataset / f"pred_{horizon}.pt"
    else:
        path = ABLATED / dataset / f"pred_{horizon}" / arm / "A2/trained_checkpoint.pt"
        if not path.is_file():
            path = OUTPUT / "ablation_anchors" / dataset / f"pred_{horizon}" / arm / "trained_checkpoint.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def reference_config(dataset, horizon):
    path = ABLATED / dataset / f"pred_{horizon}" / "environment_shuffled/run_config.json"
    if not path.is_file():
        path = OUTPUT / "ablation_configs" / dataset / f"pred_{horizon}.json"
    config = json.loads(path.read_text())
    if (config["backbone"] != "patchtst" or
            config["dataset_name"] != dataset or config["pred_len"] != horizon or
            config["prob_ablation"] != "invariant_only" or
            config["decomposition_warmup_epochs"] != 5):
        raise ValueError(f"source anchor protocol mismatch: {path}")
    return config


def candidate_valid(root, expected_anchor, lr):
    required = [root / "summary.csv", root / "run_config.json",
                root / "frozen_anchor_drift.txt"]
    required += [root / head / "test_metrics.json" for head in HEADS]
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    config = json.loads((root / "run_config.json").read_text())
    if (config.get("frozen_anchor_checkpoint") != str(expected_anchor) or
            config.get("prob_head") != "all_shared" or
            config.get("prob_condition_source") != "future_zvar" or
            config.get("prob_loss_weight_mode") != "historical_ema_maturity" or
            int(config.get("prob_num_samples", 0)) != 100 or
            abs(float(config.get("prob_head_lr", -1)) - float(lr)) > 1e-12):
        return False
    if float((root / "frozen_anchor_drift.txt").read_text()) > 1e-7:
        return False
    return all(
        "ProbTS_CRPS" in json.loads((root / head / "test_metrics.json").read_text())
        for head in HEADS
    )


def run_bank(dataset, horizon, arm, lr, gpu, config):
    chosen_anchor = anchor(dataset, horizon, arm)
    root = OUTPUT / "patchtst" / dataset / f"pred_{horizon}" / arm / f"head_lr_{lr.replace('.', 'p')}"
    if candidate_valid(root, chosen_anchor, lr):
        print(f"REUSE {dataset}/{horizon}/{arm}/{lr}", flush=True)
        return
    if (root / "run_config.json").exists():
        raise RuntimeError(f"incomplete output preserved for inspection: {root}")
    root.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(config["root_path"])
    if not dataset_root.is_absolute():
        dataset_root = DATA_PROJECT / str(dataset_root).removeprefix("./")
    env = os.environ.copy()
    env.update({
        "PROJECT_DIR": str(CODE_PROJECT), "PYTHON": str(PYTHON),
        "PREDICTIVE_ENV_RUNNER": "tools/run_fpem_head_comparison.py",
        "BACKBONE": "patchtst", "DATASET_NAME": dataset,
        "DATASET_ARCHIVE_SHA256": config["dataset_archive_sha256"],
        "DATA_ROOT": str(dataset_root), "DATA_CLASS": config["data_class"],
        "DATA_PATH": config["data_path"], "FREQ": config["freq"],
        "CYCLENET_CYCLE_LEN": str(config["cyclenet_cycle_len"]),
        "SEQ_LEN": str(config["seq_len"]), "PRED_LEN": str(horizon),
        "BATCH_SIZE": str(config["batch_size"]), "D_MODEL": str(config["d_model"]),
        "D_FF": str(config["d_ff"]), "N_HEADS": str(config["n_heads"]),
        "E_LAYERS": str(config["e_layers"]), "DROPOUT": str(config["dropout"]),
        "GPU": str(gpu), "EXPERIMENTS": "A2", "EPOCHS": "10",
        "WARMUP_EPOCHS": "0", "STAGE_EPOCHS": "2", "ENV_NUM": "3",
        "NUM_WORKERS": "0", "REPRESENTATION_CONSTRAINT": "classification",
        "DECOMPOSITION_TYPE": "complementary_gate",
        "VARIANT_FUSION_MODE": "prob_affine_flow", "PROB_HEAD": "all_shared",
        "PROB_CONDITION_MODE": "full_shape", "PROB_CONDITION_SOURCE": "future_zvar",
        "PROB_HEAD_LR": lr, "PROB_LOSS_WEIGHT_MODE": "historical_ema_maturity",
        "PROB_LOSS_MIN_WEIGHT": "0.05", "PROB_NUM_SAMPLES": "100",
        "PROB_SAMPLE_CHUNK": "10", "PROB_FLOW_STEPS": "12",
        "MAX_PROB_EVAL_BATCHES": "0", "FUTURE_PATCH_LEN": "16",
        "LAMBDA_FUTURE_H": "0", "LAMBDA_VARIANT_ANCHOR": "0",
        "DIFFERENTIAL_LR": "1", "FIXED_GAMMA_BETA_LR": "1",
        "LR": "1e-4", "LR_BACKBONE": "2e-5",
        "LR_INV_HEAD": str(config["lr_inv_head"]),
        "LR_DECOMPOSER": str(config["lr_decomposer"]),
        "LR_ENV_HEAD": str(config["lr_env_head"]),
        "LR_VARIANT": str(config["lr_variant"]),
        "LR_GAMMA_BETA_BASE": "1e-4", "LR_RELIABILITY": "3e-4",
        "LAMBDA_DOMAIN": str(config["lambda_domain"]),
        "FROZEN_ANCHOR_CHECKPOINT": str(chosen_anchor), "OUTPUT": str(root),
    })
    command = ["bash", str(CODE_PROJECT / "scripts/run_predictive_env_iv_patchtst.sh")]
    with (root / "run.log").open("w") as handle:
        result = subprocess.run(command, cwd=CODE_PROJECT, env=env,
                                stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"head bank failed ({result.returncode}); see {root / 'run.log'}")
    if not candidate_valid(root, chosen_anchor, lr):
        raise RuntimeError(f"head bank incomplete after training: {root}")
    print(f"DONE {dataset}/{horizon}/{arm}/{lr} GPU={gpu}", flush=True)


def summarize_case(dataset, horizon):
    selected, _ = selection(dataset, horizon)
    rows = []
    for head in HEADS:
        lr = selected[head]["best_head_lr"]
        values = {}
        for arm in ARMS:
            root = OUTPUT / "patchtst" / dataset / f"pred_{horizon}" / arm / f"head_lr_{lr.replace('.', 'p')}"
            if not candidate_valid(root, anchor(dataset, horizon, arm), lr):
                raise RuntimeError(f"missing prior-selected head result: {root}/{head}")
            values[arm] = json.loads((root / head / "test_metrics.json").read_text())
        row = {"dataset": dataset, "pred_len": horizon, "head": head,
               "fixed_prior_head_lr": lr, "anchor_epoch": 10,
               "selection_split": "test_prior_experiment"}
        for arm in ARMS:
            for metric in ("ProbTS_CRPS", "CRPS_empirical", "PICP_95",
                           "MPIW_95", "inv_MSE", "inv_MAE", "mean_MSE"):
                row[f"{arm}_{metric}"] = values[arm][metric]
        row["shuffled_minus_predictive_ProbTS_CRPS"] = (
            row["environment_shuffled_ProbTS_CRPS"] - row["predictive_ProbTS_CRPS"])
        row["no_domain_minus_predictive_ProbTS_CRPS"] = (
            row["no_domain_ProbTS_CRPS"] - row["predictive_ProbTS_CRPS"])
        rows.append(row)
    root = OUTPUT / "patchtst" / dataset / f"pred_{horizon}"
    with (root / "crps_ablation_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "crps_ablation_comparison.json").write_text(
        json.dumps(rows, indent=2) + "\n")


def run_case(dataset, horizon, gpu):
    config = reference_config(dataset, horizon)
    selected, lrs = selection(dataset, horizon)
    assert set(selected) == set(HEADS)
    for arm in ARMS:
        for lr in lrs:
            run_bank(dataset, horizon, arm, lr, gpu, config)
    summarize_case(dataset, horizon)
    print(f"COMPLETE {dataset}/{horizon} GPU={gpu}", flush=True)


def gpu_ready(gpu):
    result = subprocess.run(
        ["nvidia-smi", "--id", str(gpu), "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return int(result.stdout.strip()) <= 256


def worker(gpu, cases):
    failed = []
    for dataset, horizon in cases:
        while not gpu_ready(gpu):
            time.sleep(30)
        try:
            run_case(dataset, horizon, gpu)
        except Exception as exc:
            print(f"FAILED {dataset}/{horizon} GPU={gpu}: {exc}", flush=True)
            failed.append(f"{dataset}/{horizon}: {exc}")
    return failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("plan", "case", "queue"))
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--horizon", type=int, choices=HORIZONS)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    cases = [(dataset, horizon) for horizon in HORIZONS for dataset in DATASETS]
    if args.mode == "plan":
        for dataset, horizon in cases:
            _, lrs = selection(dataset, horizon)
            print(dataset, horizon, ",".join(lrs))
        return
    if args.mode == "case":
        if not args.dataset or not args.horizon:
            parser.error("case requires --dataset and --horizon")
        run_case(args.dataset, args.horizon, args.gpu)
        return
    GPUs = [int(item) for item in args.gpus.split(",")]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    archive_hash = sha256(DATA_PROJECT / "dataset/all_datasets.zip")
    for dataset, horizon in cases:
        if reference_config(dataset, horizon)["dataset_archive_sha256"] != archive_hash:
            raise RuntimeError(f"dataset archive mismatch: {dataset}/{horizon}")
    buckets = [[] for _ in GPUs]
    for index, item in enumerate(cases):
        buckets[index % len(GPUs)].append(item)
    print(f"QUEUE cases={len(cases)} GPUs={GPUs} root={OUTPUT}", flush=True)
    with ThreadPoolExecutor(max_workers=len(GPUs)) as pool:
        futures = [pool.submit(worker, gpu, items)
                   for gpu, items in zip(GPUs, buckets)]
        failures = [issue for future in as_completed(futures)
                    for issue in future.result()]
    if failures:
        (OUTPUT / "failed_cases.txt").write_text("\n".join(failures) + "\n")
        raise SystemExit(f"{len(failures)} cases failed; see failed_cases.txt")
    print("ALL_COMPLETE", len(cases), flush=True)


if __name__ == "__main__":
    main()
