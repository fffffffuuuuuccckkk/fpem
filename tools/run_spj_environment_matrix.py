"""Restartable SJP environment-supervision matrix using existing A0 anchors.

Search selection is VAL invariant MSE. TEST is reported only from the selected
run. Shuffled/no-domain controls use that same selected LR and A0 reference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "ExchangeRate", "Weather")
BACKBONES = ("patchtst", "moderntcn", "itransformer")
HORIZONS = (24, 48, 96, 192, 336, 720)
HEAD_LRS = ("2e-5", "5e-5", "1e-4")
FIELDS = ("case", "backbone", "dataset", "pred_len", "base_config",
          "reference", "server", "gpu", "estimated_cost")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def case_key(backbone, dataset, horizon):
    return f"{backbone}_{dataset}_{horizon}"


def inventory(project, server):
    root = Path(project).resolve()
    candidates = []
    templates = {}
    for config in root.glob("results/fpem_no_future_anchor_*_search*/**/A0/run_config.json"):
        try:
            data = json.loads(config.read_text())
            backbone, dataset, horizon = (data.get("backbone"), data.get("dataset_name"),
                                          int(data.get("pred_len", -1)))
            if backbone not in BACKBONES or dataset not in DATASETS or horizon not in HORIZONS:
                continue
            reference = Path(data.get("reference_checkpoint") or "")
            if not reference.is_absolute():
                reference = root / reference
            if not reference.is_file():
                reference = config.parent.parent / "shared_reference.pt"
            if not reference.is_file():
                continue
            templates.setdefault((backbone, dataset), config)
            candidates.append((backbone, dataset, horizon, config, reference))
        except (KeyError, TypeError, ValueError, OSError):
            continue
    # Older 24-step experiments stored A0 separately. Use an existing same-
    # dataset run configuration as template; the checkpoint signature is the
    # authoritative source for architecture, seed and horizon at execution.
    patterns = (
        "results/foil_shifts_a0_protocol_20261011/*/*/pred_24/shared_reference.pt",
        "results/fpem_probabilistic/unified_lr_search_20261010/*/*/pred_24/anchor/shared_reference.pt",
        "results/fpem_probabilistic/itransformer_main_20261012/*/*/pred_24/anchor/shared_reference.pt",
    )
    for pattern in patterns:
        for reference in root.glob(pattern):
            parts = reference.relative_to(root).parts
            try:
                pred_pos = next(i for i, part in enumerate(parts) if part == "pred_24")
                backbone, dataset = parts[pred_pos - 2:pred_pos]
                config = reference.parent / "run_config.json"
                if not config.is_file():
                    config = templates.get((backbone, dataset))
                if (backbone in BACKBONES and dataset in DATASETS
                        and config is not None and config.is_file()):
                    candidates.append((backbone, dataset, 24, config, reference))
            except (StopIteration, ValueError):
                continue
    chosen = {}
    for backbone, dataset, horizon, config, reference in candidates:
        key = case_key(backbone, dataset, horizon)
        # Prefer a run configuration co-located with the reference.
        quality = int(config.parent == reference.parent or
                      config.parent.parent == reference.parent)
        row = {"case": key, "backbone": backbone, "dataset": dataset,
               "pred_len": horizon, "base_config": str(config),
               "reference": str(reference), "server": server,
               "gpu": "", "estimated_cost": "", "quality": quality}
        if key not in chosen or quality > chosen[key]["quality"]:
            chosen[key] = row
    return [{k: row[k] for k in FIELDS} for row in chosen.values()]


def plan(inventories, output):
    available = {}
    for path in inventories:
        for row in read_csv(path):
            available.setdefault(row["case"], []).append(row)
    lane_load = {(server, gpu): 0.0 for server in range(3) for gpu in range(4)}
    chosen = []
    def cost(row):
        dataset_factor = {"ETTh1": 1, "ETTh2": 1, "ETTm1": 4,
                          "ETTm2": 4, "ExchangeRate": 2,
                          "Weather": 5}[row["dataset"]]
        return dataset_factor * (1.5 if row["backbone"] == "moderntcn" else 1.0)
    for key in sorted(available, key=lambda k: -cost(available[k][0])):
        options = available[key]
        candidate = min(
            ((lane_load[(int(row["server"]), gpu)], int(row["server"]), gpu, row)
             for row in options for gpu in range(4)),
            key=lambda value: (value[0], value[1], value[2]),
        )
        _, server, gpu, row = candidate
        planned = {**row, "server": server, "gpu": gpu,
                   "estimated_cost": cost(row)}
        chosen.append(planned)
        lane_load[(server, gpu)] += cost(row) * (len(HEAD_LRS) + 2)
    write_csv(output, chosen, FIELDS)
    required = {case_key(b, d, h) for b in BACKBONES for d in DATASETS[:5]
                for h in HORIZONS}
    print(json.dumps({"planned": len(chosen), "core_90_missing": sorted(required - set(available)),
                      "server_counts": {s: sum(int(r["server"]) == s for r in chosen)
                                        for s in range(3)},
                      "lane_load": {f"{s}:{g}": round(v, 1)
                                    for (s, g), v in lane_load.items()}}, indent=2))


def rebalance(manifest, output, transfer_plan):
    rows = read_csv(manifest)
    lane_load = {(server, gpu): 0.0 for server in range(3) for gpu in range(4)}
    for row in rows:
        lane_load[(int(row["server"]), int(row["gpu"]))] += float(row["estimated_cost"]) * 5
    movable = sorted((row for row in rows if row["server"] == "1"
                      and row["backbone"] == "moderntcn"
                      and float(row["estimated_cost"]) >= 6),
                     key=lambda row: -float(row["estimated_cost"]))[:8]
    transfers = []
    for row in movable:
        old_gpu = int(row["gpu"])
        cost = float(row["estimated_cost"]) * 5
        lane_load[(1, old_gpu)] -= cost
        target_gpu = min(range(4), key=lambda gpu: lane_load[(2, gpu)])
        lane_load[(2, target_gpu)] += cost
        destination = Path("/var/tmp/fpem_spj_inputs_20261019") / row["case"]
        transfers.append({"case": row["case"], "source_server": "1",
                          "source_config": row["base_config"],
                          "source_reference": row["reference"],
                          "target_config": str(destination / "base_run_config.json"),
                          "target_reference": str(destination / "shared_reference.pt")})
        row.update(server="2", gpu=str(target_gpu),
                   base_config=transfers[-1]["target_config"],
                   reference=transfers[-1]["target_reference"])
    write_csv(output, rows, FIELDS)
    write_csv(transfer_plan, transfers, list(transfers[0]))
    print(json.dumps({"moved": len(transfers),
                      "lane_load": {f"{s}:{g}": round(v, 1)
                                    for (s, g), v in lane_load.items()}}, indent=2))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def base_overrides(row, output, lr, *, mode="predictive", probability=False):
    import torch
    payload = torch.load(row["reference"], map_location="cpu", weights_only=False)
    signature = payload.get("signature")
    if not isinstance(signature, dict) or "state_dict" not in payload:
        raise ValueError(f"invalid A0 checkpoint: {row['reference']}")
    if (signature.get("backbone") != row["backbone"] or
            signature.get("dataset_name") != row["dataset"] or
            int(signature.get("pred_len", -1)) != int(row["pred_len"])):
        raise ValueError(f"A0 identity mismatch: {row['reference']}")
    overrides = {key: value for key, value in signature.items()
                 if key not in ("enc_in", "protocol_version")}
    overrides.update({
        "gpu": 0,  # CUDA_VISIBLE_DEVICES remaps the selected physical GPU to cuda:0.
        "reference_checkpoint": row["reference"],
        "require_reference_checkpoint": True,
        "decomposition_type": "supervised_joint_private",
        "representation_constraint": "classification",
        "environment_supervision": "shuffled" if mode == "shuffled" else "predictive",
        "lambda_domain": 0.0 if mode == "no_domain" else 0.1,
        "lambda_prob_env": 0.0 if mode == "no_domain" else 0.1,
        "env_num": 3,
        "experiments": "A2",
        "epochs": 10,
        "stage_epochs": 2,
        "decomposition_warmup_epochs": 5,
        "variant_fusion_mode": "prob_affine_flow" if probability else "inv_only",
        "prob_ablation": ("unconditional_flow" if mode == "no_zvar" else "full")
                         if probability else "invariant_only",
        "prob_loss_weight_mode": "constant" if mode == "no_ema" else "historical_ema_maturity",
        "select_invariant_on_val": True,
        "track_epoch_inv_test": False,
        "save_final_checkpoint": True,
        "differential_lr": True,
        "fixed_gamma_beta_lr": True,
        "lr_inv_head": float(lr),
        "lr_decomposer": 1e-4,
        "lr_env_head": 1e-4,
        "lr_variant": 1e-4,
        "lr_gamma_beta_base": 1e-4,
        "lr_reliability": 3e-4,
        "lambda_future_h": 0.0,
        "lambda_variant_anchor": 0.0,
        "prob_num_samples": 100,
        "prob_sample_chunk": 10,
        "prob_flow_steps": 12,
        "max_prob_eval_batches": 0,
        "num_workers": 0,
        "output": str(output),
    })
    return overrides, sha256(row["reference"])


def completed(path, expected):
    path = Path(path)
    if not (path / "run_complete").is_file():
        return False
    run_config = json.loads((path / "run_config.json").read_text())
    metrics = json.loads((path / "A2/metrics_and_diagnostics.json").read_text())
    selected = json.loads((path / "A2/validation_selection.json").read_text())
    for key, value in expected.items():
        if run_config.get(key) != value:
            raise ValueError(f"completed protocol mismatch {path}: {key}")
    if not all(k in metrics for k in ("inv_MSE", "inv_only_MAE")):
        raise ValueError(f"missing invariant metrics: {path}")
    if "MSE" not in selected:
        raise ValueError(f"missing VAL selection: {path}")
    return True


def verify_output(path, expected):
    path = Path(path)
    for filename in ("run_config.json", "A2/metrics_and_diagnostics.json",
                     "A2/validation_selection.json"):
        if not (path / filename).is_file():
            raise RuntimeError(f"missing output {path / filename}")
    config = json.loads((path / "run_config.json").read_text())
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"protocol mismatch {path}: {key}")
    summarize(path)


def launch(project, python, row, root, lr, mode="predictive", probability=False):
    label = ("prob_" if probability else "point_") + mode
    candidate = Path(root) / row["case"] / label / f"lr_{lr.replace('.', 'p')}"
    overrides, digest = base_overrides(row, candidate, lr, mode=mode,
                                       probability=probability)
    expected = {key: overrides[key] for key in (
        "decomposition_type", "representation_constraint", "environment_supervision",
        "lambda_domain", "env_num", "variant_fusion_mode", "prob_ablation",
        "prob_loss_weight_mode", "lr_inv_head", "reference_checkpoint")}
    if completed(candidate, expected):
        return candidate, digest, "reused"
    for attempt in range(20):
        target = candidate if attempt == 0 else Path(str(candidate) + f"_retry{attempt}")
        if target.exists():
            if completed(target, expected):
                return target, digest, "reused"
            continue
        target.mkdir(parents=True)
        overrides["output"] = str(target)
        command = [python, "-u", "tools/run_predictive_env_iv_patchtst.py",
                   "--base_run_config", row["base_config"],
                   "--run_overrides_json", json.dumps(overrides)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(row["gpu"])
        with (target / "run.log").open("w") as handle:
            status = subprocess.run(command, cwd=project, env=env,
                                    stdout=handle, stderr=subprocess.STDOUT,
                                    check=False).returncode
        if status:
            raise RuntimeError(f"exit={status}, log={target / 'run.log'}")
        # Never mark an incomplete run reusable.
        verify_output(target, expected)
        (target / "run_complete").touch()
        return target, digest, "new"
    raise RuntimeError(f"retry slots exhausted: {candidate}")


def summarize(path):
    metrics = json.loads((Path(path) / "A2/metrics_and_diagnostics.json").read_text())
    selected = json.loads((Path(path) / "A2/validation_selection.json").read_text())
    prob_path = Path(path) / "A2/test_probabilistic_metrics.json"
    probability = json.loads(prob_path.read_text()) if prob_path.is_file() else {}
    return {"val_mse": selected["MSE"], "test_mse": metrics["inv_MSE"],
            "test_mae": metrics["inv_only_MAE"],
            "probts_crps": probability.get("ProbTS_CRPS"),
            "path": str(path)}


def run(project, python, manifest, server, output):
    rows = [r for r in read_csv(manifest) if int(r["server"]) == server]
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    def worker(gpu):
        for row in [r for r in rows if int(r["gpu"]) == gpu]:
            print(f"START {row['case']} gpu={gpu}", flush=True)
            try:
                candidates = []
                for lr in HEAD_LRS:
                    path, digest, state = launch(project, python, row, root, lr)
                    candidates.append({"lr": lr, "reference_sha256": digest,
                                       **summarize(path)})
                    print(f"{state.upper()} {row['case']} {lr} val={candidates[-1]['val_mse']}", flush=True)
                best = min(candidates, key=lambda x: (x["val_mse"], float(x["lr"])))
                write_csv(root / row["case"] / "lr_candidates.csv", candidates,
                          list(candidates[0]))
                (root / row["case"] / "selection.json").write_text(
                    json.dumps({"case": row["case"], "best": best,
                                "selection_split": "VAL MSE",
                                "reference_sha256": best["reference_sha256"]}, indent=2))
                for mode in ("shuffled", "no_domain"):
                    path, _, state = launch(project, python, row, root, best["lr"], mode=mode)
                    print(f"{state.upper()} {row['case']} {mode} {summarize(path)}", flush=True)
                if int(row["pred_len"]) in (24, 48):
                    for mode in ("predictive", "shuffled", "no_domain", "no_zvar", "no_ema"):
                        path, _, state = launch(project, python, row, root, best["lr"],
                                                mode=mode, probability=True)
                        print(f"{state.upper()} {row['case']} prob_{mode} {summarize(path)}", flush=True)
                (root / row["case"] / "case_complete").touch()
                print(f"DONE {row['case']}", flush=True)
            except Exception as error:
                print(f"FAIL {row['case']}: {error}", flush=True)
                with (root / "failures.log").open("a") as handle:
                    handle.write(f"{row['case']}: {error}\n")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, range(4)))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("--project", required=True)
    inv.add_argument("--server", type=int, required=True)
    inv.add_argument("--output", required=True)
    planner = sub.add_parser("plan")
    planner.add_argument("--inventories", nargs=3, required=True)
    planner.add_argument("--output", required=True)
    balance = sub.add_parser("rebalance")
    balance.add_argument("--manifest", required=True)
    balance.add_argument("--output", required=True)
    balance.add_argument("--transfer_plan", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--project", required=True)
    runner.add_argument("--python", required=True)
    runner.add_argument("--manifest", required=True)
    runner.add_argument("--server", type=int, required=True)
    runner.add_argument("--output", required=True)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--inventory", required=True)
    smoke.add_argument("--case", default="patchtst_ETTh1_96")
    smoke.add_argument("--project", required=True)
    smoke.add_argument("--python", required=True)
    smoke.add_argument("--output", required=True)
    smoke.add_argument("--probability", action="store_true")
    args = parser.parse_args()
    if args.command == "inventory":
        rows = inventory(args.project, args.server)
        write_csv(args.output, rows, FIELDS)
        print(f"inventory server={args.server} count={len(rows)}")
    elif args.command == "plan":
        plan(args.inventories, args.output)
    elif args.command == "rebalance":
        rebalance(args.manifest, args.output, args.transfer_plan)
    elif args.command == "smoke":
        row = next(r for r in read_csv(args.inventory) if r["case"] == args.case)
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        overrides, _ = base_overrides(row, output, HEAD_LRS[1],
                                      probability=args.probability)
        overrides.update(epochs=1, stage_epochs=1, eiil_steps=2,
                         decomposition_warmup_epochs=0,
                         max_train_batches=1, max_eval_batches=1,
                         diagnostic_samples=16, num_workers=0,
                         prob_num_samples=4, prob_sample_chunk=2,
                         prob_flow_steps=2, max_prob_eval_batches=1)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        command = [args.python, "-u", "tools/run_predictive_env_iv_patchtst.py",
                   "--base_run_config", row["base_config"],
                   "--run_overrides_json", json.dumps(overrides)]
        status = subprocess.run(command, cwd=args.project, env=env,
                                stdout=(output / "run.log").open("w"),
                                stderr=subprocess.STDOUT, check=False).returncode
        if status:
            raise RuntimeError(f"smoke exit={status}; inspect {output / 'run.log'}")
        verify_output(output, {"decomposition_type": "supervised_joint_private",
                               "representation_constraint": "classification"})
        print(json.dumps(summarize(output)))
    else:
        run(args.project, args.python, args.manifest, args.server, args.output)


if __name__ == "__main__":
    main()
