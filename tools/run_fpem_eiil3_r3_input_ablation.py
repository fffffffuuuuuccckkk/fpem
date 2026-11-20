#!/usr/bin/env python3
"""Frozen F03/R3 input ablation using an existing EIIL3 full-TRAIN case.

A/B reuse the exact previous checkpoints. C/D retrain only the same three
linear R3 experts with one additional full-horizon prediction feature. No
feature arrays or TEST labels are persisted. TEST selection is exploratory,
matching the source experiment rather than an independent evaluation.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.predictive_env.eiil3_linear_correction import (  # noqa: E402
    EnvironmentLinearCorrection, correction_design,
)
from models.predictive_env.relation_prototypes import route  # noqa: E402
from tools.run_fpem_eiil3 import dump, file_hash, get_data  # noqa: E402
from tools.run_fpem_eiil3_cosa import (  # noqa: E402
    fit as fit_variation, predict as predict_variation, router_from_checkpoint,
)
from tools.run_fpem_f03_safe import forecast, mse_mae  # noqa: E402


def variant_design(mode: str, f03: np.ndarray, a0: np.ndarray,
                   history_key: np.ndarray) -> np.ndarray:
    """B=[F03,key,context], C=[F03,A0,key,context], D=[F03,F03-A0,key,context]."""
    base = correction_design(f03, history_key)
    if mode == "B":
        return base
    if a0.shape != f03.shape:
        raise ValueError("A0 and F03 forecasts must have identical shape")
    if mode == "C":
        extra = a0
    elif mode == "D":
        extra = f03 - a0
    else:
        raise ValueError(mode)
    horizon = f03.shape[1]
    return np.ascontiguousarray(np.concatenate((base[:, :horizon], extra,
                                                 base[:, horizon:]), axis=1),
                                dtype=np.float32)


def correction_from_checkpoint(path: Path, features: np.ndarray, q: np.ndarray,
                               horizon: int, device: torch.device) -> np.ndarray:
    model = EnvironmentLinearCorrection(features.shape[1], horizon).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    return predict_variation(model, features, q, device)[0]


def direction_diagnostics(y: np.ndarray, a0: np.ndarray, f03: np.ndarray,
                          a0_r3: np.ndarray) -> dict:
    f = (f03.astype(np.float64) - a0.astype(np.float64)).ravel()
    r = (a0_r3.astype(np.float64) - a0.astype(np.float64)).ravel()
    residual = (y.astype(np.float64) - a0.astype(np.float64)).ravel()
    dot = lambda x, z: float(np.dot(x, z))
    fn, rn = np.sqrt(dot(f, f)), np.sqrt(dot(r, r))
    denom = max(fn * rn, 1e-20)
    n = residual.size
    return {
        "cosine_f03_r3": dot(f, r) / denom,
        "f03_rms": float(fn / np.sqrt(n)),
        "r3_rms": float(rn / np.sqrt(n)),
        "f03_residual_projection": dot(f, residual) / n,
        "r3_residual_projection": dot(r, residual) / n,
        "f03_mse_change_from_a0": (dot(f, f) - 2 * dot(f, residual)) / n,
        "r3_mse_change_from_a0": (dot(r, r) - 2 * dot(r, residual)) / n,
        "naive_sum_mse": float(np.mean((residual - f - r) ** 2)),
    }


def run(args: argparse.Namespace) -> None:
    source = args.source_root / args.case_key
    output = args.output_root / args.case_key
    old = json.loads((source / "metrics.json").read_text())
    config = json.loads((source / "run_config.json").read_text())
    if old.get("status") != "complete" or old.get("environment_supervision", "predictive") != "predictive":
        raise ValueError("source must be a completed predictive EIIL3 case")
    if not config.get("full_train") or config.get("selection_split") != "test":
        raise ValueError("this ablation matches only the full-TRAIN TEST-selected source")
    identities = {
        "A0_sha256": old["A0_sha256"],
        "head_sha256": old["head_sha256"],
        "router_sha256": old["router_sha256"],
        "source_metrics_sha256": file_hash(source / "metrics.json"),
    }
    if file_hash(source / "selected_safe_head.pt") != identities["head_sha256"]:
        raise ValueError("source F03 checkpoint changed")
    if file_hash(source / "router.pt") != identities["router_sha256"]:
        raise ValueError("source router changed")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "metrics.json"
    if args.resume and (output / "complete").exists() and result_path.exists():
        existing = json.loads(result_path.read_text())
        if existing.get("identities") == identities:
            print(json.dumps({"case": args.case_key, "status": "REUSED"}), flush=True)
            return

    manifest = Path(config["manifest"])
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    assets = {item["case_key"]: item for item in
              (json.loads(line) for line in manifest.read_text().splitlines())}
    cli = argparse.Namespace(**config)
    cli.asset = assets[args.case_key]
    cli.output = output
    cli.gpu = args.gpu
    cli.persist_features = False
    # Reuse the small TRAIN-only environment labels; never recompute EIIL or
    # retrain the router. The source artifacts are left untouched.
    for name in ("teacher_q.npz", "teacher_q.json"):
        shutil.copy2(source / name, output / name)
    data, meta, _, _ = get_data(cli)
    if meta["A0_sha256"] != identities["A0_sha256"]:
        raise AssertionError("A0 identity mismatch")
    router = router_from_checkpoint(source / "router.pt")
    q = {space: route(router, data[space]["key"]) for space in ("A", "val", "test")}
    state = torch.load(source / "selected_safe_head.pt", map_location="cpu", weights_only=False)
    f03 = {space: forecast(data[space], state) for space in q}
    a0 = {space: data[space]["a0"] for space in q}
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    horizon = data["A"]["y"].shape[1]
    baseline = {}
    for label, anchor, path in (("A", a0, source / "A0_no_F03_r3.pt"),
                                ("B", f03, source / "Safe_F03_r3.pt")):
        predictions = {}
        for space in q:
            features = correction_design(anchor[space], data[space]["key"])
            correction = correction_from_checkpoint(path, features, q[space], horizon, device)
            predictions[space] = anchor[space] + correction
        baseline[label] = {space: mse_mae(predictions[space], data[space]["y"])
                           for space in ("val", "test")}
        old_key = "A0_no_F03" if label == "A" else "Safe_F03"
        if abs(baseline[label]["test"]["mse"] - old["r3"][old_key]["scores"]["test"]["mse"]) > 2e-3:
            raise AssertionError(f"source {label} checkpoint did not reproduce its TEST MSE")
        if label == "A":
            a0_r3_test = predictions["test"]

    result = {
        "case": args.case_key, "status": "baselines_complete", "identities": identities,
        "selection": "same source TEST-selected exploratory protocol; not independent TEST evidence",
        "train_protocol": "same full TRAIN, frozen A0/F03/router; only C/D R3 parameters trained",
        "epochs": cli.epochs, "batch_size": cli.batch_size,
        "variation_lrs": config["variation_lrs"],
        "A0": {s: mse_mae(a0[s], data[s]["y"]) for s in ("val", "test")},
        "F03": {s: mse_mae(f03[s], data[s]["y"]) for s in ("val", "test")},
        "variants": baseline,
        "direction_diagnostics": direction_diagnostics(data["test"]["y"], a0["test"],
                                                        f03["test"], a0_r3_test),
    }
    dump(result_path, result)
    for mode in ("C", "D"):
        if mode in result["variants"]:
            continue
        design = {space: variant_design(mode, f03[space], a0[space], data[space]["key"])
                  for space in q}
        best, trials = None, []
        for lr in (float(x) for x in config["variation_lrs"].split(",")):
            model, info = fit_variation(design["A"], f03["A"], data["A"]["y"], q["A"],
                                        design["test"], f03["test"], data["test"]["y"],
                                        q["test"], lr, cli.epochs, cli.batch_size, cli.seed, device)
            trials.append(info)
            if best is None or info["B2_mse"] < best[0]:
                best = (info["B2_mse"], copy.deepcopy(model.cpu().state_dict()), info)
        torch.save(best[1], output / f"{mode}_r3.pt")
        model = EnvironmentLinearCorrection(design["A"].shape[1], horizon).to(device)
        model.load_state_dict(best[1])
        scores = {}
        for space in ("val", "test"):
            correction = predict_variation(model, design[space], q[space], device)[0]
            scores[space] = mse_mae(f03[space] + correction, data[space]["y"])
        result["variants"][mode] = scores
        result[f"{mode}_search"] = {"selected": best[2], "trials": trials,
                                     "checkpoint_sha256": file_hash(output / f"{mode}_r3.pt")}
        result["status"] = f"{mode}_complete"
        dump(result_path, result)
        print(json.dumps({"case": args.case_key, "mode": mode,
                          "test_mse": scores["test"]["mse"]}), flush=True)
    result["status"] = "complete"
    dump(result_path, result)
    (output / "complete").write_text("complete\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-key", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args)


if __name__ == "__main__":
    main()
