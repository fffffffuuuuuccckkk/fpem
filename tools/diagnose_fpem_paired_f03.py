#!/usr/bin/env python3
"""Read-only F03 diagnostic using the original A0 and selected B checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.predictive_env.direct_invariant_forecaster import DirectInvariantForecaster
from tools.run_fpem_f03_r3_invariance import collect, prepare
from tools.run_fpem_paired_prototype import head_predict, point_metric, roles
from tools.run_fpem_shared_factorization_g1r3 import load_experiment

METHODS = ("F03", "F03_cons", "F03_cons_inv")


def rows_to_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def head_drift(initial: torch.Tensor, trained: torch.Tensor) -> dict:
    if initial.shape != trained.shape:
        raise ValueError("A0/F03 head weight shape differs")
    baseline = initial.detach().double().norm().item()
    difference = (trained.detach().double() - initial.detach().double()).norm().item()
    return {"a0_weight_l2": baseline, "weight_delta_l2": difference,
            "relative_weight_drift": difference / max(baseline, 1e-12)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    options = parser.parse_args()
    out = options.output
    if (out / "complete").exists():
        print("ALREADY_COMPLETE", out, flush=True)
        return
    torch.set_num_threads(4)
    config = json.loads((options.experiment / "run_config.json").read_text(encoding="utf-8"))
    source = config["cli"]
    base = json.loads(Path(source["base_run_config"]).read_text(encoding="utf-8"))
    cli = argparse.Namespace(base_run_config=source["base_run_config"], gpu=options.gpu,
                             seed=base["seed"], allow_legacy_reference_signature=source.get(
                                 "allow_legacy_reference_signature", False))
    args, model, loaders, device, a0_sha = load_experiment(cli)
    if a0_sha != config["a0_sha256"]:
        raise AssertionError("A0 SHA differs from original experiment")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("A0 is not frozen")

    raw = {}
    for role, stride in (("train", source["train_stride"]),
                         ("val", source["val_stride"]),
                         ("test", source["test_stride"])):
        raw[role] = collect(model, loaders[role].dataset.dataset, args, device, stride)
        print("COLLECT", role, len(raw[role]["ids"]), flush=True)
    scale = raw["train"]["x"].float().var(1, unbiased=False).add(1e-5).sqrt()
    floor = (scale.median(0).values * .25).clamp_min(.01)
    data = {role: prepare(record, floor) for role, record in raw.items()}
    groups, _, _, audit = roles(data["train"], data["val"], args.seq_len + args.pred_len)
    blocks = {"A": groups["A_bank_head"], "B": groups["B_transfer_reliability"],
              "VAL": data["val"], "TEST": data["test"]}
    tokens, width = blocks["A"]["hidden"].shape[-2:]
    initial = DirectInvariantForecaster(tokens, width, args.pred_len, 0,
                                        model.h_forecast_head).to(device)
    init_pred = head_predict(initial, blocks["B"], device)
    init_error = (init_pred - blocks["B"]["base_hist_norm"]).abs().max().item()
    if init_error > 1e-4:
        raise AssertionError(f"F03 epoch 0 does not reproduce A0: {init_error}")

    epoch_rows = []
    for method in METHODS:
        metadata = json.loads((options.experiment / "B_F03" / method / "metrics.json").read_text(
            encoding="utf-8"))
        baseline = point_metric(blocks["B"]["base_hist_norm"], blocks["B"])
        epoch_rows.append({"method": method, "epoch": 0, "checkpoint": "A0 initialization",
                           "a0_B_mse": baseline["mse"], "a0_B_mae": baseline["mae"],
                           "f03_B_mse": point_metric(init_pred, blocks["B"])["mse"],
                           "f03_B_mae": point_metric(init_pred, blocks["B"])["mae"],
                           "selected_by_B": False})
        for entry in metadata["trace"]:
            epoch_rows.append({"method": method, "epoch": entry["epoch"],
                               "checkpoint": "trace only; weights not retained",
                               "a0_B_mse": baseline["mse"], "a0_B_mae": baseline["mae"],
                               "f03_B_mse": entry["transfer"]["mse"],
                               "f03_B_mae": entry["transfer"]["mae"],
                               "selected_by_B": entry["epoch"] == metadata["best_epoch"]})
    rows_to_csv(out / "epoch0_B_by_epoch.csv", epoch_rows)

    metric_rows, drift_rows = [], []
    for name, block in blocks.items():
        a0 = point_metric(block["base_hist_norm"], block)
        metric_rows.append({"group": name, "method": "A0", "count": len(block["ids"]),
                            "mse": a0["mse"], "mae": a0["mae"], "delta_mse_vs_a0": 0.0})
    for method in METHODS:
        state = torch.load(options.experiment / "B_F03" / method / "head.pt",
                           map_location="cpu", weights_only=False)
        head = DirectInvariantForecaster(tokens, width, args.pred_len, 0,
                                         model.h_forecast_head).to(device)
        head.load_state_dict(state, strict=True)
        drift = head_drift(model.h_forecast_head.weight, head.head.weight)
        bias0, bias1 = model.h_forecast_head.bias, head.head.bias
        drift_rows.append({"method": method, "selected_epoch": json.loads(
            (options.experiment / "B_F03" / method / "metrics.json").read_text(
                encoding="utf-8"))["best_epoch"], **drift,
                           "relative_bias_drift": (bias1.double() - bias0.double()).norm().item()
                           / max(bias0.double().norm().item(), 1e-12)})
        for name, block in blocks.items():
            measured = point_metric(head_predict(head, block, device), block)
            a0 = next(row for row in metric_rows if row["group"] == name and row["method"] == "A0")
            metric_rows.append({"group": name, "method": method, "count": len(block["ids"]),
                                "mse": measured["mse"], "mae": measured["mae"],
                                "delta_mse_vs_a0": measured["mse"] - a0["mse"]})
        del head
        torch.cuda.empty_cache()
    rows_to_csv(out / "head_drift.csv", drift_rows)
    rows_to_csv(out / "group_mse_mae.csv", metric_rows)

    selection = json.loads((options.experiment / "C_point" / "metrics.json").read_text(
        encoding="utf-8"))
    chosen = selection["selected_head_on_B"]
    best_b = min(selection["head_scores_B"], key=selection["head_scores_B"].get)
    if chosen != best_b:
        raise AssertionError("selected_head_on_B disagrees with minimum recorded B MSE")
    select_rows = []
    for method in METHODS:
        lookup = {(row["group"], row["method"]): row for row in metric_rows}
        select_rows.append({"method": method, "selected_on_B": method == chosen,
                            "B_mse": lookup[("B", method)]["mse"],
                            "VAL_mse": lookup[("VAL", method)]["mse"],
                            "TEST_mse": lookup[("TEST", method)]["mse"],
                            "VAL_delta_vs_a0": lookup[("VAL", method)]["delta_mse_vs_a0"],
                            "TEST_delta_vs_a0": lookup[("TEST", method)]["delta_mse_vs_a0"]})
    rows_to_csv(out / "selection.csv", select_rows)
    summary = {"a0_sha256": a0_sha, "source_experiment": str(options.experiment),
               "selected_head_on_B": chosen, "epoch0_max_abs_vs_a0_B": init_error,
               "time_roles": audit, "full_pipeline_oof": False,
               "limitations": ["Only B MSE/MAE was logged for all epochs; intermediate head weights were not saved.",
                               "A0 was trained on all original TRAIN; A and B are downstream diagnostics, not full-pipeline OOF.",
                               "TEST is reported only, never used for this experiment's checkpoint selection."]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
    (out / "complete").write_text("complete\n", encoding="utf-8")
    print("COMPLETE", out, "selected", chosen, flush=True)


if __name__ == "__main__":
    main()
