"""Single-variable A0 prediction-preservation pilot for a frozen backbone."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from models.predictive_env.direct_invariant_forecaster import DirectInvariantForecaster
from tools.run_fpem_direct_invariant_search import collect, metrics, predict
from tools.run_fpem_f03_lr_search import arguments, source_baseline, train_one
from tools.run_fpem_shared_factorization_g1r3 import load_experiment


def main():
    cli = arguments()
    out = Path(cli.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    args, model, loaders, device, sha = load_experiment(cli)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("A0 must remain frozen")
    datasets = {name: loader.dataset.dataset for name, loader in loaders.items()}
    data = {name: collect(model, datasets[name], args, device,
                          {"train": cli.train_stride, "val": cli.val_stride,
                           "test": cli.test_stride}[name])
            for name in ("train", "val", "test")}
    baseline = source_baseline(args, sha)
    candidates = [baseline]
    for weight in (0.1, 1.0):
        candidate = train_one(3e-4, data, model, args, cli, device,
                              out / f"anchor_{weight:g}", anchor_lambda=weight)
        candidates.append(candidate)
    chosen = min(candidates, key=lambda item: item["val"]["mse"])
    if "test" not in chosen:
        hidden = data["train"]["hidden"]
        net = DirectInvariantForecaster(hidden.shape[-2], hidden.shape[-1],
                                       args.pred_len, pretrained_head=model.h_forecast_head).to(device)
        net.load_state_dict(torch.load(chosen["state_file"], map_location=device,
                                       weights_only=False))
        chosen["test"] = metrics(data["test"], predict(net, data["test"], device))
    report = {"backbone": args.backbone, "dataset": args.dataset_name,
              "pred_len": args.pred_len, "reference_sha256": sha,
              "a0_val": metrics(data["val"], data["val"]["base"]),
              "a0_test": metrics(data["test"], data["test"]["base"]),
              "candidates": [{k: v for k, v in item.items() if k != "history"}
                             for item in candidates],
              "selected": {k: v for k, v in chosen.items() if k != "history"},
              "selection": "VAL MSE selects epoch and anchor lambda; TEST report-only",
              "note": "The only new variable is the TRAIN prediction-preservation coefficient; no inference module."}
    (out / "selection.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "complete").write_text("complete\n", encoding="utf-8")
    print("F03 ANCHOR SEARCH COMPLETE", out, flush=True)


if __name__ == "__main__":
    main()
