"""Read-only compact manifest for the G1+R3 extension on one host."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "results/fpem_dual_pattern_memory/g1_r3_extended_test_search"
point = []
prob = []
for path in sorted(root.glob("*/*/pred_*/search/metrics.json")):
    data = json.loads(path.read_text())
    point.append({
        "backbone": data["backbone"], "dataset": data["dataset"],
        "pred_len": data["pred_len"], "a0_mse": data["a0"]["mse"],
        "a0_mae": data["a0"]["mae"], "g1_mse": data["g1"]["mse"],
        "g1_mae": data["g1"]["mae"], "final_mse": data["g1_r3"]["mse"],
        "final_mae": data["g1_r3"]["mae"], "g1_ridge": data["g1"]["ridge"],
        "g1_scale": data["g1"]["scale"], "best_trial": data["best_trial"],
        "trials_run": data["trials_run"], "test_samples": data["test_samples"],
    })
for path in sorted(root.glob("*/*/pred_*/probability_search_ema_lr/metrics.json")):
    data = json.loads(path.read_text())
    best = data["best"]
    test = best["test"]
    parts = path.parts
    pred_len = int(path.parent.parent.name.split("_")[-1])
    trial = path.parent / ("flow_lr_%g_trace.json" % best["lr"])
    trace = json.loads(trial.read_text())[-1] if trial.exists() else {}
    prob.append({
        "backbone": path.parent.parent.parent.parent.name,
        "dataset": path.parent.parent.parent.name,
        "pred_len": pred_len, "lr": best["lr"],
        "sample_scale": best["sample_scale"],
        "inv_mse": test["inv_MSE"], "inv_mae": test["inv_MAE"],
        "mean_mse": test["mean_MSE"], "mean_mae": test["mean_MAE"],
        "crps_empirical": test["CRPS"],
        "probts_crps": test["ProbTS_CRPS"],
        "picp95": test["PICP_95"], "mpiw95": test["MPIW_95"],
        "wis": test["WIS"], "samples_evaluated": test["samples_evaluated"],
        "final_maturity": trace.get("maturity"),
        "final_effective_lr": trace.get("effective_prob_lr"),
    })
print(json.dumps({"point": point, "prob": prob}))
