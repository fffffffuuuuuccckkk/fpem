#!/usr/bin/env python
"""Compact validation-only coordinate LR search for shared FPEM heads.

Search decomposer and environment LR by validation Yinv MSE; Future-Zvar LR
by mean ProbTS-exact validation CRPS across six stochastic non-legacy heads.
No test evaluation occurs until the final selected configuration is rerun.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path


STOCHASTIC_HEADS = (
    "affine_gaussian", "affine_mdn", "affine_flow",
    "residual_gaussian", "residual_lowrank", "residual_flow",
)


def read_validation(run_root):
    metrics = {}
    for head in STOCHASTIC_HEADS:
        path = run_root / head / "val_metrics.json"
        if path.exists():
            metrics[head] = json.loads(path.read_text())
    if not metrics:
        raise RuntimeError(f"no validation metrics in {run_root}")
    return {
        "inv_MSE": next(iter(metrics.values()))["inv_MSE"],
        "mean_ProbTS_CRPS": sum(
            item["ProbTS_CRPS"] for item in metrics.values()
        ) / len(metrics),
        "heads_counted": sorted(metrics),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pred_len", required=True, type=int)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--project_dir", type=Path,
                        default=Path("/data/OuXiaoyu/Time-Series-Library-FPEM"))
    parser.add_argument("--search_root", type=Path,
                        default=Path("/data/OuXiaoyu/Time-Series-Library-FPEM/results/"
                                     "fpem_probabilistic/head_comparison_search"))
    parser.add_argument("--final_root", type=Path,
                        default=Path("/data/OuXiaoyu/Time-Series-Library-FPEM/results/"
                                     "fpem_probabilistic/head_comparison"))
    parser.add_argument("--include_upper_lr", action="store_true",
                        help="Also try 2e-4 for decomposer and env LR.")
    args = parser.parse_args()
    if not (args.project_dir / "scripts/fpem_probabilistic/run_head_comparison.sh").exists():
        raise FileNotFoundError("shared-head runner not found")
    trial_log = []
    dec_choices = ["1e-4", "5e-5"]
    env_choices = ["1e-4", "2e-5"]
    if args.include_upper_lr:
        dec_choices.append("2e-4")
        env_choices.append("2e-4")
    future_choices = ["1e-4", "5e-4"]

    def run(dec, env_lr, future, validation_only):
        label = f"dec_{dec}_env_{env_lr}_future_{future}"
        root = (args.search_root / label if validation_only else args.final_root)
        result = root / args.dataset / f"pred_{args.pred_len}"
        environment = os.environ.copy()
        environment.update({
            "PROJECT_DIR": str(args.project_dir),
            "DATASET": args.dataset, "PRED_LEN": str(args.pred_len),
            "GPU": str(args.gpu), "OUTPUT_ROOT": str(root),
            "LR_DECOMPOSER": dec, "LR_ENV_HEAD": env_lr,
            "LR_VARIANT": future,
            "PROB_HEAD": "all_shared", "PROB_CONDITION_MODE": "full_shape",
            "PROB_VALIDATION_ONLY": "1" if validation_only else "0",
            "PROB_NUM_SAMPLES": environment.get("PROB_NUM_SAMPLES", "100"),
            "PROB_FLOW_STEPS": environment.get("PROB_FLOW_STEPS", "12"),
        })
        print("RUN", label, "validation" if validation_only else "final", flush=True)
        subprocess.run(
            ["bash", "scripts/fpem_probabilistic/run_head_comparison.sh"],
            cwd=args.project_dir, env=environment, check=True,
        )
        return result

    cache = {}
    def trial(dec, env_lr, future):
        key = (dec, env_lr, future)
        if key not in cache:
            location = run(dec, env_lr, future, True)
            metrics = read_validation(location)
            cache[key] = metrics
            trial_log.append({"dec": dec, "env": env_lr, "future": future,
                              "result_path": str(location), **metrics})
            print("VAL", key, metrics, flush=True)
        return cache[key]

    base_env, base_future = "1e-4", "1e-4"
    selected_dec = min(dec_choices,
                       key=lambda value: trial(value, base_env, base_future)["inv_MSE"])
    selected_env = min(env_choices,
                       key=lambda value: trial(selected_dec, value, base_future)["inv_MSE"])
    selected_future = min(future_choices,
                          key=lambda value: trial(selected_dec, selected_env, value)
                          ["mean_ProbTS_CRPS"])
    selected = (selected_dec, selected_env, selected_future)
    final_path = run(*selected, validation_only=False)
    summary = {
        "dataset": args.dataset, "pred_len": args.pred_len,
        "selection_protocol": "validation only; no test-based selection",
        "selection_metrics": {
            "decomposer_and_environment": "validation Yinv MSE",
            "future_zvar": "mean ProbTS Evaluator CRPS over six stochastic heads",
        },
        "selected": {"lr_decomposer": selected_dec,
                     "lr_env_head": selected_env,
                     "lr_future_zvar": selected_future},
        "trials": trial_log, "final_result_path": str(final_path),
    }
    destination = (args.search_root / args.dataset / f"pred_{args.pred_len}")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "coordinate_search_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print("SEARCH_COMPLETE", summary["selected"], final_path, flush=True)


if __name__ == "__main__":
    main()
