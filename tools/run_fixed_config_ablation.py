#!/usr/bin/env python3
"""Run one long-horizon control from the exact recorded FPEM configuration."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-config", type=Path, required=True)
    parser.add_argument("--arm", choices=("no_domain", "environment_shuffled"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    project = Path.cwd()
    best = json.loads(args.best_config.read_text())
    if int(best["pred_len"]) not in (96, 192, 336, 720):
        raise ValueError("long-horizon control expects 96/192/336/720")
    metrics_path = project / best["metrics_path"]
    source_config = metrics_path.parent.parent / "run_config.json"
    source = json.loads(source_config.read_text())
    if source["representation_constraint"] != "classification":
        raise ValueError("source must use classification + GRL")
    if source["decomposition_type"] != "complementary_gate":
        raise ValueError("source must use complementary_gate")
    if source["variant_fusion_mode"] != "horizon_future_var":
        raise ValueError("source must be the long-horizon FPEM main experiment")
    reference = project / source["reference_checkpoint"]
    if not reference.is_file():
        raise FileNotFoundError(reference)
    target = args.output_root / best["backbone"] / best["dataset"] / f"pred_{best['pred_len']}" / args.arm
    complete = target / "run_complete"
    metrics = target / "A2" / "metrics_and_diagnostics.json"
    if complete.is_file() and metrics.is_file():
        print(f"SKIP complete {target}", flush=True)
        return
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Incomplete output is preserved for inspection: {target}")
    runner = project / "tools" / "run_predictive_env_iv_patchtst.py"
    help_text = subprocess.check_output([sys.executable, str(runner), "--help"], text=True)
    options = set(re.findall(r"--([a-zA-Z0-9_]+)", help_text))
    override = {"gpu": args.gpu, "output": str(target)}
    if args.arm == "no_domain":
        override["lambda_domain"] = 0.0
    else:
        override["environment_supervision"] = "shuffled"
    run = dict(source)
    run.update(override)
    run["reference_checkpoint"] = str(reference)
    run["require_reference_checkpoint"] = True
    unknown = sorted(set(run) - options - {"enc_in"})
    if unknown:
        raise ValueError(f"Source configuration has unsupported runner options: {unknown}")
    if args.validate_only:
        print(f"VALID {args.arm} {best['backbone']} {best['dataset']} {best['pred_len']} GPU={args.gpu}")
        return
    target.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(runner)]
    for key, value in run.items():
        if key == "enc_in" or key not in options:
            continue
        if isinstance(value, bool):
            if value:
                command.append(f"--{key}")
        elif value is not None:
            command.extend((f"--{key}", str(value)))
    protocol = {
        "arm": args.arm,
        "source_best_config": str(args.best_config),
        "source_run_config": str(source_config),
        "source_metrics": str(metrics_path),
        "reference_checkpoint": str(reference),
        "overrides": override,
        "no_new_lr_search": True,
    }
    (target / "ablation_protocol.json").write_text(json.dumps(protocol, indent=2))
    with (target / "run.log").open("w") as handle:
        result = subprocess.run(command, cwd=project, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode or not metrics.is_file():
        raise RuntimeError(f"Control failed; inspect {target / 'run.log'}")
    report = json.loads(metrics.read_text())
    original = json.loads(metrics_path.read_text())
    summary = {
        "arm": args.arm, "backbone": best["backbone"], "dataset": best["dataset"],
        "pred_len": int(best["pred_len"]),
        "original_inv_MSE": original.get("inv_only_MSE", original.get("inv_MSE")),
        "original_inv_MAE": original.get("inv_only_MAE"),
        "ablation_inv_MSE": report.get("inv_only_MSE", report.get("inv_MSE")),
        "ablation_inv_MAE": report.get("inv_only_MAE"),
        "original_full_MSE": original.get("MSE"),
        "original_full_MAE": original.get("MAE"),
        "ablation_full_MSE": report.get("MSE"),
        "ablation_full_MAE": report.get("MAE"),
        "source_original": str(metrics_path), "source_ablation": str(metrics),
    }
    (target / "comparison.json").write_text(json.dumps(summary, indent=2))
    complete.touch()
    print(f"DONE {target}", flush=True)


if __name__ == "__main__":
    main()
