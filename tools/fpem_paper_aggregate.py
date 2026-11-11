#!/usr/bin/env python3
"""Non-destructive, protocol-aware inventory of the FPEM paper experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import torch


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "ExchangeRate", "Electricity", "Weather")
BACKBONES = ("patchtst", "itransformer", "moderntcn")
HORIZONS = (96, 192, 336, 720)
PROB_DATASETS = ("Electricity", "Solar", "ExchangeRate", "ILI", "Weather")
PROB_HORIZONS = (24, 48)
POINT_FIELDS = (
    "backbone", "dataset", "pred_len", "a0_mse", "a0_mae", "g1_mse", "g1_mae",
    "final_mse", "final_mae", "g1_improvement_pct", "r3_improvement_pct",
    "final_improvement_pct", "source", "selection_protocol", "test_windows",
    "g1_ridge", "g1_scale", "r3_best_trial", "status",
)
PROB_FIELDS = (
    "backbone", "dataset", "pred_len", "calibration", "CRPS", "ProbTS_CRPS",
    "ProbTS_CRPS_Sum", "WIS", "PICP_95", "MPIW_95", "inv_MSE", "inv_MAE",
    "mean_MSE", "mean_MAE", "PICP_95_near", "PICP_95_far", "evaluated_windows",
    "total_test_windows", "test_stride", "num_samples", "flow_training_residual",
    "source", "status",
)


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def pct(before: object, after: object) -> float:
    if not finite(before) or not finite(after) or float(before) <= 0:
        return math.nan
    return 100 * (float(before) - float(after)) / float(before)


def point_rows(repo: Path) -> tuple[list[dict], list[dict]]:
    full_root = repo / "results/fpem_dual_pattern_memory/g1_r3_full_testlr_search"
    extended_root = repo / "results/fpem_dual_pattern_memory/g1_r3_extended_test_search"
    raw = []
    for source, path in (("60-case test-LR search", full_root / "all_results.csv"),
                         ("24-case extension", extended_root / "point_results.csv")):
        for row in read_csv(path):
            raw.append((source, str(path), row))
    found = {}
    for source, path, row in raw:
        try:
            key = (row["backbone"].lower(), row["dataset"], int(row["pred_len"]))
        except (KeyError, ValueError):
            continue
        if key[0] not in BACKBONES or key[1] not in DATASETS or key[2] not in HORIZONS:
            continue
        if key in found:
            continue
        final_mse = row.get("g1_r3_mse", row.get("final_mse"))
        final_mae = row.get("g1_r3_mae", row.get("final_mae"))
        values = (row.get("a0_mse"), row.get("a0_mae"), row.get("g1_mse"),
                  row.get("g1_mae"), final_mse, final_mae)
        if not all(finite(v) for v in values):
            continue
        case_dir = Path(path).parent / key[0] / key[1] / f"pred_{key[2]}"
        artifact_dir = case_dir if source.startswith("60-case") else case_dir / "search"
        model_file = artifact_dir / "model.pt"
        metrics_file = artifact_dir / "metrics.json"
        run_file = artifact_dir / "run_config.json"
        status = "verified"
        try:
            metrics = json.loads(metrics_file.read_text())
            config = json.loads(run_file.read_text())
            checkpoint = torch.load(model_file, map_location="cpu", weights_only=False)
            if (metrics.get("backbone") != key[0] or metrics.get("dataset") != key[1]
                    or int(metrics.get("pred_len", -1)) != key[2]
                    or config.get("reference_sha256") != checkpoint.get("reference_sha256")
                    or metrics.get("reference_sha256") != checkpoint.get("reference_sha256")
                    or int(config.get("cli", {}).get("test_stride", -1)) != 1
                    or int(metrics.get("test_samples", 0)) <= 0):
                status = "incomplete: config/checkpoint/TEST protocol mismatch"
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            status = f"incomplete: {type(exc).__name__}"
        found[key] = {
            "backbone": key[0], "dataset": key[1], "pred_len": key[2],
            "a0_mse": row["a0_mse"], "a0_mae": row["a0_mae"],
            "g1_mse": row["g1_mse"], "g1_mae": row["g1_mae"],
            "final_mse": final_mse, "final_mae": final_mae,
            "g1_improvement_pct": pct(row["a0_mse"], row["g1_mse"]),
            "r3_improvement_pct": pct(row["g1_mse"], final_mse),
            "final_improvement_pct": pct(row["a0_mse"], final_mse),
            "source": path, "selection_protocol": "TEST-driven exploratory",
            "test_windows": metrics.get("test_samples", "") if status == "verified" else "",
            "g1_ridge": row.get("g1_ridge", ""),
            "g1_scale": row.get("g1_scale", ""),
            "r3_best_trial": row.get("r3_best_trial", row.get("best_trial", "")),
            "status": status,
        }
    missing = []
    for backbone in BACKBONES:
        for dataset in DATASETS:
            for horizon in HORIZONS:
                key = (backbone, dataset, horizon)
                if key not in found or found[key]["status"] != "verified":
                    missing.append({"phase": "point", "backbone": backbone,
                                    "dataset": dataset, "pred_len": horizon,
                                    "reason": "missing metrics or failed checkpoint/protocol audit"})
    return sorted(found.values(), key=lambda r: (r["backbone"], r["dataset"], r["pred_len"])), missing


def probability_rows(root: Path) -> tuple[list[dict], list[dict]]:
    found = {}
    for path in root.glob("**/metrics.json"):
        try:
            content = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "results" not in content or "total_test_windows" not in content:
            continue
        key = (content.get("backbone"), content.get("dataset"), content.get("pred_len"))
        if key[0] != "patchtst" or key[1] not in PROB_DATASETS or key[2] not in PROB_HORIZONS:
            continue
        for calibration, metrics in content["results"].items():
            valid = (all(finite(metrics.get(field)) for field in ("CRPS", "ProbTS_CRPS", "WIS", "PICP_95", "MPIW_95"))
                     and metrics.get("evaluated_windows") == content["total_test_windows"]
                     and metrics.get("test_stride") == 1 and metrics.get("num_samples", 0) >= 100)
            row = {**metrics, "backbone": key[0], "dataset": key[1], "pred_len": key[2],
                   "calibration": calibration, "flow_training_residual": content.get("flow_training_residual", "unknown"),
                   "source": str(path), "status": "verified" if valid else "incomplete"}
            # Prefer the primary copy; imported mirrors can have a duplicate result.
            found.setdefault((key, calibration), row)
    rows = sorted(found.values(), key=lambda r: (r["dataset"], r["pred_len"], r["calibration"]))
    missing = []
    for dataset in PROB_DATASETS:
        for horizon in PROB_HORIZONS:
            key = (("patchtst", dataset, horizon), "selected_scale")
            if key not in found or found[key]["status"] != "verified":
                missing.append({"phase": "probability_full_test", "backbone": "patchtst",
                                "dataset": dataset, "pred_len": horizon,
                                "reason": "full stride-1 100-sample metrics missing/invalid"})
    return rows, missing


def tex_num(value: object) -> str:
    return f"{float(value):.4f}" if finite(value) else "--"


def latex_point(rows: list[dict], field: str) -> str:
    suffix = "mse" if field == "mse" else "mae"
    lines = [r"\begin{tabular}{llrrrrrrrrr}", r"\toprule",
             r"Dataset & Horizon & \multicolumn{3}{c}{PatchTST} & \multicolumn{3}{c}{iTransformer} & \multicolumn{3}{c}{ModernTCN} \\",
             r" & & A0 & G1 & G1+R3 & A0 & G1 & G1+R3 & A0 & G1 & G1+R3 \\", r"\midrule"]
    lookup = {(r["backbone"], r["dataset"], r["pred_len"]): r for r in rows}
    for dataset in DATASETS:
        for horizon in HORIZONS:
            parts = [dataset, str(horizon)]
            for backbone in BACKBONES:
                row = lookup.get((backbone, dataset, horizon), {})
                vals = [row.get(f"{prefix}_{suffix}") for prefix in ("a0", "g1", "final")]
                valid = [float(v) for v in vals if finite(v)]
                best = min(valid) if valid else None
                for value in vals:
                    txt = tex_num(value)
                    parts.append(r"\textbf{" + txt + "}" if best is not None and finite(value) and abs(float(value) - best) < 1e-10 else txt)
            lines.append(" & ".join(parts) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}",
                  "% TEST-driven exploratory hyperparameter selection; not an independent generalization estimate."])
    return "\n".join(lines) + "\n"


def latex_probability(rows: list[dict]) -> str:
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Dataset & Horizon & CRPS & ProbTS CRPS & PICP95 & MPIW95 \\", r"\midrule"]
    for row in rows:
        if row["calibration"] != "selected_scale" or row["status"] != "verified":
            continue
        lines.append(" & ".join([row["dataset"], str(row["pred_len"]),
                                 tex_num(row["CRPS"]), tex_num(row["ProbTS_CRPS"]),
                                 tex_num(row["PICP_95"]), tex_num(row["MPIW_95"])]) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}",
                  "% Flow source may have TRAIN in-sample residual fitting and sparse TEST-based model selection; see audit."])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--result_root", type=Path, required=True)
    args = parser.parse_args()
    repo, root = args.repo.resolve(), args.result_root.resolve()
    out = root / "paper_ready"
    out.mkdir(parents=True, exist_ok=True)
    point, point_missing = point_rows(repo)
    probability, prob_missing = probability_rows(root)
    write_csv(out / "point_main_mse.csv", point, POINT_FIELDS)
    write_csv(out / "point_main_mae.csv", point, POINT_FIELDS)
    write_csv(out / "point_improvement.csv", point, POINT_FIELDS)
    write_csv(out / "probability_main_results.csv", probability, PROB_FIELDS)
    write_csv(out / "probability_interval_metrics.csv", probability, PROB_FIELDS)
    write_csv(out / "probability_full_test_audit.csv", probability, PROB_FIELDS)
    write_csv(out / "incomplete_cases.csv", point_missing + prob_missing,
              ("phase", "backbone", "dataset", "pred_len", "reason"))
    failures = []
    for log in root.glob("**/queue.log"):
        try:
            tail = log.read_text(encoding="utf-8", errors="replace")[-3000:]
        except OSError:
            continue
        lines = [line for line in tail.splitlines() if line.strip()]
        if lines and lines[-1].startswith("FAILED "):
            failures.append({"log": str(log), "last_failure": lines[-1]})
    write_csv(out / "failures.csv", failures, ("log", "last_failure"))
    inventory = []
    inventory.extend({"phase": "point", "backbone": r["backbone"], "dataset": r["dataset"],
                      "pred_len": r["pred_len"], "status": r["status"], "source": r["source"],
                      "test_windows": r["test_windows"]} for r in point)
    inventory.extend({"phase": "probability", "backbone": r["backbone"], "dataset": r["dataset"],
                      "pred_len": r["pred_len"], "status": r["status"], "source": r["source"],
                      "test_windows": r["evaluated_windows"]} for r in probability)
    write_csv(out / "experiment_inventory.csv", inventory,
              ("phase", "backbone", "dataset", "pred_len", "status", "source", "test_windows"))
    all_result_fields = ("phase", "backbone", "dataset", "pred_len", "calibration",
                         "a0_mse", "a0_mae", "g1_mse", "g1_mae", "final_mse", "final_mae",
                         "CRPS", "ProbTS_CRPS", "PICP_95", "MPIW_95", "WIS",
                         "status", "source", "test_windows")
    all_results = [{"phase": "point", **r, "test_windows": r["test_windows"]} for r in point]
    all_results += [{"phase": "probability", **r, "test_windows": r["evaluated_windows"]}
                    for r in probability]
    write_csv(out / "all_results.csv", all_results, all_result_fields)
    baseline_rows = []
    for path in (repo / "results/fpem_classic_baselines_completion_20261013/baseline_summary.csv",
                 repo / "results/fpem_classic_baselines_h48_h192/baseline_summary.csv"):
        for row in read_csv(path):
            try:
                horizon = int(row["pred_len"])
            except (KeyError, ValueError):
                continue  # Ignore old '*.incomplete.*' directory artifacts.
            if row.get("dataset") not in DATASETS or horizon not in HORIZONS:
                continue
            if not finite(row.get("MSE")) or not finite(row.get("MAE")):
                continue
            baseline_rows.append({"method": row["method"], "dataset": row["dataset"],
                                  "pred_len": horizon, "MSE": row["MSE"], "MAE": row["MAE"],
                                  "source": str(path), "protocol_status": "metric-only; checkpoint/split audit pending"})
    unique_baselines = {}
    for row in baseline_rows:
        unique_baselines.setdefault((row["method"], row["dataset"], row["pred_len"]), row)
    write_csv(out / "baseline_audit.csv", list(unique_baselines.values()),
              ("method", "dataset", "pred_len", "MSE", "MAE", "source", "protocol_status"))
    abc_root = repo / "results/fpem_abc_mechanism_search_20261031"
    abc_status = read_csv(abc_root / "case_status.csv")
    abc_new = root / "abc_patchtst_etth1_96"
    if (abc_new / "paper_verified.complete").is_file() and all(
        (abc_new / filename).is_file() for filename in
        ("A_all_results.csv", "B_all_results.csv", "C_probability_results.csv")
    ):
        abc_status.append({"backbone": "patchtst", "dataset": "ETTh1", "pred_len": "96",
                           "planned": "A_B_C", "A_complete": "True", "B_complete": "True",
                           "B_low_cost_complete": "True", "C_complete": "True"})
    write_csv(out / "abc_case_status.csv", abc_status,
              ("backbone", "dataset", "pred_len", "planned", "A_complete", "B_complete",
               "B_low_cost_complete", "C_complete"))
    abc_prob = read_csv(abc_root / "C_probability_results.csv") + read_csv(abc_new / "C_probability_results.csv")
    if abc_prob:
        fields = tuple(dict.fromkeys(key for row in abc_prob for key in row))
        write_csv(out / "probability_condition_ablation.csv", abc_prob, fields)
        write_csv(out / "probability_ablation.csv", abc_prob, fields)
    abc_point = read_csv(abc_root / "B_all_results.csv") + read_csv(abc_new / "B_all_results.csv")
    if abc_point:
        fields = tuple(dict.fromkeys(key for row in abc_point for key in row))
        write_csv(out / "point_ablation.csv", abc_point, fields)
    abc_shuffle = []
    for path in root.glob("**/abc_test_shuffle/**/metrics.csv"):
        for row in read_csv(path):
            abc_shuffle.append({**row, "source": str(path)})
    if abc_shuffle:
        fields = tuple(dict.fromkeys(key for row in abc_shuffle for key in row))
        write_csv(out / "abc_test_condition_shuffle.csv", abc_shuffle, fields)
    distribution_baselines = []
    for path in root.glob("**/simple_distributions/metrics.csv"):
        if not (path.parent / "verified.complete").is_file():
            continue
        for row in read_csv(path):
            distribution_baselines.append({**row, "source": str(path),
                                           "protocol_status": "full TEST; TRAIN in-sample residual distribution"})
    for path in root.glob("**/independent_baselines/*/test_metrics.json"):
        if not (path.parent / "paper_verified.complete").is_file():
            continue
        try:
            row = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        valid = (row.get("evaluated_windows") == row.get("total_test_windows")
                 and row.get("num_samples", 0) >= 100
                 and all(finite(row.get(key)) for key in ("CRPS", "ProbTS_CRPS", "PICP_95", "MPIW_95")))
        if valid:
            distribution_baselines.append({**row, "source": str(path),
                                           "protocol_status": "full TEST; independent project adapter"})
    if distribution_baselines:
        fields = tuple(dict.fromkeys(key for row in distribution_baselines for key in row))
        write_csv(out / "probability_baseline_audit.csv", distribution_baselines, fields)
    (out / "table1_point_mse.tex").write_text(latex_point(point, "mse"), encoding="utf-8")
    (out / "table2_point_mae.tex").write_text(latex_point(point, "mae"), encoding="utf-8")
    (out / "table3_probability_crps.tex").write_text(latex_probability(probability), encoding="utf-8")
    (out / "table4_probability_intervals.tex").write_text(latex_probability(probability), encoding="utf-8")
    wins = Counter(r["backbone"] for r in point if r["status"] == "verified"
                   and float(r["final_mse"]) < float(r["a0_mse"]))
    point_complete = sum(r["status"] == "verified" for r in point)
    prob_complete = len({(r["dataset"], r["pred_len"]) for r in probability
                         if r["calibration"] == "selected_scale" and r["status"] == "verified"})
    summary = ["# FPEM 论文实验阶段盘点", "",
               f"- 长程点预测 checkpoint、配置、指标核验通过：{point_complete}/84；各 Backbone 胜过 A0 组数：{dict(wins)}。",
               f"- 短程完整 TEST 概率复评：{prob_complete}/10（5 数据集 × 24/48）。",
               f"- 点预测缺口：{len(point_missing)}；概率缺口：{len(prob_missing)}；已汇总独立点预测 Baseline：{len(unique_baselines)} 行；已验证概率基线：{len(distribution_baselines)} 行。",
               "- 既有 C12 shuffled_fused 只在 TRAIN 打乱条件，TEST 推理条件未打乱；不可称为 TEST-shuffle 消融。",
               "- 点预测既有超参数按 TEST 选择；本轮 TEST 属探索开发集。",
               "- 既有 Flow 使用 TRAIN 样本内点残差训练；不应与严格 OOF Flow 混称同一协议。",
               "- 仅当完整 TEST stride=1 且采样数≥100、窗口数一致时，概率指标计为已验证。",
               "- 本脚本只汇总可验证的既有指标；缺少的 Baseline、消融、效率、图表不得视为已完成。",
               ""]
    (out / "paper_experiment_summary.md").write_text("\n".join(summary), encoding="utf-8")
    (out / "paper_ready_checklist.md").write_text(
        "# 验收清单\n\n"
        f"- [{'x' if point_complete == 84 else ' '}] 84 组点预测 checkpoint/指标/协议核验\n"
        f"- [{'x' if prob_complete == 10 else ' '}] 10 组完整 TEST 概率复评\n"
        "- [ ] A0/G1/G1+R3/独立概率 Baseline 公平对照\n"
        "- [ ] G1/R3/Flow 消融及 TEST 条件错配\n"
        "- [ ] 非对称/条件/Conformal 区间校准\n"
        "- [ ] 效率、机制图、随机种子重复\n", encoding="utf-8")
    (out / "reproduce_commands.md").write_text(
        "# 当前队列与复评命令\n\n"
        "以下命令应在对应服务器的 `/data/OuXiaoyu/Time-Series-Library-FPEM` 下执行；"
        "运行中队列用 `screen -ls` 查看。\n\n"
        "```bash\n"
        "bash scripts/run_paper_weather.sh 24 0\n"
        "bash scripts/run_paper_weather.sh 48 1\n"
        "bash scripts/run_paper_full_eval_queue.sh Electricity 24 0 /data/OuXiaoyu/miniconda3/envs/basicts/bin/python\n"
        "bash scripts/run_paper_simple_baselines_queue.sh Electricity 24 0 /data/OuXiaoyu/miniconda3/envs/basicts/bin/python\n"
        "bash scripts/run_paper_independent_baselines_queue.sh Electricity 24 0 /data/OuXiaoyu/miniconda3/envs/basicts/bin/python\n"
        "python tools/fpem_paper_aggregate.py --result_root results/fpem_paper_completion_20261101\n"
        "python tools/fpem_paper_figures.py --result_root results/fpem_paper_completion_20261101\n"
        "```\n\n完整案例矩阵及服务器分配见各队列日志。\n", encoding="utf-8")
    print(json.dumps({"point": point_complete, "point_missing": len(point_missing),
                      "prob_full_cases": prob_complete, "prob_missing": len(prob_missing)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
