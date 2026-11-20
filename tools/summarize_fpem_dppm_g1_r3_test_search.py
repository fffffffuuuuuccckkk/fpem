"""Summarize the three-host exploratory TEST-selected G1+R3 search."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/fpem_dual_pattern_memory/g1_r3_full_testlr_search"
BACKBONES = ("patchtst", "itransformer", "moderntcn")
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "ExchangeRate")
HORIZONS = (96, 192, 336, 720)


def main():
    rows = []
    missing = []
    for backbone in BACKBONES:
        for dataset in DATASETS:
            for horizon in HORIZONS:
                case = BASE / backbone / dataset / f"pred_{horizon}"
                if not (case / "complete").is_file():
                    missing.append(f"{backbone}/{dataset}/{horizon}")
                    continue
                result = json.loads((case / "metrics.json").read_text(encoding="utf-8"))
                rows.append({
                    "backbone": backbone, "dataset": dataset, "pred_len": horizon,
                    "a0_mse": result["a0"]["mse"], "a0_mae": result["a0"]["mae"],
                    "g1_mse": result["g1"]["mse"], "g1_mae": result["g1"]["mae"],
                    "g1_r3_mse": result["g1_r3"]["mse"],
                    "g1_r3_mae": result["g1_r3"]["mae"],
                    "g1_ridge": result["g1"]["ridge"],
                    "g1_scale": result["g1"]["scale"],
                    "r3_best_trial": result["best_trial"],
                    "r3_head_lr": result["best_optimizer"]["head_lr"],
                    "r3_dictionary_ratio": result["best_optimizer"]["dictionary_ratio"],
                    "r3_basis_ratio": result["best_optimizer"]["basis_ratio"],
                    "r3_route_ratio": result["best_optimizer"]["route_ratio"],
                    "r3_offset": result["best_offset"],
                    "trials_run": result["trials_run"],
                    "advanced_search_pruned": result["advanced_search_pruned"],
                    "gain_vs_a0": result["a0"]["mse"] - result["g1_r3"]["mse"],
                    "gain_r3_vs_g1": result["g1"]["mse"] - result["g1_r3"]["mse"],
                })
    BASE.mkdir(parents=True, exist_ok=True)
    with (BASE / "all_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else
                                ["backbone", "dataset", "pred_len"])
        writer.writeheader()
        writer.writerows(rows)
    (BASE / "missing_cases.txt").write_text("\n".join(missing) + "\n", encoding="utf-8")
    groups = defaultdict(list)
    for row in rows:
        groups[row["backbone"]].append(row)
    lines = ["# G1+R3 全量 TEST 选参搜索", "",
             f"完成 {len(rows)}/60；缺失 {len(missing)}。", "",
             "注意：本轮 G1 正则、缩放、R3 优化参数及 NULL 偏置均按 TEST MSE 选择，"
             "因此 TEST 是探索性开发集，不能作为独立泛化证据。",
             "G1/R3 仅在 TRAIN 拟合；VAL 只用于 R3 可靠性校准。", "",
             "| Backbone | 完成 | G1 胜 A0 | G1+R3 胜 A0 | R3 胜 G1 | 高级搜索剪枝 | 平均 A0 MSE | 平均 G1 MSE | 平均最终 MSE |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for backbone in BACKBONES:
        group = groups[backbone]
        if not group:
            continue
        n = len(group)
        lines.append(f"| {backbone} | {n} | "
                     f"{sum(r['a0_mse'] > r['g1_mse'] for r in group)} | "
                     f"{sum(r['gain_vs_a0'] > 0 for r in group)} | "
                     f"{sum(r['gain_r3_vs_g1'] > 0 for r in group)} | "
                     f"{sum(r['advanced_search_pruned'] for r in group)} | "
                     f"{sum(r['a0_mse'] for r in group)/n:.6f} | "
                     f"{sum(r['g1_mse'] for r in group)/n:.6f} | "
                     f"{sum(r['g1_r3_mse'] for r in group)/n:.6f} |")
    lines.extend(["", "各组 TEST 最优的 G1 Ridge/scale、R3 学习率与优化配置见 `all_results.csv`；"
                  "所有候选见各实验目录的 `g1_test_trials.csv` 和 `r3_test_trials.csv`。"])
    weaker = [f"{r['backbone']}/{r['dataset']}-{r['pred_len']}"
              for r in rows if r["gain_vs_a0"] <= 0]
    lines.append("未超过 A0：" + ("、".join(weaker) if weaker else "无") + "。")
    (BASE / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("complete", len(rows), "missing", len(missing))


if __name__ == "__main__":
    main()
