"""Collect completed raw-data diagnostics without selecting on TEST."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("results/fpem_visible_variation"))
    args = p.parse_args()
    rows = []
    for path in sorted(args.root.glob("*_seed*/metrics.csv")):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                row["source"] = str(path.parent)
                rows.append(row)
    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (args.root / "all_metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    lines = ["# 可见历史变化：原始数据 Pilot", "",
             "仅用原始时间序列；未读取或训练 A0/F03/R3/Flow。所有阈值、K-means 和标准化只在 TRAIN 拟合；超参数只用 VAL 选择。",
             "TEST 仅报告，不用于选方法。图中未来响应用于离线标签和事后展示，不进入历史特征或配对检索。", "",
             "| 配置 | pair-u AUC | pair-(u,z) AUC | shuffle-z AUC | u+z NDCG@5 | random NDCG@5 |", "|---|---:|---:|---:|---:|---:|"]
    grouped = {}
    for row in rows:
        if row.get("split") == "test":
            grouped.setdefault((row["dataset"], row["horizon"]), {})[(row["task"], row["method"])] = row
    for (dataset, horizon), group in sorted(grouped.items()):
        def value(task, method, field):
            raw = group.get((task, method), {}).get(field, "")
            try:
                return f"{float(raw):.3f}"
            except (ValueError, TypeError):
                return "NA"
        lines.append(f"| {dataset}-{horizon} | {value('B_pair_class','pair_u','auc')} | "
                     f"{value('B_pair_class','pair_u_z','auc')} | "
                     f"{value('B_pair_class','pair_u_shuffled_z','auc')} | "
                     f"{value('C_rank','pair_u_z','ndcg5')} | "
                     f"{value('C_rank','random','ndcg5')} |")
    lines += ["", "分类只保留 TRAIN 未来分歧的两端四分位，不能当作全体 pair 的判别率。",
              "置信区间按不重叠时间事件块 Bootstrap；独立事件过少时记 NA。",
              "相关性或分类增益只支持历史可见变化与未来响应存在统计关系，不证明因果漂移或 R3 的有效性。",
              "", "完整逐配置指标：`all_metrics.csv`；每个子目录另有图、TEST 配对和阈值配置。"]
    (args.root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
