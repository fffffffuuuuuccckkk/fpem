#!/usr/bin/env python3
"""Merge completed prototype-quality pilots without altering source results."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    names = ("summary", "per_prototype", "per_time_block", "mapping_probe",
             "mapping_probe_candidates", "similarity_bins")
    collected = {name: [] for name in names}
    finished = []
    for case in sorted(root.glob("pilots/*/*/pred_*")):
        if not (case / "complete").exists():
            continue
        finished.append(str(case))
        for name in names:
            source = case / (name + ".csv")
            if source.is_file():
                for row in read(source):
                    row = {"case_path": str(case), **row}
                    if "backbone" not in row:
                        row["backbone"] = case.parent.parent.name
                        row["dataset"] = case.parent.name
                        row["pred_len"] = case.name.replace("pred_", "")
                    collected[name].append(row)
    for name, rows in collected.items():
        write(root / (name + ".csv"), rows)
    full = collected["summary"]
    phase_differences = []
    for case in finished:
        real = next((row for row in full if row["case_path"] == case and
                     row["method"] == "Raw + Weak Phase"), None)
        shuffled = next((row for row in full if row["case_path"] == case and
                         row["method"] == "Raw + Shuffled Phase"), None)
        if real is None or shuffled is None:
            continue
        difference = {"case_path": case, "backbone": real["backbone"],
                      "dataset": real["dataset"], "pred_len": real["pred_len"]}
        for metric in ("FVE_C", "future_lift_C", "prototype_mapping_gain_vs_global_C",
                       "phase_concentration_A_mean"):
            try:
                difference[metric + "_real_minus_shuffled"] = float(real[metric]) - float(shuffled[metric])
            except (ValueError, TypeError):
                difference[metric + "_real_minus_shuffled"] = "NaN"
        phase_differences.append(difference)
    write(root / "phase_control_difference.csv", phase_differences)
    summary = ["# 原型提取质量对比", "",
               f"已完成 {len(finished)} 个数据配置、{len(full)} 个方法结果。完整列与缺失原因见 `summary.csv`；未完成配置不进入排名。", "",
               "| Backbone/数据/长度 | 方法 | FVE C ↑ | FutureLift C ↑ | Prototype Ridge 相对 Global C 增益 ↑ | 跨块未来中心漂移 ↓ | Raw 覆盖 P95 ↓ | K_eff ↑ | 提取+Probe 秒 ↓ | 独立 C 事件 |",
               "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(value):
        try:
            number = float(value)
            if number != number:
                return "不可估计"
            return f"{number:.4f}"
        except (TypeError, ValueError):
            return str(value)
    for row in full:
        setting = f"{row['backbone']}/{row['dataset']}-{row['pred_len']}"
        values = ("FVE_C", "future_lift_C", "prototype_mapping_gain_vs_global_C",
                  "mean_future_center_drift_BC", "coverage_common_raw_p95_C",
                  "k_eff_A", "train_total_seconds", "independent_eval_events")
        summary.append("| " + " | ".join([setting, row["method"]] + [fmt(row.get(k)) for k in values]) + " |")
    summary += ["", "不把这些指标加权为总分；正负收益须结合 `mapping_probe.csv`、`per_time_block.csv` 和独立事件数判断。",
                "所有方法共用同一 A/B/C 时间切分：A 拟合表示/原型/Probe，B 仅选 Ridge λ，C 作后续时间块评价。",
                "注意 A0 已在完整原始 TRAIN 上预训练，故这是新增原型/映射的时间外推诊断，不是完整流水线独立测试。"]
    (root / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    readme = """# FPEM 原型质量诊断指标说明

本目录只做原型与轻量线性 Probe，不训练 F03、R3 或 Residual Flow。每个案例固定 8 原型、seed 2022，使用同一 TRAIN 窗口与 A/B/C 时间切分，A/B、B/C 间按 `seq_len + pred_len` 清洗。C 是新增模块的 out-of-fit 时间块；A0 本身仍曾在完整 TRAIN 预训练，因此不是全系统 OOF。所有未来值以历史窗口均值和历史尺度归一化，绝不用未来自身标准差。评价时只用历史归属，真实未来仅作离线标签。

- `FVE_C = 1 - MSE(Y_C, prototype_future_center_A[label(X_C)]) / MSE(Y_C, global_future_center_A)`；无量纲，越高越好，可为负。原型内未来 MSE 是归一化未来的逐点均方误差，越低越好。
- `future_center_distance_rmse_A` 是训练原型未来中心两两 RMS 距离；`between_prototype_variance_A` 是按归属质量加权的未来中心相对总体中心方差；方差比以 A 未来总方差为分母。越大表示区分度，但不保证预测价值。
- `FutureLift = 1 - 同原型未来配对 MSE / 历史 Raw 距离匹配的异原型随机配对 MSE`。先限制同变量、预测标签窗互不重叠，再按 Raw 历史 MSE 的 10 个等量箱匹配随机对照。至少 3 个有效箱、50 个匹配对才报告；否则 NaN，原因在 `future_lift_reason_C`。重复配对并非独立事件，推断须看事件支持。
- `mapping_benefit_consistency_BC` 是各原型相对全局未来中心的收益在 B/C 之间的 Pearson 相关。至少 3 个共同有支持的原型才计算；越高越稳定。`mean_future_center_drift_BC` 是 B/C 未来中心相对 A 中心的加权 MSE，越低越好。
- 覆盖距离是最接近原型的每维均方距离；`own` 在各方法自身标准化表示中，不可跨方法直接比；`common_raw` 使用 A 统计标准化的共同 Raw 64 维空间，可横向比较。P90/P95/最差 10% 越低越好。
- `K_eff = exp(-sum_k p_k log p_k)`，范围 1–8，越高表示利用更均衡；`collapsed` 表示有原型低于 2% 或 5 行窗口。`independent_event_support` 按起点相隔至少 `seq_len + pred_len` 贪心计数，不把高度重叠窗口当独立证据。
- Silhouette、Davies–Bouldin、Calinski–Harabasz 均用同一 Raw 空间和相同采样点的原型标签计算；分别越高、越低、越高越好。它们描述历史聚类，不代表未来映射有效。
- 未来相似度为 `exp(-历史尺度归一化未来 MSE)`，历史相似度为 `exp(-Raw 64 维 MSE)`；`similarity_bins.csv` 按历史相似度 20 等量箱给出 C 上同原型未来相似度描述统计。
- Phase-Only 周期由 TRAIN A 历史时序一阶差分功率谱自动估计。Raw + Weak Phase 只在 Raw 最近/次近距离接近时用最多 0.15 权重的相位圆距离；Shuffled Phase 结构相同，仅打乱 A/B/C 相位。相位直方图和集中度仅描述归属。
- Probe 均用共同 Raw 64 维特征、相同 Ridge λ 候选 `{0.1,1,10,100}`，A 拟合，B 选 λ，C 评价。Prototype Ridge 每原型独立线性映射，Shared Mapping Probe 是按原型等权的单一线性映射，Random Group Ridge 是随机标签对照。`gain_vs_global_C = (MSE_global - MSE_probe)/MSE_global`；越高越好。不同方法的 Global Ridge 重复出现是相同对照，不应当视为不同模型。
- 参数量将表示投影参数和原型库浮点存储量分列。`train_total_seconds` 是特征拟合、原型拟合和 Probe 拟合的时间，不含已有 A0 训练；GPU 峰值是 A0 特征提取峰值，所有方法共用一次提取。

`per_prototype.csv` 保存每原型各块支持、未来误差、漂移、未来相似度分位数与相位分布。`per_time_block.csv` 保存各块 FutureLift、FVE、覆盖与利用率。NaN 表示支持不足或数学上无法定义，不应填零。长预测设置独立事件数可能小于原型数，因此这轮只用于筛选后续值得验证的方向，不作显著性结论。
"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    print("\n".join(summary), flush=True)


if __name__ == "__main__":
    main()
