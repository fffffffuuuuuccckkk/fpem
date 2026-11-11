#!/usr/bin/env python3
"""Merge independent pilots and print the user-facing wide comparison table."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FILES = ("summary", "per_prototype", "per_time_block", "mapping_probe",
         "mapping_probe_candidates", "routing_diagnostics", "operator_diagnostics")
TABLE = ("Case", "Method", "FVE", "FutureLift", "Future_MSE", "Operator_Within",
         "Operator_Between", "Operator_Drift", "Route_Agreement", "K_eff",
         "Independent_Events", "Ridge_MSE", "Ridge_MAE", "Gain_vs_Global",
         "Gain_vs_Random", "Train_Time", "Peak_VRAM")


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write(path, data):
    if not data:
        return
    fields = list(dict.fromkeys(key for row in data for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader(); writer.writerows(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    merged = {name: [] for name in FILES}
    status = []
    for case in sorted(root.glob("pilots/*/*/pred_*/seed_*")):
        case_key = f"{case.parent.parent.parent.name}/{case.parent.parent.name}-{case.parent.name.replace('pred_', '')}"
        if (case / "complete").is_file():
            state = "complete"
            for name in FILES:
                path = case / (name + ".csv")
                if path.is_file():
                    merged[name].extend({"case_path": str(case), "case": case_key, **row}
                                        for row in rows(path))
        elif (case / "failed.json").is_file():
            state = "failed"
        else:
            state = "running_or_pending"
        reason = ""
        if (case / "failed.json").is_file():
            reason = json.loads((case / "failed.json").read_text(encoding="utf-8")).get("reason", "")
        status.append({"case": case_key, "seed": case.name.replace("seed_", ""),
                       "status": state, "reason": reason, "path": str(case)})
    for name, data in merged.items():
        write(root / (name + ".csv"), data)
    write(root / "run_status.csv", status)
    def display(value):
        if value is None or value == "":
            return "NA"
        try:
            number = float(value)
            if number != number:
                return "NA"
            return f"{number:.4f}"
        except (ValueError, TypeError):
            return str(value)
    text = ["# 历史—未来映射原型提取质量总表", "",
            f"已完成 {sum(row['status']=='complete' for row in status)}/{len(status)} 个运行配置。完整列见 `summary.csv`；失败与待运行见 `run_status.csv`。", "",
            "| " + " | ".join(TABLE) + " |",
            "|" + "|".join(["---"] * len(TABLE)) + "|"]
    for row in merged["summary"]:
        if row.get("status") != "ok":
            text.append("| " + " | ".join([row.get("case", ""), row.get("method", ""),
                *("NA" for _ in TABLE[2:])]) + " |")
            continue
        values = [row.get("case", ""), row.get("method", "")]
        values += [display(row.get(name)) for name in TABLE[2:]]
        text.append("| " + " | ".join(values) + " |")
    text += ["", "不做综合加权排名。`NA` 表示无法可靠估计或尚无结果；看 `operator_diagnostics.csv` 和 `FutureLift_reason` 确认原因。",
             "P2 若独立事件不足，将明确列为 `not_estimable`，不会把重叠窗口当成有效局部算子样本。",
             "A0 来自完整原始 TRAIN 预训练；这里只检验新增原型/映射在 TRAIN 时间块的外推，不是完整流水线 OOF。"]
    (root / "summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    if merged["summary"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = 0
        for row in merged["summary"]:
            try:
                x, y = float(row["Operator_Drift"]), float(row["Ridge_MSE"])
                if x != x or y != y:
                    continue
            except (ValueError, TypeError, KeyError):
                continue
            ax.scatter(x, y)
            ax.annotate(row.get("case", "") + " " + row.get("method", ""), (x, y), fontsize=6)
            plotted += 1
        ax.set(xlabel="Cross-time operator behavior drift (lower better)",
               ylabel="Later-block full-future Ridge MSE (lower better)",
               title=f"Mapping accuracy vs operator stability; {plotted} estimable arms")
        fig.tight_layout(); fig.savefig(root / "mapping_stability_pareto.png", dpi=160); plt.close(fig)
    readme = """# 指标与协议

本轮固定 K=8，seed=2022；A0 只读。TRAIN 按时间切 A（早期教师/原型）、B1（学生/Probe 拟合）、B2（Ridge 正则选择与学生诊断）、C（晚期时间外推）。相邻块完整历史+未来窗口互不重叠。所有目标除关系教师和离线诊断外，归一化仅使用当前历史均值与尺度。C 路由只用历史，绝不使用 C 未来。A0 原先在完整 TRAIN 训练过，因此不是全系统 OOF。

- FVE = 1 - C 上按历史学生软归属合成的 A 原型未来中心 MSE / A 总体未来中心 MSE；越大越好。原型内未来 MSE 是历史尺度归一化后的逐点 MSE，越小越好。未来相似度定义为 exp(-两条未来的归一化 MSE)，分位数见 `per_prototype.csv`。
- FutureLift = 1 - 同原型未来配对 MSE / Raw 历史距离分箱匹配的异原型随机配对 MSE。配对限同变量、完整窗口不重叠；至少 3 个有效箱及 50 对，否则 NA 并记录原因。它是描述统计，不把重复配对当独立事件。
- Between-Prototype Variance 是 A 软归属质量加权的原型未来中心方差；中心距离是两两未来中心 RMS。越大只说明区分度，非预测能力。
- Operator Within/Between 是局部 4×4 仿射算子在双方均有训练输入支持的公共参考历史状态上的输出 MSE；Operator Separation=Between/Within。无足够共同参考或独立事件时 NA。Operator Drift 是 C 离线拟合算子相对同原型 A 算子的函数距离；C 真值只用于此诊断，从不改变 C 归属。
- P2 的局部邻域仅按历史四维状态最近邻检索，贪心排除完整窗口重叠；至少 d_x+2 个独立事件且 Gram 条件数 <1e6 才标为可靠。公共参考输入来自 A 中心区域；若少于 4 个共有支持参考点，不报告映射距离有效。
- Route Agreement 是 B2 软教师与历史学生的 Top-1 一致率；KL、A/B1/B2 利用率及缺失原因见 `routing_diagnostics.csv`。P0 直接按历史 KMeans 路由，无学生网络。C 不计算未来教师归属。
- Ridge MSE/MAE 使用共同 Raw 64 输入、完整未来输出。Global、Prototype、Shared、Random 四种 Probe 只改变软原型权重，使用同一 B1 拟合、B2 选择 λ∈{0.1,1,10,100}、C 评价。Gain=(MSE_control-MSE_prototype)/MSE_control；正数越大越好。Operator Transfer Probe 独立预测四维未来统计，不能与完整未来 Ridge MSE 直接比数值。
- 原型覆盖距离在共同 TRAIN-A 标准化 Raw 64 空间可横向比较；方法原生空间距离不得直接横比。Silhouette 越高、Davies–Bouldin 越低、Calinski–Harabasz 越高，但它们只描述历史划分。K_eff=exp(-Σp log p)，范围 1–8；独立事件按起点间隔至少 seq_len+pred_len 贪心计数。
- 关系指纹的均值/趋势/周期幅值/相位/频谱能量/波动比均由 A 历史尺度计算；周期由 A 原始时间序列差分功率谱估计，弱周期有相位有效性掩码。P2 状态维度固定 4；低容量学生路由 64→32→8。训练时间不含已有 A0 训练；峰值显存含只读 A0 特征提取。CPU 峰值是进程 RSS。

`summary.csv` 保存全部列，`per_prototype.csv`、`per_time_block.csv`、`mapping_probe.csv`、`routing_diagnostics.csv`、`operator_diagnostics.csv` 保存细节。NA 不填零。长预测 720 的独立事件数可能不足，此时不能基于重叠窗口断言算子稳定。
"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    print("\n".join(text), flush=True)


if __name__ == "__main__":
    main()
