"""Summarize independent F03/R3 pilots without using TEST for selection."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    rows = []
    for path in sorted(root.rglob("metrics.json")):
        if path.parent.name not in {
            "F03", "P01", "F03_I1", "F03_I2", "F03_I3",
            "P01_I1", "P01_I2", "P01_I3", "F03_I1_random",
            "F03_I1_shuffled", "F03_I1_time", "F03_I1_eiil",
            "F03_I3_random", "F03_I3_shuffled", "F03_I3_time",
            "F03_I3_eiil"}:
            continue
        item = json.loads(path.read_text(encoding="utf-8"))
        config = json.loads((path.parents[1] / "run_config.json").read_text(encoding="utf-8"))
        if config["cli"]["epochs"] != 10 or config["cli"]["train_stride"] != 16:
            continue  # one-epoch/strided interface smoke is never a result
        rows.append({"backbone":config["backbone"], "dataset":config["dataset"],
            "pred_len":config["pred_len"], "seed":config["cli"]["seed"],
            "method":item["method"], "lr":item["selected"]["lr"],
            "lambda":item["selected"]["lambda"],
            "gradient_normalization":config["cli"].get("gradient_normalization","l2"),
            "support_weighting":config["cli"].get("support_weighting","effective"),
            "constraint_start_epoch":config["cli"].get("constraint_start_epoch",1),
            "val_mse":item["val_mse"],
            "a0_mse":item["a0"]["mse"], "a0_mae":item["a0"]["mae"],
            "shared_mse":item["invariant"]["mse"],
            "shared_mae":item["invariant"]["mae"],
            "shared_gain_vs_a0":item["a0"]["mse"]-item["invariant"]["mse"],
            "full_mse":item["r3"]["mse"], "full_mae":item["r3"]["mae"],
            "r3_gain":item["invariant"]["mse"]-item["r3"]["mse"],
            "reliability_mean":item["reliability_mean"],
            "r3_beta_mean":item["r3_beta_mean"],
            "i3_teacher_active":item["i3_teacher_active"],
            "a0_sha256":item["A0_sha256"],
            "reliability_protocol":item["reliability_protocol"],
            "metrics_path":str(path)})
    root.mkdir(parents=True, exist_ok=True)
    dest = root / "all_results.csv"
    if rows:
        with dest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    by_case = {}
    for row in rows:
        key=(row["backbone"],row["dataset"],row["pred_len"],row["seed"])
        by_case.setdefault(key,{})[row["method"]]=row
    report=["# F03 + R3 原型不变学习 pilot", "",
      "按 VAL 选择 LR、lambda 与 Epoch；TEST 只作探索性报告，不参与选择。",
      "共享头与变化分支使用同一 TRAIN 拟合的 R3。每个 F03 锚点重新估计可靠性。", ""]
    for case, methods in by_case.items():
        report += [f"## {case[0]} / {case[1]}-{case[2]} / seed {case[3]}", "",
          "| 方法 | A0 MSE | 共享 MSE | +R3 MSE | R3 增量 | VAL MSE |", 
          "|---|---:|---:|---:|---:|---:|"]
        for name, row in sorted(methods.items()):
            report.append(f"| {name} | {row['a0_mse']:.6f} | {row['shared_mse']:.6f} | {row['full_mse']:.6f} | {row['r3_gain']:+.6f} | {row['val_mse']:.6f} |")
        report.append("")
    report += ["## I3 真实原型与同结构负对照（配对差值）", "",
               "正差值表示真实 R3 原型的 MSE 更低；这些仍是探索性 TEST。", "",
               "| 配置/种子 | 对照 | VAL 对照−真实 | TEST 对照−真实 |", 
               "|---|---|---:|---:|"]
    for case, methods in by_case.items():
        real = methods.get("F03_I3")
        if not real:
            continue
        for mode in ("random", "shuffled", "time"):
            control = methods.get("F03_I3_"+mode)
            if control:
                label = f"{case[0]}/{case[1]}-{case[2]}/{case[3]}"
                report.append(f"| {label} | {mode} | {control['val_mse']-real['val_mse']:+.6f} | {control['shared_mse']-real['shared_mse']:+.6f} |")
    report.append("")
    report += ["I1 与 V-REx 风险方差思想接近；I2 是分组预测头梯度共同最优约束，不能称为 Fishr 完整复现。",
               "I3 的时间外推审计见各 case 的 i3_transfer.json；参数共识本身不等于不变映射。",
               "若缺少同协议 EPMD checkpoint，本表不伪造 EPMD 数值。"]
    (root / "summary.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    print(f"wrote {len(rows)} method rows to {dest}")


if __name__ == "__main__":
    main()
