"""Aggregate completed DPPM offline diagnostic cases without selecting by TEST."""
from __future__ import annotations

import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/"results/fpem_dual_pattern_memory/diagnostics"
CASES=("patchtst_ETTh1_720","itransformer_ETTm1_720","moderntcn_ETTm2_720")


def main():
    rows=[]
    for case in CASES:
        folder=ROOT/case
        if not (folder/"diagnostic_complete").exists():
            raise RuntimeError(f"not complete: {case}")
        summary=json.loads((folder/"summary.json").read_text())
        modules=json.loads((folder/"module_metrics.json").read_text())
        corrected=json.loads((folder/"purged_module_metrics.json").read_text())
        capacity=json.loads((folder/"prototype_capacity.json").read_text())
        rows.append((case,summary,modules,corrected,capacity))
    lines=["# DPPM 第一批离线诊断（D0–D6、D9–D11）","",
      "训练目标仅来自 TRAIN；R3 强度仅用前半 VAL 选择。TEST 最优系数只作为事后诊断，不参与部署。",
      "旧 pilot 的 5 时间块 OOF 没有清除跨块重叠预测窗口；新对照采用 6 块，按标签观测终点过滤来源，2/3 块选方向、4 块校可靠性、5 块独立评估。",
      "A0 曾在完整 TRAIN 训练，因此此处‘OOF’仅指记忆模块，不是完整系统 OOF。","",
      "| 配置 | 旧 OOF 重叠源 | A0 TEST | 原型独立 TEST | B3 TEST | B6 TEST | 清理后 B6 TEST | VAL 选 R3 系数 | R3 TEST | 清理后可靠片段 |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for case,s,_,_,_ in rows:
        lines.append(f"| {case} | {s['overlapping_source_windows_in_old_oof']} | {s['a0_test_mse']:.6f} | {s['stable_memory_test_mse']:.6f} | {s['b3_test_mse']:.6f} | {s['b6_test_mse']:.6f} | {s['purged_b6_test_mse']:.6f} | {s['val_selected_r3_alpha']:.3f} | {s['r3_val_scaled_test_mse']:.6f} | {s['purged_reliability_nonzero_count']}/64 |")
    lines += ["","## 容量与诊断","",
      "原型未来头使用闭式 Ridge，**没有学习率**；先检查原型数、正则强度及样本外方向，不做无意义的 LR 网格。",""]
    for case,s,old,new,capacity in rows:
        old_oof=old["TRAIN_memory_OOF_fold4"]["metrics"]
        new_oof=new["TRAIN_memory_OOF_fold5"]["metrics"]
        lines += [f"- {case}: 稳定 rank={s['stable_rank']}，VAL eta={s['eta']}；原始可靠性正收益 {s['reliability_positive_gain_count']}/64，清理后 {s['purged_reliability_positive_gain_count']}/64；稳定/变化头参数量 {capacity['stable_head_parameters']}/{capacity['variation_head_parameters']}。",
            f"  - 旧 OOF B3/B6 MSE={old_oof['b3']['mse']:.6f}/{old_oof['b6']['mse']:.6f}；独立清理后 OOF B3/B6 MSE={new_oof['b3']['mse']:.6f}/{new_oof['b6']['mse']:.6f}。"]
    lines += ["","详细证据：各 case 的 `code_audit.json`、`module_metrics.json`、`purged_module_metrics.json`、`reliability_64.csv`、`purged_reliability_64.json`、`chronological_purge_audit.json`、`worst20_test.json`。",
      "","边界平滑的零填充会把常数可靠性 1 在两端衰减为 0.6；新代码改成复制边界填充，旧结果仍保留。",
      "","本批只诊断并修复因果切分与边界数值问题。D7/D8 的变化输入对照、B7–B13、优化搜索和概率 Flow 均未启动；需先根据上述证据决定是否有可迁移信号。"]
    (ROOT/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(ROOT/"summary.md")


if __name__=="__main__":main()
