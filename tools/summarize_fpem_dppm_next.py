"""Summarize completed sequential DPPM V/C/R3 pilot without TEST selection."""
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/"results/fpem_dual_pattern_memory/next_pilot"
CASES=("patchtst_ETTh1_720","itransformer_ETTm1_720","moderntcn_ETTm2_720")


def main():
    rows=[]
    for case in CASES:
        folder=ROOT/case
        if not (folder/"next_pilot_complete").exists():
            raise RuntimeError(f"pending {case}")
        with (folder/"all_results.csv").open(newline="",encoding="utf-8") as file:
            rows.extend(csv.DictReader(file))
    with (ROOT/"all_results.csv").open("w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    lines=["# DPPM V/C/R3 分阶段试验","",
      "所有 Backbone 与 A0 checkpoint 冻结；复用 TRAIN 预测缓存。OOF 仅指记忆模块，6 个连续块执行来源未来终点过滤。参数及单一缩放仅用前半 VAL 选择；TEST 排名是探索性，未用于选择。","",
      "| Backbone/数据 | A0 TEST MSE | 最佳 V（VAL 选） | V TEST | 最佳 C（VAL 选） | C TEST | R3-A | R3-B 打乱 | R3-C 全局 | R3-D 单库 |",
      "|---|---:|---|---:|---|---:|---:|---:|---:|---:|"]
    for case in CASES:
        subset=[r for r in rows if r["case"]==case]
        v=min((r for r in subset if r["stage"]=="V" and r["representation"]!="V3"),
              key=lambda r:(round(float(r["val_mse"]),6),float(r["oof_candidate_mse"])))
        c=min((r for r in subset if r["stage"]=="C"),key=lambda r:(round(float(r["val_mse"]),6),float(r["oof_candidate_mse"])))
        rr={r["variant"]:r for r in subset if r["stage"]=="R3"}
        lines.append(f"| {case} | {float(v['a0_test_mse']):.6f} | {v['variant']} | {float(v['test_mse']):.6f} | {c['variant']} | {float(c['test_mse']):.6f} | {float(rr['R3_A_current']['test_mse']):.6f} | {float(rr['R3_B_shuffle_xy']['test_mse']):.6f} | {float(rr['R3_C_global']['test_mse']):.6f} | {float(rr['R3_D_single_bank']['test_mse']):.6f} |")
        lines += ["",f"- {case}: VAL 最佳变化描述 {v['representation']} / {v['mapping_input']}；映射容量 {c['variant']}。对应 OOF 原型候选收益 V={float(v['oof_candidate_gain']):+.6f}、C={float(c['oof_candidate_gain']):+.6f}；变化库匹配熵 V={float(v['match_entropy']):.3f}，独立支持最小值 {v['min_independent_events']}。",""]
    lines += ["详细 TRAIN/OOF/VAL/TEST MSE/MAE、原型参数量、独立支持、VAL 缩放网格及按 horizon/样本分层增益在各 case 的 JSON 和 CSV 中。", "",
      "本试验不训练概率 Flow，也不使用 TEST 选择正式方案。若 R3-A 不能持续超过打乱配对、全局线性和单库对照，不能将收益归因于双模式记忆。"]
    (ROOT/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(ROOT/"summary.md")


if __name__=="__main__":main()
