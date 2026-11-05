"""Collect completed extractor pilots without selecting methods on TEST."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/fpem_dual_pattern_memory/extractor_pilot"
CASES = ("patchtst_ETTh1_720", "itransformer_ETTm1_720", "moderntcn_ETTm2_720")


def save_csv(name, rows):
    if not rows:
        return
    path = RESULT / name
    with path.open("w", newline="", encoding="utf-8") as f:
        fields = sorted(set().union(*(set(row) for row in rows)))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main():
    RESULT.mkdir(parents=True, exist_ok=True)
    all_rows, proto, mapping, group, efficiency = [], [], [], [], []
    completed = []
    blocked = []
    for case in CASES:
        for mode_index in range(9):
            mode = f"P{mode_index}"
            folder = RESULT / case / mode
            if (folder/"blocked.json").exists():
                blocked.append((case, mode, json.loads((folder/"blocked.json").read_text())))
            if not (folder/"extractor_complete").exists():
                continue
            completed.append((case, mode))
            result = json.loads((folder/"point_results.json").read_text())
            extraction = json.loads((folder/"extractor_statistics.json").read_text())
            bank = json.loads((folder/"bank_statistics.json").read_text())
            oof = json.loads((folder/"mapping_oof.json").read_text())
            cost = json.loads((folder/"efficiency.json").read_text())
            for variant, test in {**result["test"], **result["controls"]}.items():
                val = result["val"].get(variant,{})
                all_rows.append({"case":case,"method":mode,"variant":variant,
                    "val_mse":val.get("mse",""),"val_mae":val.get("mae",""),
                    "test_mse":test.get("mse",""),"test_mae":test.get("mae",""),
                    "a0_test_mse":result["test"]["a0"]["mse"],
                    "eta":result["eta"],"null":result["null"]})
            for branch in ("stable","rare"):
                info = bank[branch]
                for j, support in enumerate(info["support"]):
                    proto.append({"case":case,"method":mode,"branch":branch,
                        "prototype":j,"raw_support":support,
                        "effective_support":info["effective_support"][j],
                        "independent_events":extraction.get(branch+"_independent_events",[""]*len(info["support"]))[j],
                        "match_entropy":extraction.get(branch+"_match_entropy",""),
                        "train_mapping_mse":extraction.get(branch+"_train_mapping_mse",""),
                        "block_support":json.dumps(info["block_support"]),
                        "independent_event_total":extraction.get("independent_event_total", ""),
                        "candidate_independent_event_support":json.dumps(extraction.get("candidate_independent_events",[])),
                        "candidate_time_coverage":json.dumps(extraction.get("candidate_block_coverage",[]))})
            mapping.append({"case":case,"method":mode,
                "a0_memory_oof_mse":oof.get("a0_oof_mse",""),
                "stable_memory_oof_mse":oof.get("stable_oof_mse",""),
                "rare_candidate_memory_oof_mse":oof.get("rare_candidate_oof_mse",""),
                "stable_oof_gain":oof.get("stable_oof_gain",""),
                "rare_oof_gain":oof.get("rare_oof_gain","")})
            with (folder/"common_rare_group_results.csv").open(newline="",encoding="utf-8") as f:
                for item in csv.DictReader(f):
                    group.append({"case":case,"method":mode,**item})
            efficiency.append({"case":case,"method":mode,**cost,
                               "extractor_build_seconds":extraction.get("build_seconds", "")})
    save_csv("all_results.csv", all_rows)
    save_csv("prototype_statistics.csv", proto)
    save_csv("mapping_oof_results.csv", mapping)
    save_csv("common_rare_group_results.csv", group)
    save_csv("efficiency.csv", efficiency)
    lines = ["# DPPM 原型提取比较（阶段性）", "",
      "同一 A0、原 DPPM 未来映射/NULL/可靠性/融合；参数只用前半 VAL 选择。",
      "TRAIN 的 A0 预测来自全 TRAIN 预训练模型，因此本表的时间块 OOF 仅指记忆模块，不是全系统严格 OOF。",
      "未重训 Residual Flow。TEST 结果仅报告，不能据此挑选正式方法。", "",
      f"已完成 {len(completed)} 组；安全跳过 {len(blocked)} 组。", "",
      "| 配置 | 方法 | A0 TEST MSE | 完整 TEST MSE | 稳定 OOF 增益 | 稀有 OOF 增益 | 真实配对优于打乱 | 完整≠稀有单支 |",
      "|---|---|---:|---:|---:|---:|---|---|" ]
    by = {(r["case"],r["method"],r["variant"]):r for r in all_rows}
    om = {(r["case"],r["method"]):r for r in mapping}
    for case, mode in completed:
        full, rare = by[(case,mode,"full")], by[(case,mode,"rare")]
        shuffle = by.get((case,mode,"shuffled_xy"))
        o = om[(case,mode)]
        better = "未跑" if shuffle is None else ("是" if float(full["test_mse"]) < float(shuffle["test_mse"])-1e-8 else "否")
        distinct = "是" if abs(float(full["test_mse"])-float(rare["test_mse"]))>1e-8 else "否"
        lines.append(f"| {case} | {mode} | {float(full['a0_test_mse']):.6f} | {float(full['test_mse']):.6f} | {float(o['stable_oof_gain']):.6f} | {float(o['rare_oof_gain']):.6f} | {better} | {distinct} |")
    if blocked:
        lines.extend(("", "## 未运行的安全边界"))
        for case,mode,reason in blocked:
            lines.append(f"- {case} {mode}：{reason['reason']}。不能以全 TRAIN A0 训练内误差冒充预测弱点 OOF。")
    lines.extend(("", "只有同时通过真实配对、独立分支、时间块 OOF 与低复杂度检查，才能主张双原型有额外价值。"))
    (RESULT/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"summarized {len(completed)} completed, {len(blocked)} blocked")


if __name__ == "__main__":
    main()
