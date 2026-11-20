"""Incremental cross-case roll-up for the A/B/C mechanism queue."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.fpem_abc_mechanism import write_rows


def arguments():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",required=True)
    parser.add_argument("--source",required=True)
    return parser.parse_args()


def read_csv(path):
    if not path.exists():return []
    with path.open(encoding="utf-8",newline="") as file:
        return list(csv.DictReader(file))


def numeric(value):
    try:return float(value)
    except (ValueError,TypeError):return float("nan")


def summary_best(rows,key,low=False):
    valid=[row for row in rows if np.isfinite(numeric(row.get(key)))]
    return (min(valid,key=lambda r:numeric(r[key])) if low else
            max(valid,key=lambda r:numeric(r[key]))) if valid else None


def plotting(root,point,rows):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    folder=root/"figures";folder.mkdir(exist_ok=True)
    if point:
        labels=[f"{r['backbone']}\n{r['dataset']}-{r['pred_len']}" for r in point]
        x=np.arange(len(point))
        a0=np.array([numeric(r["A0_MSE"]) for r in point])
        g1=np.array([numeric(r["G1_MSE"]) for r in point])
        r3=np.array([numeric(r["G1_R3_MSE"]) for r in point])
        fig,ax=plt.subplots(figsize=(max(8,len(point)*.35),5))
        ax.plot(x,a0,label="A0",linewidth=1);ax.plot(x,g1,label="G1",linewidth=1)
        ax.plot(x,r3,label="G1+R3",linewidth=1)
        ax.set_xticks(x,labels,rotation=90);ax.set_ylabel("TEST MSE")
        ax.legend();fig.tight_layout();fig.savefig(folder/"point_60_cases.png",dpi=130);plt.close(fig)
    for name,code,y,filename in (("A","A2","gain_mse","A_forward_transfer.png"),
                                  ("B","B1","gain_mse","B_density_gain.png"),
                                  ("C_risk","C1","test_corr_squared_error","C_disagreement_risk.png")):
        chosen=[r for r in rows.get(name,[]) if r.get("experiment")==code and
                np.isfinite(numeric(r.get(y)))]
        if not chosen:continue
        labels=[f"{r['backbone']}-{r['dataset']}-{r['pred_len']}" for r in chosen]
        fig,ax=plt.subplots(figsize=(max(7,len(chosen)*.5),4))
        ax.bar(np.arange(len(chosen)),[numeric(r[y]) for r in chosen]);ax.axhline(0,color="gray",lw=.5)
        ax.set_xticks(np.arange(len(chosen)),labels,rotation=90)
        ax.set_ylabel(y);fig.tight_layout();fig.savefig(folder/filename,dpi=130);plt.close(fig)


def main():
    cli=arguments();root=Path(cli.root);source=Path(cli.source)
    root.mkdir(parents=True,exist_ok=True)
    point=[];groups={name:[] for name in ("A","B","C_risk","C_probability")}
    case_status=[]
    for metrics_file in sorted(source.glob("*/*/pred_*/metrics.json")):
        case=metrics_file.parent.relative_to(source)
        try:metrics=json.loads(metrics_file.read_text())
        except (json.JSONDecodeError,OSError):continue
        backbone,dataset,pred_folder=case.parts
        horizon=int(pred_folder.split("_")[-1])
        entry={"backbone":backbone,"dataset":dataset,"pred_len":horizon,
               "A0_MSE":metrics["a0"]["mse"],"A0_MAE":metrics["a0"]["mae"],
               "G1_MSE":metrics["g1"]["mse"],"G1_MAE":metrics["g1"]["mae"],
               "G1_R3_MSE":metrics["g1_r3"]["mse"],
               "G1_R3_MAE":metrics["g1_r3"]["mae"],
               "G1_gain_vs_A0":metrics["a0"]["mse"]-metrics["g1"]["mse"],
               "R3_gain_vs_G1":metrics["g1"]["mse"]-metrics["g1_r3"]["mse"],
               "TEST_selected_exploratory":True,
               "point_checkpoint":str(metrics_file.parent/"model.pt")}
        point.append(entry)
        out=root/case
        full=(backbone,dataset,horizon) in {
            ("patchtst","ETTh1",720),("patchtst","ETTh1",96),
            ("itransformer","ETTm1",720),("itransformer","ETTm1",96),
            ("moderntcn","ETTm2",720),("moderntcn","ETTm2",96)}
        status={"backbone":backbone,"dataset":dataset,"pred_len":horizon,
                "planned":"full_ABC" if full else "A_B_low_cost",
                "A_complete":(out/"A.complete").exists(),
                "B_complete":(out/"B.complete").exists(),
                "B_low_cost_complete":(out/"B_low_cost.complete").exists(),
                "C_complete":(out/"C.complete").exists()}
        case_status.append(status)
        for name,file in (("A","A_all_results.csv"),("B","B_all_results.csv"),
                          ("C_risk","C_risk_results.csv"),
                          ("C_probability","C_probability_results.csv")):
            groups[name].extend(read_csv(out/file))
        groups["B"].extend(read_csv(out/"B_retrain_trials.csv"))
    write_rows(root/"per_case_metrics.csv",point)
    for name,filename in (("A","A_all_results.csv"),("B","B_all_results.csv"),
                          ("C_risk","C_risk_results.csv"),
                          ("C_probability","C_probability_results.csv")):
        write_rows(root/filename,groups[name])
        key="test_AUSE" if name=="C_risk" else "CRPS" if name=="C_probability" else "gain_mse"
        ranked=[row for row in groups[name] if np.isfinite(numeric(row.get(key)))]
        ranked.sort(key=lambda row:numeric(row[key]),reverse=name in ("A","B"))
        write_rows(root/(name+"_ranking.csv"),ranked)
    write_rows(root/"case_status.csv",case_status)
    failures=[]
    source_fail=root/"failures_raw.tsv"
    if source_fail.exists():
        for line in source_fail.read_text(errors="replace").splitlines():
            parts=line.split("\t")
            failures.append({"case":parts[0] if parts else "", "attempt":parts[1] if len(parts)>1 else "",
                             "exit_code":parts[2] if len(parts)>2 else "",
                             "at":parts[3] if len(parts)>3 else ""})
    write_rows(root/"failures.csv",failures)
    plotting(root,point,groups)
    manifest={"source_root":str(source),"result_root":str(root),"cases":case_status,
              "test_selection":"exploratory developer-set; not independent generalization",
              "point_model_semantics":"frozen A0 + existing G1/R3 checkpoint unchanged",
              "risk_oof_caveat":"G1/R3 cross-fitted forward on TRAIN, but frozen A0 was originally fit on TRAIN",
              "C12_test_stride":16,"C12_samples":100,"C12_flow_steps":12,
              "low_cost_all_60":True,"full_six":True}
    (root/"experiment_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    full_complete=sum(s["A_complete"] and s["B_complete"] and s["C_complete"] for s in case_status if s["planned"]=="full_ABC")
    low_complete=sum(s["A_complete"] and s["B_low_cost_complete"] for s in case_status if s["planned"]=="A_B_low_cost")
    best_a=summary_best([r for r in groups["A"] if r.get("experiment")=="A2"],"gain_mse")
    best_b=summary_best([r for r in groups["B"] if r.get("experiment")=="B5"],"gain_mse")
    best_c=summary_best(groups["C_risk"],"test_AUSE",low=True)
    best_prob=summary_best(groups["C_probability"],"CRPS",low=True)
    lines=["# G1/R3/Unresolved Future 三层机制实验", "",
        f"进度：完整 ABC {full_complete}/6；其余低成本 A/B {low_complete}/{max(0,len(case_status)-6)}；现有点预测指标 {len(point)} 组。",
        "", "本轮允许按 TEST 筛选，所有 TEST 排名均是探索性开发结果，不可直接作为独立泛化证据。",
        "冻结 A0 和既有 G1/R3 checkpoint；未修改原点预测方法。", "",
        "## 当前机制证据", ""]
    for title,row,keys in (("A 跨时间迁移",best_a,("backbone","dataset","pred_len","gain_mse")),
                            ("B 专家错配负对照",best_b,("backbone","dataset","pred_len","gain_mse")),
                            ("C 风险信号",best_c,("backbone","dataset","pred_len","feature","test_AUSE")),
                            ("C 概率条件",best_prob,("backbone","dataset","pred_len","mode","CRPS"))):
        lines.append(f"- {title}："+("、".join(f"{k}={row.get(k)}" for k in keys) if row else "尚无有效结果"))
    lines += ["", "尚未完成全部配置前，不判定 H1/H2/H3 成立，也不把单项 TEST 最优宣称为机制证明。",
              "R3 必须与正确配对、错配重训和简单映射对照成对比较；C 必须与历史新颖度及无条件 Flow 比较。",
              "概率 Flow 的训练目标来自严格时间前向 G1/R3 OOF；冻结 A0 曾由 TRAIN 训练，因此不等于整条 A0+G1+R3 链的完全外推残差。",
              "", "图：figures/point_60_cases.png、A_forward_transfer.png、B_density_gain.png、C_disagreement_risk.png；每个完整案例另有 A1_g1_block_gain.png。",
              "表：per_case_metrics.csv、A_all_results.csv、B_all_results.csv、C_risk_results.csv、C_probability_results.csv。", ""]
    (root/"summary.md").write_text("\n".join(lines),encoding="utf-8")
    print("AGGREGATED",len(point),full_complete,low_complete)


if __name__=="__main__":main()
