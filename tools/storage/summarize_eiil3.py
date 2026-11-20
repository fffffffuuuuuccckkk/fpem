"""Read-only EIIL3 point/Flow accounting from per-case JSON, never stale aggregate CSV."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter,defaultdict
from pathlib import Path

POINT=("A0","I3_ERM","I3_SoftIRM_EIIL3","I3_SoftIRM_ShuffleQ",
       "I3_GlobalRidge","I3_R3_EIIL3","I3_R3_ShuffleQ")
FLOW=("Flow_U0","Flow_U1","Flow_Ushuffle")


def value(obj,*keys):
    for key in keys:
        if not isinstance(obj,dict):return None
        obj=obj.get(key)
    return obj if isinstance(obj,(float,int)) else None


def summarize(root,out):
    rows=[]
    manifest=[json.loads(line) for line in (root/"manifest.jsonl").read_text().splitlines() if line]
    for item in manifest:
        case=root/item["case_key"]
        point_file=case/"point_metrics.json"
        point=json.loads(point_file.read_text()) if point_file.is_file() else None
        has_point=bool(point and (case/"point_complete").exists())
        flow_status=json.loads((case/"flow_status.json").read_text()) if (case/"flow_status.json").exists() else None
        flow_results={name:json.loads((case/(name+".json")).read_text())
                      for name in FLOW if (case/(name+".json")).is_file()}
        row={"case_key":item["case_key"],"backbone":item["backbone"],
             "dataset":item["dataset"],"pred_len":item["pred_len"],"seed":item["seed"],
             "manifest_status":item["status"],"manifest_reason":item.get("missing_reason",""),
             "teacher_source":point.get("teacher_source","") if point else "",
             "point_state":"complete" if has_point else ("partial" if case.exists() else "not_started"),
             "flow_state":("complete" if (case/"flow_complete").exists() and len(flow_results)==3
                 else flow_status.get("status") if flow_status else
                 "partial" if flow_results else "not_started"),
             "flow_reason":flow_status.get("reason","") if flow_status else "",
             "flow_modes":",".join(flow_results)}
        if point:
            for name in POINT:
                row[name+"_MSE"]=value(point,"scores","test",name,"mse")
                row[name+"_MAE"]=value(point,"scores","test",name,"mae")
            a0=row["A0_MSE"];r3=row["I3_R3_EIIL3_MSE"]
            row["R3_gain_vs_A0_pct"]=(a0-r3)/a0*100 if a0 and r3 is not None else None
            row["ridge_alpha_selected"]=point.get("ridge_alpha_selected")
            row["u_test_mean"]=value(point,"u_test","mean")
            row["I3_best_epoch"]=value(point,"I3_SoftIRM_EIIL3","best_epoch")
            row["I3_lambda_inv"]=value(point,"I3_SoftIRM_EIIL3","lambda_inv")
        for name,entry in flow_results.items():
            for key in ("CRPS","WIS","PICP_95","MPIW_95"):
                row[name+"_"+key]=value(entry,"TEST",key)
        rows.append(row)
    out.mkdir(parents=True,exist_ok=True)
    columns=list(dict.fromkeys(key for row in rows for key in row))
    with (out/"current_status.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    complete=[r for r in rows if r["point_state"]=="complete"]
    flow=[r for r in rows if r["flow_state"]=="complete"]
    def less(left,right,items):return sum(r.get(left) is not None and r.get(right) is not None and r[left]<r[right]-1e-9 for r in items)
    def mean(key,items):
        nums=[r[key] for r in items if r.get(key) is not None]
        return statistics.mean(nums) if nums else None
    result={"manifest":len(rows),"manifest_status":dict(Counter(r["manifest_status"] for r in rows)),
            "point_status":dict(Counter(r["point_state"] for r in rows)),
            "point_by_manifest_status":{status:dict(Counter(r["point_state"] for r in rows
                if r["manifest_status"]==status)) for status in sorted({r["manifest_status"] for r in rows})},
            "flow_status":dict(Counter(r["flow_state"] for r in rows)),
            "teacher_sources":dict(Counter(r["teacher_source"] for r in complete)),
            "I3_best_epoch_zero":sum(r.get("I3_best_epoch")==0 for r in complete),
            "I3_lambda_zero":sum(r.get("I3_lambda_inv")==0 for r in complete),
            "I3_exact_A0_MSE":sum(r.get("I3_SoftIRM_EIIL3_MSE")==r.get("A0_MSE") for r in complete),
            "point_wins":{"I3_vs_A0":less("I3_SoftIRM_EIIL3_MSE","A0_MSE",complete),
                "R3_vs_A0":less("I3_R3_EIIL3_MSE","A0_MSE",complete),
                "R3_vs_I3":less("I3_R3_EIIL3_MSE","I3_SoftIRM_EIIL3_MSE",complete),
                "R3_vs_GlobalRidge":less("I3_R3_EIIL3_MSE","I3_GlobalRidge_MSE",complete),
                "R3_vs_ShuffleQ":less("I3_R3_EIIL3_MSE","I3_R3_ShuffleQ_MSE",complete)},
            "point_by_backbone":{},"flow_wins":{
                "U1_vs_U0":less("Flow_U1_CRPS","Flow_U0_CRPS",flow),
                "U1_vs_Ushuffle":less("Flow_U1_CRPS","Flow_Ushuffle_CRPS",flow)},
            "flow_mean_crps":{mode:mean(mode+"_CRPS",flow) for mode in FLOW},
            "flow_U1_minus_U0_crps_max_abs":max((abs(r["Flow_U1_CRPS"]-r["Flow_U0_CRPS"])
                for r in flow),default=None),
            "missing_flow_after_point":[r["case_key"]+" ["+r["flow_state"]+"]" for r in complete
                                        if r["flow_state"]!="complete"],
            "partial_point":[r["case_key"] for r in rows if r["point_state"]=="partial"],
            "best_point_gains":[{"case":r["case_key"],"gain_pct":r["R3_gain_vs_A0_pct"],
                "A0_MSE":r["A0_MSE"],"R3_MSE":r["I3_R3_EIIL3_MSE"]}
                for r in sorted(complete,key=lambda x:-x["R3_gain_vs_A0_pct"])[:5]],
            "worst_point_gains":[{"case":r["case_key"],"gain_pct":r["R3_gain_vs_A0_pct"],
                "A0_MSE":r["A0_MSE"],"R3_MSE":r["I3_R3_EIIL3_MSE"]}
                for r in sorted(complete,key=lambda x:x["R3_gain_vs_A0_pct"])[:5]],
            "completed_cases":[r["case_key"] for r in complete]}
    for backbone in sorted({r["backbone"] for r in rows}):
        group=[r for r in complete if r["backbone"]==backbone]
        result["point_by_backbone"][backbone]={"complete":len(group),
            "I3_vs_A0":less("I3_SoftIRM_EIIL3_MSE","A0_MSE",group),
            "R3_vs_A0":less("I3_R3_EIIL3_MSE","A0_MSE",group),
            "R3_gain_vs_A0_mean_pct":mean("R3_gain_vs_A0_pct",group)}
    (out/"current_summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);a=p.parse_args();summarize(a.root,a.out)
