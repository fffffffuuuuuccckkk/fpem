#!/usr/bin/env python3
"""Incremental, non-destructive point/Flow table for the EIIL3 queue."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

VARIANTS=("A0","I3_ERM","I3_SoftIRM_EIIL3","I3_SoftIRM_ShuffleQ",
          "I3_GlobalRidge","I3_R3_EIIL3","I3_R3_ShuffleQ")
FLOW=("Flow_U0","Flow_U1","Flow_Ushuffle")
FIELDS=("dataset","backbone","seq_len","pred_len","seed","K","env_source",
        "A0_sha","router_hash","I3_hash","R3_hash","variant","stage","status",
        "MSE","MAE","CRPS","WIS","PICP95","MPIW95","reason")


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS,extrasaction="ignore")
        writer.writeheader();writer.writerows(rows)
    os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path("results/fpem_eiil3_20261110"))
    cli=p.parse_args();root=cli.root
    manifest=[json.loads(line) for line in (root/"manifest.jsonl").read_text().splitlines()]
    rows=[];finished=0;skipped_flow=0
    for asset in manifest:
        case=root/asset["case_key"]
        base={"dataset":asset["dataset"],"backbone":asset["backbone"],
              "seq_len":asset["seq_len"],"pred_len":asset["pred_len"],
              "seed":asset["seed"],"K":3,"A0_sha":asset.get("A0_sha256","")}
        point=case/"point_metrics.json"
        if not point.is_file():
            rows.append({**base,"stage":"point","variant":"ALL",
                         "status":"BLOCKED" if asset["status"]=="MISSING_A0" else "PENDING",
                         "reason":asset.get("missing_reason","")})
            continue
        metric=json.loads(point.read_text());finished+=1
        common={**base,"env_source":metric["teacher_source"],
                "A0_sha":metric["A0_sha256"],"router_hash":metric["router_sha256"],
                "I3_hash":metric["I3_sha256"],"R3_hash":metric["R3_sha256"]}
        for variant in VARIANTS:
            result=metric["scores"]["test"][variant]
            rows.append({**common,"variant":variant,"stage":"point","status":"complete",
                         "MSE":result["mse"],"MAE":result["mae"]})
        status_path=case/"flow_status.json"
        if status_path.is_file():
            state=json.loads(status_path.read_text())
            if state["status"].startswith("FLOW_SKIPPED"):
                skipped_flow+=1
                rows.append({**common,"variant":"Flow_ALL","stage":"flow",
                             "status":state["status"],"reason":state["reason"]})
                continue
        for mode in FLOW:
            path=case/(mode+".json")
            if path.is_file():
                value=json.loads(path.read_text());test=value["TEST"]
                rows.append({**common,"variant":mode,"stage":"flow","status":"complete",
                             "MSE":test.get("inv_MSE"),"MAE":test.get("inv_MAE"),
                             "CRPS":test.get("CRPS"),"WIS":test.get("WIS"),
                             "PICP95":test.get("PICP_95"),"MPIW95":test.get("MPIW_95")})
            else:
                rows.append({**common,"variant":mode,"stage":"flow","status":"pending"})
    write_csv(root/"all_results.csv",rows)
    summary=["# EIIL3 running summary","",
             f"Manifest cases: {len(manifest)}; point complete: {finished}; Flow skipped: {skipped_flow}.",
             "TEST is report-only for fixed runs; pilot candidate TEST ranks are exploratory.",
             "Legacy q from PredictiveEnvIV training is labeled as such, not as fixed-A0 EIIL.",""]
    for backbone in ("patchtst","itransformer","moderntcn"):
        cases=[]
        for row in manifest:
            if row["backbone"]!=backbone:continue
            path=root/row["case_key"]/"point_metrics.json"
            if not path.is_file():continue
            score=json.loads(path.read_text())["scores"]["test"]
            cases.append((score["I3_R3_EIIL3"]["mse"],score["A0"]["mse"],
                          score["I3_ERM"]["mse"],score["I3_GlobalRidge"]["mse"]))
        wins=sum(point<a0 for point,a0,_,_ in cases)
        summary.append(f"- {backbone}: {len(cases)} point cases, {wins} I3+R3 wins vs same-case A0.")
    (root/"summary.md").write_text("\n".join(summary)+"\n")
    print(summary[2])


if __name__=="__main__":main()
