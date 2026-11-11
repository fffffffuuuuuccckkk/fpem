#!/usr/bin/env python3
"""One-GPU resumable pilot-then-full queue; failures never erase other cases."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def record(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(value,default=str)+"\n")


def execute(argv,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("a") as stream:
        stream.write("COMMAND "+" ".join(argv)+"\n")
        stream.flush()
        return subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,check=False).returncode


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",type=Path,default=Path("results/fpem_eiil3_20261110/manifest.jsonl"))
    p.add_argument("--output-root",type=Path,default=Path("results/fpem_eiil3_20261110"))
    p.add_argument("--backbone",required=True,choices=("patchtst","itransformer","moderntcn"))
    p.add_argument("--horizon-group",required=True,choices=("short","long"))
    p.add_argument("--pilot-dataset",required=True)
    p.add_argument("--gpu",type=int,required=True)
    p.add_argument("--seed",type=int,default=2021)
    p.add_argument("--flow",action="store_true")
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--pilot-only",action="store_true")
    p.add_argument("--full-only",action="store_true")
    p.add_argument("--generate-missing-q",action="store_true")
    cli=p.parse_args()
    rows=[json.loads(line) for line in cli.manifest.read_text().splitlines()]
    horizons=(96,192) if cli.horizon_group=="short" else (336,720)
    horizon=96 if cli.horizon_group=="short" else 720
    pilot=f"{cli.backbone}/{cli.pilot_dataset}/pred_{horizon}/seed_{cli.seed}"
    chosen={row["case_key"]:row for row in rows}
    if pilot not in chosen or chosen[pilot]["status"]=="MISSING_A0":
        raise ValueError(f"pilot A0 missing: {pilot}")
    runlog=cli.output_root/"queue_logs"/f"{cli.backbone}_{cli.horizon_group}_gpu{cli.gpu}.log"
    status=cli.output_root/"queue_logs"/f"{cli.backbone}_{cli.horizon_group}_gpu{cli.gpu}.jsonl"
    work=[row for row in rows if row["backbone"]==cli.backbone and row["pred_len"] in horizons]
    if cli.dry_run:
        print(json.dumps({"pilot":pilot,"cases":[(x["case_key"],x["status"]) for x in work]}));return
    point=Path(__file__).with_name("run_fpem_eiil3.py")
    flow=Path(__file__).with_name("run_fpem_eiil3_flow.py")
    pilot_dir=cli.output_root/pilot
    if not cli.full_only:
        args=[sys.executable,str(point),"--manifest",str(cli.manifest),"--case-key",pilot,
              "--output-root",str(cli.output_root),"--gpu",str(cli.gpu),"--seed",str(cli.seed),
              "--pilot-search","--resume","--batch-size","32"]
        if cli.generate_missing_q:args.append("--generate-missing-q")
        start=time.time();code=execute(args,runlog)
        record(status,{"case":pilot,"phase":"pilot_point","exit_code":code,"seconds":time.time()-start})
        if code!=0 or not (pilot_dir/"point_complete").exists():
            raise RuntimeError(f"pilot failed; full queue not started: {pilot}")
        if cli.flow:
            fargs=[sys.executable,str(flow),"--case-dir",str(pilot_dir),"--gpu",str(cli.gpu),
                   "--seed",str(cli.seed),"--lrs","0.00003,0.0001,0.0003","--resume"]
            start=time.time();code=execute(fargs,runlog)
            record(status,{"case":pilot,"phase":"pilot_flow","exit_code":code,"seconds":time.time()-start})
            if code!=0:raise RuntimeError(f"pilot Flow failed; full queue not started: {pilot}")
    if cli.pilot_only:return
    pilot_metrics=json.loads((pilot_dir/"point_metrics.json").read_text())
    pilot_scores=pilot_metrics.get("scores",{}).get("B1",{})
    if not pilot_scores or not np.isfinite(pilot_scores["I3_SoftIRM_EIIL3"]["mse"]):
        raise ValueError("pilot did not produce finite B1 I3 result")
    selected={"router_lr":pilot_metrics["router"]["lr"],
              "lr_i3":pilot_metrics["I3_SoftIRM_EIIL3"]["lr"],
              "lambda_inv":pilot_metrics["I3_SoftIRM_EIIL3"]["lambda_inv"],
              "ridge_alpha":pilot_metrics["ridge_alpha_selected"],
              "pilot_case":pilot,"selection":"TRAIN B1 for head, TRAIN B2 for ridge; TEST never selects"}
    select_path=cli.output_root/"selected_configs"/f"{cli.backbone}_{cli.horizon_group}.json"
    select_path.parent.mkdir(parents=True,exist_ok=True)
    select_path.write_text(json.dumps(selected,indent=2)+"\n")
    for row in work:
        case=row["case_key"]
        if case==pilot:continue
        if row["status"]=="MISSING_A0":
            record(status,{"case":case,"phase":"full","status":"BLOCKED","reason":row["missing_reason"]})
            continue
        output=cli.output_root/case
        argv=[sys.executable,str(point),"--manifest",str(cli.manifest),"--case-key",case,
              "--output-root",str(cli.output_root),"--gpu",str(cli.gpu),"--seed",str(cli.seed),
              "--router-lr",str(selected["router_lr"]),"--lr-i3",str(selected["lr_i3"]),
              "--lambda-inv",str(selected["lambda_inv"]),"--ridge-alpha",str(selected["ridge_alpha"]),
              "--resume","--batch-size","32"]
        if cli.generate_missing_q:argv.append("--generate-missing-q")
        start=time.time();code=execute(argv,runlog)
        record(status,{"case":case,"phase":"full_point","exit_code":code,"seconds":time.time()-start})
        if code or not (output/"point_complete").exists():
            continue
        if cli.flow:
            fargs=[sys.executable,str(flow),"--case-dir",str(output),"--gpu",str(cli.gpu),
                   "--seed",str(cli.seed),"--lrs","0.0001","--resume"]
            start=time.time();code=execute(fargs,runlog)
            record(status,{"case":case,"phase":"full_flow","exit_code":code,"seconds":time.time()-start})
    record(status,{"phase":"queue_end","status":"complete_or_recorded_failures"})


if __name__=="__main__":main()
