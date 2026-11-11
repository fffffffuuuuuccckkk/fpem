#!/usr/bin/env python3
"""U0/U1/U-shuffle on one frozen EIIL3 point prediction and four purged VAL blocks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.predictive_env.probability_heads import ProbabilityHeadBank
from models.predictive_env.residual_difference.condition import ResidualDifferenceCondition
from tools.run_fpem_mechanism_fusion_flow import (
    split_four_purged,event_count,forward,iterate,evaluate,
)
from tools.run_fpem_eiil3 import file_hash,dump

MODES=("Flow_U0","Flow_U1","Flow_Ushuffle")


def shuffled_by_event(signal,start,seed):
    unique,inverse=np.unique(start,return_inverse=True)
    rng=np.random.default_rng(seed)
    groups=[np.flatnonzero(inverse==k) for k in range(len(unique))]
    if len({len(group) for group in groups})!=1:
        raise ValueError("cannot shuffle U by complete event: unequal channel counts")
    src=rng.permutation(len(unique))
    result=np.empty_like(signal)
    for target,origin in enumerate(src):
        result[groups[target]]=signal[groups[origin]]
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--case-dir",type=Path,required=True)
    p.add_argument("--gpu",type=int,default=0)
    p.add_argument("--seed",type=int,default=2021)
    p.add_argument("--epochs",type=int,default=5)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--samples",type=int,default=100)
    p.add_argument("--lrs",default="0.0001")
    p.add_argument("--resume",action="store_true")
    cli=p.parse_args();torch.set_num_threads(4)
    case=cli.case_dir
    if not (case/"point_complete").exists():
        raise FileNotFoundError("point stage incomplete")
    meta=json.loads((case/"features.json").read_text())
    metrics=json.loads((case/"point_metrics.json").read_text())
    with np.load(case/"flow_input.npz") as source:
        data={split:{field:source[f"{split}_{field}"] for field in ("key","y","start","point","u")}
              for split in ("val","test")}
    point_hash=file_hash(case/"flow_input.npz")
    try:
        parts=split_four_purged(data["val"]["start"],meta["span"])
    except ValueError as exc:
        dump(case/"flow_status.json",{"status":"FLOW_SKIPPED_INSUFFICIENT_EVENTS","reason":str(exc)})
        print("FLOW_SKIPPED",exc,flush=True);return
    counts={f"V{i}":event_count(data["val"]["start"][part],meta["span"])
            for i,part in enumerate(parts)}
    if min(counts.values())<2:
        dump(case/"flow_status.json",{"status":"FLOW_SKIPPED_INSUFFICIENT_EVENTS",
                                      "independent_events":counts,"reason":"each purged VAL block needs >=2 events"})
        print("FLOW_SKIPPED",counts,flush=True);return
    blocks=[{key:value[part] for key,value in data["val"].items()} for part in parts]
    _,train,validation,calibration=blocks
    device=torch.device(f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu")
    lrs=[float(value) for value in cli.lrs.split(",")]
    if not lrs or any(lr<=0 for lr in lrs):raise ValueError("positive Flow LRs required")
    results={}
    for mode in MODES:
        dest=case/(mode+".json")
        if cli.resume and dest.exists():
            old=json.loads(dest.read_text())
            if old.get("point_hash")==point_hash:
                results[mode]=old;continue
        features={}
        for label,block in (("train",train),("select",validation),
                            ("cal",calibration),("test",data["test"])):
            u=block["u"]
            if mode=="Flow_U0":u=np.zeros_like(u)
            elif mode=="Flow_Ushuffle":u=shuffled_by_event(u,block["start"],cli.seed+311)
            features[label]=u[:,:,None].astype(np.float32)
        best=None;trace=[]
        for lr in lrs:
            torch.manual_seed(cli.seed)
            bank=ProbabilityHeadBank(64,meta["horizon"],patch_len=16,heads=("residual_flow",)).to(device)
            adapter=ResidualDifferenceCondition(1,64,16).to(device)
            optim=torch.optim.Adam(list(bank.parameters())+list(adapter.parameters()),lr=lr)
            for epoch in range(cli.epochs):
                bank.train();adapter.train();total=0.;seen=0
                for record,signal in iterate(train,features["train"],cli.batch_size):
                    cond,future,point,scale=forward(bank,adapter,record,signal,device)
                    truth=torch.from_numpy(record["y"][:,:,None]).to(device)
                    loss,_=bank.training_losses(cond,future,point,truth,scale)["residual_flow"]
                    if not torch.isfinite(loss):raise FloatingPointError("nonfinite Flow loss")
                    optim.zero_grad(set_to_none=True);loss.backward();optim.step()
                    total+=float(loss.detach())*len(truth);seen+=len(truth)
                val,_=evaluate(bank,adapter,validation,features["select"],device,
                               min(32,cli.samples),cli.seed+601,cli.batch_size,max_rows=256)
                record={"lr":lr,"epoch":epoch+1,"train_loss":total/seen,"V2_CRPS":val["CRPS"]}
                trace.append(record)
                if best is None or val["CRPS"]<best[0]:
                    best=(val["CRPS"],lr,epoch+1,
                          {key:value.detach().cpu().clone() for key,value in bank.state_dict().items()},
                          {key:value.detach().cpu().clone() for key,value in adapter.state_dict().items()})
                print(meta["dataset"],meta["horizon"],mode,record,flush=True)
        _,lr,epoch,bank_state,adapter_state=best
        bank.load_state_dict(bank_state);adapter.load_state_dict(adapter_state)
        cal,ratios=evaluate(bank,adapter,calibration,features["cal"],device,
                            min(32,cli.samples),cli.seed+602,cli.batch_size)
        factor=float(torch.quantile(ratios,.95).clamp(.1,10))
        test,_=evaluate(bank,adapter,data["test"],features["test"],device,
                        cli.samples,cli.seed+603,cli.batch_size,calibration=factor)
        result={"case":metrics["case"],"mode":mode,"point_hash":point_hash,
                "A0_sha256":metrics["A0_sha256"],"router_sha256":metrics["router_sha256"],
                "I3_sha256":metrics["I3_sha256"],"R3_sha256":metrics["R3_sha256"],
                "selected_lr":lr,"best_epoch":epoch,"V2_CRPS":best[0],
                "V3_calibration_factor":factor,"V3_metrics":cal,"TEST":test,
                "independent_events":counts,"trials":trace,
                "protocol":"one frozen I3+R3 point; V0/V1/V2/V3 purged; TEST report-only; U is dynamic [N,T]",
                "caveat":"A0 may have used VAL; Flow residuals are out-of-added-module training, not full A0 cross-fit"}
        torch.save({"bank":bank_state,"adapter":adapter_state},case/(mode+".pt"))
        dump(dest,result);results[mode]=result
    dump(case/"flow_summary.json",results)
    if len(results)==len(MODES):(case/"flow_complete").write_text("complete\n")


if __name__=="__main__":main()
