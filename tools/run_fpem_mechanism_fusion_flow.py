#!/usr/bin/env python3
"""Residual Flow ablation on a frozen point system, using purged VAL blocks.

The frozen A0 was trained on TRAIN, including C. Therefore TRAIN-C residuals
are not full point-model OOF and must never train the Flow or its summaries.
Four nonoverlapping VAL blocks supply summaries, Flow targets, epoch selection,
and interval calibration respectively. TEST is report-only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.predictive_env.mechanism_fusion import K, independent_support
from models.predictive_env.residual_difference.condition import ResidualDifferenceCondition
from models.predictive_env.probability_heads import ProbabilityHeadBank
from tools.evaluate_fpem_probabilistic import probabilistic_metrics
from tools.run_fpem_mechanism_fusion import METHODS, score, write_json

MODES=("U0_original_condition","U1_mapping_disagreement","U2_oof_residual_summary","U3_both")
NAMES=("CRPS","WIS","PICP_95","MPIW_95","inv_MSE","inv_MAE","mean_MSE","mean_MAE")


def split_four_purged(start,span):
    """Quarter VAL by time; purge windows crossing each boundary."""
    unique=np.unique(start)
    if len(unique)<4:raise ValueError("fewer than four distinct VAL origins")
    cuts=np.quantile(unique,[.25,.5,.75]).astype(int)
    edges=[None,*cuts,None]
    blocks=[]
    for left,right in zip(edges[:-1],edges[1:]):
        mask=np.ones(len(start),dtype=bool)
        if left is not None:mask &= start>=left
        if right is not None:mask &= start+span<=right
        block=np.flatnonzero(mask)
        if not len(block):raise ValueError("empty purged VAL quarter")
        blocks.append(block)
    for earlier,later in zip(blocks[:-1],blocks[1:]):
        if int((start[earlier]+span).max())>int(start[later].min()):
            raise AssertionError("VAL block futures overlap")
    return blocks


def event_count(starts,span):
    count,last=0,-1
    for value in np.sort(np.unique(starts)):
        if value>=last:
            count+=1;last=int(value)+span
    return count


def build_summary(residual,q,start,span,patch_len=16):
    """Store only K×future-patch moments/quantiles, never query residuals."""
    n,horizon=residual.shape
    patches=math.ceil(horizon/patch_len)
    global_stats=[];local=[]
    support=independent_support(start,q,span)
    for patch in range(patches):
        values=residual[:,patch*patch_len:min((patch+1)*patch_len,horizon)].mean(1)
        center=float(values.mean());spread=float(values.std()+1e-4)
        global_stats.append([center,spread,float(np.quantile(values,.05)),
                             float(np.quantile(values,.95)),float(np.median(abs(values-np.median(values))))])
    global_stats=np.asarray(global_stats,np.float32)
    for j in range(K):
        centers=[]
        hard=q.argmax(1)==j
        for patch in range(patches):
            values=residual[:,patch*patch_len:min((patch+1)*patch_len,horizon)].mean(1)
            weight=q[:,j]
            mean=float(np.average(values,weights=weight)) if weight.sum()>1e-5 else 0.
            var=float(np.average((values-mean)**2,weights=weight)) if weight.sum()>1e-5 else 0.
            selected=values[hard]
            if len(selected):
                raw=[mean,math.sqrt(max(var,1e-8)),float(np.quantile(selected,.05)),
                     float(np.quantile(selected,.95)),float(np.median(abs(selected-np.median(selected))))]
            else:raw=global_stats[patch].tolist()
            shrink=float(support[j]/(support[j]+5.))
            centers.append(shrink*np.asarray(raw)+ (1-shrink)*global_stats[patch])
        local.append(centers)
    return np.asarray(local,np.float32),global_stats,support


def mode_features(q,candidate,point,reliability,local,support,mode,patch_len=16):
    n,horizon=point.shape
    result=np.zeros((n,horizon,14),np.float32)
    if mode==MODES[0]:return result
    ent=-(q*np.log(q+1e-8)).sum(1)/math.log(K)
    effective=q@support.astype(np.float32)
    rho=np.asarray(reliability,np.float32)
    for t in range(horizon):
        patch=min(t//patch_len,local.shape[1]-1)
        segment=min(t*8//horizon,7)
        mass=q*rho[:,segment][None]
        total=mass.sum(1)
        disagreement=np.sqrt(np.sum(mass*(candidate[:,:,t]-point[:,None,t])**2,axis=1))
        if mode in (MODES[1],MODES[3]):
            result[:,t,0]=disagreement
            result[:,t,1]=total
            result[:,t,2]=ent
            result[:,t,3]=effective/10.
            result[:,t,4]=1-total
        if mode in (MODES[2],MODES[3]):
            mixture=q@local[:,patch,:]
            result[:,t,5:10]=mixture
            result[:,t,10]=ent
            result[:,t,11]=effective/10.
            result[:,t,12]=1-total
            result[:,t,13]=float(np.max(support))/10.
    return result


def segments(data,pred,q,candidate,span,local,support,rho,mode):
    return mode_features(q,candidate,pred,rho,local,support,mode)


def forward(bank,adapter,batch,feature,device):
    point=torch.from_numpy(batch["point"][:, :,None]).to(device)
    length=point.shape[1]
    patch_count=math.ceil(length/16)
    future=torch.from_numpy(batch["key"][:,None,None,:]).to(device).expand(-1,1,patch_count,-1)
    scale=point.new_ones(len(point),1,1)
    condition=bank.condition(future,point,scale)
    condition=adapter(condition,torch.from_numpy(feature[:,None]).to(device))
    return condition,future,point,scale


def iterate(data,feature,batch_size):
    for a in range(0,len(data["key"]),batch_size):
        b=min(a+batch_size,len(data["key"]))
        yield {key:value[a:b] for key,value in data.items()},feature[a:b]


@torch.no_grad()
def evaluate(bank,adapter,data,feature,device,samples,seed,batch_size,max_rows=0,calibration=1.):
    bank.eval();adapter.eval()
    sums={name:0. for name in NAMES};count=0;ratios=[]
    generator=torch.Generator(device=device).manual_seed(seed)
    for record,signal in iterate(data,feature,batch_size):
        if max_rows and count>=max_rows:break
        cond,future,point,scale=forward(bank,adapter,record,signal,device)
        sampled=bank.sample_head("residual_flow",cond,future,point,scale,samples,
                                 generator,chunk_size=8,steps=12)["y_samples"]
        sampled=point[:,None]+calibration*(sampled-point[:,None])
        truth=torch.from_numpy(record["y"][:,:,None]).to(device)
        metrics=probabilistic_metrics(sampled,truth,point)
        for name in NAMES:sums[name]+=float(metrics[name])*len(truth)
        lo=torch.quantile(sampled,.025,1);hi=torch.quantile(sampled,.975,1)
        ratios.append(((truth-(lo+hi)/2).abs()/((hi-lo)/2).clamp_min(1e-5)).flatten().cpu())
        count+=len(truth)
    return {name:value/count for name,value in sums.items()},torch.cat(ratios)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--case_dir",required=True);parser.add_argument("--method",choices=("G3","G4","G7"),required=True)
    parser.add_argument("--gpu",type=int,default=0);parser.add_argument("--seed",type=int,default=2022)
    parser.add_argument("--epochs",type=int,default=5);parser.add_argument("--batch_size",type=int,default=64)
    parser.add_argument("--samples",type=int,default=100);parser.add_argument("--lr",type=float,default=1e-4)
    parser.add_argument("--modes",nargs="+",choices=MODES,default=list(MODES))
    cli=parser.parse_args();torch.manual_seed(cli.seed);torch.set_num_threads(4)
    case=Path(cli.case_dir)
    meta=json.loads((case/"features.json").read_text())
    method=METHODS[cli.method];out=case/"methods"/method
    if not (out/"point_complete").exists():raise FileNotFoundError("point stage incomplete")
    with np.load(case/"features.npz") as source:
        data={split:{key:source[f"{split}_{key}"] for key in ("key","y","start")}
              for split in ("val","test")}
    with np.load(out/"point_cache.npz") as source:
        for split in data:
            data[split].update(point=source[f"{split}_pred"],q=source[f"{split}_q"],
                               candidate=source[f"{split}_candidate"])
    metrics=json.loads((out/"point_metrics.json").read_text())
    rho=np.asarray(metrics["reliability"],np.float32)
    span=meta["span"]
    try:
        quarters=split_four_purged(data["val"]["start"],span)
    except ValueError as exc:
        write_json(out/"flow_status.json",{"status":"skipped_insufficient_purged_VAL",
                                          "reason":str(exc)})
        print("FLOW_SKIPPED",method,str(exc),flush=True);return
    independent={f"V{i}":event_count(data["val"]["start"][block],span)
                 for i,block in enumerate(quarters)}
    if min(independent.values())<2:
        write_json(out/"flow_status.json",{"status":"skipped_insufficient_independent_VAL_events",
                                          "independent_events":independent,
                                          "reason":"Each of four purged VAL blocks requires two independent events"})
        print("FLOW_SKIPPED",method,independent,flush=True);return
    source,train,validation,calibration=(
        {key:value[block] for key,value in data["val"].items()} for block in quarters)
    local,global_stats,support=build_summary(source["y"]-source["point"],source["q"],source["start"],span)
    np.savez_compressed(out/"oof_residual_summary.npz",local=local,global_stats=global_stats,support=support)
    device=torch.device(f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu")
    results={}
    for mode in cli.modes:
        path=out/(mode+"_flow.json")
        if path.exists():results[mode]=json.loads(path.read_text());continue
        signals={label:mode_features(block["q"],block["candidate"],block["point"],rho,
                                     local,support,mode)
                 for label,block in (("train",train),("val",validation),("cal",calibration),("test",data["test"]))}
        torch.manual_seed(cli.seed)
        bank=ProbabilityHeadBank(64,meta["horizon"],patch_len=16,heads=("residual_flow",)).to(device)
        adapter=ResidualDifferenceCondition(14,64,16).to(device)
        optim=torch.optim.Adam(list(bank.parameters())+list(adapter.parameters()),lr=cli.lr)
        best=float("inf");trace=[];best_state=None
        for epoch in range(cli.epochs):
            bank.train();adapter.train();total=0.;count=0
            for record,signal in iterate(train,signals["train"],cli.batch_size):
                cond,future,point,scale=forward(bank,adapter,record,signal,device)
                truth=torch.from_numpy(record["y"][:,:,None]).to(device)
                loss,_=bank.training_losses(cond,future,point,truth,scale)["residual_flow"]
                if not torch.isfinite(loss):raise FloatingPointError("nonfinite Flow loss")
                optim.zero_grad(set_to_none=True);loss.backward();optim.step()
                total+=float(loss.detach())*len(truth);count+=len(truth)
            val,_=evaluate(bank,adapter,validation,signals["val"],device,min(32,cli.samples),
                           cli.seed+601,cli.batch_size,max_rows=256)
            trace.append({"epoch":epoch+1,"train_loss":total/count,"val_crps":val["CRPS"]})
            if val["CRPS"]<best:
                best=val["CRPS"]
                best_state={"bank":{k:v.detach().cpu().clone() for k,v in bank.state_dict().items()},
                            "adapter":{k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()},
                            "epoch":epoch+1}
            print(mode,"epoch",epoch+1,"val_crps",val["CRPS"],flush=True)
        bank.load_state_dict(best_state["bank"]);adapter.load_state_dict(best_state["adapter"])
        cal,ratios=evaluate(bank,adapter,calibration,signals["cal"],device,min(32,cli.samples),
                            cli.seed+602,cli.batch_size)
        factor=float(torch.quantile(ratios,.95).clamp(.1,10))
        test,_=evaluate(bank,adapter,data["test"],signals["test"],device,cli.samples,
                        cli.seed+603,cli.batch_size,calibration=factor)
        result={"mode":mode,"best_epoch":best_state["epoch"],"val_crps":best,
                "calibration_factor":factor,"calibration":cal,"test":test,"trace":trace,
                "independent_events":independent,"point_source_hash":metrics["source_group_sha256"],
                "protocol":"Four purged VAL blocks: V0 summary, V1 Flow targets, V2 epoch selection, V3 interval calibration; frozen TRAIN-fitted point; TEST report-only",
                "provenance_caveat":"A0 checkpoint may have used VAL for model selection; these are out-of-training, not fully cross-fitted A0 OOF residuals"}
        torch.save(best_state,out/(mode+"_flow.pt"));write_json(path,result);results[mode]=result
    write_json(out/"flow_summary.json",results)
    if all((out/(mode+"_flow.json")).exists() for mode in MODES):
        (out/"flow_complete").write_text("complete\n")


if __name__=="__main__":main()
