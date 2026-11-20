"""Frozen-A0 G1+R3 full point-forecast matrix; no probability path."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.predictive_env.dual_pattern_memory.rare_learning import RarePatternLearner
from models.predictive_env.dual_pattern_memory.forecast_fusion import (
    prototype_segment_reliability, variation_forecast)
from models.predictive_env.dual_pattern_memory.prototype_bank import normalized_pairs
from tools.run_fpem_shared_factorization_g1r3 import load_experiment,write_json
from tools.run_predictive_env_iv_patchtst import IndexedDataset

def history_key(x: torch.Tensor, size: int) -> torch.Tensor:
    series=x.transpose(1,2).float()
    normal=(series-series.mean(-1,keepdim=True))/series.var(-1,keepdim=True,unbiased=False).add(1e-5).sqrt()
    return F.interpolate(normal.reshape(-1,1,series.shape[-1]),size=size,
                         mode="linear",align_corners=False).reshape(x.shape[0],x.shape[2],size)

@torch.no_grad()
def collect_split(model,loader,device):
    parts={name:[] for name in ("x","target","base","ids")}
    for x,y,ids,cycle in loader:
        x=x.float().to(device)
        cycle=cycle.to(device)
        hidden,mean,std=model.encode(x,cycle)
        base=model.frozen_h_head(hidden,cycle).permute(0,2,1)*std+mean
        for key,value in (("x",x),("target",y[:,-model.pred_len:]),
                          ("base",base),("ids",ids)):
            parts[key].append(value.detach().cpu())
    return {key:torch.cat(value) for key,value in parts.items()}

RIDGES=(1e-4,1e-3,1e-2,1e-1,1.,10.,100.,1000.,10000.)
SCALES=(0.,.05,.1,.2,.5,1.)
OFFSETS=(-3.,-1.,0.,1.,3.)
LR={"patchtst":(1e-4,.1,.1,.1),
    "itransformer":(3e-3,1.,.1,1.),
    "moderntcn":(3e-3,1.,.1,1.)}

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--base_run_config",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--gpu",type=int,default=0)
    p.add_argument("--seed",type=int,default=2021)
    p.add_argument("--train_stride",type=int,default=16)
    p.add_argument("--val_stride",type=int,default=4)
    p.add_argument("--test_stride",type=int,default=1)
    p.add_argument("--allow_legacy_reference_signature",action="store_true")
    return p.parse_args()

def subset(record,mask):
    return {key:value[mask] for key,value in record.items()}

def record(raw,floor):
    y,mean,scale=normalized_pairs(raw["x"],raw["target"],floor)
    return {"key":history_key(raw["x"],64).cpu().float(),
            "truth":y.cpu().float(),
            "base":((raw["base"].float()-mean)/scale).transpose(1,2).cpu(),
            "scale":scale.cpu()}

def metric(data,pred):
    err=(pred-data["truth"])*data["scale"].transpose(1,2)
    return {"mse":float(err.square().mean()),"mae":float(err.abs().mean())}

def residual_ridge(key,target,ridge):
    x=key.reshape(-1,key.shape[-1]).double()
    y=target.reshape(-1,target.shape[-1]).double()
    design=torch.cat((x,torch.ones(len(x),1,dtype=x.dtype)),1)
    gram=design.T@design/len(design)
    penalty=torch.eye(len(gram),dtype=x.dtype)*ridge
    penalty[-1,-1]=ridge*.01
    return torch.linalg.solve(gram+penalty,design.T@y/len(design)).float()

def ridge_query(key,weight):
    x=key.reshape(-1,key.shape[-1])
    return (torch.cat((x,torch.ones(len(x),1)),1)@weight).reshape(
        key.shape[0],key.shape[1],-1)

def collect(model,dataset,args,device,stride):
    wrapped=IndexedDataset(dataset,cycle_len=args.cyclenet_cycle_len)
    loader=DataLoader(Subset(wrapped,range(0,len(dataset),stride)),
                      batch_size=args.batch_size,shuffle=False,num_workers=0)
    with torch.no_grad():
        return collect_split(model,loader,device)

def g1_fit(early,selection):
    target=early["truth"]-early["base"]
    best=None
    for ridge in RIDGES:
        weight=residual_ridge(early["key"],target,ridge)
        delta=ridge_query(selection["key"],weight)
        for scale in SCALES:
            value=metric(selection,selection["base"]+scale*delta)
            if best is None or value["mse"]<best["val_mse"]:
                best={"ridge":ridge,"scale":scale,"val_mse":value["mse"],
                      "val_mae":value["mae"],"weight":weight}
    return best

def full_prediction(data,g1,learner,reliability,offset):
    anchor=data["base"]+g1["scale"]*ridge_query(data["key"],g1["weight"])
    query=learner.query(data["key"],offset)
    empty=anchor.new_zeros(anchor.shape[-1],0)
    full,beta=variation_forecast(anchor,query["candidate"],query["weight"],
                                  reliability,empty,True,8)
    return full,anchor,query,beta

def main():
    cli=parse_args()
    output=Path(cli.output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/"complete").exists():
        print("SKIP complete",output,flush=True)
        return
    torch.manual_seed(cli.seed)
    torch.set_num_threads(4)
    started=time.monotonic()
    args,model,loaders,device,sha=load_experiment(cli)
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise AssertionError("A0 Backbone must remain frozen")
    raw_train=collect(model,loaders["train"].dataset.dataset,args,device,cli.train_stride)
    starts=raw_train["ids"].long()
    first_val_history=len(loaders["train"].dataset.dataset.data_x)-args.seq_len
    valid=starts+args.seq_len+args.pred_len<=first_val_history
    raw_train=subset(raw_train,valid)
    starts=starts[valid]
    historical_std=raw_train["x"].float().var(1,unbiased=False).add(1e-5).sqrt()
    floor=(historical_std.median(0).values*.25).clamp_min(.01)
    train=record(raw_train,floor)
    del raw_train
    raw_val=collect(model,loaders["val"].dataset.dataset,args,device,cli.val_stride)
    val_ids=raw_val["ids"].long()
    val=record(raw_val,floor)
    del raw_val
    cut=len(loaders["val"].dataset.dataset.data_x)//2
    # VAL labels may overlap the later TEST *history* (which is observable),
    # but must never enter the TEST target period.
    first_test_target_start=len(loaders["val"].dataset.dataset.data_x)
    early_mask=val_ids+args.seq_len+args.pred_len<=cut
    select_mask=(val_ids>=cut)&(val_ids+args.seq_len+args.pred_len<=first_test_target_start)
    early,selection=subset(val,early_mask),subset(val,select_mask)
    fit_train=train
    protocol="G1 fit on early purged VAL; G1/r3 config and NULL selected on later purged VAL; TEST full stride1 report only"
    if min(len(early["key"]),len(selection["key"]))<8:
        # Some long-horizon custom VAL segments cannot be bisected with a
        # full-horizon embargo. Use strictly earlier TRAIN blocks instead.
        split=int(starts.max())//2
        fit_mask=starts+args.seq_len+args.pred_len<=split
        hold_mask=(starts>=split)&(starts+args.seq_len+args.pred_len<=first_val_history)
        select_mask=val_ids+args.seq_len+args.pred_len<=first_test_target_start
        if min(int(fit_mask.sum()),int(hold_mask.sum()),int(select_mask.sum()))<8:
            raise ValueError("insufficient purged TRAIN/VAL support")
        if int((starts[fit_mask]+args.seq_len+args.pred_len).max())>int(starts[hold_mask].min()):
            raise AssertionError("TRAIN fit future overlaps TRAIN holdout history")
        if int((starts[hold_mask]+args.seq_len+args.pred_len).max())>first_val_history:
            raise AssertionError("TRAIN holdout future overlaps VAL history")
        fit_train=subset(train,fit_mask)
        early=subset(train,hold_mask)
        selection=subset(val,select_mask)
        g1=g1_fit(fit_train,early)
        protocol="Long-horizon fallback: G1/R3 fit on early purged TRAIN, reliability on later purged TRAIN, NULL on full eligible VAL; TEST report only"
    else:
        if int((val_ids[early_mask]+args.seq_len+args.pred_len).max())>int(val_ids[select_mask].min()):
            raise AssertionError("early VAL future overlaps selection history")
        g1=g1_fit(early,selection)
    if int((val_ids[select_mask]+args.seq_len+args.pred_len).max())>first_test_target_start:
        raise AssertionError("selection VAL future overlaps TEST targets")
    head_lr,dict_ratio,basis_ratio,route_ratio=LR[args.backbone]
    opt={"head_lr":head_lr,"dictionary_ratio":dict_ratio,
         "basis_ratio":basis_ratio,"route_ratio":route_ratio,
         "optimizer":"adam","schedule":"constant"}
    reference=residual_ridge(fit_train["key"],fit_train["truth"],100.)
    learner=RarePatternLearner("R3",patterns=8,rank=8,seed=cli.seed,
                               epochs=10,optimizer_config=opt)
    learner.fit(fit_train["key"],fit_train["truth"],{
        "reference_map":reference,"starts":starts if fit_train is train else starts[fit_mask],
        "span":args.seq_len+args.pred_len,"device":"cpu",
        "validation":(early["key"],early["truth"])})
    early_anchor=early["base"]+g1["scale"]*ridge_query(early["key"],g1["weight"])
    early_query=learner.query(early["key"])
    reliability,evidence=prototype_segment_reliability(
        early_anchor,early_query["candidate"],early["truth"],early_query["weight"],8)
    choice=None
    for offset in OFFSETS:
        pred,anchor,query,beta=full_prediction(selection,g1,learner,reliability,offset)
        value=metric(selection,pred)
        if choice is None or value["mse"]<choice["val_mse"]:
            choice={"offset":offset,"val_mse":value["mse"],"val_mae":value["mae"]}
    raw_test=collect(model,loaders["test"].dataset.dataset,args,device,cli.test_stride)
    test=record(raw_test,floor)
    del raw_test
    pred,anchor,query,beta=full_prediction(test,g1,learner,reliability,choice["offset"])
    early_mix=(early_query["weight"][...,None]*early_query["candidate"]).sum(2)+\
              early_query["null_weight"][...,None]*early_anchor
    result={"backbone":args.backbone,"dataset":args.dataset_name,"pred_len":args.pred_len,
            "a0":metric(test,test["base"]),"g1":metric(test,anchor),
            "g1_r3":metric(test,pred),"val":choice,
            "g1_ridge":g1["ridge"],"g1_scale":g1["scale"],
            "r3_optimizer":opt,"r3_early_candidate_mse":metric(early,early_mix)["mse"],
            "g1_early_mse":metric(early,early_anchor)["mse"],
            "reliability_mean":float(reliability.mean()),
            "actual_effect_mean":float((pred-anchor).abs().mean()),
            "active_sample_ratio":float((beta.abs().sum((1,2))>0).float().mean()),
            "null_mean":float(query["null_weight"].mean()),
            "reference_sha256":sha,"train_samples":len(fit_train["key"]),
            "early_val_samples":len(early["key"]),"selection_val_samples":len(selection["key"]),
            "test_samples":len(test["key"]),"elapsed_seconds":time.monotonic()-started,
            "protocol":protocol,
            "caveat":"TEST-selected A0 catalogs are exploratory; G1 early-VAL fit makes reliability in-sample for G1"}
    torch.save({"rare":learner.export(),"g1_weight":g1["weight"],
                "g1_scale":g1["scale"],"reliability":reliability,
                "offset":choice["offset"],"reference_sha256":sha},output/"model.pt")
    write_json(output/"run_config.json",{"cli":vars(cli),"reference_sha256":sha,
               "dataset":args.dataset_name,"backbone":args.backbone,
               "pred_len":args.pred_len,"r3_optimizer":opt,
               "g1_ridge":g1["ridge"],"g1_scale":g1["scale"]})
    write_json(output/"metrics.json",result)
    write_json(output/"reliability_evidence.json",evidence)
    (output/"complete").write_text("complete\n",encoding="utf-8")
    print("COMPLETE",args.backbone,args.dataset_name,args.pred_len,
          result["g1_r3"],flush=True)

if __name__=="__main__":
    main()
