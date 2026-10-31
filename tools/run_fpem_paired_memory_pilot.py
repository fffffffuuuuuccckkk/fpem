"""TRAIN-causal real-pair memory and direct-concatenation point Pilot.

The probability/risk stage has a separate entry point so it cannot silently
consume validation labels while fitting point predictors. A0 is never retrained.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.predictive_env.paired_memory import PairedMemoryBank
from models.predictive_env.direct_concat import DirectConcatPredictor
from tools.run_fpem_predictive_memory import anchor_records
from tools.run_fpem_shared_factorization import load_experiment, train_loader, write_json


def arguments():
    p=argparse.ArgumentParser()
    p.add_argument("--base_run_config",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--mode",choices=("A0","A1","A2","A3","A4","A5",
                                     "B0","B1","B2","B3","B4","B5","B6","B7"),required=True)
    p.add_argument("--hidden_view",choices=("all","common","rare","mixed","random"),default="mixed")
    p.add_argument("--memory_cache",default="")
    p.add_argument("--anchor_cache",default="")
    p.add_argument("--gpu",type=int,default=0)
    p.add_argument("--seed",type=int,default=2021)
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--common_k",type=int,default=3)
    p.add_argument("--rare_k",type=int,default=2)
    p.add_argument("--history_size",type=int,default=64)
    p.add_argument("--lr_retrieval",type=float,default=1e-4)
    p.add_argument("--lr_residual",type=float,default=1e-4)
    p.add_argument("--allow_legacy_reference_signature",action="store_true")
    p.add_argument("--max_batches",type=int,default=0)
    return p.parse_args()


def memory_from_cache(cli,args,model,loaders,device,ref_hash):
    cache=Path(cli.memory_cache) if cli.memory_cache else None
    if cache is not None and cache.is_file():
        payload=torch.load(cache,map_location="cpu",weights_only=False)
        if (payload["reference_sha256"] != ref_hash or
                payload["seq_len"] != args.seq_len or payload["pred_len"] != args.pred_len):
            raise ValueError("paired memory cache does not match A0/data")
        memory=PairedMemoryBank(args.seq_len,args.pred_len,cli.history_size)
        # The bank starts with zero-sized buffers; install their checkpoint
        # shapes before loading, rather than relying on strict=False (which
        # does not suppress tensor shape mismatches).
        for name,value in payload["memory"].items():
            setattr(memory,name,value.clone())
        return memory.to(device),payload["audit"]
    anchor=Path(cli.anchor_cache) if cli.anchor_cache else None
    if anchor is not None and anchor.is_file():
        existing=torch.load(anchor,map_location="cpu",weights_only=False)
        if existing.get("reference_sha256") != ref_hash:
            raise ValueError("TRAIN anchor cache does not match A0 checkpoint")
        records=existing["records"]
    else:
        records=anchor_records(model,loaders["train"].dataset.dataset,args,device)
    memory=PairedMemoryBank(args.seq_len,args.pred_len,cli.history_size)
    audit=memory.fit(records)
    if cache is not None:
        cache.parent.mkdir(parents=True,exist_ok=True)
        torch.save({"memory":memory.state_dict(),"reference_sha256":ref_hash,
                    "seq_len":args.seq_len,"pred_len":args.pred_len,"audit":audit},cache)
    return memory.to(device),audit


def view_for_mode(cli):
    return {"A1":"random","A2":"all","A3":"common","A4":"rare",
            "A5":"mixed"}.get(cli.mode,cli.hidden_view)


def future_mode(cli):
    return {"B1":"future","B2":"residual","B3":"shuffled",
            "B4":"future","B5":"future","B6":"future",
            "B7":"future"}.get(cli.mode,"none")


def forward(model,predictor,memory,x,cycle,ids,args,cli,split,val_start):
    with torch.no_grad():
        hidden,mean,std=model.encode(x,cycle)
        a0_norm=model.frozen_h_head(hidden,cycle)
        a0=a0_norm.permute(0,2,1)*std+mean
    if cli.mode=="A0":
        return a0,a0,a0.new_zeros(a0.shape),None
    view=view_for_mode(cli)
    retrieved=memory.retrieve(x,view,cli.common_k,cli.rare_k,ids,split,val_start)
    alternate=None
    if cli.mode in {"B4","B5","B6","B7"}:
        other={"B4":"common","B5":"rare","B6":"mixed","B7":"random"}[cli.mode]
        alternate=memory.retrieve(x,other,cli.common_k,cli.rare_k,ids,split,val_start)
    full,point,delta=predictor(model,hidden,retrieved,mean,std,cycle,
                               future_mode(cli),alternate)
    return full,point,delta,retrieved


@torch.no_grad()
def evaluate(model,predictor,memory,loader,args,cli,split,val_start,
             selection_only=False):
    started=time.monotonic()
    model.eval()
    if predictor is not None:
        predictor.eval()
    total=mae=a0_mse=a0_mae=point_mse=delta_abs=0.
    count=nsamples=0
    retrieved_valid=retrieved_distance=future_disagreement=0.
    selection_limit=max(1,len(loader.dataset)//2) if selection_only else None
    for batch_index,(x,y,ids,cycle) in enumerate(loader):
        if cli.max_batches and batch_index>=cli.max_batches:
            break
        if selection_limit is not None:
            valid=ids<selection_limit
            if not bool(valid.any()):
                break
            x,y,ids,cycle=x[valid],y[valid],ids[valid],cycle[valid]
        x=x.float().to(next(model.parameters()).device)
        y=y[:,-args.pred_len:].float().to(x.device)
        cycle=cycle.to(x.device)
        output,point,delta,retrieved=forward(model,predictor,memory,x,cycle,ids,args,cli,split,val_start)
        if retrieved is not None:
            valid=retrieved["valid"]
            retrieved_valid+=(float(valid.float().mean()) if valid.numel() else 0.)*len(x)
            retrieved_distance+=float((retrieved["distance"]*valid).sum()/valid.sum().clamp_min(1))*len(x)
            nvalid=valid.sum(-1).clamp_min(1)
            mean_future=retrieved["future"].sum(2)/nvalid[...,None]
            disagreement=((retrieved["future"]-mean_future[:,:,None]).square()*
                          valid[...,None]).sum((2,3))/(nvalid*args.pred_len)
            future_disagreement+=float(disagreement.mean())*len(x)
        hidden,mean,std=model.encode(x,cycle)
        a0=model.frozen_h_head(hidden,cycle).permute(0,2,1)*std+mean
        total+=float((output-y).square().sum());mae+=float((output-y).abs().sum())
        a0_mse+=float((a0-y).square().sum());a0_mae+=float((a0-y).abs().sum())
        point_mse+=float((point-y).square().sum());delta_abs+=float(delta.abs().sum())
        count+=y.numel();nsamples+=len(x)
    if count==0:
        raise RuntimeError("evaluation split has no observations")
    return {"mse":total/count,"mae":mae/count,"point_mse":point_mse/count,
            "a0_mse":a0_mse/count,"a0_mae":a0_mae/count,
            "correction_abs_mean":delta_abs/count,"evaluated_samples":nsamples,
            "retrieved_valid_fraction":retrieved_valid/nsamples,
            "retrieval_distance_mean":retrieved_distance/nsamples,
            "retrieved_future_disagreement":future_disagreement/nsamples,
            "evaluation_seconds":time.monotonic()-started}


def main():
    cli=arguments()
    torch.set_num_threads(min(4,torch.get_num_threads()))
    out=Path(cli.output);out.mkdir(parents=True,exist_ok=True)
    if (out/"run_complete").is_file():
        print("already complete",flush=True);return
    torch.manual_seed(cli.seed);np.random.seed(cli.seed)
    args,model,loaders,device,ref_hash=load_experiment(cli)
    val_start=len(loaders["train"].dataset.dataset.data_x)-args.seq_len
    memory,audit=memory_from_cache(cli,args,model,loaders,device,ref_hash)
    write_json(out/"memory_audit.json",audit)
    write_json(out/"run_config.json",{"cli":vars(cli),"reference_sha256":ref_hash,
               "val_first_historical_start":val_start,
               "time_policy":"TRAIN source future complete before query history starts; VAL source complete before first VAL history; TEST full TRAIN",
               "checkpoint_selection":"first half of VAL; second half reserved for interval calibration"})
    if cli.mode=="A0":
        result={"val_selection":evaluate(model,None,memory,loaders["val"],args,cli,"val",val_start,True),
                "val_full":evaluate(model,None,memory,loaders["val"],args,cli,"val",val_start),
                "test":evaluate(model,None,memory,loaders["test"],args,cli,"test",val_start),
                "best_epoch":0,"reference_sha256":ref_hash}
        write_json(out/"result.json",result);(out/"run_complete").touch();return
    token_num=model.head_linear.in_features//model.d_model
    predictor=DirectConcatPredictor(model.h_forecast_head,token_num,model.d_model,
                                    args.pred_len,cli.common_k+cli.rare_k).to(device)
    optimizer=torch.optim.Adam([
        {"params":predictor.hidden_head.head.weight,"lr":cli.lr_retrieval},
        {"params":predictor.residual_head.parameters(),"lr":cli.lr_residual}],
        weight_decay=0.)
    train=train_loader(loaders["train"].dataset.dataset,args,cli.seed)
    history=[];best=math.inf;started=time.monotonic()
    for epoch in range(cli.epochs):
        model.eval();predictor.train();train_total=0.;batches=0
        for batch_index,(x,y,ids,cycle) in enumerate(train):
            if cli.max_batches and batch_index>=cli.max_batches:break
            x=x.float().to(device);cycle=cycle.to(device)
            y=y[:,-args.pred_len:].float().to(device)
            output,_,_,_=forward(model,predictor,memory,x,cycle,ids,args,cli,"train",val_start)
            loss=F.mse_loss(output,y)
            if not torch.isfinite(loss):raise FloatingPointError("nonfinite loss")
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(),5.)
            optimizer.step();train_total+=float(loss.detach());batches+=1
        selection=evaluate(model,predictor,memory,loaders["val"],args,cli,"val",val_start,True)
        row={"epoch":epoch+1,"train_mse":train_total/max(1,batches),
             "val_selection_mse":selection["mse"],"val_selection_mae":selection["mae"]}
        history.append(row);print(json.dumps(row),flush=True)
        write_json(out/"train_history.json",history)
        if selection["mse"]<best:
            best=selection["mse"]
            torch.save({"predictor":predictor.state_dict(),"best_epoch":epoch+1,
                        "val_selection":selection,"reference_sha256":ref_hash,
                        "memory_reference":str(cli.memory_cache)},out/"best.pt")
    checkpoint=torch.load(out/"best.pt",map_location=device,weights_only=False)
    predictor.load_state_dict(checkpoint["predictor"])
    val_full=evaluate(model,predictor,memory,loaders["val"],args,cli,"val",val_start)
    test=evaluate(model,predictor,memory,loaders["test"],args,cli,"test",val_start)
    result={"mode":cli.mode,"hidden_view":view_for_mode(cli),
            "future_mode":future_mode(cli),"best_epoch":checkpoint["best_epoch"],
            "val_selection":checkpoint["val_selection"],"val_full":val_full,
            "test":test,"train_history":history,"training_seconds":time.monotonic()-started,
            "reference_sha256":ref_hash,
            "new_trainable_parameters":sum(p.numel() for p in predictor.parameters() if p.requires_grad),
            "active_trainable_parameters":predictor.hidden_head.head.weight.numel()-
                args.pred_len*model.h_forecast_head.in_features+
                (sum(p.numel() for p in predictor.residual_head.parameters())
                 if future_mode(cli)!="none" else 0),
            "gpu_peak_memory_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else 0,
            "selection_note":"VAL first half only; TEST exploratory, not used for checkpoint"}
    write_json(out/"result.json",result);(out/"run_complete").touch()


if __name__=="__main__":main()
