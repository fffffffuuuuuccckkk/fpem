"""Time-forward OOF mapping-transfer risk and split-calibrated intervals.

This stage must start only after the paired-memory point Pilot finishes.
Pseudo-unseen blocks are TRAIN chronological holdouts. No TEST target enters
retrieval, risk fitting, checkpoint selection or interval calibration.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.predictive_env.paired_memory import PairedMemoryBank
from models.predictive_env.direct_concat import DirectConcatPredictor
from models.predictive_env.unseen_risk import (
    ConformalCalibrator,RiskScaleHead,fit_transfer_regression,
    interval_metrics,risk_features)
from tools.run_predictive_env_iv_patchtst import IndexedDataset
from tools.run_fpem_paired_memory_pilot import forward,memory_from_cache
from tools.run_fpem_shared_factorization import load_experiment,write_json


def arguments():
    p=argparse.ArgumentParser()
    p.add_argument("--base_run_config",required=True)
    p.add_argument("--point_run",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--anchor_cache",required=True)
    p.add_argument("--memory_cache",required=True)
    p.add_argument("--gpu",type=int,default=0)
    p.add_argument("--seed",type=int,default=2021)
    p.add_argument("--history_size",type=int,default=64)
    p.add_argument("--allow_legacy_reference_signature",action="store_true")
    p.add_argument("--oof_epochs",type=int,default=10)
    p.add_argument("--max_batches",type=int,default=0)
    return p.parse_args()


def predictor_for(model,args,runcli,device):
    tokens=model.head_linear.in_features//model.d_model
    return DirectConcatPredictor(model.h_forecast_head,tokens,model.d_model,
                                 args.pred_len,runcli.common_k+runcli.rare_k).to(device)


def _loader(dataset,args,indices,shuffle=False,seed=2021):
    wrapped=IndexedDataset(dataset,cycle_len=args.cyclenet_cycle_len)
    gen=torch.Generator().manual_seed(seed)
    return DataLoader(Subset(wrapped,list(indices)),batch_size=args.batch_size,
                      shuffle=shuffle,generator=gen if shuffle else None,num_workers=0)


def _paired_source(records,mask):
    return {key:value[mask] for key,value in records.items()
            if key in ("x","target","a0_pred","hidden","starts")}


@torch.no_grad()
def collect(model,predictor,memory,loader,args,runcli,device,split,val_start,
            keep_prediction=False):
    model.eval();predictor.eval()
    feats=[];gain=[];errors=[];pred=[];truth=[];residual_scales=[]
    for batch_index,(x,y,ids,cycle) in enumerate(loader):
        if runcli.max_batches and batch_index>=runcli.max_batches:break
        x=x.float().to(device);cycle=cycle.to(device)
        target=y[:,-args.pred_len:].float().to(device)
        output,_,_,retrieved=forward(model,predictor,memory,x,cycle,ids,args,
                                    runcli,split,val_start)
        if retrieved is None:
            raise RuntimeError("risk stage requires a retrieved point model")
        features=risk_features(retrieved)
        hidden,mean,std=model.encode(x,cycle)
        baseline=model.frozen_h_head(hidden,cycle)
        count=retrieved["valid"].sum(-1).clamp_min(1)
        retrieved_residual=(retrieved["residual"]*retrieved["valid"][...,None]).sum(2)/count[...,None]
        naive=((baseline+retrieved_residual).permute(0,2,1)*std+mean)
        a0=baseline.permute(0,2,1)*std+mean
        transfer=((a0-target).square().mean(1)-
                  (naive-target).square().mean(1))
        feats.append(features.cpu());gain.append(transfer.cpu())
        errors.append((output-target).square().mean(1).cpu())
        # A fixed historical-residual interval baseline, not target-driven.
        residual_scales.append(retrieved["residual"].std(2,unbiased=False).mean(-1).cpu())
        if keep_prediction:
            pred.append(output.cpu());truth.append(target.cpu())
    if not feats:
        raise RuntimeError("OOF/VAL/TEST collector received no samples")
    result={"features":torch.cat(feats),"transfer_gain":torch.cat(gain),
            "error_mse":torch.cat(errors),
            "retrieved_residual_scale":torch.cat(residual_scales)}
    if keep_prediction:
        result["prediction"]=torch.cat(pred);result["truth"]=torch.cat(truth)
    return result


def fit_risk(features,error_mse,seed,epochs=120,lr=1e-3):
    torch.manual_seed(seed)
    model=RiskScaleHead(features.shape[-1])
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    x=features.float().reshape(-1,features.shape[-1]).cpu()
    y=error_mse.float().reshape(-1).cpu()
    for _ in range(epochs):
        model.train();optimizer.zero_grad(set_to_none=True)
        scale=model(x)
        loss=(torch.log(scale)+.5*y/scale.square()).mean()
        if not torch.isfinite(loss):raise FloatingPointError("risk scale NLL not finite")
        loss.backward();optimizer.step()
    model.eval()
    return model


def add_transfer(features,coefficient):
    pred=features@coefficient[:-1]+coefficient[-1]
    return torch.cat((features,pred[...,None]),-1)


def oof_training(cli,model,datasets,args,device,ref_hash,runcli,records,out):
    n=len(datasets["train"])
    segments=((int(.4*n),int(.6*n)),(int(.6*n),int(.8*n)),
              (int(.8*n),n))
    collected=[];records_out=[]
    for fold,(lo,hi) in enumerate(segments):
        fold_cache=out/f"oof_fold_{fold}.pt"
        if fold_cache.is_file():
            saved=torch.load(fold_cache,map_location="cpu",weights_only=False)
            if (saved.get("reference_sha256")==ref_hash and
                    saved.get("point_run")==str(cli.point_run) and
                    saved.get("oof_epochs")==cli.oof_epochs and
                    saved.get("max_batches")==cli.max_batches and
                    saved.get("holdout")==[lo,hi]):
                collected.append(saved["item"])
                records_out.append(saved["record"])
                print(json.dumps({"oof_fold":fold,"reused":True}),flush=True)
                continue
        source_limit=lo-args.seq_len-args.pred_len
        if source_limit<2:raise RuntimeError("OOF source interval too small")
        mask=(records["starts"]+args.seq_len+args.pred_len)<=lo
        source=_paired_source(records,mask)
        bank=PairedMemoryBank(args.seq_len,args.pred_len,runcli.history_size)
        audit=bank.fit(source);bank=bank.to(device)
        predictor=predictor_for(model,args,runcli,device)
        optimizer=torch.optim.Adam([
            {"params":predictor.hidden_head.head.weight,"lr":runcli.lr_retrieval},
            {"params":predictor.residual_head.parameters(),"lr":runcli.lr_residual}],weight_decay=0.)
        train=_loader(datasets["train"],args,range(source_limit+1),
                      shuffle=True,seed=cli.seed+fold)
        for epoch in range(cli.oof_epochs):
            predictor.train();total=0.;batches=0
            for batch_index,(x,y,ids,cycle) in enumerate(train):
                if cli.max_batches and batch_index>=cli.max_batches:break
                x=x.float().to(device);cycle=cycle.to(device)
                y=y[:,-args.pred_len:].float().to(device)
                output,_,_,_=forward(model,predictor,bank,x,cycle,ids,args,
                                     runcli,"train",None)
                loss=F.mse_loss(output,y)
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(predictor.parameters(),5.)
                optimizer.step();total+=float(loss.detach());batches+=1
            print(json.dumps({"oof_fold":fold,"epoch":epoch+1,
                              "source_train_mse":total/max(batches,1)}),flush=True)
        heldout=_loader(datasets["train"],args,range(lo,hi))
        item=collect(model,predictor,bank,heldout,args,runcli,device,
                     "train",None,keep_prediction=False)
        collected.append(item)
        records_out.append({"fold":fold,"holdout_start":lo,"holdout_end":hi,
                            "source_last_complete_end":int(source["starts"].max()+args.seq_len+args.pred_len),
                            "source_memory":audit,"holdout_mse":float(item["error_mse"].mean())})
        assert records_out[-1]["source_last_complete_end"]<=lo
        torch.save({"reference_sha256":ref_hash,"point_run":str(cli.point_run),
                    "oof_epochs":cli.oof_epochs,"holdout":[lo,hi],
                    "max_batches":cli.max_batches,
                    "item":item,"record":records_out[-1]},fold_cache)
        write_json(out/"oof_progress.json",records_out)
    merged={key:torch.cat([item[key] for item in collected])
            for key in ("features","transfer_gain","error_mse","retrieved_residual_scale")}
    return merged,records_out


def main():
    cli=arguments();torch.set_num_threads(min(4,torch.get_num_threads()))
    out=Path(cli.output);out.mkdir(parents=True,exist_ok=True)
    if (out/"run_complete").is_file():
        print("already complete",flush=True);return
    point=Path(cli.point_run)
    if not (point/"run_complete").is_file():
        raise RuntimeError("point run must finish before risk stage")
    saved=json.loads((point/"run_config.json").read_text(encoding="utf-8"))
    runcli=SimpleNamespace(**saved["cli"])
    if runcli.mode=="A0":raise ValueError("retrieval risk needs non-A0 point mode")
    runcli.max_batches=cli.max_batches
    args,model,loaders,device,ref_hash=load_experiment(cli)
    if ref_hash != saved["reference_sha256"]:
        raise ValueError("risk/point A0 checkpoint mismatch")
    memory,_=memory_from_cache(cli,args,model,loaders,device,ref_hash)
    anchor=torch.load(cli.anchor_cache,map_location="cpu",weights_only=False)
    if anchor.get("reference_sha256")!=ref_hash:
        raise ValueError("OOF TRAIN anchor checkpoint mismatch")
    records=anchor["records"]
    point_checkpoint=torch.load(point/"best.pt",map_location=device,weights_only=False)
    predictor=predictor_for(model,args,runcli,device)
    predictor.load_state_dict(point_checkpoint["predictor"])
    write_json(out/"run_config.json",{"cli":vars(cli),"point_run":str(point),
               "point_best_epoch":point_checkpoint["best_epoch"],
               "reference_sha256":ref_hash,"source":"TRAIN chronological OOF",
               "calibration":"latter half of VAL; disjoint from point checkpoint selection"})
    oof,folds=oof_training(cli,model,{"train":loaders["train"].dataset.dataset},
                           args,device,ref_hash,runcli,records,out)
    coeff=fit_transfer_regression(oof["features"],oof["transfer_gain"])
    oof_features=add_transfer(oof["features"],coeff)
    risk_models={
        "novelty":fit_risk(oof_features[...,:1],oof["error_mse"],cli.seed+10),
        "disagreement":fit_risk(oof_features[...,1:3],oof["error_mse"],cli.seed+11),
        "full":fit_risk(oof_features,oof["error_mse"],cli.seed+12),
    }
    torch.save({"transfer_coeff":coeff,"risk_models":{k:m.state_dict() for k,m in risk_models.items()},
                "reference_sha256":ref_hash,"point_run":str(point)},out/"risk.pt")
    val_dataset=loaders["val"].dataset.dataset
    half=len(val_dataset)//2
    val_loader=_loader(val_dataset,args,range(half,len(val_dataset)))
    val_start=len(loaders["train"].dataset.dataset.data_x)-args.seq_len
    calibration=collect(model,predictor,memory,val_loader,args,runcli,device,
                        "val",val_start,keep_prediction=True)
    test=collect(model,predictor,memory,loaders["test"],args,runcli,device,
                 "test",val_start,keep_prediction=True)
    rows=[]
    for name in ("fixed","retrieved_residual","novelty","disagreement","full"):
        if name=="fixed":
            value=oof["error_mse"].mean().sqrt().clamp_min(1e-4)
            vscale=torch.full_like(calibration["error_mse"],float(value))
            tscale=torch.full_like(test["error_mse"],float(value))
        elif name=="retrieved_residual":
            vscale=calibration["retrieved_residual_scale"].clamp_min(.01)
            tscale=test["retrieved_residual_scale"].clamp_min(.01)
        else:
            def select(features):
                full=add_transfer(features,coeff)
                if name=="novelty":return full[...,:1]
                if name=="disagreement":return full[...,1:3]
                return full
            with torch.no_grad():
                vscale=risk_models[name](select(calibration["features"]))
                tscale=risk_models[name](select(test["features"]))
        calibrator=ConformalCalibrator(alpha=.05)
        cal=calibrator.fit(calibration["prediction"],calibration["truth"],vscale)
        lower,upper=calibrator.interval(test["prediction"],tscale)
        metrics=interval_metrics(test["prediction"],test["truth"],tscale,lower,upper)
        rows.append({"model":name,"calibration_q":cal["simultaneous_q"],**metrics})
    with (out/"interval_results.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)
    risk_score=risk_models["full"](oof_features).detach()
    oof_error=oof["error_mse"].sqrt()
    correlation=float(torch.corrcoef(torch.stack((risk_score.flatten(),oof_error.flatten())))[0,1])
    summary={"point_run":str(point),"source":"TRAIN chronological OOF",
             "oof_folds":folds,"oof_scale_error_correlation":correlation,
             "calibration_split":"VAL latter half only","interval_results":rows,
             "limitations":"Conformal simultaneous coverage assumes exchangeability; no arbitrary drift guarantee"}
    write_json(out/"result.json",summary)
    (out/"run_complete").touch()


if __name__=="__main__":main()
