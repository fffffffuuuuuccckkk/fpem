"""Strict TRAIN OOF CFIP and same-budget existing affine Flow B5/B6 pilot.

This is separate from the point-prediction queue.  A core-only A0 is trained
from fresh weights, then a core-only B3 point model.  Its predictions on the
later, target-disjoint TRAIN calibration block train CFIP.  The full-TRAIN B3
point checkpoint is used only for the final Flow comparison.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.PatchTST_PredictiveEnvIV import Model
from models.predictive_env.csif import CommonSpecificFactors, ConsensusLatentFusion
from models.predictive_env.conditional_innovation_prior import (
    ConditionalFutureInnovationPrior, InnovationConditionAdapter,
    strict_chronological_indices,
)
from models.predictive_env.horizon_future_variant import HorizonFutureVariant
from models.predictive_env.probability_heads import ProbabilityHeadBank
from tools.evaluate_fpem_probabilistic import probabilistic_metrics
from tools.run_fpem_csif import forward_parts
from tools.run_fpem_shared_factorization import load_experiment, prediction, write_json
from tools.run_predictive_env_iv_patchtst import (
    IndexedDataset, build_data, model_config, parse_args as legacy_parse_args,
)


def arguments():
    p=argparse.ArgumentParser()
    p.add_argument("--base_run_config",required=True)
    p.add_argument("--point_checkpoint",required=True,
                   help="B3 best.pt trained on full TRAIN, used only in Flow stage")
    p.add_argument("--output",required=True)
    p.add_argument("--gpu",type=int,default=0)
    p.add_argument("--seed",type=int,default=2021)
    p.add_argument("--core_epochs",type=int,default=10)
    p.add_argument("--point_epochs",type=int,default=10)
    p.add_argument("--prior_epochs",type=int,default=10)
    p.add_argument("--flow_epochs",type=int,default=10)
    p.add_argument("--lr_core",type=float,default=1e-4)
    p.add_argument("--lr_head",type=float,default=5e-5)
    p.add_argument("--lr_factor",type=float,default=1e-4)
    p.add_argument("--lr_prior",type=float,default=1e-3)
    p.add_argument("--lr_flow",type=float,default=1e-4)
    p.add_argument("--samples",type=int,default=32)
    p.add_argument("--flow_steps",type=int,default=12)
    p.add_argument("--max_batches",type=int,default=0)
    p.add_argument("--allow_legacy_reference_signature",action="store_true")
    return p.parse_args()


def base_data(cli):
    saved=json.loads(Path(cli.base_run_config).read_text())
    original=sys.argv
    try:
        sys.argv=[original[0]]
        defaults=vars(legacy_parse_args())
    finally:
        sys.argv=original
    defaults.update(saved)
    defaults.update({"num_workers":0,"seed":cli.seed})
    args=SimpleNamespace(**defaults)
    datasets,_,_,_=build_data(args)
    device=torch.device(f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu")
    return args,datasets,device


def fresh_core_model(args,seed,device):
    # Never initialize from the A0 trained on the complete TRAIN period.
    torch.manual_seed(seed+419)
    cfg=model_config(args,"A0")
    if args.backbone=="moderntcn":
        for field in ("modern_tcn_patch_size","modern_tcn_patch_stride",
                      "modern_tcn_num_stages","modern_tcn_ffn_ratio",
                      "modern_tcn_large_size","modern_tcn_small_size",
                      "modern_tcn_downsample_ratio","modern_tcn_head_dropout"):
            setattr(cfg,field,getattr(args,field))
    return Model(cfg).to(device)


def subset_loader(dataset,indices,args,seed,shuffle):
    indexed=IndexedDataset(dataset,cycle_len=args.cyclenet_cycle_len)
    return DataLoader(Subset(indexed,indices),batch_size=args.batch_size,
                      shuffle=shuffle,num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


def core_forecast(model,x,cycle):
    hidden,mean,std=model.encode(x,cycle)
    return prediction(model,hidden,mean,std,cycle,"active")


def prepare_strict_core(cli,args,datasets,device,outdir):
    train_dataset=datasets["train"]
    core,cal,cut=strict_chronological_indices(len(train_dataset),args.seq_len,args.pred_len)
    # The standard loader fits its scaler on all TRAIN.  Refit on the earlier
    # core time points so even preprocessing cannot inspect calibration values.
    raw=train_dataset.inverse_transform(train_dataset.data_x).astype(np.float32)
    core_mean=raw[:cut].mean(0)
    core_std=raw[:cut].std(0).clip(1e-6)
    core_scaled=(raw-core_mean)/core_std
    train_dataset.data_x=core_scaled
    train_dataset.data_y=core_scaled
    metadata={"strict_train_only":True,"trained_from_scratch":True,
              "core_target_last":max(i+args.seq_len+args.pred_len-1 for i in core),
              "calibration_target_first":min(i+args.seq_len for i in cal),
              "boundary":cut,"core_count":len(core),"calibration_count":len(cal),
              "core_fraction_requested":0.8,
              "core_only_scaler_mean":core_mean.tolist(),
              "core_only_scaler_std":core_std.tolist()}
    if metadata["core_target_last"]>=metadata["calibration_target_first"]:
        raise RuntimeError("TRAIN OOF target overlap")
    core_a0_path=outdir/"core_only_a0.pt"
    core_point_path=outdir/"core_only_b3.pt"
    model=fresh_core_model(args,cli.seed,device)
    if core_a0_path.exists():
        state=torch.load(core_a0_path,map_location=device,weights_only=False)
        if state.get("metadata")!=metadata:
            raise RuntimeError("core-only A0 provenance mismatch")
        model.load_state_dict(state["model"])
    else:
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cli.lr_core)
        for epoch in range(cli.core_epochs):
            model.train();total=0.;count=0
            for step,(x,y,ids,cycle) in enumerate(subset_loader(train_dataset,core,args,cli.seed+epoch,True)):
                if cli.max_batches and step>=cli.max_batches:break
                x=x.float().to(device);y=y[:,-args.pred_len:].float().to(device);cycle=cycle.to(device)
                loss=(core_forecast(model,x,cycle)-y).square().mean()
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                total+=float(loss.detach())*len(x);count+=len(x)
            print(json.dumps({"stage":"core_a0","epoch":epoch+1,"train_mse":total/count}),flush=True)
        model.eval()
        torch.save({"model":model.state_dict(),"metadata":metadata,
                    "epochs":cli.core_epochs},core_a0_path)
    if core_point_path.exists():
        state=torch.load(core_point_path,map_location=device,weights_only=False)
        if state.get("metadata")!=metadata:raise RuntimeError("core-only B3 provenance mismatch")
        model.load_state_dict(state["model"])
        factor=CommonSpecificFactors(model.d_model,"memory").to(device)
        fusion=ConsensusLatentFusion(model.d_model,True).to(device)
        factor.load_state_dict(state["factor"]);fusion.load_state_dict(state["fusion"])
        return model,factor,fusion,metadata,cal
    model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    model.head_linear.requires_grad_(True)
    torch.manual_seed(cli.seed+3401)
    factor=CommonSpecificFactors(model.d_model,"memory").to(device)
    fusion=ConsensusLatentFusion(model.d_model,True).to(device)
    optimizer=torch.optim.Adam([{"params":model.head_linear.parameters(),"lr":cli.lr_head},
                                {"params":factor.parameters(),"lr":cli.lr_factor},
                                {"params":fusion.parameters(),"lr":cli.lr_factor}])
    for epoch in range(cli.point_epochs):
        model.eval();model.head_dropout.train();factor.train();fusion.train();total=0.;count=0
        for step,(x,y,ids,cycle) in enumerate(subset_loader(train_dataset,core,args,cli.seed+900+epoch,True)):
            if cli.max_batches and step>=cli.max_batches:break
            x=x.float().to(device);y=y[:,-args.pred_len:].float().to(device);cycle=cycle.to(device)
            out=forward_parts(model,factor,fusion,x,cycle)
            loss=(out["point"]-y).square().mean()+.05*(out["common"]-y).square().mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            factor.observe_train_batch(out["activations"].detach(),ids,args.seq_len)
            total+=float(loss.detach())*len(x);count+=len(x)
        memory=factor.finish_train_epoch()
        print(json.dumps({"stage":"core_b3","epoch":epoch+1,
                          "train_loss":total/count,**memory}),flush=True)
    model.eval();factor.eval();fusion.eval()
    torch.save({"model":model.state_dict(),"factor":factor.state_dict(),
                "fusion":fusion.state_dict(),"metadata":metadata,
                "epochs":cli.point_epochs},core_point_path)
    return model,factor,fusion,metadata,cal


@torch.no_grad()
def calibration_cache(model,factor,fusion,dataset,indices,args,device,max_batches):
    model.eval();factor.eval();fusion.eval()
    arrays={key:[] for key in ("common","specific","support","residual")}
    for step,(x,y,ids,cycle) in enumerate(subset_loader(dataset,indices,args,args.seed,False)):
        if max_batches and step>=max_batches:break
        x=x.float().to(device);y=y[:,-args.pred_len:].float().to(device);cycle=cycle.to(device)
        out=forward_parts(model,factor,fusion,x,cycle)
        _,_,std=model.encode(x,cycle)
        residual=((y-out["point"])/std.clamp_min(1e-4)).transpose(1,2)
        arrays["common"].append(out["common_hidden"].mean(2).cpu())
        arrays["specific"].append(out["specific_hidden"].mean(2).cpu())
        arrays["support"].append(out["support"].mean(2).cpu())
        arrays["residual"].append(residual.cpu())
    return {key:torch.cat(value) for key,value in arrays.items()}


def fit_prior(cli,args,datasets,device,outdir,model,factor,fusion,metadata,cal):
    path=outdir/"strict_cfip.pt"
    prior=ConditionalFutureInnovationPrior(model.d_model,args.pred_len).to(device)
    if path.exists():
        state=torch.load(path,map_location=device,weights_only=False)
        if state.get("metadata")!=metadata:raise RuntimeError("CFIP OOF provenance mismatch")
        prior.load_state_dict(state["prior"]);prior.eval()
        return prior,state["summary"]
    records=calibration_cache(model,factor,fusion,datasets["train"],cal,args,device,cli.max_batches)
    count=len(records["residual"]);split=max(1,int(.8*count))
    if split>=count:raise ValueError("calibration holdout too small")
    optimizer=torch.optim.Adam(prior.parameters(),lr=cli.lr_prior)
    best=math.inf;history=[]
    for epoch in range(cli.prior_epochs):
        prior.train();gen=torch.Generator().manual_seed(cli.seed+epoch+4000)
        order=torch.randperm(split,generator=gen)
        losses=[]
        for ids in order.split(args.batch_size):
            common=records["common"][ids].to(device).unsqueeze(2)
            specific=records["specific"][ids].to(device).unsqueeze(2)
            support=records["support"][ids].to(device).unsqueeze(2)
            residual=records["residual"][ids].to(device)
            distribution=prior(common,specific,support,factor.popularity().detach())
            loss=prior.nll(distribution,residual)
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            losses.append(float(loss.detach()))
        prior.eval();valid=[];pi=[];normal=[];novel=[]
        with torch.no_grad():
            for ids in torch.arange(split,count).split(args.batch_size):
                distribution=prior(records["common"][ids].to(device).unsqueeze(2),
                    records["specific"][ids].to(device).unsqueeze(2),
                    records["support"][ids].to(device).unsqueeze(2),factor.popularity().detach())
                valid.append(float(prior.nll(distribution,records["residual"][ids].to(device))))
                pi.append(float(distribution["pi_novel"].mean()))
                normal.append(float(distribution["sigma_normal"].mean()))
                novel.append(float(distribution["sigma_novel"].mean()))
        row={"epoch":epoch+1,"train_nll":sum(losses)/len(losses),
             "val_nll":sum(valid)/len(valid),"pi_novel":sum(pi)/len(pi),
             "sigma_normal":sum(normal)/len(normal),"sigma_novel":sum(novel)/len(novel)}
        history.append(row);print(json.dumps({"stage":"prior",**row}),flush=True)
        if row["val_nll"]<best:
            best=row["val_nll"]
            torch.save({"prior":prior.state_dict(),"metadata":metadata,
                        "summary":row,"history":history},path)
    state=torch.load(path,map_location=device,weights_only=False)
    prior.load_state_dict(state["prior"]);prior.eval()
    write_json(outdir/"cfip_summary.json",{"metadata":metadata,"best":state["summary"],
                                           "history":history})
    return prior,state["summary"]


def load_full_point(cli,args,device):
    saved_args,model,loaders,device,digest=load_experiment(cli)
    checkpoint=torch.load(cli.point_checkpoint,map_location=device,weights_only=False)
    if checkpoint.get("reference_sha256")!=digest:
        raise RuntimeError("B3 point checkpoint and full A0 identity mismatch")
    model.load_state_dict(checkpoint["model"])
    factor=CommonSpecificFactors(model.d_model,"memory").to(device)
    fusion=ConsensusLatentFusion(model.d_model,True).to(device)
    factor.load_state_dict(checkpoint["factor"])
    fusion.load_state_dict(checkpoint["fusion"])
    for module in (model,factor,fusion):
        module.eval()
        for parameter in module.parameters():parameter.requires_grad_(False)
    return model,factor,fusion,loaders,digest


def flow_condition(model,factor,fusion,future_model,bank,prior,adapter,x,cycle,enabled):
    with torch.no_grad():
        out=forward_parts(model,factor,fusion,x,cycle)
        _,_,std=model.encode(x,cycle)
        if enabled:
            dist=prior(out["common_hidden"],out["specific_hidden"],out["support"],
                       factor.popularity())
            innovation=prior.features(dist)
        else:innovation=None
    future=future_model.encode_future(out["specific_hidden"].detach())[0]
    condition=bank.condition(future,out["point"].detach(),std.detach())
    condition=adapter(condition,innovation,enabled=enabled)
    return condition,future,out["point"].detach(),std.detach()


@torch.no_grad()
def evaluate_flow(model,factor,fusion,future_model,bank,prior,adapter,loader,args,
                  device,enabled,samples,steps,max_batches,seed):
    future_model.eval();bank.eval();adapter.eval()
    sums={};seen=0
    for step,(x,y,ids,cycle) in enumerate(loader):
        if max_batches and step>=max_batches:break
        x=x.float().to(device);y=y[:,-args.pred_len:].float().to(device);cycle=cycle.to(device)
        condition,future,y_inv,std=flow_condition(model,factor,fusion,future_model,
                                                   bank,prior,adapter,x,cycle,enabled)
        gen=torch.Generator(device=device).manual_seed(seed+step*1009)
        paths=bank.sample_head("affine_flow",condition,future,y_inv,std,
                               samples,gen,chunk_size=min(10,samples),steps=steps)
        metrics=probabilistic_metrics(paths["y_samples"],y,y_inv)
        for key in ("inv_MSE","inv_MAE","mean_MSE","mean_MAE","CRPS","PICP_95",
                    "MPIW_95","WIS","EnergyScore","sample_std_mean"):
            sums[key]=sums.get(key,0.)+float(metrics[key])*len(x)
        seen+=len(x)
    return {key:value/seen for key,value in sums.items()}


def train_flow(cli,args,device,outdir,model,factor,fusion,loaders,prior,digest,mode):
    trial=outdir/mode;trial.mkdir(parents=True,exist_ok=True)
    if (trial/"run_complete").exists():
        return json.loads((trial/"result.json").read_text())
    enabled=mode=="B5"
    torch.manual_seed(cli.seed+8701)
    future_model=HorizonFutureVariant(model.d_model,args.pred_len,
                                     future_patch_len=16,environment_count=0).to(device)
    bank=ProbabilityHeadBank(model.d_model,args.pred_len,patch_len=16,
                             heads=("affine_flow",)).to(device)
    adapter=InnovationConditionAdapter(model.d_model,16).to(device)
    params=list(future_model.parameters())+list(bank.parameters())
    if enabled:params+=list(adapter.parameters())
    optimizer=torch.optim.Adam(params,lr=cli.lr_flow)
    best=math.inf;history=[]
    for epoch in range(cli.flow_epochs):
        future_model.train();bank.train();adapter.train()
        training=subset_loader(loaders["train"].dataset.dataset,
                               list(range(len(loaders["train"].dataset))),
                               args,cli.seed+epoch,True)
        total=0.;count=0
        for step,(x,y,ids,cycle) in enumerate(training):
            if cli.max_batches and step>=cli.max_batches:break
            x=x.float().to(device);y=y[:,-args.pred_len:].float().to(device);cycle=cycle.to(device)
            condition,future,y_inv,std=flow_condition(model,factor,fusion,
                future_model,bank,prior,adapter,x,cycle,enabled)
            loss,details=bank.training_losses(condition,future,y_inv,y,std)["affine_flow"]
            if not bool(torch.isfinite(loss.detach())):raise FloatingPointError("nonfinite Flow loss")
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            total+=float(loss.detach())*len(x);count+=len(x)
        val=evaluate_flow(model,factor,fusion,future_model,bank,prior,adapter,
            loaders["val"],args,device,enabled,min(cli.samples,32),cli.flow_steps,
            cli.max_batches,cli.seed+19001)
        row={"epoch":epoch+1,"train_flow_loss":total/count,"val":val}
        history.append(row);print(json.dumps({"stage":mode,**row}),flush=True)
        if val["CRPS"]<best:
            best=val["CRPS"]
            torch.save({"future":future_model.state_dict(),"bank":bank.state_dict(),
                        "adapter":adapter.state_dict(),"epoch":epoch+1,
                        "val":val,"reference_sha256":digest},trial/"best.pt")
    saved=torch.load(trial/"best.pt",map_location=device,weights_only=False)
    future_model.load_state_dict(saved["future"]);bank.load_state_dict(saved["bank"])
    adapter.load_state_dict(saved["adapter"])
    test=evaluate_flow(model,factor,fusion,future_model,bank,prior,adapter,
        loaders["test"],args,device,enabled,cli.samples,cli.flow_steps,
        cli.max_batches,cli.seed+25007)
    result={"mode":mode,"val":saved["val"],"test":test,
            "best_epoch":saved["epoch"],"history":history,
            "reference_sha256":digest,"point_checkpoint":cli.point_checkpoint,
            "prior_training":"strict TRAIN-only core/calibration split",
            "flow_base_noise":"unchanged standard Gaussian"}
    write_json(trial/"result.json",result)
    (trial/"run_complete").touch()
    return result


def main():
    cli=arguments();outdir=Path(cli.output);outdir.mkdir(parents=True,exist_ok=True)
    args,datasets,device=base_data(cli)
    model,factor,fusion,metadata,cal=prepare_strict_core(cli,args,datasets,device,outdir)
    prior,prior_summary=fit_prior(cli,args,datasets,device,outdir,model,factor,fusion,metadata,cal)
    full_model,full_factor,full_fusion,loaders,digest=load_full_point(cli,args,device)
    results={}
    for mode in ("B6","B5"):
        results[mode]=train_flow(cli,args,device,outdir,full_model,full_factor,
                                 full_fusion,loaders,prior,digest,mode)
    write_json(outdir/"comparison.json",{"strict_oof":metadata,
        "cfip":prior_summary,"modes":{m:{"val":r["val"],"test":r["test"]}
                                       for m,r in results.items()}})
    (outdir/"run_complete").touch()


if __name__=="__main__":main()
