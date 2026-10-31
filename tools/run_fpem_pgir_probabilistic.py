"""PGIR-specific strict TRAIN-OOF CFIP followed by the existing affine Flow.

The core A0, prototype bank and mask never observe calibration targets. The
full-TRAIN P2 point checkpoint is used only for the final Flow training/test.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.predictive_env.pgir import PGIRMask
from models.predictive_env.ptm import PrototypeBank
from models.predictive_env.conditional_innovation_prior import (
    ConditionalFutureInnovationPrior, InnovationConditionAdapter,
    strict_chronological_indices)
from models.predictive_env.horizon_future_variant import HorizonFutureVariant
from models.predictive_env.probability_heads import ProbabilityHeadBank
from tools.evaluate_fpem_probabilistic import probabilistic_metrics
from tools.run_fpem_csif_probabilistic import (
    base_data, fresh_core_model, subset_loader, core_forecast)
from tools.run_fpem_ptm import base_forecast
from tools.run_fpem_shared_factorization import load_experiment, write_json


class CoreView:
    """Only the early TRAIN time points are visible to prototype fitting."""
    set_type = 0
    def __init__(self, dataset, cut):
        self.seq_len = int(dataset.seq_len)
        self.pred_len = int(dataset.pred_len)
        self.data_x = np.asarray(dataset.data_x[:cut]).copy()
        self.data_y = np.asarray(dataset.data_y[:cut]).copy()
    def __len__(self):
        return max(0, len(self.data_x)-self.seq_len-self.pred_len+1)


def condition_features(x, selected, info):
    # [B,C,8], historical inputs only; no future target or inferred test q.
    past = x.transpose(1, 2)
    proto = info["query"]["x_proto"].transpose(1, 2)
    mean = past.mean(-1, keepdim=True)
    scale = past.std(-1, unbiased=False, keepdim=True).clamp_min(1e-4)
    z = (past-mean)/scale
    residual = (past-proto)/scale
    common = torch.stack((z.mean(-1), z.std(-1, unbiased=False),
                          z[..., 0], z[..., -1],
                          (z[..., -1]-z[..., 0]),
                          (z[..., 1:]-z[..., :-1]).abs().mean(-1),
                          z.amax(-1), z.amin(-1)), -1)
    specific = torch.stack((residual.mean(-1), residual.std(-1, unbiased=False),
                            residual.abs().mean(-1), residual[..., 0],
                            residual[..., -1], residual.amax(-1),
                            residual.amin(-1),
                            info["mask"].detach().mean(-1)), -1)
    support = info["invariant"].detach().mean(-1, keepdim=True)
    return common.detach().unsqueeze(2), specific.detach().unsqueeze(2), support.unsqueeze(2)


def pgir_forward(model, bank, selector, x, cycle, *, core=False):
    selected, info = selector(x, bank, mode="P2")
    if core:
        y_inv = core_forecast(model, selected, cycle)
    else:
        y_inv = base_forecast(model, selected, cycle)
    # Same historical standardization convention in CFIP and Flow.
    _, _, std = model.encode(selected, cycle)
    return y_inv, std.clamp_min(1e-4), selected, info


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--base_run_config", required=True)
    p.add_argument("--point_checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gpu", type=int, default=3)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--core_epochs", type=int, default=10)
    p.add_argument("--mask_epochs", type=int, default=10)
    p.add_argument("--prior_epochs", type=int, default=10)
    p.add_argument("--flow_epochs", type=int, default=10)
    p.add_argument("--lr_core", type=float, default=1e-4)
    p.add_argument("--lr_mask", type=float, default=1e-3)
    p.add_argument("--lr_prior", type=float, default=1e-3)
    p.add_argument("--lr_flow", type=float, default=1e-4)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--flow_steps", type=int, default=12)
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--allow_legacy_reference_signature", action="store_true")
    return p.parse_args()


def prepare_core(cli, args, datasets, device, out):
    dataset = datasets["train"]
    core, cal, cut = strict_chronological_indices(len(dataset), args.seq_len, args.pred_len)
    raw = dataset.inverse_transform(dataset.data_x).astype(np.float32)
    center = raw[:cut].mean(0)
    scale = raw[:cut].std(0).clip(1e-6)
    data = (raw-center)/scale
    dataset.data_x = data; dataset.data_y = data
    view = CoreView(dataset, cut)
    metadata = {"core_target_last": max(i+args.seq_len+args.pred_len-1 for i in core),
                "cal_target_first": min(i+args.seq_len for i in cal),
                "core_count": len(core), "cal_count": len(cal), "cut": cut,
                "core_scaler_center": center.tolist(), "core_scaler_scale": scale.tolist(),
                "seed": cli.seed, "core_epochs": cli.core_epochs, "mask_epochs": cli.mask_epochs}
    if metadata["core_target_last"] >= metadata["cal_target_first"]:
        raise RuntimeError("OOF targets overlap")
    path = out/"core_only_pgir.pt"
    model = fresh_core_model(args, cli.seed, device)
    bank = PrototypeBank(args.pred_len, 8, 64).to(device)
    selector = PGIRMask(16).to(device)
    if path.exists():
        state = torch.load(path, map_location=device, weights_only=False)
        if state["metadata"] != metadata: raise RuntimeError("core checkpoint provenance mismatch")
        model.load_state_dict(state["model"])
        bank.load_state_dict(state["bank"])
        selector.load_state_dict(state["selector"])
        return model, bank, selector, metadata, cal
    optimizer = torch.optim.Adam(model.parameters(), lr=cli.lr_core)
    for epoch in range(cli.core_epochs):
        model.train(); total = count = 0
        for step, (x,y,_,cycle) in enumerate(subset_loader(dataset, core, args, cli.seed+epoch, True)):
            if cli.max_batches and step >= cli.max_batches: break
            x=x.float().to(device); y=y[:,-args.pred_len:].float().to(device); cycle=cycle.to(device)
            loss=(core_forecast(model,x,cycle)-y).square().mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total+=float(loss.detach())*len(x); count+=len(x)
        print(json.dumps({"stage":"core_A0","epoch":epoch+1,"mse":total/max(count,1)}),flush=True)
    model.eval().requires_grad_(False)
    fit=bank.fit_train(view, seed=cli.seed)
    threshold=selector.fit_threshold(bank,view,device)
    optimizer=torch.optim.Adam(selector.parameters(),lr=cli.lr_mask)
    for epoch in range(cli.mask_epochs):
        selector.train(); total=count=0
        for step,(x,y,_,cycle) in enumerate(subset_loader(dataset,core,args,cli.seed+900+epoch,True)):
            if cli.max_batches and step>=cli.max_batches:break
            x=x.float().to(device);y=y[:,-args.pred_len:].float().to(device);cycle=cycle.to(device)
            forecast,_,_,info=pgir_forward(model,bank,selector,x,cycle,core=True)
            pred=(forecast-y).square().mean()
            noninv=1-info["invariant"]
            sparse=(noninv*info["probability"]).sum()/noninv.sum().clamp_min(1)
            p=info["probability"]
            temporal=(p[...,1:]-p[...,:-1]).abs().mean() if p.shape[-1]>1 else pred*0
            loss=pred+0.005*sparse+0.001*temporal
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            total+=float(pred.detach())*len(x);count+=len(x)
        print(json.dumps({"stage":"core_PGIR","epoch":epoch+1,"mse":total/max(count,1)}),flush=True)
    model.eval();selector.eval();bank.eval()
    torch.save({"model":model.state_dict(),"bank":bank.state_dict(),
                "selector":selector.state_dict(),"metadata":metadata,
                "bank_fit":fit,"threshold":threshold},path)
    return model,bank,selector,metadata,cal


@torch.no_grad()
def calibration_cache(cli, model, bank, selector, dataset, indices, args, device):
    arrays={k:[] for k in ("common","specific","support","residual")}
    for step,(x,y,_,cycle) in enumerate(subset_loader(dataset,indices,args,cli.seed,False)):
        if cli.max_batches and step>=cli.max_batches:break
        x=x.float().to(device);y=y[:,-args.pred_len:].float().to(device);cycle=cycle.to(device)
        y_inv,std,selected,info=pgir_forward(model,bank,selector,x,cycle,core=True)
        common,specific,support=condition_features(x,selected,info)
        arrays["common"].append(common[:, :, 0].cpu())
        arrays["specific"].append(specific[:, :, 0].cpu())
        arrays["support"].append(support[:, :, 0].cpu())
        arrays["residual"].append(((y-y_inv)/std).transpose(1,2).cpu())
    return {k:torch.cat(v) for k,v in arrays.items()}


def fit_prior(cli,args,dataset,device,out,model,bank,selector,metadata,cal):
    path=out/"strict_oof_cfip.pt"
    prior=ConditionalFutureInnovationPrior(8,args.pred_len).to(device)
    if path.exists():
        state=torch.load(path,map_location=device,weights_only=False)
        if state["metadata"]!=metadata:raise RuntimeError("CFIP provenance mismatch")
        prior.load_state_dict(state["prior"]);prior.eval()
        return prior,state["summary"]
    records=calibration_cache(cli,model,bank,selector,dataset,cal,args,device)
    n=len(records["residual"]);split=max(1,int(0.8*n))
    if split>=n:raise ValueError("not enough OOF calibration records")
    optimizer=torch.optim.Adam(prior.parameters(),lr=cli.lr_prior)
    best=math.inf
    for epoch in range(cli.prior_epochs):
        prior.train(); losses=[]
        order=torch.randperm(split,generator=torch.Generator().manual_seed(cli.seed+epoch+4000))
        for ids in order.split(args.batch_size):
            common=records["common"][ids].to(device).unsqueeze(2)
            specific=records["specific"][ids].to(device).unsqueeze(2)
            support=records["support"][ids].to(device).unsqueeze(2)
            dist=prior(common,specific,support,torch.tensor(0.5,device=device))
            loss=prior.nll(dist,records["residual"][ids].to(device))
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            losses.append(float(loss.detach()))
        prior.eval();valid=[]
        with torch.no_grad():
            for ids in torch.arange(split,n).split(args.batch_size):
                dist=prior(records["common"][ids].to(device).unsqueeze(2),
                           records["specific"][ids].to(device).unsqueeze(2),
                           records["support"][ids].to(device).unsqueeze(2),
                           torch.tensor(0.5,device=device))
                valid.append(float(prior.nll(dist,records["residual"][ids].to(device))))
        row={"epoch":epoch+1,"train_nll":sum(losses)/len(losses),
             "oof_val_nll":sum(valid)/len(valid)}
        print(json.dumps({"stage":"PGIR_CFIP",**row}),flush=True)
        if row["oof_val_nll"]<best:
            best=row["oof_val_nll"]
            torch.save({"prior":prior.state_dict(),"metadata":metadata,"summary":row},path)
    saved=torch.load(path,map_location=device,weights_only=False)
    prior.load_state_dict(saved["prior"]);prior.eval()
    return prior,saved["summary"]


def load_full_point(cli,args,device):
    _,model,loaders,device,digest=load_experiment(cli)
    state=torch.load(cli.point_checkpoint,map_location=device,weights_only=False)
    if state.get("reference_sha256")!=digest:raise RuntimeError("P2/A0 identity mismatch")
    selector=PGIRMask(16).to(device)
    bank=PrototypeBank(args.pred_len,8,64).to(device)
    selector.load_state_dict(state["selector"])
    bank.load_state_dict(state["bank"])
    model.eval().requires_grad_(False);selector.eval().requires_grad_(False);bank.eval()
    return model,bank,selector,loaders,digest


def flow_condition(model,bank,selector,future_model,head_bank,prior,adapter,x,cycle,enabled):
    with torch.no_grad():
        y_inv,std,selected,info=pgir_forward(model,bank,selector,x,cycle)
        common,specific,support=condition_features(x,selected,info)
        if enabled:
            dist=prior(common,specific,support,torch.tensor(0.5,device=x.device))
            innovation=prior.features(dist)
        else: innovation=None
    future=future_model.encode_future(specific)[0]
    condition=head_bank.condition(future,y_inv.detach(),std.detach())
    condition=adapter(condition,innovation,enabled=enabled)
    return condition,future,y_inv.detach(),std.detach()


@torch.no_grad()
def evaluate_flow(model,proto,selector,future,heads,prior,adapter,loader,args,device,
                  samples,steps,max_batches,seed):
    future.eval();heads.eval();adapter.eval()
    sums={};seen=0
    for step,(x,y,_,cycle) in enumerate(loader):
        if max_batches and step>=max_batches:break
        x=x.float().to(device);y=y[:,-args.pred_len:].float().to(device);cycle=cycle.to(device)
        condition,fvar,y_inv,std=flow_condition(model,proto,selector,future,heads,
                                                prior,adapter,x,cycle,True)
        generator=torch.Generator(device=device).manual_seed(seed+step*1009)
        paths=heads.sample_head("affine_flow",condition,fvar,y_inv,std,
                                samples,generator,chunk_size=min(10,samples),steps=steps)
        metric=probabilistic_metrics(paths["y_samples"],y,y_inv)
        for key in ("inv_MSE","inv_MAE","mean_MSE","mean_MAE","CRPS","PICP_95",
                    "MPIW_95","WIS","EnergyScore","sample_std_mean"):
            sums[key]=sums.get(key,0.)+float(metric[key])*len(x)
        seen+=len(x)
    if not seen:raise RuntimeError("empty probability evaluation")
    return {k:v/seen for k,v in sums.items()}


def main():
    cli=parse();out=Path(cli.output);out.mkdir(parents=True,exist_ok=True)
    args,datasets,device=base_data(cli)
    core_model,core_bank,core_selector,metadata,cal=prepare_core(cli,args,datasets,device,out)
    prior,prior_summary=fit_prior(cli,args,datasets["train"],device,out,
                                  core_model,core_bank,core_selector,metadata,cal)
    del core_model, core_bank, core_selector
    if device.type == "cuda": torch.cuda.empty_cache()
    model,proto,selector,loaders,digest=load_full_point(cli,args,device)
    torch.manual_seed(cli.seed+8701)
    future=HorizonFutureVariant(8,args.pred_len,future_patch_len=16,environment_count=0).to(device)
    heads=ProbabilityHeadBank(8,args.pred_len,patch_len=16,heads=("affine_flow",)).to(device)
    adapter=InnovationConditionAdapter(8,16).to(device)
    optimizer=torch.optim.Adam(list(future.parameters())+list(heads.parameters())+
                               list(adapter.parameters()),lr=cli.lr_flow)
    history=[];best=math.inf;start=time.perf_counter()
    for epoch in range(cli.flow_epochs):
        future.train();heads.train();adapter.train();total=count=0
        for step,(x,y,_,cycle) in enumerate(loaders["train"]):
            if cli.max_batches and step>=cli.max_batches:break
            x=x.float().to(device);y=y[:,-args.pred_len:].float().to(device);cycle=cycle.to(device)
            condition,fvar,y_inv,std=flow_condition(model,proto,selector,future,heads,
                                                     prior,adapter,x,cycle,True)
            loss,_=heads.training_losses(condition,fvar,y_inv,y,std)["affine_flow"]
            if not torch.isfinite(loss):raise FloatingPointError("non-finite PGIR Flow loss")
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            total+=float(loss.detach())*len(x);count+=len(x)
        val=evaluate_flow(model,proto,selector,future,heads,prior,adapter,
                          loaders["val"],args,device,min(cli.samples,32),cli.flow_steps,
                          cli.max_batches,cli.seed+19001)
        row={"epoch":epoch+1,"train_flow_loss":total/max(count,1),"val":val}
        history.append(row);print(json.dumps(row),flush=True)
        if val["CRPS"]<best:
            best=val["CRPS"]
            torch.save({"future":future.state_dict(),"heads":heads.state_dict(),
                        "adapter":adapter.state_dict(),"epoch":epoch+1,
                        "val":val,"reference_sha256":digest},out/"best.pt")
    saved=torch.load(out/"best.pt",map_location=device,weights_only=False)
    future.load_state_dict(saved["future"]);heads.load_state_dict(saved["heads"])
    adapter.load_state_dict(saved["adapter"])
    test=evaluate_flow(model,proto,selector,future,heads,prior,adapter,
                       loaders["test"],args,device,cli.samples,cli.flow_steps,
                       cli.max_batches,cli.seed+25007)
    result={"mode":"P4","val":saved["val"],"test":test,
            "best_epoch":saved["epoch"],"history":history,
            "point_checkpoint":cli.point_checkpoint,
            "prior_training":"strict TRAIN-only core/calibration; own core A0/bank/PGIR",
            "oof":metadata,"prior_summary":prior_summary,
            "training_seconds":time.perf_counter()-start}
    write_json(out/"result.json",result)
    write_json(out/"run_config.json",vars(cli))
    (out/"run_complete").touch()


if __name__=="__main__":
    main()
