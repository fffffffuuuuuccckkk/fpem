"""Fair 10-epoch PGIR T1-T5 pilot, frozen A0, VAL-selected checkpoint."""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.predictive_env.ptm import PrototypeBank
from models.predictive_env.pgir_tricks import PGIRTrickMask
from tools.pgir_gain_labels import build_gain_cache
from tools.run_fpem_ptm import base_forecast, read_or_fit_bank
from tools.run_fpem_shared_factorization import load_experiment, train_loader, write_json


def parse():
    p=argparse.ArgumentParser()
    p.add_argument("--base_run_config",required=True)
    p.add_argument("--bank_checkpoint",required=True)
    p.add_argument("--core_p4_dir",required=True)
    p.add_argument("--gain_cache",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--mode",choices=("T1","T2","T3","T4","T5"),required=True)
    p.add_argument("--local",action="store_true")
    p.add_argument("--gain_supervision",action="store_true")
    p.add_argument("--boundary",action="store_true")
    p.add_argument("--recent",action="store_true")
    p.add_argument("--gpu",type=int,default=0)
    p.add_argument("--seed",type=int,default=2021)
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--lr_mask",type=float,default=1e-3)
    p.add_argument("--lr_feature",type=float,default=None)
    p.add_argument("--lr_head",type=float,default=None)
    p.add_argument("--output_bias",type=float,default=0.1)
    p.add_argument("--normalize_features",action="store_true")
    p.add_argument("--schedule",choices=("constant","cosine_warmup"),default="constant")
    p.add_argument("--ema_decay",type=float,default=0.0)
    p.add_argument("--evaluate_checkpoint",default="")
    p.add_argument("--calibrate_threshold",action="store_true")
    p.add_argument("--segment_len",type=int,default=16)
    p.add_argument("--max_replace",type=int,default=2)
    p.add_argument("--recent_penalty",type=float,default=0.25)
    p.add_argument("--lambda_replace",type=float,default=0.001)
    p.add_argument("--lambda_temporal",type=float,default=0.001)
    p.add_argument("--lambda_gain",type=float,default=0.05)
    p.add_argument("--max_batches",type=int,default=0)
    p.add_argument("--allow_legacy_reference_signature",action="store_true")
    return p.parse_args()


def forward(model,bank,selector,x,cycle,cli):
    selected,info=selector(x,bank,local=cli.local,boundary=cli.boundary,
                           recent=cli.recent)
    return base_forecast(model,selected,cycle),selected,info


@torch.no_grad()
def evaluate(model,bank,selector,loader,args,device,cli):
    model.eval();selector.eval()
    mse=mae=a0mse=a0mae=replaced=active=positive=gain=0.
    count=elements=series=0
    for step,(x,y,_,cycle) in enumerate(loader):
        if cli.max_batches and step>=cli.max_batches:break
        x=x.float().to(device);y=y[:,-args.pred_len:].float().to(device);cycle=cycle.to(device)
        pred,_,info=forward(model,bank,selector,x,cycle,cli)
        a0=base_forecast(model,x,cycle)
        mse+=float((pred-y).square().sum());mae+=float((pred-y).abs().sum())
        a0mse+=float((a0-y).square().sum());a0mae+=float((a0-y).abs().sum())
        per_gain=(a0-y).square().mean((1,2))-(pred-y).square().mean((1,2))
        gain+=float(per_gain.sum());positive+=float((per_gain>0).sum())
        replaced+=float(info["replace_count"].sum())
        active+=float((info["replace_count"]>0).sum())
        count+=len(x);series+=len(x)*x.shape[-1];elements+=y.numel()
    if not count:raise RuntimeError("empty validation/test loader")
    return {"mse":mse/elements,"mae":mae/elements,
            "a0_mse":a0mse/elements,"a0_mae":a0mae/elements,
            "mean_gain":gain/count,"positive_gain_ratio":positive/count,
            "mean_replaced_patches_per_series":replaced/series,
            "fraction_series_intervened":active/series}


def main():
    cli=parse()
    out=Path(cli.output);out.mkdir(parents=True,exist_ok=True)
    args,model,loaders,device,digest=load_experiment(cli)
    train_dataset=loaders["train"].dataset.dataset
    bank=PrototypeBank(args.pred_len,8,64)
    read_or_fit_bank(bank,train_dataset,cli.bank_checkpoint,args,
                     argparse.Namespace(**{**vars(cli),"k":8,"history_size":64}),digest)
    bank.to(device).eval()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cli.seed+5739)
        selector=PGIRTrickMask(cli.segment_len,cli.max_replace,cli.recent_penalty,
                               cli.output_bias,cli.normalize_features).to(device)
    threshold=selector.fit_threshold(bank,train_dataset,device)
    feature_stats=selector.fit_feature_normalizer(bank,train_dataset,device)
    model.eval().requires_grad_(False)
    x0,_,_,cycle0=next(iter(loaders["train"]))
    x0=x0.float().to(device);cycle0=cycle0.to(device)
    with torch.no_grad():
        initial,selected,_=forward(model,bank,selector,x0,cycle0,cli)
        a0=base_forecast(model,x0,cycle0)
        diff=float((initial-a0).abs().max())
    if not torch.equal(selected,x0) or diff>1e-4:
        raise AssertionError(f"PGIR trick initial A0 equality failed: {diff}")
    oof=None
    if cli.gain_supervision and not cli.evaluate_checkpoint:
        oof=build_gain_cache(cli,train_dataset,bank,selector,args,device,
                             cli.core_p4_dir,cli.gain_cache,digest)
        if oof["features"].shape[-1]!=5 or not oof["sample_count"]:
            raise AssertionError("empty/malformed OOF patch-gain supervision")
    write_json(out/"run_config.json",{"cli":vars(cli),"reference_sha256":digest,
        "train_only_threshold":threshold,"initial_a0_max_abs_difference":diff,
        "train_only_feature_stats":feature_stats,
        "checkpoint_selection":"VAL MSE","backbone_frozen":True,
        "oof_gain_labels":None if oof is None else {
            "core_oof":oof["core_oof"],"sample_count":oof["sample_count"],
            "stride":oof["stride"],"beneficial_fraction":oof["beneficial_fraction"],
            "raw_delta_mean":oof["raw_delta_mean"],"clipped_fraction":oof["clipped_fraction"]}})
    lr_feature=cli.lr_mask if cli.lr_feature is None else cli.lr_feature
    lr_head=cli.lr_mask if cli.lr_head is None else cli.lr_head
    optimizer=torch.optim.Adam([
        {"params":selector.net[:2].parameters(),"lr":lr_feature},
        {"params":selector.net[-1].parameters(),"lr":lr_head}],lr=cli.lr_mask)
    training=None if cli.evaluate_checkpoint else train_loader(train_dataset,args,cli.seed)
    if not (0 <= cli.ema_decay < 1):
        raise ValueError("ema_decay must be in [0,1)")
    ema_selector=copy.deepcopy(selector) if cli.ema_decay else None
    total_steps=(min(len(training),cli.max_batches) if cli.max_batches else len(training))*cli.epochs if training else 0
    warmup=max(1,math.ceil(0.1*total_steps))
    def schedule_factor(step):
        if step<warmup:return (step+1)/warmup
        progress=min(1.,(step-warmup+1)/max(1,total_steps-warmup))
        return 0.5*(1+math.cos(math.pi*progress))
    scheduler=(torch.optim.lr_scheduler.LambdaLR(optimizer,schedule_factor)
               if cli.schedule=="cosine_warmup" and training else None)
    history=[];best=math.inf;start=time.perf_counter()
    for epoch in range(0 if cli.evaluate_checkpoint else cli.epochs):
        selector.train();model.eval();total=count=0
        for step,(x,y,_,cycle) in enumerate(training):
            if cli.max_batches and step>=cli.max_batches:break
            x=x.float().to(device);y=y[:,-args.pred_len:].float().to(device);cycle=cycle.to(device)
            pred,_,info=forward(model,bank,selector,x,cycle,cli)
            pred_loss=(pred-y).square().mean()
            noninv=1-info["invariant"]
            replace_cost=(noninv*(1-info["probability"])).sum()/noninv.sum().clamp_min(1)
            probability=info["probability"]
            temporal=(probability[...,1:]-probability[...,:-1]).abs().mean() if probability.shape[-1]>1 else pred_loss*0
            loss=pred_loss+cli.lambda_replace*replace_cost+cli.lambda_temporal*temporal
            gain_loss=pred_loss.new_zeros(())
            if oof is not None:
                gen=torch.Generator(device="cpu").manual_seed(cli.seed+epoch*100003+step)
                indices=torch.randint(len(oof["target"]),(len(x),),generator=gen)
                features=oof["features"][indices].to(device)
                target=oof["target"][indices].to(device)
                eligible=1-oof["invariant"][indices].to(device)
                bs,ch,ns,_=features.shape
                features=selector.prepare_features(features)
                logits=selector.net(features.reshape(bs*ch,ns,5).transpose(1,2))
                logits=logits[:,0].reshape(bs,ch,ns)
                gain_loss=(F.smooth_l1_loss(logits,target,reduction="none")*eligible).sum()/eligible.sum().clamp_min(1)
                loss=loss+cli.lambda_gain*gain_loss
            if not torch.isfinite(loss):raise FloatingPointError("nonfinite trick loss")
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(),1.0)
            optimizer.step()
            if scheduler is not None:scheduler.step()
            if ema_selector is not None:
                with torch.no_grad():
                    for ema_param,param in zip(ema_selector.parameters(),selector.parameters()):
                        ema_param.lerp_(param,1-cli.ema_decay)
            total+=float(pred_loss.detach())*len(x);count+=len(x)
        val=evaluate(model,bank,selector,loaders["val"],args,device,cli)
        selected_kind="raw"
        if ema_selector is not None:
            ema_val=evaluate(model,bank,ema_selector,loaders["val"],args,device,cli)
            if ema_val["mse"]<val["mse"]:
                val=ema_val;selected_kind="ema"
        row={"epoch":epoch+1,"train_mse":total/max(count,1),
             "val_mse":val["mse"],"val_mae":val["mae"],
             "intervened":val["fraction_series_intervened"],
             "checkpoint_kind":selected_kind,
             "lr_feature":optimizer.param_groups[0]["lr"],
             "lr_head":optimizer.param_groups[1]["lr"]}
        history.append(row);write_json(out/"history.json",history)
        print(json.dumps(row),flush=True)
        if val["mse"]<best:
            best=val["mse"]
            selected=ema_selector if selected_kind=="ema" else selector
            torch.save({"selector":selected.state_dict(),"epoch":epoch+1,
                        "checkpoint_kind":selected_kind,
                        "reference_sha256":digest},out/"best.pt")
    saved=torch.load(cli.evaluate_checkpoint or out/"best.pt",map_location=device,weights_only=False)
    if saved.get("reference_sha256")!=digest:
        raise RuntimeError("PGIR checkpoint and A0 identity mismatch")
    selector.load_state_dict(saved["selector"])
    if cli.calibrate_threshold:
        choices=[-1e9,-0.2,-0.1,0.,0.1,0.2]
        val_by_threshold=[]
        for value in choices:
            selector.decision_threshold=value
            measure=evaluate(model,bank,selector,loaders["val"],args,device,cli)
            val_by_threshold.append({"threshold":value,"val_mse":measure["mse"],
                                     "fraction_series_intervened":measure["fraction_series_intervened"]})
        winner=min(val_by_threshold,key=lambda row:(row["val_mse"],abs(row["threshold"])))
        selector.decision_threshold=winner["threshold"]
        write_json(out/"threshold_validation.json",val_by_threshold)
    val=evaluate(model,bank,selector,loaders["val"],args,device,cli)
    test=evaluate(model,bank,selector,loaders["test"],args,device,cli)
    write_json(out/"result.json",{"mode":cli.mode,"best_epoch":saved["epoch"],
        "val":val,"test":test,"history":history,
        "checkpoint_kind":saved.get("checkpoint_kind","raw"),
        "decision_threshold":selector.decision_threshold,
        "lr_feature":lr_feature,"lr_head":lr_head,
        "training_seconds":time.perf_counter()-start,
        "checkpoint":str(cli.evaluate_checkpoint or out/"best.pt")})
    (out/"run_complete").touch()


if __name__=="__main__":main()
