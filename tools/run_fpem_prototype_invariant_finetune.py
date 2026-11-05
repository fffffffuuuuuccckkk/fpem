"""Stage J2: live-H partial/full fine-tuning with a frozen R3 router."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader,Subset

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.predictive_env.dual_pattern_memory.rare_learning import RarePatternLearner
from models.predictive_env.prototype_guided_invariant import PrototypeGuidedSharedHead
from models.predictive_env.prototype_invariance_losses import (
    prototype_risk_penalty,shared_optimum_penalty,conditional_agreement)
from tools.run_fpem_dppm_g1_r3_full import history_key
from models.predictive_env.prototype_shared_subspace import pairwise_prototype_error_loss
from models.predictive_env.prototype_mapping_consensus import predict_mapping
from tools.run_fpem_prototype_invariant_search import method_spec,build_head,write_json
from tools.run_fpem_shared_factorization_g1r3 import load_experiment


def parser():
    p=argparse.ArgumentParser()
    p.add_argument("--base_run_config",required=True)
    p.add_argument("--shared_root",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--gpu",type=int,default=0)
    p.add_argument("--seed",type=int,default=2021)
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--train_stride",type=int,default=16)
    p.add_argument("--val_stride",type=int,default=4)
    p.add_argument("--test_stride",type=int,default=1)
    p.add_argument("--allow_legacy_reference_signature",action="store_true")
    return p.parse_args()


def select_shared(root):
    names=("p01","p02","risk_var","risk_dro","shared_grad_norm",
           "shared_grad_variance","shared_irm","conditional","subspace",
           "ref_global","ref_prototype","pairwise","shared_private",
           "shared_private_irm","adversarial_weak","adversarial_strong",
           "map_projection","map_shared_irm")
    rows=[]
    for name in names:
        path=root/name/"metrics.json"
        if path.exists() and (root/name/"model.pt").exists():
            row=json.loads(path.read_text())
            rows.append((row["val"]["mse"],name))
    if not rows:
        raise FileNotFoundError("no trained J1 shared heads")
    return min(rows)[1]


def encoder_modules(model):
    return [m for m in (model.patch_embedding,model.inverted_embedding,
                        model.encoder,model.modern_tcn_backbone) if m is not None]


def enable_backbone(model,mode):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if mode=="frozen":
        return []
    if mode=="full":
        modules=encoder_modules(model)
    elif mode=="partial":
        if model.encoder is not None and hasattr(model.encoder,"attn_layers"):
            modules=[model.encoder.attn_layers[-1]]
        elif model.modern_tcn_backbone is not None:
            modules=[model.modern_tcn_backbone.stages[-1]]
        else:
            raise ValueError("no identifiable last backbone block")
    else:
        raise ValueError(mode)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


def make_head(root,method,model,args,device):
    rank,reference,pretrain,residual=method_spec(method)
    state=torch.load(root/method/"model.pt",map_location="cpu",weights_only=False)
    # Saved head state determines token count exactly, including ModernTCN.
    key="forecaster.head.weight"
    tokens=state[key].shape[1]//(rank or model.d_model)
    head=build_head(method,tokens,model.d_model,args.pred_len)
    head.load_state_dict(state)
    return head.to(device)


@torch.no_grad()
def evaluate(model,head,learner,loader,device):
    model.eval();head.eval()
    total_sq=total_abs=elements=0
    for x,y,_,cycle in loader:
        x,y,cycle=x.float().to(device),y.float().to(device),cycle.to(device)
        h,mean,std=model.encode(x,cycle)
        q=learner.query(history_key(x.cpu(),64))["weight"].to(device)
        base=model.frozen_h_head(h,cycle)
        pred=head(h,q,base).transpose(1,2)*std+mean
        err=pred-y[:,-pred.shape[1]:]
        total_sq+=float(err.square().sum());total_abs+=float(err.abs().sum())
        elements+=err.numel()
    return {"mse":total_sq/elements,"mae":total_abs/elements}


def run_one(cli,method,mode):
    out=Path(cli.output)/mode
    if (out/"complete").exists():
        return json.loads((out/"metrics.json").read_text())
    out.mkdir(parents=True,exist_ok=True)
    args,model,loaders,device,sha=load_experiment(cli)
    learner=RarePatternLearner.from_export(torch.load(Path(cli.shared_root)/"r3.pt",
                                             map_location="cpu",weights_only=False))
    head=make_head(Path(cli.shared_root),method,model,args,device)
    mapping_state=None
    if method in ("map_projection","map_shared_irm"):
        mapping_state=torch.load(Path(cli.shared_root)/"map_consensus/model.pt",
                                 map_location="cpu",weights_only=False)
    parameters=enable_backbone(model,mode)
    if mode=="frozen":
        return {"mode":mode,"skipped":"J1 result is the frozen-head control"}
    optimizer=torch.optim.Adam([{"params":list(head.parameters()),"lr":3e-4},
                                 {"params":parameters,"lr":2e-5}])
    generator=torch.Generator().manual_seed(cli.seed)
    best=None;trace=[];started=time.monotonic()
    train_data=loaders["train"].dataset
    train_indices=torch.arange(0,len(train_data),cli.train_stride)
    val_loader=DataLoader(Subset(loaders["val"].dataset,
        range(0,len(loaders["val"].dataset),cli.val_stride)),
        batch_size=args.batch_size,num_workers=0)
    test_loader=DataLoader(Subset(loaders["test"].dataset,
        range(0,len(loaders["test"].dataset),cli.test_stride)),
        batch_size=args.batch_size,num_workers=0)
    for epoch in range(cli.epochs):
        model.train();head.train()
        order=train_indices[torch.randperm(len(train_indices),generator=generator)]
        for selected in order.split(args.batch_size):
            items=[train_data[int(i)] for i in selected]
            x=torch.stack([torch.as_tensor(item[0]) for item in items]).float().to(device)
            y=torch.stack([torch.as_tensor(item[1]) for item in items]).float().to(device)
            cycle=torch.tensor([int(item[3]) for item in items],device=device)
            h,mean,std=model.encode(x,cycle)
            q=learner.query(history_key(x.detach().cpu(),64))["weight"].to(device).detach()
            base=model.frozen_h_head(h,cycle).detach()
            pred=head(h,q,base)
            truth=((y[:,-args.pred_len:]-mean)/std).transpose(1,2)
            loss=(pred-truth).square().mean()
            if method=="risk_var":
                regular,_=prototype_risk_penalty(pred,truth,q,"variance")
                loss=loss+.01*regular
            elif method=="risk_dro":
                regular,_=prototype_risk_penalty(pred,truth,q,"dro")
                loss=loss+.01*regular
            elif method in ("shared_grad_norm","shared_grad_variance","shared_irm"):
                variant={"shared_grad_norm":"norm","shared_grad_variance":"variance",
                         "shared_irm":"irm"}[method]
                regular,_=shared_optimum_penalty(pred,truth,q,head.head_weight,variant)
                loss=loss+.01*regular
            elif method=="conditional":
                loss=loss+.01*conditional_agreement(pred,truth,q,h.mean(2))
            elif method=="pairwise":
                loss=loss+.01*pairwise_prototype_error_loss(
                    pred,truth,q,selected.to(device),args.seq_len+args.pred_len)
            elif method.startswith("shared_private"):
                loss=loss+.01*head.split_penalty(h)
                if method=="shared_private_irm":
                    regular,_=shared_optimum_penalty(pred,truth,q,head.head_weight,"irm")
                    loss=loss+.01*regular
            elif method.startswith("adversarial_"):
                strength=.1 if method.endswith("weak") else 1.
                logits=head.adversarial_logits(h,strength)
                has_route=q.sum(-1)>1e-8
                norm_q=q/q.sum(-1,keepdim=True).clamp_min(1e-8)
                ce=-(norm_q*logits.log_softmax(-1)).sum(-1)
                if has_route.any():loss=loss+.01*ce[has_route].mean()
            elif method in ("map_projection","map_shared_irm"):
                with torch.no_grad():
                    projected=h.detach().cpu() @ mapping_state["basis"]
                    teacher=predict_mapping(projected,mapping_state["fitted"],"consensus").to(device)
                loss=loss+.01*(pred-teacher).square().mean()
                if method=="map_shared_irm":
                    regular,_=shared_optimum_penalty(pred,truth,q,head.head_weight,"irm")
                    loss=loss+.01*regular
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite fine-tuning objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters+list(head.parameters()),1.)
            optimizer.step()
        val=evaluate(model,head,learner,val_loader,device)
        trace.append({"epoch":epoch+1,**val})
        if best is None or val["mse"]<best["val"]["mse"]:
            best={"epoch":epoch+1,"val":val,
                  "model":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                  "head":{k:v.detach().cpu().clone() for k,v in head.state_dict().items()}}
        print(mode,epoch+1,val,flush=True)
    model.load_state_dict(best.pop("model"));head.load_state_dict(best.pop("head"))
    test=evaluate(model,head,learner,test_loader,device)
    result={"method":method,"mode":mode,"val":best["val"],"test":test,
            "best_epoch":best["epoch"],"trainable_backbone_parameters":sum(p.numel() for p in parameters),
            "trainable_head_parameters":sum(p.numel() for p in head.parameters()),
            "elapsed_seconds":time.monotonic()-started,"reference_sha256":sha,
            "selection":"VAL checkpoint; exploratory TEST report", "trace":trace}
    torch.save({"backbone_state":model.state_dict(),"head_state":head.state_dict(),
                "reference_sha256":sha},out/"model.pt")
    write_json(out/"metrics.json",result)
    (out/"complete").write_text("complete\n")
    return result


def main():
    cli=parser();root=Path(cli.shared_root)
    method=select_shared(root)
    rows=[]
    for mode in ("partial","full"):
        row=run_one(cli,method,mode)
        rows.append(row)
    write_json(Path(cli.output)/"selection.json",{"J1_best":method,"rows":rows})


if __name__=="__main__":main()
