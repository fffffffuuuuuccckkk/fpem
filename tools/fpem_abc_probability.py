"""C12: equal-capacity detached residual Flow condition comparisons."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from models.predictive_env.probability_heads import ProbabilityHeadBank
from tools.evaluate_fpem_probabilistic import probabilistic_metrics
from tools.fpem_abc_mechanism import write_rows


MODES=("base_condition","history_key","expert_disagreement",
       "retrieved_future","OOF_residual","all_fused","shuffled_fused")
FEATURES={
    "expert_disagreement":["C1_expert_var","C2_null_disagreement",
        "C3_pair_l1","C3_pair_l2","C4_near_disagreement","C4_far_disagreement"],
    "retrieved_future":["C5_neighbour_future_var","C5_neighbour_future_mad",
        "C5_neighbour_future_width","C5_neighbour_future_support"],
    "OOF_residual":["C6_neighbour_OOF_var","C6_neighbour_OOF_mad",
        "C6_neighbour_OOF_width","C6_neighbour_OOF_support"],
}
FEATURES["all_fused"]=FEATURES["expert_disagreement"]+FEATURES[
    "retrieved_future"]+FEATURES["OOF_residual"]
FEATURES["shuffled_fused"]=FEATURES["all_fused"]
DIM=len(FEATURES["all_fused"])


def to_raw(data, normal):
    return (normal*data["scale"].transpose(1,2)+
            data["history_mean"].transpose(1,2)).transpose(1,2).float()


def make_data(data, point, features, oof_mean, oof_std):
    cols=FEATURES["all_fused"]
    values=np.stack([features[name] for name in cols],-1)
    values=np.nan_to_num((values-oof_mean)/oof_std,nan=0,posinf=0,neginf=0)
    return {"future":data["key"].float(),"y_inv":to_raw(data,point),
            "truth":to_raw(data,data["truth"]),"scale":data["scale"].float(),
            "features":torch.from_numpy(values).float(),"ids":data["ids"],
            "risk_score":torch.from_numpy(values[...,0]+values[...,-4]).float().mean(-1)}


def packet(record, ids, device, mode, train=False):
    future=record["future"][ids].to(device)
    point=record["y_inv"][ids].to(device).detach()
    scale=record["scale"][ids].to(device).detach()
    truth=record["truth"][ids].to(device)
    extras=record["features"][ids].to(device).clone()
    if mode=="shuffled_fused" and train:
        # Only TRAIN features are permuted; temporal blocks stay intact.
        n=len(extras)
        if n>1: extras=extras.roll(1,0)
    elif mode in ("base_condition","history_key"):
        extras.zero_()
    else:
        allowed=set(FEATURES[mode]); mask=torch.tensor(
            [name in allowed for name in FEATURES["all_fused"]],device=device)
        extras*=mask
    if mode=="base_condition": future=torch.zeros_like(future)
    patches=math.ceil(point.shape[1]/16)
    future=future.unsqueeze(2).expand(-1,-1,patches,-1)
    return future,point,truth,scale,extras.unsqueeze(2).expand(-1,-1,patches,-1)


class ConditionedResidualFlow(nn.Module):
    def __init__(self,horizon):
        super().__init__()
        self.bank=ProbabilityHeadBank(64,horizon,patch_len=16,heads=("residual_flow",))
        self.extra=nn.Linear(DIM,64)
        nn.init.zeros_(self.extra.weight); nn.init.zeros_(self.extra.bias)

    def condition(self,future,point,scale,extras):
        return self.bank.condition(future,point,scale)+self.extra(extras)


def score(model,record,device,mode,seed=2021,sample_scale=1.,stride=1,
          risk_scale=None,high_threshold=None):
    model.eval(); count=0; totals={}; high=[]; high_hits=0.; high_count=0
    generator=torch.Generator(device=device).manual_seed(seed+401)
    indices=torch.arange(0,len(record["ids"]),stride)
    for ids in indices.split(4):
        future,point,truth,scale,extra=packet(record,ids,device,mode)
        with torch.no_grad():
            condition=model.condition(future,point,scale,extra)
            sample=model.bank.sample_head("residual_flow",condition,future,
                  point,scale,100,generator,chunk_size=8,steps=12)["y_samples"]
            scale_effect=sample_scale
            if risk_scale is not None:
                scale_effect=sample_scale*risk_scale[ids].to(device)[:,None,None]
            sample=point[:,None]+scale_effect*(sample-point[:,None])
            result=probabilistic_metrics(sample,truth,point)
        for key in ("CRPS","ProbTS_CRPS","WIS","PICP_95","MPIW_95",
                    "inv_MSE","inv_MAE","mean_MSE","mean_MAE",
                    "sample_std_mean","sample_diversity"):
            totals[key]=totals.get(key,0.)+float(result[key])*len(ids)
        lo=torch.quantile(sample,.025,dim=1); hi=torch.quantile(sample,.975,dim=1)
        covered=(truth>=lo)&(truth<=hi)
        if high_threshold is not None:
            risk=record["risk_score"][ids].to(device)>=high_threshold
            high_hits+=float(covered[risk].float().sum())
            high_count+=int(covered[risk].numel())
        for j,part in enumerate(torch.tensor_split(torch.arange(truth.shape[1],device=device),4)):
            key=f"PICP_95_bin{j+1}"
            totals[key]=totals.get(key,0.)+float(covered[:,part].float().mean())*len(ids)
        high.append({"trajectory_coverage":float(((truth>=lo)&(truth<=hi)).all((1,2)).float().mean()),
                     "count":len(ids),"error":float((truth-point).square().mean())})
        count+=len(ids)
    output={key:value/count for key,value in totals.items()}
    output["trajectory_coverage_95"] = sum(x["trajectory_coverage"]*x["count"] for x in high)/count
    output["high_risk_PICP_95"]=high_hits/high_count if high_count else float("nan")
    output["evaluated_windows"]=count
    return output


def train_flow(record,mode,device,seed,horizon,epochs=8):
    torch.manual_seed(seed)
    model=ConditionedResidualFlow(horizon).to(device)
    optim=torch.optim.Adam(model.parameters(),lr=1e-4)
    trace=[]
    for epoch in range(epochs):
        model.train(); total=0.; n=0
        order=torch.randperm(len(record["ids"]),generator=torch.Generator().manual_seed(seed+epoch))
        for ids in order.split(8):
            future,point,truth,scale,extra=packet(record,ids,device,mode,train=True)
            condition=model.condition(future,point,scale,extra)
            loss,_=model.bank.training_losses(condition,future,point,truth,scale)["residual_flow"]
            optim.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            optim.step(); total+=float(loss.detach())*len(ids); n+=len(ids)
        trace.append({"epoch":epoch+1,"flow_loss":total/max(n,1)})
        print("C12 EPOCH",mode,epoch+1,trace[-1],flush=True)
    return model,trace


def score_scale_control(train,cal,test,distribution):
    # OOF TRAIN residuals only.  Gaussian and Student-t use the same
    # history-conditional Ridge error-scale head, no TEST target fitting.
    from sklearn.linear_model import Ridge
    x=train["features"].reshape(-1,DIM).numpy()
    y=(train["truth"]-train["y_inv"])/train["scale"]
    target=y.square().mean(1).reshape(-1).numpy()
    mean=x.mean(0); std=x.std(0).clip(.01)
    model=Ridge(alpha=10.).fit((x-mean)/std,np.log(target+1e-5))
    rows=[]
    for record in (cal,test):
        features=record["features"].reshape(-1,DIM).numpy()
        variance=np.exp(model.predict((features-mean)/std)).reshape(
            len(record["ids"]),-1).clip(1e-5,100.)
        rows.append(torch.from_numpy(variance).float())
    return rows


def evaluate_scale_control(record,variance,distribution,stride=16):
    generator=torch.Generator().manual_seed(2021+401)
    total={};count=0
    for ids in torch.arange(0,len(record["ids"]),stride).split(8):
        point=record["y_inv"][ids]
        truth=record["truth"][ids]
        n,t,c=point.shape
        noise=torch.randn(n,100,t,c,generator=generator)
        if distribution=="student_t":
            chi=torch.randn(n,100,t,c,4,generator=generator).square().sum(-1)
            noise=noise/torch.sqrt(chi/4)
        sigma=variance[ids].sqrt()*record["scale"][ids,0]
        samples=point[:,None]+noise*sigma[:,None,None]
        scores=probabilistic_metrics(samples,truth,point)
        for key in ("CRPS","WIS","PICP_95","MPIW_95","inv_MSE","mean_MSE"):
            total[key]=total.get(key,0.)+float(scores[key])*n
        lo=samples.quantile(.025,dim=1);hi=samples.quantile(.975,dim=1)
        total["trajectory_coverage_95"]=total.get("trajectory_coverage_95",0.)+float(
            ((truth>=lo)&(truth<=hi)).all((1,2)).float().sum())
        count+=n
    return {key:value/count for key,value in total.items()}|{"evaluated_windows":count}


def run_c12(data,meta,cache,output):
    if (output/"C12.complete").exists(): return
    oof=cache["oof"]
    columns=FEATURES["all_fused"]
    vals=np.stack([cache["features_oof"][key] for key in columns],-1)
    mu=np.nan_to_num(vals.mean((0,1)),nan=0)
    std=np.nan_to_num(vals.std((0,1)),nan=1).clip(.01)
    point_oof=oof["prediction"]
    point_test=cache["compact_test"]["full"]
    training=make_data(oof["data"],point_oof,cache["features_oof"],mu,std)
    testing=make_data(data["test"],point_test,cache["features_test"],mu,std)
    split=max(1,int(len(training["ids"])*.75))
    fit={key:value[:split] for key,value in training.items()}
    cal={key:value[split:] for key,value in training.items()}
    high_threshold=float(torch.quantile(fit["risk_score"],.8))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results=[]
    for mode in MODES:
        path=output/f"C12_{mode}.json"
        if path.exists():
            results.append(json.loads(path.read_text()));continue
        model,trace=train_flow(fit,mode,device,2021,meta["pred_len"])
        calibration=[(scale,score(model,cal,device,mode,seed=2021,sample_scale=scale,
                                  stride=2,high_threshold=high_threshold)) for scale in (.8,1.,1.2)]
        best=min(calibration,key=lambda x:x[1]["CRPS"])[0]
        test=score(model,testing,device,mode,seed=2021,sample_scale=best,stride=16,
                   high_threshold=high_threshold)
        row={"experiment":"C12","mode":mode,"test":test,
             "calibrated_scale":best,"calibration":calibration,
             "train_trace":trace,"model_parameters":sum(p.numel() for p in model.parameters()),
             "point_anchor_frozen":True,"OOF_training":True,
             "test_selection":"exploratory_TEST_CRPS","test_stride":16}
        path.write_text(json.dumps(row,indent=2,ensure_ascii=False))
        torch.save({"state_dict":model.state_dict(),"mode":mode,
                    "point_source":meta["source"],"feature_mean":mu,
                    "feature_std":std},output/f"C12_{mode}.pt")
        results.append(row)
        print("C12 COMPLETE",mode,test,flush=True)
    for distribution in ("gaussian","student_t"):
        cal_var,test_var=score_scale_control(fit,cal,testing,distribution)
        metrics=evaluate_scale_control(testing,test_var,distribution)
        results.append({"experiment":"C12","mode":distribution+"_risk_scale",
                        "test":metrics,"point_anchor_frozen":True,
                        "OOF_training":True,"test_stride":16})
    # Lightweight interval-width calibration from the same OOF risk signal;
    # it changes only sample spread, never the frozen point anchor.
    base=next(x for x in results if x["mode"]=="base_condition")
    base_model=ConditionedResidualFlow(meta["pred_len"]).to(device)
    base_payload=torch.load(output/"C12_base_condition.pt",map_location=device,weights_only=False)
    base_model.load_state_dict(base_payload["state_dict"])
    _,test_variance=score_scale_control(fit,cal,testing,"gaussian")
    reference=float(((fit["truth"]-fit["y_inv"])/fit["scale"]).square().mean())
    risk_scale=(test_variance/reference).sqrt().clamp(.5,2.)
    calibrated=score(base_model,testing,device,"base_condition",seed=2021,
                     sample_scale=base["calibrated_scale"],stride=16,
                     risk_scale=risk_scale,high_threshold=high_threshold)
    results.append({"experiment":"C12","mode":"interval_calibration_control",
                    "test":calibrated,"note":"baseline Flow with OOF-fitted risk-dependent width",
                    "calibrated_scale":base["calibrated_scale"],
                    "point_anchor_frozen":True})
    flat=[]
    for row in results:
        flat.append({"experiment":"C12","mode":row["mode"],
                     **{k:v for k,v in row["test"].items()},
                     "backbone":meta["backbone"],"dataset":meta["dataset"],
                     "pred_len":meta["pred_len"],"test_selection":"exploratory"})
    write_rows(output/"C_probability_results.csv",flat)
    write_rows(output/"C_probability_ranking.csv",sorted(flat,key=lambda x:x["CRPS"]))
    (output/"C12.complete").write_text("complete\n")
