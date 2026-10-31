"""C1-C12 unresolved-future diagnostics with causal point-model OOF targets."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from models.predictive_env.dual_pattern_memory.forecast_fusion import prototype_segment_reliability
from models.predictive_env.dual_pattern_memory.rare_learning import RarePatternLearner
from models.predictive_env.probability_heads import ProbabilityHeadBank
from tools.evaluate_fpem_probabilistic import probabilistic_metrics
from tools.fpem_abc_mechanism import (
    chronological_blocks, gain_row, nearest, pearson, residual_correction,
    sample_error, simple_key_drift, write_rows,
)
from tools.run_fpem_dppm_g1_r3_full import residual_ridge, ridge_query, subset


def original_units(data, normalized):
    scale = data["scale"].transpose(1, 2)
    return (normalized * scale + data["history_mean"].transpose(1, 2)).transpose(1, 2)


def anchor(data, state):
    return data["base"] + state["g1_scale"]*ridge_query(data["key"],state["g1_weight"])


def causal_point_oof(train, meta, saved, output):
    cache=output/"C_point_OOF.pt"
    if cache.exists():
        return torch.load(cache,map_location="cpu",weights_only=False)
    from tools.run_fpem_abc_mechanism import predict_custom
    blocks=chronological_blocks(train["ids"],4)
    source_model=RarePatternLearner.from_export(saved["rare"])
    config=saved["rare"].get("optimizer_config",{})
    rows=[]; records=[]; states=[]
    for fold in (2,3):
        fit=torch.cat(blocks[:fold-1]); cal=blocks[fold-1]; hold=blocks[fold]
        fit=fit[train["ids"][fit]+meta["span"]<=train["ids"][cal].min()]
        cal=cal[train["ids"][cal]+meta["span"]<=train["ids"][hold].min()]
        if min(len(fit),len(cal),len(hold))<8:
            raise ValueError(f"strict forward OOF fold {fold} has insufficient support")
        if int(train["ids"][fit].max()+meta["span"])>int(train["ids"][cal].min()):
            raise AssertionError("OOF fit future overlaps calibration history")
        if int(train["ids"][cal].max()+meta["span"])>int(train["ids"][hold].min()):
            raise AssertionError("OOF calibration future overlaps evaluation history")
        g1=residual_ridge(train["key"][fit],
                          (train["truth"]-train["base"])[fit],.1)
        state={"g1_weight":g1,"g1_scale":saved["g1_scale"],
               "offset":saved["offset"]}
        reference=residual_ridge(train["key"][fit],train["truth"][fit],100.)
        learner=RarePatternLearner("R3",patterns=saved["rare"]["patterns"],
                                   rank=saved["rare"]["rank"],seed=2021,
                                   epochs=int(config.get("epochs",10)),optimizer_config=config)
        learner.fit(train["key"][fit],train["truth"][fit],{
            "reference_map":reference,"starts":train["ids"][fit],
            "span":meta["span"],"device":"cpu"})
        cal_data=subset(train,torch.isin(torch.arange(len(train["ids"])),cal))
        cal_anchor=anchor(cal_data,state)
        cal_query=learner.query(cal_data["key"])
        reliability,_=prototype_segment_reliability(cal_anchor,cal_query["candidate"],
                              cal_data["truth"],cal_query["weight"],8)
        state["reliability"]=reliability
        hold_data=subset(train,torch.isin(torch.arange(len(train["ids"])),hold))
        prediction=predict_custom(hold_data,state,learner)
        compact=residual_correction(hold_data,{"weight":g1,"scale":saved["g1_scale"]},
                                    learner,reliability,saved["offset"])
        records.append({"data":hold_data,"prediction":prediction,"compact":compact})
        states.append({"fold":fold,"model":learner.export(),"g1_weight":g1,
                       "reliability":reliability,"fit_future_end":int(
                           train["ids"][fit].max()+meta["span"]),
                       "cal_future_end":int(train["ids"][cal].max()+meta["span"]),
                       "hold_start":int(train["ids"][hold].min())})
        rows.append({"fold":fold,"fit_count":len(fit),"cal_count":len(cal),
                     "hold_count":len(hold),"fit_future_end":states[-1]["fit_future_end"],
                     "cal_future_end":states[-1]["cal_future_end"],
                     "hold_history_start":states[-1]["hold_start"],
                     "point_mse":float(sample_error(hold_data,prediction).mean()),
                     "a0_frozen_TRAIN_fit_caveat":True})
    write_rows(output/"C_OOF_audit.csv",rows)
    combined={"data":{key:torch.cat([item["data"][key] for item in records])
                      for key in records[0]["data"]},
              "prediction":torch.cat([item["prediction"] for item in records]),
              "compact":{key:torch.cat([item["compact"][key] for item in records])
                         for key in records[0]["compact"]},
              "states":states}
    torch.save(combined,cache)
    return combined


def retrieved_stats(source, query, source_ids, query_ids, span, k=8):
    channels=source["key"].shape[1]
    flat_source_ids=source_ids.repeat_interleave(channels)
    flat_query_ids=query_ids.repeat_interleave(channels)
    ids,dist,valid=nearest(source["key"],query["key"],k=k,
        source_ids=flat_source_ids,query_ids=flat_query_ids,span=span)
    if (flat_source_ids[ids] + span > flat_query_ids[:,None])[valid].any():
        raise AssertionError("retrieval violates source_future_end <= query_history_start")
    flat_future=source["truth"].reshape(-1,source["truth"].shape[-1])
    result_parts=[]
    for begin in range(0,len(ids),128):
        stop=min(len(ids),begin+128)
        ix=ids[begin:stop]; good=valid[begin:stop]
        fut=flat_future[ix]
        count=good.sum(1).clamp_min(1)
        masked=fut.masked_fill(~good[...,None],float("inf"))
        ordered=masked.sort(1).values
        horizon=fut.shape[-1]
        def quantile(values,p):
            index=(count.float()-1)*p
            lo=index.floor().long();hi=index.ceil().long()
            x=values.gather(1,lo[:,None,None].expand(-1,1,horizon))[:,0]
            y=values.gather(1,hi[:,None,None].expand(-1,1,horizon))[:,0]
            return x+(y-x)*(index-lo)[:,None]
        center=quantile(ordered,.5)
        q10=quantile(ordered,.1);q90=quantile(ordered,.9)
        deviations=(fut-center[:,None]).abs().masked_fill(~good[...,None],float("inf"))
        mad=quantile(deviations.sort(1).values,.5).mean(-1)
        avg=fut.masked_fill(~good[...,None],0).sum(1)/count[:,None]
        variance=((fut-avg[:,None]).square()*good[...,None]).sum(1)/count[:,None]
        event=flat_source_ids[ix]//span
        independent=torch.tensor([len(event[j,good[j]].unique()) for j in range(len(ix))])
        result=torch.stack([variance.mean(-1),mad,(q90-q10).mean(-1),
            q90.mean(-1),q10.mean(-1),independent.float(),
            torch.nan_to_num(dist[begin:stop,0],nan=0,posinf=0)],1)
        result[~good.any(1)]=0
        result_parts.append(result)
    result=torch.cat(result_parts).numpy()
    return result.reshape(len(query["key"]),channels,-1),ids,valid


def oof_residual_retrieval(oof, query, query_ids, span):
    source=oof["data"]
    correction=oof["prediction"]
    residual=source["truth"]-correction
    copy={**source,"truth":residual}
    return retrieved_stats(copy,query,source["ids"],query_ids,span)


def risk_features(data, compact, train, oof, meta, test=False):
    n,c=data["key"].shape[:2]
    ids=data["ids"] + (meta["test_start_absolute"] if test else 0)
    train_future,neighbour_ix,valid=retrieved_stats(train,data,train["ids"],ids,meta["span"])
    residual_future,_,_=oof_residual_retrieval(oof,data,ids,meta["span"])
    # C1-C4, C9 are strictly history / fitted-model functions.
    feature={
        "C1_expert_var":compact["expert_variance"].mean(-1).numpy(),
        "C2_null_disagreement":compact["null_variance"].mean(-1).numpy(),
        "C3_pair_l1":compact["pair_l1"].mean(-1).numpy(),
        "C3_pair_l2":compact["pair_l2"].mean(-1).numpy(),
        "C3_pair_max":compact["pair_max"].amax(-1).numpy(),
        "C3_pair_iqr":compact["pair_iqr"].mean(-1).numpy(),
        "C4_near_disagreement":compact["seg_disagreement"][...,:2].mean(-1).numpy(),
        "C4_far_disagreement":compact["seg_disagreement"][...,-2:].mean(-1).numpy(),
        "C4_low_frequency":compact["frequency_low_disagreement"].numpy(),
        "C4_high_frequency":compact["frequency_high_disagreement"].numpy(),
        "C9_novelty":simple_key_drift(train["key"],data["key"])[:,None].repeat(c,axis=1),
        "C9_null":compact["null"].numpy(),
        "C9_entropy":compact["route_entropy"].numpy(),
        "C9_support":compact["effective_support"].numpy(),
        "C9_reliability":compact["reliability"].numpy(),
    }
    for j,name in enumerate(("var","mad","width","positive_tail","negative_tail","support","distance")):
        feature[f"C5_neighbour_future_{name}"]=train_future[...,j]
        feature[f"C6_neighbour_OOF_{name}"]=residual_future[...,j]
    feature["C9_nearest_distance"]=train_future[...,-1]
    feature["C9_independent_support"]=train_future[...,5]
    return feature,train_future,residual_future


def group_auc(scores,labels):
    from sklearn.metrics import roc_auc_score,average_precision_score
    if len(np.unique(labels))<2:
        return float("nan"),float("nan")
    return float(roc_auc_score(labels,scores)),float(average_precision_score(labels,scores))


def ause(scores,error):
    scores=np.asarray(scores).reshape(-1); error=np.asarray(error).reshape(-1)
    if len(scores)<10:return float("nan")
    points=np.linspace(.05,1,20)
    predicted=np.argsort(scores)
    oracle=np.argsort(error)
    difference=[error[predicted[:max(1,int(len(error)*p))]].mean()-
                error[oracle[:max(1,int(len(error)*p))]].mean() for p in points]
    return float(np.trapz(difference,points))


def risk_fit_predict(x_train,y_train,x_test,kind):
    x_train=np.nan_to_num(x_train,nan=0,posinf=0,neginf=0)
    x_test=np.nan_to_num(x_test,nan=0,posinf=0,neginf=0)
    mu=x_train.mean(0); std=x_train.std(0).clip(.01)
    x=(x_train-mu)/std; xt=(x_test-mu)/std
    if kind=="ridge":
        from sklearn.linear_model import Ridge
        model=Ridge(alpha=10.).fit(x,y_train)
    elif kind=="mlp":
        from sklearn.neural_network import MLPRegressor
        model=MLPRegressor(hidden_layer_sizes=(16,),max_iter=120,
                           random_state=2021,early_stopping=False).fit(x,y_train)
    elif kind=="lightgbm":
        import lightgbm as lgb
        model=lgb.LGBMRegressor(n_estimators=80,num_leaves=7,learning_rate=.05,
                                 verbosity=-1,random_state=2021).fit(x,y_train)
    else: raise ValueError(kind)
    return model.predict(xt)


def c_risk_train(data,meta,saved,learner,output):
    train,test=data["train"],data["test"]
    oof=causal_point_oof(train,meta,saved,output)
    print("C OOF READY",len(oof["data"]["key"]),flush=True)
    compact_test=residual_correction(test,{"weight":saved["g1_weight"],
        "scale":saved["g1_scale"]},learner,saved["reliability"],saved["offset"])
    features_test,future_test,residual_test=risk_features(
        test,compact_test,train,oof,meta,test=True)
    features_oof,future_oof,residual_oof=risk_features(
        oof["data"],oof["compact"],train,oof,meta,test=False)
    target_test=((test["truth"]-compact_test["full"])*test["scale"].transpose(1,2)).square().mean(-1).numpy()
    target_oof=((oof["data"]["truth"]-oof["prediction"])*oof["data"]["scale"].transpose(1,2)).square().mean(-1).numpy()
    event_cut=float(np.quantile(target_oof,.9))
    rows=[]
    names=list(features_test)
    for name in names:
        v=features_test[name]
        auc,ap=group_auc(v.reshape(-1),(target_test>event_cut).reshape(-1))
        rows.append({"experiment":name.split("_")[0],"feature":name,
                     "kind":"single_signal","test_corr_squared_error":pearson(v,target_test),
                     "test_AUSE":ause(v,target_test),"test_AUROC":auc,"test_AUPRC":ap,
                     "event_threshold_from_TRAIN_OOF":event_cut})
    simple=[name for name in names if name.startswith("C9_")]
    disagreement=[name for name in names if name.startswith(("C1_","C2_","C3_","C4_","C5_","C6_"))]
    groups={"simple_history":simple,"expert_disagreement":disagreement[:8],
            "retrieved_future":[x for x in names if x.startswith("C5_")],
            "retrieved_OOF":[x for x in names if x.startswith("C6_")],
            "all_conditions":names}
    for group,fields in groups.items():
        x_train=np.stack([features_oof[f] for f in fields],-1).reshape(-1,len(fields))
        x_test=np.stack([features_test[f] for f in fields],-1).reshape(-1,len(fields))
        for target_kind in ("squared","absolute","high_event"):
            y=target_oof.reshape(-1)
            if target_kind=="absolute": y=np.sqrt(y)
            elif target_kind=="high_event": y=(y>event_cut).astype(float)
            truth=target_test.reshape(-1)
            if target_kind=="absolute": truth=np.sqrt(truth)
            elif target_kind=="high_event": truth=(truth>event_cut).astype(float)
            for model in ("ridge","mlp","lightgbm"):
                try:
                    pred=risk_fit_predict(x_train,y,x_test,model)
                    auc,ap=group_auc(pred,(target_test>event_cut).reshape(-1))
                    rows.append({"experiment":"C10","feature":group,"kind":target_kind,
                                 "model":model,"test_AUSE":ause(pred,truth),
                                 "test_AUROC":auc,"test_AUPRC":ap,
                                 "test_MSE":float(np.mean((pred-truth)**2)),
                                 "event_threshold_from_TRAIN_OOF":event_cut})
                except (ImportError,ValueError) as error:
                    rows.append({"experiment":"C10","feature":group,"kind":target_kind,
                                 "model":model,"status":"unavailable","reason":str(error)})
    print("C RISK MODELS READY",len(rows),flush=True)
    # C11: shuffle disagreement only within temporal blocks and novelty bins.
    rng=np.random.default_rng(2021)
    for repeat in range(20):
        fields=groups["all_conditions"]
        shuffled={k:features_oof[k].copy() for k in fields}
        n=len(oof["data"]["key"])
        novelty=features_oof["C9_novelty"].mean(1)
        cut=np.quantile(novelty,[.33,.67])
        bins=np.digitize(novelty,cut)
        block=np.floor(np.arange(n)*4/max(1,n)).astype(int)
        for label in np.unique(4*bins+block):
            ids=np.where(4*bins+block==label)[0]
            if len(ids)<2: continue
            perm=rng.permutation(ids)
            for key in disagreement:
                shuffled[key][ids]=shuffled[key][perm]
        x_train=np.stack([shuffled[f] for f in fields],-1).reshape(-1,len(fields))
        x_test=np.stack([features_test[f] for f in fields],-1).reshape(-1,len(fields))
        pred=risk_fit_predict(x_train,target_oof.reshape(-1),x_test,"ridge")
        rows.append({"experiment":"C11","feature":"within_block_novelty_shuffle",
                     "kind":"squared","model":"ridge","repeat":repeat,
                     "test_AUSE":ause(pred,target_test),
                     "test_MSE":float(np.mean((pred-target_test.reshape(-1))**2))})
    print("C SHUFFLE READY",flush=True)
    # C7: do not force GMM fit on <8 independent neighbours.
    try:
        from sklearn.mixture import GaussianMixture
        for label,source in (("future",train),("OOF",oof["data"])):
            scores=[]
            subset_ids=np.linspace(0,len(test["key"])-1,min(64,len(test["key"]))).astype(int)
            source_future=source["truth"] if label=="future" else source["truth"]-oof["prediction"]
            lookup,_,valid=nearest(source["key"],test["key"][subset_ids],k=8,
                source_ids=source["ids"].repeat_interleave(test["key"].shape[1]),
                query_ids=(test["ids"][subset_ids]+meta["test_start_absolute"]).repeat_interleave(
                    test["key"].shape[1]),span=meta["span"])
            flat=source_future.reshape(-1,source_future.shape[-1])
            # One scalar per independent historical trajectory, not five
            # mixed summary statistics masquerading as distribution samples.
            values_by_query=flat[lookup].mean(-1).numpy()
            for i in range(len(values_by_query)):
                    source_events=(source["ids"].repeat_interleave(test["key"].shape[1])
                                   [lookup[i,valid[i]]]//meta["span"]).unique()
                    if len(source_events)<4: continue
                    values=values_by_query[i].reshape(-1,1)
                    one=GaussianMixture(1,random_state=2021).fit(values)
                    two=GaussianMixture(2,random_state=2021).fit(values)
                    scores.append(one.bic(values)-two.bic(values))
            rows.append({"experiment":"C7","feature":label+"_GMM_BIC",
                         "kind":"diagnostic","effective_queries":len(scores),
                         "bimodal_preferred_ratio":float((np.asarray(scores)>0).mean()) if scores else float("nan")})
    except ImportError:
        rows.append({"experiment":"C7","status":"unavailable","reason":"scikit-learn absent"})
    print("C GMM READY",flush=True)
    # C8: fold models have different training times; compare their TEST forecasts.
    from tools.run_fpem_abc_mechanism import predict_custom
    variants=[]
    for fold in oof["states"]:
        alt=RarePatternLearner.from_export(fold["model"])
        local={"g1_weight":fold["g1_weight"],"g1_scale":saved["g1_scale"],
               "offset":saved["offset"],"reliability":fold["reliability"]}
        variants.append(predict_custom(test,local,alt))
    instability=torch.stack(variants).var(0,unbiased=False).mean(-1).numpy()
    features_test["C8_model_instability"]=instability
    rows.append({"experiment":"C8","feature":"between_training_blocks",
                 "kind":"diagnostic","test_corr_squared_error":pearson(instability,target_test),
                 "test_AUSE":ause(instability,target_test),
                 "test_MSE":float(instability.mean())})
    for row in rows:
        row.update({k:meta[k] for k in ("backbone","dataset","pred_len")})
        row["selection_protocol"]="TEST_exploratory"
    write_rows(output/"C_risk_results.csv",rows)
    write_rows(output/"C_risk_ranking.csv",sorted(rows,key=lambda r:r.get("test_AUSE",float("inf"))))
    cache={"oof":oof,"compact_test":compact_test,"features_oof":features_oof,
           "features_test":features_test,"event_cut":event_cut}
    torch.save(cache,output/"C_risk_cache.pt")
    return cache


def run_risk_studies(data,meta,state,learner,output):
    cache_path=output/"C_risk_cache.pt"
    cache=torch.load(cache_path,map_location="cpu",weights_only=False) if cache_path.exists() else \
        c_risk_train(data,meta,state,learner,output)
    run_probability_studies(data,meta,cache,output)


def run_probability_studies(data,meta,cache,output):
    """Same-capacity detached residual Flow conditions, plus two scale controls."""
    from tools.fpem_abc_probability import run_c12
    run_c12(data,meta,cache,output)
