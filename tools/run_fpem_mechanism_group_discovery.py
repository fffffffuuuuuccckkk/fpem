#!/usr/bin/env python3
"""TRAIN-only mechanism grouping and low-cost diagnostics; no fusion training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
import numpy as np
import torch
from scipy.fft import dct
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.predictive_env.mechanism_grouping import (
    K,fit_mixture,fit_rff_mixture,fit_fingerprint,fit_fingerprint_cluster,
    fingerprint_posterior,make_rff,posterior,predict_linear,weighted_linear,
)
from models.predictive_env.relation_prototypes import softmax_neg,squared_distance,train_router,route,teacher_quality,independent_count
from tools.run_fpem_prototype_quality import partition,rows,estimate_period,cluster_metrics
from tools.run_fpem_relation_prototype_diagnostics import split_middle,flatten_history,mean_sq,mae,soft_centers
from tools.run_fpem_simple_mapping_prototypes import pair_pool,future_lift,fit_local,route_predict,corr
from tools.run_fpem_f03_r3_invariance import collect,prepare
from tools.run_fpem_shared_factorization_g1r3 import load_experiment

METHODS=("G1_Switching_Ridge","G2_HRM_lite","G3_KerHRM_lite","G4_ANM_MM_lite",
         "G5_SSCM_lite","G6_CCSL_proxy","G7_Ridge_Update","G8_A0_Gradient_Direction",
         "G8_A0_Gradient_Amplitude","B0_Raw_History","B1_Random_Balanced","B2_ShuffleY_G1")
ALPHAS=(.1,1.,10.,100.)
BLOCKS=("A","B1","B2","C")


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+"\n",encoding="utf-8")


def data_case(cli,out):
    cache=out/"case_cache.npz"; meta=out/"case_cache.json"
    if cache.exists() and meta.exists():
        arrays=np.load(cache)
        blocks={block:{field:arrays[f"{block}_{field}"] for field in ("raw","history","y","ids","channels","phase","hidden","base_y")}
                for block in BLOCKS}
        return blocks,json.loads(meta.read_text())
    saved=json.loads(Path(cli.base_run_config).read_text())
    prototype_seed=cli.seed
    cli.seed=int(saved["seed"])
    try: args,model,loaders,device,a0_sha=load_experiment(cli)
    finally: cli.seed=prototype_seed
    if any(p.requires_grad for p in model.parameters()):raise AssertionError("A0 not frozen")
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
    raw=collect(model,loaders["train"].dataset.dataset,args,device,cli.stride)
    peak=int(torch.cuda.max_memory_allocated(device)) if device.type=="cuda" else 0
    floor_std=raw["x"].float().var(1,unbiased=False).add(1e-5).sqrt()
    floor=(floor_std.median(0).values*.25).clamp_min(.01)
    prepared=prepare(raw,floor)
    span=args.seq_len+args.pred_len
    masks,audit=partition(prepared,span)
    period,_=estimate_period(loaders["train"].dataset.dataset.data_x[:audit["cut_A"]])
    blocks={name:rows(prepared,mask,period,args.seq_len) for name,mask in masks.items()}
    for name,mask in masks.items():
        blocks[name]["history"]=flatten_history(prepared,mask)
        n=int(mask.sum());c=raw["key"].shape[1]
        blocks[name]["base_y"]=prepared["base_hist_norm"][mask].numpy().reshape(n*c,-1).astype(np.float32)
    blocks["B1"],blocks["B2"]=split_middle(blocks.pop("B"),span)
    np.savez_compressed(cache,**{f"{name}_{field}":array for name in BLOCKS for field,array in blocks[name].items()})
    meta_value={"case":f"{args.backbone}/{args.dataset_name}-{args.pred_len}","dataset":args.dataset_name,
                "backbone":args.backbone,"pred_len":args.pred_len,"seed":prototype_seed,"a0_seed":args.seed,
                "a0_sha256":a0_sha,"span":span,"stride":cli.stride,"temporal_audit":audit,
                "peak_vram_bytes":peak,"A0_training":"reused frozen checkpoint; no retraining",
                "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()}
    dump(meta,meta_value)
    return blocks,meta_value


def representation(blocks,out):
    cache=out/"representation_cache.npz"
    if cache.exists():
        arrays=np.load(cache)
        return ({b:arrays[f"{b}_z"] for b in BLOCKS},
                {b:arrays[f"{b}_v"] for b in BLOCKS},
                {b:arrays[f"{b}_a0v"] for b in BLOCKS},
                {b:arrays[f"{b}_raw"] for b in BLOCKS},int(arrays["z_dim"]))
    h_scale=StandardScaler().fit(blocks["A"]["hidden"])
    dim=16 if len(blocks["A"]["hidden"])>=128 and blocks["A"]["hidden"].shape[1]>=16 else 8
    reducer=PCA(n_components=dim,svd_solver="randomized",random_state=2022).fit(h_scale.transform(blocks["A"]["hidden"]))
    zraw={b:reducer.transform(h_scale.transform(blocks[b]["hidden"])) for b in BLOCKS}
    zscale=StandardScaler().fit(zraw["A"])
    z={b:zscale.transform(zraw[b]).astype(np.float32) for b in BLOCKS}
    rawscale=StandardScaler().fit(blocks["A"]["raw"])
    common={b:rawscale.transform(blocks[b]["raw"]).astype(np.float32) for b in BLOCKS}
    def coefficients(y):return dct(y,type=2,norm="ortho",axis=1)[:,:8]
    vscale=StandardScaler().fit(coefficients(blocks["A"]["y"]))
    v={b:vscale.transform(coefficients(blocks[b]["y"])).astype(np.float32) for b in BLOCKS}
    a0v={b:vscale.transform(coefficients(blocks[b]["base_y"])).astype(np.float32) for b in BLOCKS}
    np.savez_compressed(cache,z_dim=dim,**{f"{b}_{field}":value for b in BLOCKS
                               for field,value in (("z",z[b]),("v",v[b]),("a0v",a0v[b]),("raw",common[b]))})
    return z,v,a0v,common,dim


def shuffled_by_event(values,ids,channels,span,seed):
    rng=np.random.default_rng(seed)
    event=(ids//span).astype(int)
    unique=np.unique(event)
    destination=rng.permutation(unique)
    mapping=dict(zip(unique,destination))
    lookup={}
    for index,(ev,ch) in enumerate(zip(event,channels)):
        lookup.setdefault((ev,ch),[]).append(index)
    result=np.empty_like(values)
    for index,(ev,ch) in enumerate(zip(event,channels)):
        candidates=lookup.get((mapping[ev],ch))
        if not candidates: candidates=lookup[(ev,ch)]
        chosen=min(candidates,key=lambda j:abs(int(ids[j]%span)-int(ids[index]%span)))
        result[index]=values[chosen]
    return result


def forward_ridge_reference(z,v,ids,span):
    """Earlier-fold reference for later A windows; first fold is flagged in-sample."""
    order=np.argsort(ids,kind="stable")
    edges=np.quantile(ids,[0,.25,.5,.75,1])
    result=np.empty_like(v)
    for fold in range(4):
        take=(ids>=edges[fold]) & (ids<edges[fold+1] if fold<3 else ids<=edges[fold+1])
        source=(ids+span<=edges[fold]) if fold>0 else take
        if source.sum()<max(16,z.shape[1]+2):source=(ids<edges[fold]) if fold>0 else take
        if source.sum()<max(16,z.shape[1]+2):source=take
        model=weighted_linear(z[source],v[source],np.ones(source.sum()),10.)
        result[take]=predict_linear(z[take],model)
    return result


def group_method(method,z,v,a0v,common,blocks,cfg,seed,rff,prior=None):
    start=time.perf_counter();metadata={"method":method,"K":K}
    teacher={}
    if method.startswith(("G1_","G2_","G5_","G6_","G3_","G4_","B2_")):
        actual_method="G1_Switching_Ridge" if method.startswith("B2_") else method
        use_rff=actual_method.startswith(("G3_","G4_"))
        design=rff if use_rff else z
        target=v["A"]
        if method.startswith("B2_"):
            target=shuffled_by_event(v["A"],blocks["A"]["ids"],blocks["A"]["channels"],cfg["span"],seed+901)
        fit=(fit_rff_mixture(actual_method,design["A"],target,seed=seed)
             if use_rff else fit_mixture(actual_method,design["A"],target,seed=seed))
        teacher["A"]=fit["q_train"]
        for block in ("B1","B2","C"):
            target_block=v[block]
            if method.startswith("B2_") and block!="C":
                target_block=shuffled_by_event(v[block],blocks[block]["ids"],blocks[block]["channels"],cfg["span"],seed+901)
            teacher[block]=posterior(design[block],target_block,fit["models"],fit["priors"],fit["variance"],temperature=1.5)
        metadata.update({k:fit[k] for k in ("parameter_count","support_difference","fit_seconds","feature_definition")})
        metadata["mapping_signature"]={"coefficients_shape":list(fit["models"].shape),
                                      "coefficient_norms":[float(np.linalg.norm(x)) for x in fit["models"]],
                                      "noise_min":float(np.min(fit["variance"])),"noise_max":float(np.max(fit["variance"]))}
    elif method.startswith(("G7_","G8_")):
        if method.startswith("G7_"):
            reference_A=forward_ridge_reference(z["A"],v["A"],blocks["A"]["ids"],cfg["span"])
            global_coeff=weighted_linear(z["A"],v["A"],np.ones(len(v["A"])),10.)
            ref={b:reference_A if b=="A" else predict_linear(z[b],global_coeff) for b in BLOCKS}
            signed=False;metadata["reference"]="TRAIN-A forward-fold Global Ridge; first fold in-sample"
        else:
            ref=a0v;signed=method.endswith("Direction")
            metadata["reference"]="frozen A0 in shared DCT space"
        fp={b:fit_fingerprint(z[b],v[b],ref[b],signed_only=signed) for b in BLOCKS}
        fit=fit_fingerprint_cluster(fp["A"],seed=seed)
        teacher={b:fingerprint_posterior(fit,fp[b]) for b in BLOCKS}
        metadata.update({k:fit[k] for k in ("parameter_count","fit_seconds","feature_definition")})
        metadata["mapping_signature"]={"fingerprint_shape":list(fp["A"].shape),
                                       "pca_components":int(fit["reducer"].n_components_),
                                       "reference":metadata["reference"]}
    elif method=="B0_Raw_History":
        km=MiniBatchKMeans(n_clusters=K,n_init=3,random_state=seed,batch_size=256).fit(common["A"])
        distance={b:squared_distance(common[b],km.cluster_centers_) for b in BLOCKS}
        temp=float(np.median(distance["A"].min(axis=1))+.05)
        teacher={b:softmax_neg(distance[b],temp) for b in BLOCKS}
        metadata.update(parameter_count=int(km.cluster_centers_.size),fit_seconds=time.perf_counter()-start,
                        feature_definition="Raw 64D historical key only; negative control",
                        mapping_signature={"center_shape":list(km.cluster_centers_.shape)})
    else:
        if prior is None:raise ValueError("B1 needs G1 priors")
        rng=np.random.default_rng(seed+553)
        teacher={b:prior[b][rng.permutation(len(prior[b]))] for b in BLOCKS}
        metadata.update(parameter_count=0,fit_seconds=0.,
                        feature_definition="permuted G1 soft rows; same group-mass distribution",
                        mapping_signature={"source":"G1_Switching_Ridge permuted within block"})
    return teacher,metadata


def probe_one(x,y,q,alpha):
    models=fit_local(x,y,q,alpha)
    return models


def evaluation(method,teacher,student,blocks,z,v,common,cfg,meta,seed,pool):
    xfit,yfit=common["A"],blocks["A"]["y"]
    # B2 is later than the student-fit block B1 and chooses one alpha jointly.
    alpha_scores=[]
    for alpha in ALPHAS:
        models=probe_one(xfit,yfit,teacher["A"],alpha)
        prediction,_=route_predict(models,common["B2"],student["B2"])
        alpha_scores.append((mean_sq(prediction,blocks["B2"]["y"]),alpha))
    _,alpha=min(alpha_scores)
    models=probe_one(xfit,yfit,teacher["A"],alpha)
    pred,outputs=route_predict(models,common["C"],student["C"])
    oracle,_=route_predict(models,common["C"],teacher["C"])
    global_scores=[]
    for candidate in ALPHAS:
        model=Ridge(alpha=candidate).fit(xfit,yfit)
        global_scores.append((mean_sq(model.predict(common["B2"]),blocks["B2"]["y"]),candidate))
    _,ga=min(global_scores)
    global_pred=Ridge(alpha=ga).fit(xfit,yfit).predict(common["C"])
    rng=np.random.default_rng(seed+662)
    random_a=teacher["A"][rng.permutation(len(teacher["A"]))]
    random_b2=student["B2"][rng.permutation(len(student["B2"]))]
    random_c=student["C"][rng.permutation(len(student["C"]))]
    random_scores=[]
    for candidate in ALPHAS:
        models_r=probe_one(xfit,yfit,random_a,candidate)
        rp,_=route_predict(models_r,common["B2"],random_b2)
        random_scores.append((mean_sq(rp,blocks["B2"]["y"]),candidate))
    _,ra=min(random_scores)
    rp,_=route_predict(probe_one(xfit,yfit,random_a,ra),common["C"],random_c)
    target=blocks["C"]["y"]
    mse=mean_sq(pred,target);gmse=mean_sq(global_pred,target);rmse=mean_sq(rp,target)
    # A/B2 mapping responses use the same z input and v DCT output, with
    # Hungarian alignment for arbitrary prototype permutation.
    map_a=[weighted_linear(z["A"],v["A"],teacher["A"][:,k],10.) for k in range(K)]
    map_b=[weighted_linear(z["B2"],v["B2"],teacher["B2"][:,k],10.) for k in range(K)]
    reference=z["A"][np.linspace(0,len(z["A"])-1,min(128,len(z["A"])),dtype=int)]
    resp_a=np.stack([predict_linear(reference,x) for x in map_a],axis=1)
    resp_b=np.stack([predict_linear(reference,x) for x in map_b],axis=1)
    distance=np.array([[np.mean((resp_a[:,i]-resp_b[:,j])**2) for j in range(K)] for i in range(K)])
    left,right=linear_sum_assignment(distance)
    drift=float(np.mean(distance[left,right]))
    # In-sample weighted response error is distinct from the cross-block
    # mapping drift above; report both rather than duplicating one statistic.
    fitted_a=np.stack([predict_linear(z["A"],x) for x in map_a],axis=1)
    within=float(np.mean(np.sum(teacher["A"][:,:,None]*(fitted_a-v["A"][:,None,:])**2,axis=1)))
    between=float(np.mean([np.mean((resp_a[:,i]-resp_a[:,j])**2) for i in range(K) for j in range(i+1,K)]))
    centers=soft_centers(blocks["A"]["y"],teacher["A"])
    future_center_pred=student["C"]@centers
    fve=1-mean_sq(future_center_pred,target)/max(mean_sq(target,blocks["A"]["y"].mean(axis=0)),1e-9)
    lift,lift_reason,pair=future_lift(pool,student["C"].argmax(axis=1),seed)
    mass=teacher["A"].mean(axis=0)
    keff=float(np.exp(-np.sum(mass*np.log(mass+1e-9))))
    coverage=squared_distance(common["C"],soft_centers(common["A"],teacher["A"])).min(axis=1)
    disagreement=np.mean(np.einsum("nk,nkt->nt",student["C"],(outputs-pred[:,None])**2),axis=1)
    error=np.mean((pred-target)**2,axis=1)
    rho=float(spearmanr(disagreement,error).statistic) if np.std(disagreement)>1e-9 else None
    support=[independent_count(blocks["A"]["ids"][teacher["A"].argmax(axis=1)==k],cfg["span"]) for k in range(K)]
    cluster=cluster_metrics(common["A"],teacher["A"].argmax(axis=1))
    router_b2=teacher_quality(teacher["B2"],student["B2"])
    summary={"case":cfg["case"],"method":method,"seed":seed,"status":"ok",
             "independent_events_C":independent_count(blocks["C"]["ids"],cfg["span"]),
             "k_eff":keff,"collapsed":bool(np.min(mass)<.02),"within_mapping":within,
             "between_mapping":between,"cross_time_drift":drift,"future_lift":lift,
             "future_lift_reason":lift_reason,"fve":float(fve),
             "teacher_only_mse_C":mean_sq(oracle,target),"student_routed_ridge_mse_C":mse,
             "student_routed_ridge_mae_C":mae(pred,target),"global_ridge_mse_C":gmse,
             "random_ridge_mse_C":rmse,"gain_vs_global_C":1-mse/max(gmse,1e-9),
             "gain_vs_random_C":1-mse/max(rmse,1e-9),
             "uncertainty_error_spearman_C":rho,"router_agreement_B2":router_b2["top1_agreement"],
             "router_soft_kl_B2":router_b2["soft_kl"],
             "coverage_mean_C":float(coverage.mean()),"coverage_p95_C":float(np.quantile(coverage,.95)),
             "min_prototype_independent_events_A":min(support),
             "raw_silhouette_A":cluster["silhouette_raw"],
             "raw_davies_bouldin_A":cluster["davies_bouldin_raw"],
             "raw_calinski_harabasz_A":cluster["calinski_harabasz_raw"],
             "fit_time_sec":meta["fit_seconds"],"grouping_parameters":meta["parameter_count"],
             "router_parameters":meta["router_parameters"],"router_time_sec":meta["router_time_sec"],
             "peak_vram_bytes":cfg["peak_vram_bytes"],
             "peak_cpu_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             "a0_sha256":cfg["a0_sha256"]}
    per_proto=[]
    for k in range(K):
        for block in BLOCKS:
            hard=teacher[block].argmax(axis=1)==k
            per_proto.append({"case":cfg["case"],"method":method,"seed":seed,"block":block,"prototype":k,
                              "soft_mass":float(teacher[block][:,k].sum()),"hard_rows":int(hard.sum()),
                              "independent_events":independent_count(blocks[block]["ids"][hard],cfg["span"]),
                              "future_center_mse":mean_sq(blocks[block]["y"][hard],centers[k]) if hard.any() else None,
                              "mapping_norm_A":float(np.linalg.norm(map_a[k])),
                              "mapping_drift_A_to_B2":float(distance[k,right[k]]) if block=="B2" else None})
    per_block=[]
    for block in BLOCKS:
        y=blocks[block]["y"]
        p=student[block]@centers
        per_block.append({"case":cfg["case"],"method":method,"seed":seed,"block":block,
                          "independent_events":independent_count(blocks[block]["ids"],cfg["span"]),
                          "fve":1-mean_sq(p,y)/max(mean_sq(y,blocks["A"]["y"].mean(axis=0)),1e-9),
                          "future_center_mse":mean_sq(p,y),
                          "student_entropy":float(np.mean(-np.sum(student[block]*np.log(student[block]+1e-9),axis=1)))})
    probes=[{"case":cfg["case"],"method":method,"seed":seed,"probe":label,"C_mse":value,"alpha":a}
            for label,value,a in (("Global Ridge",gmse,ga),("Prototype Student Routed Ridge",mse,alpha),
                                  ("Prototype Teacher Oracle (nondeployable)",mean_sq(oracle,target),alpha),
                                  ("Random Group Ridge",rmse,ra))]
    router=[{"case":cfg["case"],"method":method,"seed":seed,"block":b,
             "teacher_future_used_for_offline_audit":b=="C",**teacher_quality(teacher[b],student[b])} for b in BLOCKS]
    edges=np.unique(np.quantile(disagreement,np.linspace(0,1,11)))
    bins=[]
    if len(edges)>2:
        labels=np.digitize(disagreement,edges[1:-1])
        for i in range(len(edges)-1):
            mask=labels==i
            if mask.any():bins.append({"case":cfg["case"],"method":method,"seed":seed,"bin":i,
                                       "count":int(mask.sum()),"disagreement_mean":float(disagreement[mask].mean()),
                                       "squared_error_mean":float(error[mask].mean())})
    return summary,per_proto,per_block,probes,router,bins,support


def save_assignments(path,teacher,blocks,cfg):
    names=("A","B1","B2")
    q=np.concatenate([teacher[b] for b in names]).astype(np.float32)
    ids=np.concatenate([blocks[b]["ids"] for b in names]).astype(np.int64)
    channels=np.concatenate([blocks[b]["channels"] for b in names]).astype(np.int16)
    time_block=np.concatenate([np.repeat(b,len(blocks[b]["ids"])) for b in names])
    sample_id=np.array([f'{cfg["dataset"]}:TRAIN:{start}:{channel}' for start,channel in zip(ids,channels)])
    np.savez_compressed(path,sample_id=sample_id,window_start=ids,channel=channels,time_block=time_block,
                        q_teacher=q,hard_group=q.argmax(axis=1).astype(np.int8),confidence=q.max(axis=1),
                        event_id=(ids//cfg["span"]).astype(np.int64),event_start=ids,
                        event_end=ids+cfg["span"],valid_mask=np.ones(len(q),dtype=bool))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--base_run_config",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--gpu",type=int,default=0);parser.add_argument("--stride",type=int,default=8)
    parser.add_argument("--seed",type=int,default=2022)
    parser.add_argument("--allow_legacy_reference_signature",action="store_true")
    cli=parser.parse_args();out=Path(cli.output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    blocks,cfg=data_case(cli,out)
    z,v,a0v,common,z_dim=representation(blocks,out)
    cfg["z_dim"]=z_dim;cfg["v_dim"]=8;dump(out/"case_cache.json",cfg)
    rff=make_rff(z["A"],z,seed=cli.seed)
    pool=pair_pool(blocks["C"],cfg["span"],cli.seed+113)
    prior=None
    for method in METHODS:
        directory=out/"methods"/method
        if (directory/"complete").exists():
            print("SKIP",method,flush=True)
            if method=="G1_Switching_Ridge":
                prior={b:np.load(directory/"teacher_cache.npz")[b] for b in BLOCKS}
            continue
        directory.mkdir(parents=True,exist_ok=True)
        print("START",method,flush=True)
        started=time.perf_counter()
        try:
            teacher,meta=group_method(method,z,v,a0v,common,blocks,cfg,cli.seed,rff,prior)
            if method=="G1_Switching_Ridge":prior=teacher
            # Router is diagnostic only; later window must not mistake it for
            # a trained F03/R3 component.
            router,router_time,router_parameters=train_router(np.concatenate((z["A"],z["B1"])),
                  np.concatenate((teacher["A"],teacher["B1"])),seed=cli.seed,epochs=40)
            student={b:route(router,z[b]) for b in BLOCKS}
            meta["router_time_sec"]=router_time;meta["router_parameters"]=router_parameters
            summary,per_proto,per_block,probes,router_diag,bins,support=evaluation(
                method,teacher,student,blocks,z,v,common,cfg,meta,cli.seed,pool)
            summary["total_time_sec"]=time.perf_counter()-started
            save_assignments(directory/"assignments_train.npz",teacher,blocks,cfg)
            np.savez_compressed(directory/"teacher_cache.npz",**teacher)
            meta.update({"A0_sha256":cfg["a0_sha256"],"dataset":cfg["dataset"],"pred_len":cfg["pred_len"],
                         "seed":cli.seed,"split_and_embargo":cfg["temporal_audit"],
                         "prototype_valid_support":support,
                         "group_priors":teacher["A"].mean(axis=0).tolist(),
                         "git_commit":cfg["git_commit"],
                         "normalization":"history-only scale; z PCA and v DCT StandardScaler fitted on TRAIN A",
                         "C_teacher":"offline diagnostic only; excluded from assignments_train.npz"})
            dump(directory/"prototype_meta.json",meta)
            dump(directory/"metrics.json",{"summary":summary,"per_prototype":per_proto,"per_time_block":per_block,
                                          "mapping_probe":probes,"router_diagnostic":router_diag,"ambiguity_bins":bins})
            (directory/"complete").write_text("complete\n")
            print("COMPLETE",method,"student_MSE",summary["student_routed_ridge_mse_C"],flush=True)
        except Exception as error:
            dump(directory/"failed.json",{"method":method,"error":repr(error)})
            print("FAILED",method,repr(error),flush=True)
    print("CASE_FINISHED",cfg["case"],flush=True)


if __name__=="__main__":main()
