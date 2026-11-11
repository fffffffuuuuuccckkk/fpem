#!/usr/bin/env python3
"""All 255 simple mapping-descriptor prototypes, one frozen-A0 data case.

A learns descriptor centers; A+B1 TRAIN labels teach the history-only router.
B1 fits Ridge; B2 chooses alpha; C is a chronologically later holdout.
Each combination is checkpointed separately and may be resumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.predictive_env.relation_prototypes import softmax_neg, squared_distance, train_router, route, teacher_quality, independent_count
from tools.fpem_simple_mapping_descriptors import NAMES, DIMS, combinations, descriptors, robust_groups, combine
from tools.run_fpem_prototype_quality import partition, rows, estimate_period
from tools.run_fpem_relation_prototype_diagnostics import split_middle, flatten_history, mean_sq, mae, soft_centers
from tools.run_fpem_f03_r3_invariance import collect, prepare
from tools.run_fpem_shared_factorization_g1r3 import load_experiment

ALPHAS = (.1, 1., 10., 100.)
BLOCKS = ("A", "B1", "B2", "C")


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, allow_nan=False, indent=2, default=number) + "\n", encoding="utf-8")
    temp.replace(path)


def number(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(str(type(value)))


def safe(value):
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    return value


def data_case(args, cli, out):
    cache = out / "case_cache.npz"
    metadata = out / "case_cache.json"
    if cache.exists() and metadata.exists():
        arrays = np.load(cache)
        blocks = {name: {field: arrays[f"{name}_{field}"] for field in ("raw", "history", "y", "ids", "channels", "phase")}
                  for name in BLOCKS}
        return blocks, json.loads(metadata.read_text())
    prototype_seed = cli.seed
    cli.seed = int(args["seed"])
    try:
        args, model, loaders, device, a0_sha = load_experiment(cli)
    finally:
        cli.seed = prototype_seed
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("A0 must be frozen")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    raw = collect(model, loaders["train"].dataset.dataset, args, device, cli.stride)
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    floor_std = raw["x"].float().var(1, unbiased=False).add(1e-5).sqrt()
    floor = (floor_std.median(0).values * .25).clamp_min(.01)
    prepared = prepare(raw, floor)
    span = args.seq_len + args.pred_len
    masks, audit = partition(prepared, span)
    period, source = estimate_period(loaders["train"].dataset.dataset.data_x[:audit["cut_A"]])
    blocks = {name: rows(prepared, mask, period, args.seq_len) for name, mask in masks.items()}
    for name in blocks:
        blocks[name]["history"] = flatten_history(prepared, masks[name])
        blocks[name].pop("hidden", None)
    blocks["B1"], blocks["B2"] = split_middle(blocks.pop("B"), span)
    np.savez_compressed(cache, **{f"{name}_{field}": array for name in BLOCKS for field, array in blocks[name].items()})
    cfg = {"case": f"{args.backbone}/{args.dataset_name}-{args.pred_len}", "seed": prototype_seed,
           "a0_seed": args.seed,
           "span": span, "period": period, "period_source": source, "a0_sha256": a0_sha,
           "temporal_audit": audit, "peak_vram_bytes_data_collection": peak,
           "protocol": "TRAIN A/B1/B2/C purged by full history+future span; only C metrics are holdout",
           "training_scope": "original TRAIN only; A0 reused, not retrained"}
    dump(metadata, cfg)
    return blocks, cfg


def pair_pool(block, span, seed):
    rng = np.random.default_rng(seed)
    n = len(block["y"])
    left, right = rng.integers(n, size=150000), rng.integers(n, size=150000)
    okay = (block["channels"][left] == block["channels"][right]) & (np.abs(block["ids"][left] - block["ids"][right]) >= span)
    left, right = left[okay][:8000], right[okay][:8000]
    if len(left) < 50:
        return None
    hx = np.mean((block["raw"][left] - block["raw"][right]) ** 2, axis=1)
    fy = np.mean((block["y"][left] - block["y"][right]) ** 2, axis=1)
    edges = np.unique(np.quantile(hx, np.linspace(0, 1, 11)))
    bins = np.digitize(hx, edges[1:-1]) if len(edges) >= 4 else np.zeros(len(hx), int)
    return left, right, bins, fy


def future_lift(pool, labels, seed):
    if pool is None:
        return None, "fewer than 50 non-overlapping same-variable pairs", None
    left, right, bins, future_mse = pool
    same = labels[left] == labels[right]
    rng = np.random.default_rng(seed)
    a, b = [], []
    for index in np.unique(bins):
        good = np.flatnonzero((bins == index) & same)
        control = np.flatnonzero((bins == index) & ~same)
        if min(len(good), len(control)) < 10:
            continue
        count = min(len(good), len(control), 200)
        a.extend(future_mse[rng.choice(good, count, replace=False)])
        b.extend(future_mse[rng.choice(control, count, replace=False)])
    if len(a) < 50:
        return None, "insufficient history-distance-matched control pairs", None
    return float(1 - np.mean(a) / max(np.mean(b), 1e-12)), "ok", {
        "within_future_similarity_p10": float(np.quantile(np.exp(-future_mse[same]), .1)) if same.any() else None,
        "within_future_similarity_median": float(np.median(np.exp(-future_mse[same]))) if same.any() else None,
        "within_future_similarity_p90": float(np.quantile(np.exp(-future_mse[same]), .9)) if same.any() else None,
        "matched_pairs": len(a)}


def fit_local(x, y, q, alpha):
    global_model = Ridge(alpha=alpha).fit(x, y)
    local = []
    for k in range(8):
        weights = q[:, k]
        if weights.sum() < max(8, x.shape[1] / 4):
            local.append(global_model)
        else:
            local.append(Ridge(alpha=alpha).fit(x, y, sample_weight=weights))
    return local


def local_outputs(models, x):
    return np.stack([model.predict(x) for model in models], axis=1).astype(np.float32)


def route_predict(models, x, q):
    outputs = local_outputs(models, x)
    return np.einsum("nk,nkt->nt", q, outputs), outputs


def corr(a, b):
    a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
    if len(a) < 8 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def run_combo(names, blocks, features, fitted, common, pool, global_result, cfg, cli):
    started = time.perf_counter()
    tag = "+".join(names)
    matrix = {block: combine(features[block], fitted, names) for block in BLOCKS}
    invalid = [name for name in names if not fitted[name][2].any()]
    if invalid:
        raise ValueError("near-constant feature group(s): " + ",".join(invalid))
    cluster_start = time.perf_counter()
    km = MiniBatchKMeans(n_clusters=8, n_init=3, batch_size=256, random_state=cli.seed,
                         max_iter=150).fit(matrix["A"])
    dist = {block: squared_distance(matrix[block], km.cluster_centers_) for block in BLOCKS}
    temperature = float(np.median(dist["A"].min(axis=1)) + .05)
    teacher = {block: softmax_neg(dist[block], temperature) for block in BLOCKS}
    cluster_seconds = time.perf_counter() - cluster_start
    router_start = time.perf_counter()
    network, _, params = train_router(np.concatenate((common["A"], common["B1"])),
                                      np.concatenate((teacher["A"], teacher["B1"])),
                                      seed=cli.seed, epochs=40)
    student = {block: route(network, common[block]) for block in BLOCKS}
    router_seconds = time.perf_counter() - router_start
    fit_x, fit_y, fit_q = common["B1"], blocks["B1"]["y"], student["B1"]
    alpha_scores = []
    probe_start = time.perf_counter()
    for alpha in ALPHAS:
        models = fit_local(fit_x, fit_y, fit_q, alpha)
        pred, _ = route_predict(models, common["B2"], student["B2"])
        alpha_scores.append((mean_sq(pred, blocks["B2"]["y"]), alpha))
    selected_b2, selected_alpha = min(alpha_scores)
    models = fit_local(fit_x, fit_y, fit_q, selected_alpha)
    pred_c, candidates_c = route_predict(models, common["C"], student["C"])
    teacher_pred_c, _ = route_predict(models, common["C"], teacher["C"])
    c_y = blocks["C"]["y"]
    prot_mse = mean_sq(pred_c, c_y)
    teacher_mse = mean_sq(teacher_pred_c, c_y)
    random_q = {}
    rng = np.random.default_rng(cli.seed + 1901)
    for block in ("B1", "B2", "C"):
        random_q[block] = student[block][rng.permutation(len(student[block]))]
    random_scores = []
    for alpha in ALPHAS:
        random_models = fit_local(fit_x, fit_y, random_q["B1"], alpha)
        random_pred, _ = route_predict(random_models, common["B2"], random_q["B2"])
        random_scores.append((mean_sq(random_pred, blocks["B2"]["y"]), alpha))
    _, random_alpha = min(random_scores)
    random_models = fit_local(fit_x, fit_y, random_q["B1"], random_alpha)
    random_pred_c, _ = route_predict(random_models, common["C"], random_q["C"])
    random_mse = mean_sq(random_pred_c, c_y)
    probe_seconds = time.perf_counter() - probe_start
    # Mapping behavior is compared in a common 4-segment output space, not via
    # unpaired C labels or C-fitted deployment models.
    chunk = lambda y: np.stack([part.mean(axis=1) for part in np.array_split(y, 4, axis=1)], axis=1)
    map_b1 = fit_local(common["B1"], chunk(blocks["B1"]["y"]), teacher["B1"], 10.)
    map_b2 = fit_local(common["B2"], chunk(blocks["B2"]["y"]), teacher["B2"], 10.)
    map_c = fit_local(common["C"], chunk(c_y), teacher["C"], 10.)
    reference = common["A"][np.linspace(0, len(common["A"])-1, min(128, len(common["A"])), dtype=int)]
    response_b1 = local_outputs(map_b1, reference)
    response_b2 = local_outputs(map_b2, reference)
    response_c = local_outputs(map_c, reference)
    within = float(np.mean((response_b1 - response_b2) ** 2))
    between = float(np.mean([np.mean((response_b1[:, i] - response_b1[:, j]) ** 2)
                             for i in range(8) for j in range(i+1, 8)]))
    drift = float(np.mean((response_b1 - response_c) ** 2))
    center_future = soft_centers(blocks["A"]["y"], teacher["A"])
    center_history = soft_centers(common["A"], teacher["A"])
    p = teacher["A"].mean(axis=0)
    k_eff = float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0]))))
    fve = float(1 - mean_sq(student["C"] @ center_future, c_y) /
                max(mean_sq(c_y, blocks["A"]["y"].mean(axis=0)), 1e-9))
    lift, lift_reason, pair_stats = future_lift(pool, student["C"].argmax(axis=1), cli.seed)
    raw_distance = squared_distance(common["C"], center_history).min(axis=1)
    uncertainty = np.mean(np.einsum("nk,nkt->nt", student["C"],
                                    (candidates_c - pred_c[:, None]) ** 2), axis=1)
    squared_error = np.mean((pred_c - c_y) ** 2, axis=1)
    disagreement_corr = corr(uncertainty, squared_error)
    routing = [dict(method=tag, block=block, **teacher_quality(teacher[block], student[block]),
                    entropy=float(np.mean(-np.sum(student[block]*np.log(student[block]+1e-9), axis=1))),
                    teacher_future_for_offline_diagnostic=block == "C") for block in BLOCKS]
    masses = student["C"].sum(axis=0)
    per_proto = []
    for block in BLOCKS:
        for k in range(8):
            hard = student[block].argmax(axis=1) == k
            phase = blocks[block]["phase"][hard]
            response = response_b1 if block == "B1" else response_b2 if block == "B2" else response_c if block == "C" else None
            per_proto.append(dict(method=tag, block=block, prototype=k,
                                  soft_mass=float(student[block][:,k].sum()),hard_rows=int(hard.sum()),
                                  independent_events=independent_count(blocks[block]["ids"][hard],cfg["span"]),
                                  future_mse=mean_sq(blocks[block]["y"][hard],center_future[k]) if hard.any() else None,
                                  mapping_drift_from_B1=float(np.mean((response_b1[:,k]-response[:,k])**2)) if response is not None else None,
                                  phase_concentration=float(abs(np.exp(2j*np.pi*phase).mean())) if hard.any() else None,
                                  phase_histogram=np.histogram(phase,bins=np.linspace(0,1,9))[0].tolist()))
    per_block = []
    for block in BLOCKS:
        q = student[block]
        target = blocks[block]["y"]
        center_pred = q @ center_future
        baseline = blocks["A"]["y"].mean(axis=0)
        per_block.append(dict(method=tag, block=block, fve=float(1-mean_sq(center_pred,target)/max(mean_sq(target,baseline),1e-9)),
                              future_mse=mean_sq(center_pred,target),
                              independent_events=independent_count(blocks[block]["ids"],cfg["span"]),
                              k_eff=float(np.exp(-np.sum(q.mean(axis=0)*np.log(q.mean(axis=0)+1e-9)))),
                              mapping_behavior_drift_from_B1=float(np.mean((response_b1 -
                                  (response_b1 if block=="B1" else response_b2 if block=="B2" else response_c))**2)) if block in ("B1","B2","C") else None))
    try:
        subset = np.linspace(0,len(matrix["A"])-1,min(500,len(matrix["A"])),dtype=int)
        labels = teacher["A"][subset].argmax(axis=1)
        if 1 < len(np.unique(labels)) < len(labels):
            cluster = dict(silhouette=float(silhouette_score(matrix["A"][subset],labels)),
                           davies_bouldin=float(davies_bouldin_score(matrix["A"][subset],labels)),
                           calinski_harabasz=float(calinski_harabasz_score(matrix["A"][subset],labels)))
        else:
            cluster = dict(silhouette=None,davies_bouldin=None,calinski_harabasz=None)
    except ValueError:
        cluster = dict(silhouette=None,davies_bouldin=None,calinski_harabasz=None)
    summary = dict(case=cfg["case"],seed=cli.seed,method=tag,feature_dim=int(matrix["A"].shape[1]),
                   descriptor_count=len(names),status="ok",fve_C=fve,future_lift_C=lift,
                   future_lift_reason=lift_reason,mapping_drift_C=drift,mapping_within_B1_B2=within,
                   mapping_between_B1=between,mapping_separation=between/max(within,1e-9),
                   route_agreement_B2=routing[2]["top1_agreement"],soft_kl_B2=routing[2]["soft_kl"],
                   k_eff_A=k_eff,independent_events_A=independent_count(blocks["A"]["ids"],cfg["span"]),
                   independent_events_C=independent_count(blocks["C"]["ids"],cfg["span"]),
                   global_ridge_mse_C=global_result["mse_C"],prototype_ridge_mse_C=prot_mse,
                   prototype_ridge_mae_C=mae(pred_c,c_y),teacher_ridge_mse_C=teacher_mse,
                   random_ridge_mse_C=random_mse,
                   gain_vs_global_C=1-prot_mse/max(global_result["mse_C"],1e-9),
                   gain_vs_random_C=1-prot_mse/max(random_mse,1e-9),
                   disagreement_error_correlation_C=disagreement_corr,
                   raw_coverage_mean_C=float(raw_distance.mean()),
                   raw_coverage_p90_C=float(np.quantile(raw_distance,.9)),
                   raw_coverage_p95_C=float(np.quantile(raw_distance,.95)),
                   collapsed_A=bool((p<.02).any()),
                   train_time_sec=time.perf_counter()-started,
                   descriptor_extract_time_sec=cfg["descriptor_extract_time_sec"],
                   cluster_time_sec=cluster_seconds,router_time_sec=router_seconds,probe_time_sec=probe_seconds,
                   router_parameters=params,peak_vram_bytes=cfg["peak_vram_bytes_data_collection"],
                   prototype_library_bytes=int(km.cluster_centers_.nbytes + center_future.nbytes),
                   peak_cpu_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**cluster)
    probes = [dict(method=tag,probe="Global Ridge",C_mse=global_result["mse_C"],C_mae=global_result["mae_C"],alpha=global_result["alpha"]),
              dict(method=tag,probe="Shared Mapping (same weighted objective as Global)",C_mse=global_result["mse_C"],C_mae=global_result["mae_C"],alpha=global_result["alpha"]),
              dict(method=tag,probe="Prototype Ridge deployable",C_mse=prot_mse,C_mae=mae(pred_c,c_y),alpha=selected_alpha,B2_mse=selected_b2),
              dict(method=tag,probe="Prototype Ridge teacher offline",C_mse=teacher_mse,C_mae=mae(teacher_pred_c,c_y),alpha=selected_alpha),
              dict(method=tag,probe="Random Group Ridge",C_mse=random_mse,C_mae=mae(random_pred_c,c_y),alpha=random_alpha)]
    bins = []
    edges = np.unique(np.quantile(uncertainty,np.linspace(0,1,11)))
    if len(edges)>2:
        label = np.digitize(uncertainty,edges[1:-1])
        for b in range(len(edges)-1):
            selected = label==b
            if selected.any():
                bins.append(dict(method=tag,bin=b,count=int(selected.sum()),disagreement_mean=float(uncertainty[selected].mean()),
                                 error_mse=float(squared_error[selected].mean())))
    return safe(dict(summary=summary,per_prototype=per_proto,per_time_block=per_block,
                     mapping_probe=probes,routing_diagnostics=routing,
                     disagreement_bins=bins,future_similarity=pair_stats,
                     ridge_candidates=[dict(alpha=alpha,B2_mse=score) for score,alpha in alpha_scores]))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--base_run_config",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--stride",type=int,default=8)
    parser.add_argument("--seed",type=int,default=2022)
    parser.add_argument("--start",type=int,default=0)
    parser.add_argument("--stop",type=int,default=255)
    parser.add_argument("--allow_legacy_reference_signature",action="store_true")
    cli=parser.parse_args()
    out=Path(cli.output); out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    source=json.loads(Path(cli.base_run_config).read_text())
    blocks,cfg=data_case(source,cli,out)
    raw_scale=StandardScaler().fit(blocks["A"]["raw"])
    common={name:raw_scale.transform(blocks[name]["raw"]).astype(np.float32) for name in BLOCKS}
    feature_cache=out/"descriptor_cache.npz"
    if feature_cache.exists():
        cache=np.load(feature_cache)
        features={block:{name:cache[f"{block}_{name}"] for name in NAMES} for block in BLOCKS}
    else:
        started=time.perf_counter()
        features={}
        validity={}
        for block in BLOCKS:
            features[block],validity[block]=descriptors(blocks[block]["history"],blocks[block]["y"],cfg["period"])
        cfg["descriptor_extract_time_sec"]=time.perf_counter()-started
        cfg["descriptor_validity"]=validity
        dump(out/"case_cache.json",cfg)
        np.savez_compressed(feature_cache,**{f"{block}_{name}":features[block][name] for block in BLOCKS for name in NAMES})
    cfg.setdefault("descriptor_extract_time_sec",0.0)
    fitted=robust_groups(features["A"],NAMES)
    dump(out/"feature_validity.json",{name:{"valid_mask":fitted[name][2].tolist(),"feature_dim":DIMS[int(name[1:])-1]} for name in NAMES})
    global_file=out/"global_probe.json"
    if global_file.exists():
        global_result=json.loads(global_file.read_text())
    else:
        scores=[]
        for alpha in ALPHAS:
            model=Ridge(alpha=alpha).fit(common["B1"],blocks["B1"]["y"])
            scores.append((mean_sq(model.predict(common["B2"]),blocks["B2"]["y"]),alpha))
        _,alpha=min(scores)
        model=Ridge(alpha=alpha).fit(common["B1"],blocks["B1"]["y"])
        prediction=model.predict(common["C"])
        global_result=dict(alpha=alpha,mse_C=mean_sq(prediction,blocks["C"]["y"]),
                           mae_C=mae(prediction,blocks["C"]["y"]),B2_mse=min(scores)[0])
        dump(global_file,global_result)
    pool=pair_pool(blocks["C"],cfg["span"],cli.seed+90)
    groups=combinations()
    if len(groups)!=255 or len(set(groups))!=255:
        raise AssertionError("combination registry must contain 255 unique entries")
    for index in range(cli.start,min(cli.stop,len(groups))):
        names=groups[index]
        tag="+".join(names)
        target=out/"combos"/(f"{index:03d}_"+tag+".json")
        if target.exists():
            print("SKIP",index,tag,flush=True);continue
        print("START",index,tag,flush=True)
        try:
            result=run_combo(names,blocks,features,fitted,common,pool,global_result,cfg,cli)
            dump(target,result)
            print("COMPLETE",index,tag,"MSE",result["summary"]["prototype_ridge_mse_C"],flush=True)
        except Exception as error:
            failure=out/"failures"/(f"{index:03d}_"+tag+".json")
            dump(failure,dict(index=index,method=tag,error=repr(error)))
            print("FAILED",index,tag,repr(error),flush=True)
    print("SHARD_FINISHED",cli.start,cli.stop,flush=True)


if __name__=="__main__":
    main()
