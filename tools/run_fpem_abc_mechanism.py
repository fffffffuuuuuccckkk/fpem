"""Mechanism diagnostics for the TEST-selected frozen-A0 G1/R3 checkpoints.

No existing model or checkpoint is changed. TEST selection here is explicitly
exploratory. Each stage has an independent completion marker for safe resume.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.predictive_env.dual_pattern_memory.rare_learning import RarePatternLearner
from models.predictive_env.dual_pattern_memory.forecast_fusion import (
    variation_forecast, prototype_segment_reliability,
)
from tools.fpem_abc_mechanism import (
    block_bootstrap_gain, by_time, chronological_blocks, effect, gain_row, nearest, pearson,
    rank_rows, residual_correction, sample_error, simple_key_drift,
    synthetic_g1, synthetic_r3, write_rows,
)
from tools.run_fpem_dppm_g1_r3_full import (
    collect, metric, record, residual_ridge, ridge_query, subset,
)
from tools.run_fpem_shared_factorization_g1r3 import load_experiment


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--phase", choices=("A", "B", "C", "all"), default="all")
    parser.add_argument("--test_stride", type=int, default=1)
    parser.add_argument("--train_stride", type=int, default=16)
    parser.add_argument("--val_stride", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--low_cost_only", action="store_true")
    parser.add_argument("--allow_legacy_reference_signature", action="store_true")
    return parser.parse_args()


def get_data(cli, output):
    source = Path(cli.checkpoint_dir)
    config = json.loads((source / "run_config.json").read_text())
    cli.base_run_config = config["cli"]["base_run_config"]
    cli.allow_legacy_reference_signature = config["cli"].get(
        "allow_legacy_reference_signature", False)
    state = torch.load(source / "model.pt", map_location="cpu", weights_only=False)
    original_metrics=json.loads((source/"metrics.json").read_text())
    cache = output / "records.pt"
    if cache.exists():
        saved = torch.load(cache, map_location="cpu", weights_only=False)
        if saved["sha"] != state["reference_sha256"]:
            raise ValueError("A0 checkpoint changed since record cache")
        saved["meta"].setdefault("selected_r3_gain",
            float(original_metrics["g1"]["mse"]-original_metrics["g1_r3"]["mse"]))
        return saved["data"], saved["meta"], state
    args, model, loaders, device, sha = load_experiment(cli)
    if sha != state["reference_sha256"]:
        raise ValueError("A0 reference SHA does not match selected G1/R3 checkpoint")
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise AssertionError("A0 must be frozen")
    raw_train = collect(model, loaders["train"].dataset.dataset, args, device, cli.train_stride)
    span = args.seq_len + args.pred_len
    cutoff = len(loaders["train"].dataset.dataset.data_x) - args.seq_len
    valid = raw_train["ids"] + span <= cutoff
    raw_train = subset(raw_train, valid)
    std = raw_train["x"].float().var(1, unbiased=False).add(1e-5).sqrt()
    floor = (std.median(0).values * .25).clamp_min(.01)
    raw_val = collect(model, loaders["val"].dataset.dataset, args, device, cli.val_stride)
    raw_test = collect(model, loaders["test"].dataset.dataset, args, device, cli.test_stride)
    data = {}
    for name, raw in (("train", raw_train), ("val", raw_val), ("test", raw_test)):
        data[name] = record(raw, floor)
        data[name]["ids"] = raw["ids"].long()
        data[name]["history_mean"] = raw["x"].float().mean(1, keepdim=True)
    meta = {"backbone": args.backbone, "dataset": args.dataset_name,
            "pred_len": args.pred_len, "seq_len": args.seq_len,
            "span": span, "sha": sha,
            "selected_r3_gain":float(original_metrics["g1"]["mse"]-
                                     original_metrics["g1_r3"]["mse"]),
            "source": str(source), "train_cutoff": cutoff,
            "test_start_absolute": len(loaders["train"].dataset.dataset.data_x)
                + len(loaders["val"].dataset.dataset.data_x) - 2*args.seq_len,
            "train_size": len(data["train"]["key"]),
            "val_size": len(data["val"]["key"]),
            "test_size": len(data["test"]["key"])}
    torch.save({"data": data, "meta": meta, "sha": sha}, cache)
    return data, meta, state


def g1(state):
    return {"weight": state["g1_weight"], "scale": float(state["g1_scale"])}


def anchor(data, state):
    return data["base"] + float(state["g1_scale"]) * ridge_query(
        data["key"], state["g1_weight"])


def add_meta(rows, meta):
    for row in rows:
        row.update({key: meta[key] for key in ("backbone", "dataset", "pred_len")})
        row["selection_protocol"] = "TEST_selected_exploratory"
    return rows


def make_plot(path, x, y, title, xlabel, ylabel):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, marker="o", linewidth=1.4)
    ax.axhline(0, color="gray", linewidth=.7)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def a_studies(data, meta, state, output):
    train, test = data["train"], data["test"]
    stable = anchor(test, state)
    delta = stable - test["base"]
    gains = effect(test, stable, test["base"])
    rows = []
    for n in (5, 10, 20):
        block = by_time(gains, n)
        for j, value in enumerate(block):
            rows.append(gain_row("A1", "TEST", value, block_count=n, block=j,
                                 worst_block_gain=float(min(x.mean() for x in block)),
                                 independent_block_unit=True))
    rows.append(gain_row("A1","TEST_block_bootstrap",gains,
                         **block_bootstrap_gain(gains)))
    make_plot(output / "figures" / "A1_g1_block_gain.png", range(20),
              [x.mean() for x in by_time(gains, 20)], "G1 gain by TEST time block",
              "TEST time block", "A0 MSE - G1 MSE")
    blocks = chronological_blocks(train["ids"], 5)
    fitted = []
    # A2: past-only source, embargo by complete source future.
    for source_index, source_block in enumerate(blocks[:-1]):
        source = source_block
        w = residual_ridge(train["key"][source],
                           (train["truth"]-train["base"])[source], .1)
        fitted.append(w)
        for target_index in range(source_index + 1, len(blocks)):
            hold = blocks[target_index]
            allowed = train["ids"][hold] >= (
                train["ids"][source].max() + meta["span"])
            hold = hold[allowed]
            if len(hold) < 4:
                continue
            piece = subset(train, torch.isin(torch.arange(len(train["ids"])), hold))
            pred = piece["base"] + state["g1_scale"] * ridge_query(piece["key"], w)
            rows.append(gain_row("A2", "TRAIN_forward", effect(piece, pred, piece["base"]),
                                 source_block=source_index, target_block=target_index,
                                 source_future_end=int(train["ids"][source].max()+meta["span"]),
                                 target_history_start=int(train["ids"][hold].min())))
    # A3 is explicitly offline leave-one-time-block-out, NOT temporal transfer.
    for j, hold in enumerate(blocks):
        fit = torch.cat([item for i, item in enumerate(blocks) if i != j])
        w = residual_ridge(train["key"][fit],
                           (train["truth"]-train["base"])[fit], .1)
        piece = subset(train, torch.isin(torch.arange(len(train["ids"])), hold))
        pred = piece["base"] + state["g1_scale"] * ridge_query(piece["key"], w)
        rows.append(gain_row("A3", "TRAIN_offline_LOBO", effect(piece, pred, piece["base"]),
                             omitted_block=j, contains_future_training=True))
    # A4: evaluate different block maps on the identical TEST history set.
    corrections = [ridge_query(test["key"], w).reshape(-1) for w in fitted]
    for i in range(len(corrections)):
        for j in range(i+1, len(corrections)):
            x, y = corrections[i], corrections[j]
            rows.append({"experiment": "A4", "split": "TEST_history_only",
                         "source_a": i, "source_b": j,
                         "direction_cosine": float(F.cosine_similarity(x, y, dim=0)),
                         "relative_distance": float((x-y).norm()/(x.norm()+1e-8)),
                         "sign_agreement": float(((x*y)>0).float().mean()),
                         "gain_mse": float("nan")})
    # A5 forward gain as a function of temporal source-target block distance.
    for item in [r for r in rows if r["experiment"] == "A2"]:
        rows.append({**item, "experiment": "A5",
                     "time_distance_blocks": item["target_block"]-item["source_block"]})
    # A6 compares same support count over different temporal spans.
    for j in range(2, len(blocks)):
        hold = blocks[j]
        eligible = train["ids"] + meta["span"] <= train["ids"][hold].min()
        before = eligible.nonzero().flatten()
        if len(before) < 8:
            continue
        budget = min(len(before), len(blocks[j-1]))
        candidates = {
            "all_past": before[torch.linspace(0, len(before)-1, budget).long()],
            "recent": before[-budget:], "earliest": before[:budget]}
        piece = subset(train, torch.isin(torch.arange(len(train["ids"])), hold))
        for label, selected in candidates.items():
            w = residual_ridge(train["key"][selected],
                               (train["truth"]-train["base"])[selected], .1)
            pred = piece["base"]+state["g1_scale"]*ridge_query(piece["key"], w)
            rows.append(gain_row("A6", "TRAIN_forward", effect(piece, pred, piece["base"]),
                                 source=label, target_block=j, support=budget))
    # A7: fixed TRAIN feature statistics; no TEST future in drift definition.
    drift = simple_key_drift(train["key"], test["key"])
    trend = (test["key"][..., -8:].mean(-1)-test["key"][..., :8].mean(-1)).abs().mean(-1).numpy()
    train_frequency=torch.fft.rfft(train["key"],dim=-1).abs().square().mean((0,1))
    test_frequency=torch.fft.rfft(test["key"],dim=-1).abs().square().mean(1)
    frequency=((test_frequency-train_frequency).square().mean(-1)/
               train_frequency.square().mean().clamp_min(1e-8)).numpy()
    for name, value in (("key_shift_mahalanobis_proxy", drift),
                        ("trend_shift", trend),("frequency_shift",frequency)):
        threshold = np.median(value)
        for group, mask in (("low", value <= threshold), ("high", value > threshold)):
            rows.append(gain_row("A7", "TEST", gains[mask],
                                 drift_kind=name, drift_group=group,
                                 feature_threshold=float(threshold)))
    # Simple Gaussian-kernel MMD, computed from history keys only, at block
    # level to avoid treating overlapping windows as independent evidence.
    src=train["key"].mean(1)[::max(1,len(train["key"])//128)][:128]
    bandwidth=torch.pdist(src).square().median().clamp_min(.01)
    mmd=[]
    for block in by_time(np.arange(len(test["key"])),10):
        dst=test["key"][block].mean(1)[::max(1,len(block)//128)][:128]
        xx=torch.exp(-torch.cdist(src,src).square()/bandwidth).mean()
        yy=torch.exp(-torch.cdist(dst,dst).square()/bandwidth).mean()
        xy=torch.exp(-torch.cdist(src,dst).square()/bandwidth).mean()
        mmd.append(float(xx+yy-2*xy))
    for j,(value,gain_block) in enumerate(zip(mmd,by_time(gains,10))):
        rows.append(gain_row("A7","TEST_block",gain_block,
                             drift_kind="RBF_MMD2",block=j,drift_value=value))
    # A8: true error enters evaluation ONLY, never the prediction path.
    error = test["truth"]-test["base"]
    alignment = (delta*error).mean((1,2)).numpy()
    over = (delta.square() > 2*delta*error).float().mean((1,2)).numpy()
    rows.append(gain_row("A8", "TEST", gains,
                         correction_error_cosine=float(F.cosine_similarity(
                             delta.reshape(len(delta),-1), error.reshape(len(error),-1),dim=-1).mean()),
                         direction_positive_ratio=float((alignment>0).mean()),
                         overcorrection_ratio=float(over.mean())))
    # A9: simpler baselines and wrong TRAIN historical-future association.
    y_train = train["truth"]-train["base"]
    intercept = y_train.mean((0,1),keepdim=True)
    period = train["ids"] % 24
    periodic = torch.stack([y_train[period==h].mean((0,1)) if (period==h).any()
                            else intercept[0,0] for h in range(24)])
    rng = torch.Generator().manual_seed(2021)
    shuffle = y_train[torch.randperm(len(y_train),generator=rng)]
    bad_map = residual_ridge(train["key"], shuffle, .1)
    variants = {
        "intercept": intercept.expand_as(delta),
        "periodic_24": periodic[(test["ids"] % 24)][:,None].expand_as(delta),
        "history_ridge": ridge_query(test["key"],state["g1_weight"]),
        "mismatched_ridge": ridge_query(test["key"],bad_map)}
    for name, candidate in variants.items():
        pred = test["base"]+state["g1_scale"]*candidate
        rows.append(gain_row("A9", "TEST", effect(test,pred,test["base"]),
                             baseline=name))
    rows.append(synthetic_g1())
    return add_meta(rows,meta)


def predict_custom(data, state, learner, reliability=None, offset=None, mode="normal",
                   permutation=None, chunk=24):
    stable = anchor(data,state)
    reliability = state["reliability"] if reliability is None else reliability
    offset = state["offset"] if offset is None else offset
    outputs = []
    rng = torch.Generator().manual_seed(2021)
    for start in range(0,len(stable),chunk):
        stop = min(len(stable),start+chunk)
        query = learner.query(data["key"][start:stop],offset)
        candidate, weight = query["candidate"],query["weight"]
        if mode == "permuted_expert":
            candidate = candidate[:,:,permutation]
        elif mode == "random_expert":
            weight = weight[:,:,torch.randperm(weight.shape[-1],generator=rng)]
        elif mode == "shuffled_horizon":
            candidate = candidate[...,torch.randperm(candidate.shape[-1],generator=rng)]
        elif mode == "no_expert":
            weight = torch.zeros_like(weight)
        elif mode == "shared_only":
            design=torch.cat((data["key"][start:stop],
                data["key"][start:stop].new_ones(*data["key"][start:stop].shape[:2],1)),-1)
            shared=design@learner.net.reference
            candidate=shared[:,:,None].expand_as(candidate)
        elif mode == "no_null":
            weight = weight/weight.sum(-1,keepdim=True).clamp_min(1e-8)
        elif mode == "dense_route":
            weight = query["score"].softmax(-1)
        empty = stable.new_zeros(stable.shape[-1],0)
        prediction,_ = variation_forecast(stable[start:stop],candidate,weight,
                                           reliability,empty,True,8)
        outputs.append(prediction)
    return torch.cat(outputs)


def b_studies(data, meta, state, learner, output, repeats):
    train,test = data["train"],data["test"]
    stable = anchor(test,state)
    stats = residual_correction(test,g1(state),learner,state["reliability"],state["offset"])
    actual = stats["full"]
    gain = effect(test,actual,stable)
    rows=[]
    rows.append(gain_row("B0","TEST",gain,**block_bootstrap_gain(gain)))
    flat_train = train["key"].reshape(-1,64)
    flat_test = test["key"].reshape(-1,64)
    ix,dist,valid = nearest(flat_train,flat_test,k=8)
    nearest_distance = dist[:,0].reshape(len(test["key"]),-1).mean(-1).numpy()
    density = np.quantile(nearest_distance,np.linspace(0,1,6))
    for group in range(5):
        mask = (nearest_distance >= density[group]) & (
            nearest_distance < density[group+1] if group<4 else nearest_distance<=density[group+1])
        rows.append(gain_row("B1","TEST",gain[mask],density_quintile=group+1,
                             count=int(mask.sum()),mean_distance=float(nearest_distance[mask].mean())))
    for field in ("null","active","reliability"):
        score = stats[field].float().mean(-1).numpy()
        cut = np.quantile(score,[0,.33,.67,1])
        for group in range(3):
            mask = (score>=cut[group]) & (score < cut[group+1] if group<2 else score<=cut[group+1])
            if mask.any():
                rows.append(gain_row("B2","TEST",gain[mask],kind=field,tercile=group+1,
                                     negative_transfer_ratio=float((gain[mask]<0).mean())))
    for segment in range(stats["seg_gain"].shape[-1]):
        rows.append({"experiment":"B3","split":"TEST","future_segment":segment+1,
                     "gain_mse":float(stats["seg_gain"][...,segment].mean()),
                     "correction_strength":float(stats["seg_effect"][...,segment].mean()),
                     "reliability":float(stats["seg_reliability"][...,segment].mean())})
    # B4: routing alternatives, all using the same selected checkpoint.
    for mode in ("normal","random_expert","no_expert"):
        pred = actual if mode=="normal" else predict_custom(test,state,learner,mode=mode)
        rows.append(gain_row("B4","TEST",effect(test,pred,stable),routing=mode,
                             intervention="inference_only"))
    # Direct historical future KNN correction uses TRAIN labels only.
    historical = (train["truth"]-anchor(train,state)).reshape(-1,meta["pred_len"])
    knn = historical[ix].mean(1).reshape_as(stable)
    rows.append(gain_row("B4","TEST",effect(test,stable+knn,stable),routing="KNN_future",
                         intervention="TRAIN_memory_only"))
    farthest=[]
    for begin in range(0,len(flat_test),256):
        distance=torch.cdist(flat_test[begin:begin+256],flat_train)
        farthest.append(distance.topk(8,largest=True).indices)
    far_index=torch.cat(farthest)
    far_correction=historical[far_index].mean(1).reshape_as(stable)
    rows.append(gain_row("B4","TEST",effect(test,stable+far_correction,stable),
                         routing="farthest_history",intervention="TRAIN_memory_only"))
    far = historical[torch.randint(0,len(historical),(len(ix),8),generator=torch.Generator().manual_seed(2021))].mean(1).reshape_as(stable)
    rows.append(gain_row("B4","TEST",effect(test,stable+far,stable),routing="random_history",
                         intervention="TRAIN_memory_only"))
    # B5 expert identity mismatch: route held fixed, 20 seeded permutations.
    for repeat in range(repeats):
        permutation=torch.randperm(learner.patterns,generator=torch.Generator().manual_seed(9000+repeat))
        pred=predict_custom(test,state,learner,mode="permuted_expert",permutation=permutation)
        rows.append(gain_row("B5","TEST",effect(test,pred,stable),repeat=repeat,
                             permutation="-".join(map(str,permutation.tolist())),
                             intervention="inference_only"))
    for mode in ("shuffled_horizon","no_expert"):
        pred=predict_custom(test,state,learner,mode=mode)
        rows.append(gain_row("B7","TEST",effect(test,pred,stable),mode=mode,
                             intervention="inference_only"))
    for mode in ("no_null","zero_reliability","dense_route","shared_only","no_expert"):
        if mode=="zero_reliability":
            pred=predict_custom(test,state,learner,reliability=torch.zeros_like(state["reliability"]))
        else:
            pred=predict_custom(test,state,learner,mode=mode)
        rows.append(gain_row("B9","TEST",effect(test,pred,stable),mode=mode,
                             intervention="inference_only_not_retrained"))
    # B8: same input/output, non-prototype controls; TRAIN labels only.
    target=train["truth"]-anchor(train,state)
    for rank in (8,16):
        x=train["key"].reshape(-1,64)
        y=target.reshape(-1,meta["pred_len"])
        u,s,vh=torch.linalg.svd(y,full_matrices=False)
        basis=vh[:rank].T
        coefficients=y@basis
        weight=residual_ridge(train["key"],coefficients.reshape_as(
            train["key"][...,:rank]),.1)
        correction=ridge_query(test["key"],weight)@basis.T
        rows.append(gain_row("B8","TEST",effect(test,stable+correction,stable),
                             control=f"global_low_rank_{rank}",
                             parameters=int(weight.numel()+basis.numel()),
                             intervention="TRAIN_fit"))
    # MLP control: same 64-history key, full future output, fixed 10-epoch budget.
    torch.manual_seed(2021)
    net=torch.nn.Sequential(torch.nn.Linear(64,16),torch.nn.GELU(),
                            torch.nn.Linear(16,meta["pred_len"]))
    optim=torch.optim.Adam(net.parameters(),lr=1e-3)
    x=train["key"].reshape(-1,64); y=target.reshape(-1,meta["pred_len"])
    for epoch in range(10):
        for part in torch.randperm(len(x),generator=torch.Generator().manual_seed(2021+epoch)).split(256):
            loss=F.mse_loss(net(x[part]),y[part]); optim.zero_grad(); loss.backward(); optim.step()
    with torch.no_grad():
        correction=net(test["key"].reshape(-1,64)).reshape_as(stable)
    rows.append(gain_row("B8","TEST",effect(test,stable+correction,stable),
                         control="global_MLP_16",parameters=sum(p.numel() for p in net.parameters()),
                         intervention="TRAIN_fit"))
    rows.append(synthetic_r3())
    # B6 and B10 require independent retraining; separate resumable stage below.
    return add_meta(rows,meta),stats,{"nearest_index":ix,"nearest_distance":dist}


def retrain_b(data, meta, state, output, repeats):
    train,val,test=data["train"],data["val"],data["test"]
    target=train["truth"]
    source=Path(output)/"B_retrain_trials.csv"
    existing=[]
    if source.exists():
        import csv
        with source.open(encoding="utf-8") as file: existing=list(csv.DictReader(file))
    done={str(row["trial"]) for row in existing}
    rows=[]
    saved=state["rare"]
    original=RarePatternLearner.from_export(saved)
    config=saved.get("optimizer_config",{})
    stable=anchor(test,state)
    # Temporal-block permutations preserve source counts and gross time context.
    blocks=chronological_blocks(train["ids"],4)
    for repeat in range(repeats):
        trial=f"B6_mismatch_{repeat:02d}"
        if trial in done: continue
        perm=torch.arange(len(target))
        for block in blocks:
            perm[block]=block[torch.randperm(len(block),generator=torch.Generator().manual_seed(3100+repeat*4+int(block[0])))]
        learner=RarePatternLearner("R3",patterns=saved["patterns"],rank=saved["rank"],
                                   seed=2021+repeat,epochs=int(config.get("epochs",10)),
                                   optimizer_config=config)
        mismatch_reference=residual_ridge(train["key"],target[perm],100.)
        learner.fit(train["key"],target[perm],{"reference_map":mismatch_reference,
                    "starts":train["ids"],"span":meta["span"],"device":"cpu",
                    "validation":(val["key"],val["truth"])})
        val_query=learner.query(val["key"])
        calibrated,_=prototype_segment_reliability(anchor(val,state),
                    val_query["candidate"],val["truth"],val_query["weight"],8)
        pred=predict_custom(test,state,learner,reliability=calibrated)
        rows.append(gain_row("B6","TEST",effect(test,pred,stable),trial=trial,
                             shuffled_within_train_time_blocks=True,
                             intervention="retrained_R3_same_budget"))
        write_rows(source,existing+add_meta(rows,meta))
    # B10: fit each TRAIN time block, evaluate on later chronological blocks.
    for j,block in enumerate(blocks[:-1]):
        trial=f"B10_block_{j}"
        if trial in done: continue
        later=torch.cat(blocks[j+1:])
        later=later[train["ids"][later]>=train["ids"][block].max()+meta["span"]]
        if len(later)<8: continue
        learner=RarePatternLearner("R3",patterns=saved["patterns"],rank=saved["rank"],
                                   seed=2021,epochs=int(config.get("epochs",10)),
                                   optimizer_config=config)
        block_reference=residual_ridge(train["key"][block],target[block],100.)
        local_state={**state,"g1_weight":residual_ridge(train["key"][block],
                         (train["truth"]-train["base"])[block],.1)}
        learner.fit(train["key"][block],target[block],{"reference_map":block_reference,
                    "starts":train["ids"][block],"span":meta["span"],"device":"cpu"})
        piece=subset(train,torch.isin(torch.arange(len(train["ids"])),later))
        # This is a forward transfer diagnostic. Use source-block reliability
        # only, never calibrate it on the held-out future block.
        source_piece=subset(train,torch.isin(torch.arange(len(train["ids"])),block))
        source_query=learner.query(source_piece["key"])
        local_reliability,_=prototype_segment_reliability(anchor(source_piece,local_state),
            source_query["candidate"],source_piece["truth"],source_query["weight"],8)
        pred=predict_custom(piece,local_state,learner,reliability=local_reliability)
        rows.append(gain_row("B10","TRAIN_forward",effect(piece,pred,anchor(piece,local_state)),
                             trial=trial,source_block=j,
                             source_future_end=int(train["ids"][block].max()+meta["span"]),
                             target_history_start=int(train["ids"][later].min()),
                             intervention="retrained_R3_same_budget"))
        write_rows(source,existing+add_meta(rows,meta))
    all_rows=existing+add_meta(rows,meta)
    mismatched=[float(row["gain_mse"]) for row in all_rows
                if str(row["trial"]).startswith("B6_mismatch_")]
    if mismatched:
        actual=meta.get("selected_r3_gain")
        if actual is not None:
            summary={"experiment":"B6","trial":"B6_empirical_significance",
                     "split":"TEST","gain_mse":float(actual),
                     "mismatch_mean_gain":float(np.mean(mismatched)),
                     "mismatch_std_gain":float(np.std(mismatched)),
                     "empirical_p_true_beats_mismatch":float((1+sum(x>=actual for x in mismatched))/
                         (len(mismatched)+1)),"repeats":len(mismatched)}
            all_rows.append(summary)
            write_rows(source,all_rows)
    return all_rows


def main():
    cli=arguments(); torch.set_num_threads(4)
    output=Path(cli.output); output.mkdir(parents=True,exist_ok=True)
    data,meta,state=get_data(cli,output)
    learner=RarePatternLearner.from_export(state["rare"])
    manifest={"source":str(cli.checkpoint_dir),"meta":meta,
              "A_codes":[f"A{i}" for i in range(1,11)],
              "B_codes":[f"B{i}" for i in range(1,12)],
              "C_codes":[f"C{i}" for i in range(1,13)],
              "test_selection_warning":"TEST is an exploratory development set",
              "A0_frozen":True,"original_model_unchanged":True}
    (output/"experiment_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    print("START",meta,flush=True)
    if cli.phase in ("A","all") and not (output/"A.complete").exists():
        rows=a_studies(data,meta,state,output)
        write_rows(output/"A_all_results.csv",rows)
        write_rows(output/"A_ranking.csv",rank_rows(rows))
        (output/"A.complete").write_text("complete\n")
        print("A COMPLETE",len(rows),flush=True)
    b_marker="B_low_cost.complete" if cli.low_cost_only else "B.complete"
    if cli.phase in ("B","all") and not (output/b_marker).exists():
        rows,stats,neighbours=b_studies(data,meta,state,learner,output,cli.repeats)
        write_rows(output/"B_all_results.csv",rows)
        torch.save({"stats":stats,"neighbours":neighbours},output/"B_compact.pt")
        retrain=[] if cli.low_cost_only else retrain_b(data,meta,state,output,cli.repeats)
        write_rows(output/"B_ranking.csv",rank_rows(rows+retrain))
        (output/b_marker).write_text("complete\n")
        print("B COMPLETE",len(rows)+len(retrain),flush=True)
    if not cli.low_cost_only and cli.phase in ("C","all") and not (output/"C.complete").exists():
        from tools.fpem_abc_risk import run_risk_studies
        run_risk_studies(data,meta,state,learner,output)
        (output/"C.complete").write_text("complete\n")
        print("C COMPLETE",flush=True)
    if all((output/f"{phase}.complete").exists() for phase in "ABC"):
        (output/"complete").write_text("complete\n")


if __name__=="__main__":
    main()
