"""F03 full-H prototype invariance with the *same* frozen R3 variation bank.

Every fitted label is TRAIN or purged VAL. TEST is evaluated once after VAL
selection and is never used for checkpoint, lambda, or reliability selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.predictive_env.direct_invariant_forecaster import DirectInvariantForecaster
from models.predictive_env.dual_pattern_memory.rare_learning import RarePatternLearner
from models.predictive_env.dual_pattern_memory.prototype_bank import normalized_pairs
from models.predictive_env.dual_pattern_memory.forecast_fusion import (
    prototype_segment_reliability, variation_forecast)
from models.predictive_env.prototype_invariance_losses import (
    prototype_risk_penalty, shared_optimum_penalty, prototype_risks)
from models.predictive_env.prototype_guided_invariant import routing_control
from models.predictive_env.eiil_environment import PredictiveConflictEnvironment
from tools.run_fpem_dppm_g1_r3_full import LR, OFFSETS, history_key, metric, residual_ridge, subset
from tools.run_fpem_shared_factorization_g1r3 import load_experiment
from tools.run_predictive_env_iv_patchtst import IndexedDataset


def cli_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_run_config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=2022)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--train_stride", type=int, default=16)
    p.add_argument("--val_stride", type=int, default=4)
    p.add_argument("--test_stride", type=int, default=1)
    p.add_argument("--lrs", default="0.0003,0.001")
    p.add_argument("--lambdas", default="0.001,0.01")
    p.add_argument("--gradient_normalization", choices=("l2", "rms"), default="l2")
    p.add_argument("--support_weighting", choices=("effective", "mass", "uniform"), default="effective")
    p.add_argument("--constraint_start_epoch", type=int, default=1)
    p.add_argument("--methods", default="F03,P01,F03_I1,F03_I2,F03_I3,P01_I1,P01_I2,P01_I3,F03_I1_random,F03_I1_shuffled,F03_I1_time,F03_I1_eiil")
    p.add_argument("--allow_legacy_reference_signature", action="store_true")
    return p.parse_args()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


@torch.no_grad()
def collect(model, dataset, args, device, stride):
    wrapped = IndexedDataset(dataset, cycle_len=args.cyclenet_cycle_len)
    loader = DataLoader(Subset(wrapped, range(0, len(dataset), stride)),
                        batch_size=args.batch_size, shuffle=False, num_workers=0)
    chunks = {k: [] for k in ("x", "hidden", "mean", "std", "truth", "base", "ids")}
    for x, y, ids, cycle in loader:
        x, cycle = x.float().to(device), cycle.to(device)
        hidden, mean, std = model.encode(x, cycle)
        base = model.frozen_h_head(hidden, cycle).transpose(1, 2) * std + mean
        values = dict(x=x, hidden=hidden, mean=mean, std=std,
                      truth=y[:, -args.pred_len:], base=base, ids=ids)
        for key, value in values.items():
            chunks[key].append(value.detach().cpu())
    return {key: torch.cat(value) for key, value in chunks.items()}


def prepare(raw, floor):
    truth, mean, scale = normalized_pairs(raw["x"], raw["truth"], floor)
    raw["key"] = history_key(raw["x"], 64).float()
    raw["r3_truth"] = truth.float()
    raw["scale"] = scale.float()
    raw["hist_mean"] = mean.float()
    raw["target_norm"] = ((raw["truth"]-raw["mean"])/raw["std"]).transpose(1, 2).float()
    raw["base_norm"] = ((raw["base"]-raw["mean"])/raw["std"]).transpose(1, 2).float()
    raw["base_hist_norm"] = ((raw["base"]-mean)/scale).transpose(1, 2).float()
    raw["truth_hist_norm"] = truth.float()
    return raw


def paired_r3(data, args, seed, output):
    """R3 is fit once, reused unchanged by all invariant and variation arms."""
    path = output / "r3.pt"
    if path.exists():
        return RarePatternLearner.from_export(torch.load(path, map_location="cpu", weights_only=False))
    train, val = data["train"], data["val"]
    starts = train["ids"].long()
    boundary = len(data["train_dataset"].data_x) - args.seq_len
    fit = starts + args.seq_len + args.pred_len <= boundary
    if not bool(fit.any()):
        raise ValueError("no eligible TRAIN samples for R3")
    # R3 checkpoint selection must not consume the later VAL calibration set.
    split = int(starts[fit].min()) + int((starts[fit].max()-starts[fit].min())*.7)
    early_train = fit & (starts + args.seq_len + args.pred_len <= split)
    later_train = fit & (starts >= split)
    fit_train = subset(train, early_train)
    valid = subset(train, later_train)
    if min(len(fit_train["key"]), len(valid["key"])) < 8:
        raise ValueError("insufficient purged R3 fit/validation support")
    if int((starts[early_train]+args.seq_len+args.pred_len).max()) > int(starts[later_train].min()):
        raise AssertionError("R3 TRAIN fit futures overlap R3 TRAIN validation histories")
    head_lr, dict_ratio, basis_ratio, route_ratio = LR[args.backbone]
    config = dict(head_lr=head_lr, dictionary_ratio=dict_ratio,
                  basis_ratio=basis_ratio, route_ratio=route_ratio,
                  optimizer="adam", schedule="constant")
    reference = residual_ridge(fit_train["key"], fit_train["r3_truth"], 100.)
    learner = RarePatternLearner("R3", patterns=8, rank=8, seed=seed,
                                 epochs=10, optimizer_config=config)
    learner.fit(fit_train["key"], fit_train["r3_truth"],
                {"reference_map": reference, "starts": fit_train["ids"],
                 "span": args.seq_len + args.pred_len, "device": "cpu",
                 "validation": (valid["key"], valid["r3_truth"])})
    torch.save(learner.export(), path)
    write_json(output / "r3_protocol.json", {"fit_count": len(fit_train["key"]),
        "validation_count": len(valid["key"]), "optimizer": config,
        "validation_source": "later purged TRAIN, separate from VAL reliability",
        "q_detached": True, "same_bank_for_invariance_and_variation": True})
    return learner


def route_data(data, learner):
    for split in ("train", "val", "test"):
        query = learner.query(data[split]["key"])
        data[split]["q"] = query["weight"].detach().float()
        data[split]["null"] = query["null_weight"].detach().float()
        data[split]["candidate"] = query["candidate"].detach().float()


def prediction(net, data, device, batch=64):
    net.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(data["hidden"]), batch):
            stop = start + batch
            h = data["hidden"][start:stop].float().to(device)
            norm = net(h).transpose(1, 2).cpu()
            parts.append(norm * data["std"][start:stop] + data["mean"][start:stop])
    return torch.cat(parts)


def consensus_prediction(data, teacher):
    """Deployable H-space consensus; uses no future label or environment q."""
    h = data["hidden"].float().permute(0, 1, 3, 2).flatten(-2)
    norm = h @ teacher["weight"].float() + teacher["bias"].float()
    return norm.transpose(1, 2) * data["std"] + data["mean"]


def weighted_prediction(net, data, device, teacher, alpha):
    base = prediction(net, data, device)
    if alpha == 0.0:
        return base
    return base + float(alpha) * (consensus_prediction(data, teacher) - base)


def learn_consensus_weight(net, train, val, teacher, device):
    """Fit one convex mixing weight on purged TRAIN holdout, select on VAL.

    The consensus teacher is fitted on an earlier TRAIN block. This weight sees
    only its later, non-overlapping TRAIN block. It does not use TEST targets.
    """
    if teacher is None:
        raise ValueError("I3 blend requires a TRAIN consensus teacher")
    fit = teacher["hold"].bool()
    if int(fit.sum()) < 12:
        raise ValueError("insufficient purged TRAIN support for blend weight")
    f_train = prediction(net, train, device)[fit]
    c_train = consensus_prediction(train, teacher)[fit]
    delta = c_train - f_train
    residual = train["truth"][fit] - f_train
    # Sufficient statistics make scalar training independent of sequence size.
    curvature = delta.square().mean().detach()
    slope = (delta * residual).mean().detach()
    f_val = prediction(net, val, device)
    c_val = consensus_prediction(val, teacher)
    alpha = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([alpha], lr=0.05)
    candidates = [0.0]
    if float(curvature) > 1e-12:
        for _ in range(80):
            optimizer.zero_grad(set_to_none=True)
            # Equivalent to mean squared error of the convex Y-space blend.
            (curvature * alpha.square() - 2.0 * slope * alpha).backward()
            optimizer.step()
            with torch.no_grad():
                alpha.clamp_(0.0, 1.0)
            candidates.append(float(alpha.detach()))
    scored = [(raw_metric(f_val + a * (c_val - f_val), val["truth"])["mse"], a)
              for a in sorted(set(candidates))]
    selected_val, selected_alpha = min(scored, key=lambda item: item[0])
    return selected_alpha, {
        "weight": selected_alpha,
        "train_holdout_count": int(fit.sum()),
        "train_unconstrained_optimum": float(slope / curvature) if float(curvature) > 1e-12 else 0.0,
        "train_curvature": float(curvature),
        "baseline_val_mse": raw_metric(f_val, val["truth"])["mse"],
        "selected_val_mse": selected_val,
        "selected_on": "VAL; weight fitted on purged later TRAIN only",
        "test_used_for_selection": False,
    }


def raw_metric(pred, truth):
    delta = (pred-truth).float()
    return {"mse": float(delta.square().mean()), "mae": float(delta.abs().mean())}


def group_q(train, method, seed):
    if method.endswith("_random"):
        return routing_control(train["q"], "random", seed=seed)
    if method.endswith("_shuffled"):
        return routing_control(train["q"], "shuffled", seed=seed)
    if method.endswith("_time"):
        return routing_control(train["q"], "time", history_ids=train["ids"])
    if method.endswith("_eiil"):
        eiil = PredictiveConflictEnvironment(len(train["hidden"]),
            env_num=train["q"].shape[-1], steps=20, seed=seed,
            quality_diagnostics=False, random_partition_repeats=1)
        eiil.update(train["base"], train["truth"],
                    torch.arange(len(train["hidden"])), source_split="train")
        return eiil.q[:, None].expand_as(train["q"]) * train["q"].sum(-1, keepdim=True)
    return train["q"].detach()


def _weighted_dual(x, y, w, ridge=0.1):
    """Full flattened H-space ridge; dual solve avoids a huge feature Gram."""
    x, y, w = x.double(), y.double(), w.double().clamp_min(0)
    mass = w.sum().clamp_min(1e-8)
    xm, ym = (w[:, None]*x).sum(0)/mass, (w[:, None]*y).sum(0)/mass
    xc, yc = x-xm, y-ym
    sw = w.sqrt()
    design = xc*sw[:, None]
    target = yc*sw[:, None]
    gram = design @ design.T / max(1, x.shape[1])
    alpha = torch.linalg.solve(gram + ridge*torch.eye(len(x), dtype=x.dtype), target)
    coef = (design.T @ alpha)/max(1, x.shape[1])
    return coef.float(), (ym-xm@coef.double()).float()


def consensus_teacher(train, seed, output, q_override=None, audit_name="i3_transfer.json"):
    """TRAIN early-fit / later-purged transfer audit; not a plain map average claim."""
    ids = train["ids"].long()
    span = train["x"].shape[1] + train["truth"].shape[1]
    lo, hi = int(ids.min()), int(ids.max())
    cut = lo + (hi-lo)//3
    fit = ids + span <= cut
    hold = ids >= cut
    if min(int(fit.sum()), int(hold.sum())) < 12:
        return None, {"reason": "insufficient temporal transfer support"}
    h = train["hidden"].float().permute(0, 1, 3, 2).flatten(-2)
    x = h.reshape(-1, h.shape[-1])
    y = train["target_norm"].reshape(-1, train["target_norm"].shape[-1])
    q_source = train["q"] if q_override is None else q_override
    q = q_source.reshape(-1, q_source.shape[-1]).detach()
    row_fit = fit[:, None].expand(-1, h.shape[1]).reshape(-1)
    row_hold = hold[:, None].expand(-1, h.shape[1]).reshape(-1)
    rows = row_fit.nonzero().flatten()
    if len(rows) > 384:
        rows = rows[torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed))[:384]]
    global_w, global_b = _weighted_dual(x[rows], y[rows], q[rows].sum(-1).clamp_min(1e-3))
    maps, biases, support = [], [], []
    for j in range(q.shape[-1]):
        w, b = _weighted_dual(x[rows], y[rows], q[rows, j])
        maps.append(w); biases.append(b); support.append(float(q[rows, j].sum()))
    masses = torch.tensor(support).clamp_min(1e-8)
    # Same H coordinates; support-weighted consensus is only a *candidate*.
    candidate_w = (torch.stack(maps)*masses[:, None, None]).sum(0)/masses.sum()
    candidate_b = (torch.stack(biases)*masses[:, None]).sum(0)/masses.sum()
    transfers = []
    hold_indices = row_hold.nonzero().flatten()
    for block in torch.tensor_split(hold_indices, 2):
        if len(block) == 0:
            continue
        target = y[block]
        baseline = train["base_norm"].reshape(-1, y.shape[-1])[block]
        global_pred = x[block] @ global_w + global_b
        consensus = x[block] @ candidate_w + candidate_b
        transfers.append({"a0": raw_metric(baseline,target)["mse"],
                          "global": raw_metric(global_pred,target)["mse"],
                          "consensus": raw_metric(consensus,target)["mse"]})
    # Require a positive gain in *each* later block vs both A0 and ordinary
    # global Ridge, otherwise do not label the map cross-temporally stable.
    accepted = bool(transfers) and all(t["consensus"] < min(t["a0"],t["global"])
                                            for t in transfers)
    audit = {"fit_rows": len(rows), "transfer": transfers,
             "support": support, "accepted": accepted,
             "baseline": "full H global Ridge and A0; EPMD separate local benchmark"}
    write_json(output / audit_name, audit)
    # Transfer failure is reported, not silently converted into an unchanged
    # F03 run: I3 remains a real intervention whose value is decided on VAL.
    return {"weight": candidate_w, "bias": candidate_b, "hold": hold}, audit


def fit_one(method, lr, lam, train, val, model, args, device, seed,
            teacher, epochs, output, grad_mode="l2", support_mode="effective",
            constraint_start_epoch=1):
    rank = 16 if method.startswith("P01") else 0
    tokens, width = train["hidden"].shape[-2:]
    torch.manual_seed(seed)
    head = DirectInvariantForecaster(tokens,width,args.pred_len,rank,
                  model.h_forecast_head if rank == 0 else None).to(device)
    if rank == 0:
        initial = prediction(head,val,device)
        max_diff = float((initial-val["base"]).abs().max())
        if max_diff > 1e-5:
            raise AssertionError(f"F03 is not A0-function-preserving: {max_diff}")
    else:
        max_diff = None
    opt = torch.optim.Adam(head.parameters(),lr=lr)
    q_train = group_q(train,method,seed).detach()
    generator = torch.Generator().manual_seed(seed)
    best, trace = None, []
    kind = "I1" if "_I1" in method else "I2" if "_I2" in method else "I3" if "_I3" in method else "base"
    for epoch in range(epochs):
        head.train()
        total, count = 0.,0
        for indices in torch.randperm(len(train["hidden"]),generator=generator).split(args.batch_size):
            h = train["hidden"][indices].float().to(device)
            normalized = head(h)
            pred = normalized.transpose(1,2)*train["std"][indices].to(device)+train["mean"][indices].to(device)
            target = train["truth"][indices].to(device)
            future = (pred-target).square().mean()
            penalty = future.new_zeros(())
            if kind in ("I1","I2") and epoch+1 >= constraint_start_epoch:
                q = q_train[indices].to(device)
                truth = train["target_norm"][indices].to(device)
                if support_mode == "effective" and kind == "I1":
                    penalty,_ = prototype_risk_penalty(normalized,truth,q,kind="variance")
                elif support_mode == "effective" and kind == "I2":
                    penalty,_ = shared_optimum_penalty(normalized,truth,q,head.head.weight,variant="norm")
                else:
                    risks,_,support=prototype_risks(normalized,truth,q)
                    mass=support["mass"].to(device)
                    weights=mass if support_mode == "mass" else (mass>0).float()
                    rho=weights/weights.sum().clamp_min(1e-8)
                    if kind == "I1":
                        center=(rho*risks).sum()
                        penalty=(rho*(risks-center).square()).sum()
                    else:
                        gradients=[torch.autograd.grad(r,head.head.weight,
                            create_graph=True,retain_graph=True)[0] for r in risks]
                        penalty=sum(rho[j]*grad.square().sum()
                                    for j,grad in enumerate(gradients))
                if kind == "I2" and grad_mode == "rms":
                    penalty=penalty/head.head.weight.numel()
            elif kind == "I3" and teacher is not None and epoch+1 >= constraint_start_epoch:
                valid = teacher["hold"][indices]
                if bool(valid.any()):
                    feature = h[valid].permute(0,1,3,2).flatten(-2)
                    wanted = (feature@teacher["weight"].to(device)+teacher["bias"].to(device)).detach()
                    penalty = (normalized[valid]-wanted).square().mean()
            loss = future + lam*penalty
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite {method} objective")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
            opt.step()
            total += float(loss.detach()); count += 1
        val_pred = prediction(head,val,device)
        row = {"epoch":epoch+1,"train_objective":total/max(count,1),**raw_metric(val_pred,val["truth"])}
        trace.append(row)
        if best is None or row["mse"] < best["val"]["mse"]:
            best = {"epoch":epoch+1,"val":row,
                    "state":{k:v.detach().cpu().clone() for k,v in head.state_dict().items()}}
        print(method,lr,lam,"epoch",epoch+1,"VAL",row["mse"],flush=True)
    head.load_state_dict(best.pop("state"))
    output.mkdir(parents=True,exist_ok=True)
    torch.save(head.state_dict(),output/"head.pt")
    write_json(output/"trace.json",trace)
    write_json(output/"selection.json",{"method":method,"lr":lr,"lambda":lam,
               "best_epoch":best["epoch"],"val":best["val"],"initial_a0_max_abs":max_diff,
               "i3_teacher_active":kind=="I3" and teacher is not None})
    return best["val"]["mse"],head


def r3_fusion(anchor, cal, choice, test, learner):
    """Reliability is re-estimated for this anchor on an isolated calibration split."""
    anchor_cal = ((anchor(cal)-cal["hist_mean"])/cal["scale"]).transpose(1,2)
    reliability,evidence=prototype_segment_reliability(anchor_cal,cal["candidate"],
                                            cal["truth_hist_norm"],cal["q"],8)
    choice_anchor = ((anchor(choice)-choice["hist_mean"])/choice["scale"]).transpose(1,2)
    best=None
    empty=choice_anchor.new_zeros(choice_anchor.shape[-1],0)
    for offset in OFFSETS:
        query=learner.query(choice["key"],offset)
        full,_=variation_forecast(choice_anchor,query["candidate"],query["weight"],
                                  reliability,empty,True,8)
        score=metric({"truth":choice["truth_hist_norm"],"scale":choice["scale"]},full)
        if best is None or score["mse"]<best["val_mse"]:
            best={"offset":offset,"val_mse":score["mse"],"val_mae":score["mae"]}
    test_anchor=((anchor(test)-test["hist_mean"])/test["scale"]).transpose(1,2)
    query=learner.query(test["key"],best["offset"])
    empty=test_anchor.new_zeros(test_anchor.shape[-1],0)
    full,beta=variation_forecast(test_anchor,query["candidate"],query["weight"],
                                 reliability,empty,True,8)
    full_raw=full.transpose(1,2)*test["scale"]+test["hist_mean"]
    return (metric({"truth":test["truth_hist_norm"],"scale":test["scale"]},test_anchor),
            metric({"truth":test["truth_hist_norm"],"scale":test["scale"]},full),
            raw_metric(full_raw,test["truth"]),best,evidence,float(beta.mean()))


def main():
    cli=cli_args(); out=Path(cli.output); out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    load_cli=argparse.Namespace(**vars(cli))
    load_cli.seed=json.loads(Path(cli.base_run_config).read_text(encoding="utf-8"))["seed"]
    args,model,loaders,device,sha=load_experiment(load_cli)
    if any(p.requires_grad for p in model.parameters()):
        raise AssertionError("A0 Backbone must be frozen")
    torch.manual_seed(cli.seed)
    datasets={k:v.dataset.dataset for k,v in loaders.items()}
    data={k:collect(model,datasets[k],args,device,
                {"train":cli.train_stride,"val":cli.val_stride,"test":cli.test_stride}[k])
          for k in ("train","val","test")}
    std=data["train"]["x"].float().var(1,unbiased=False).add(1e-5).sqrt()
    floor=(std.median(0).values*.25).clamp_min(.01)
    data={k:prepare(v,floor) for k,v in data.items()}
    data["train_dataset"],data["val_dataset"]=datasets["train"],datasets["val"]
    learner=paired_r3(data,args,cli.seed,out)
    route_data(data,learner)
    train,val,test=data["train"],data["val"],data["test"]
    teacher,audit=consensus_teacher(train,cli.seed,out)
    control_teachers={}
    for mode in ("random","shuffled","time","eiil"):
        if any(m.endswith("_I3_"+mode) for m in cli.methods.split(",")):
            q_control=group_q(train,"F03_I3_"+mode,cli.seed)
            control_teachers[mode],_=consensus_teacher(
                train,cli.seed,out,q_override=q_control,
                audit_name="i3_transfer_"+mode+".json")
    ids=val["ids"].long(); length=len(datasets["val"].data_x)
    cut=length//2
    calmask=ids+args.seq_len+args.pred_len<=cut
    choosemask=(ids>=cut)&(ids+args.seq_len+args.pred_len<=length)
    if min(int(calmask.sum()),int(choosemask.sum()))<8:
        raise ValueError("insufficient disjoint VAL calibration/selection support; "
                         "TRAIN in-sample reliability is not allowed")
    if int((ids[calmask]+args.seq_len+args.pred_len).max())>int(ids[choosemask].min()):
        raise AssertionError("VAL calibration windows overlap selection history")
    cal,choice=subset(val,calmask),subset(val,choosemask)
    reliability_protocol="early purged VAL calibration; later purged VAL NULL selection"
    write_json(out/"run_config.json",{"cli":vars(cli),"A0_sha256":sha,
        "backbone":args.backbone,"dataset":args.dataset_name,"pred_len":args.pred_len,
        "r3_shared":True,"reliability_protocol":reliability_protocol,
        "i3_transfer":audit,"test_used_for_selection":False})
    methods=[x.strip() for x in cli.methods.split(",") if x.strip()]
    if "F03_I3_BLEND" in methods and "F03" not in methods:
        methods.insert(0, "F03")
    if "F03_I3_BLEND" in methods:
        methods = ["F03"] + [x for x in methods if x != "F03"]
    if cli.constraint_start_epoch < 1 or cli.constraint_start_epoch > cli.epochs:
        raise ValueError("constraint_start_epoch must be within training epochs")
    lrs=[float(x) for x in cli.lrs.split(",")]
    lambdas=[float(x) for x in cli.lambdas.split(",")]
    summary=[]
    for method in methods:
        method_out=out/method; method_out.mkdir(exist_ok=True)
        if (method_out/"complete").exists():
            summary.append(json.loads((method_out/"metrics.json").read_text())); continue
        blend_audit = None
        if method == "F03_I3_BLEND":
            f03_metrics = json.loads((out/"F03"/"metrics.json").read_text(encoding="utf-8"))
            f03_trial = Path(f03_metrics["selected"]["trial"])
            head = DirectInvariantForecaster(train["hidden"].shape[-2],
                       train["hidden"].shape[-1],args.pred_len,0).to(device)
            head.load_state_dict(torch.load(f03_trial/"head.pt",map_location=device,weights_only=False))
            alpha,blend_audit=learn_consensus_weight(head,train,val,teacher,device)
            best={"value":blend_audit["selected_val_mse"],"trial":str(f03_trial),
                  "lr":f03_metrics["selected"]["lr"],"lambda":0.0,
                  "blend_alpha":alpha}
            write_json(method_out/"blend_weight.json",blend_audit)
            anchor=lambda block: weighted_prediction(head,block,device,teacher,alpha)
        else:
            best=None
            for lr in lrs:
                for lam in ([0.] if method in ("F03","P01") else lambdas):
                    trial=method_out/f"lr_{lr:g}_lambda_{lam:g}"
                    if (trial/"selection.json").exists() and (trial/"head.pt").exists():
                        sel=json.loads((trial/"selection.json").read_text()); value=sel["val"]["mse"]
                    else:
                        method_teacher=control_teachers.get(method.rsplit("_",1)[-1],teacher) if "_I3_" in method else teacher
                        value,_=fit_one(method,lr,lam,train,val,model,args,device,cli.seed,
                                        method_teacher,cli.epochs,trial,
                                        cli.gradient_normalization,cli.support_weighting,
                                        cli.constraint_start_epoch)
                    if best is None or value<best["value"]:
                        best={"value":value,"trial":str(trial),"lr":lr,"lambda":lam}
            head=DirectInvariantForecaster(train["hidden"].shape[-2],train["hidden"].shape[-1],
                      args.pred_len,16 if method.startswith("P01") else 0).to(device)
            head.load_state_dict(torch.load(Path(best["trial"])/"head.pt",map_location=device,weights_only=False))
            anchor=lambda block: prediction(head,block,device)
        invariant=raw_metric(anchor(test),test["truth"])
        stable_norm,full_norm,full_raw,choice_info,evidence,beta=r3_fusion(anchor,cal,choice,test,learner)
        row={"method":method,"selected":best,"val_mse":best["value"],
             "a0":raw_metric(test["base"],test["truth"]),"invariant":invariant,
             "r3":full_raw,
             "invariant_history_norm":stable_norm,"r3_history_norm":full_norm,
             "variation_gain":stable_norm["mse"]-full_norm["mse"],
             "reliability_null_selection":choice_info,"reliability_mean":float(sum(e["reliability"] for e in evidence)/len(evidence)),
             "r3_beta_mean":beta,"i3_teacher_active":teacher is not None,
             "A0_sha256":sha,"reliability_protocol":reliability_protocol}
        if blend_audit is not None:
            row["blend_weight"] = blend_audit
        write_json(method_out/"metrics.json",row)
        write_json(method_out/"reliability_evidence.json",evidence)
        (method_out/"complete").write_text("complete\n",encoding="utf-8")
        summary.append(row)
        print("COMPLETE",args.backbone,args.dataset_name,args.pred_len,method,invariant,full_norm,flush=True)
    write_json(out/"summary.json",summary)
    (out/"complete").write_text("complete\n",encoding="utf-8")


if __name__=="__main__":
    main()
