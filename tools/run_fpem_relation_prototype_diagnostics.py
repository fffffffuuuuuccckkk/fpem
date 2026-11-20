#!/usr/bin/env python3
"""P0 Raw vs P1 relation fingerprint vs P2 function-behavior operators.

Only original TRAIN is read. A: teacher/prototypes; B1: student and Probe fit;
B2: Ridge selection/routing audit; C: untouched downstream time block.
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.predictive_env.relation_prototypes import (
    softmax_neg, squared_distance, time_statistics, relation_fingerprint,
    independent_count, local_operators, operator_behaviors, train_router,
    route, teacher_quality,
)
from tools.run_fpem_prototype_quality import (
    save_csv, mean_sq, mae, partition, rows, estimate_period, pair_evidence,
    cluster_metrics,
)
from tools.run_fpem_f03_r3_invariance import collect, prepare
from tools.run_fpem_shared_factorization_g1r3 import load_experiment

METHODS = ("P0_Raw", "P1_Relation_Fingerprint", "P2_Local_Operator")
ALPHAS = (0.1, 1.0, 10.0, 100.0)
SEED = 2022


def split_middle(block, span):
    starts = block["ids"]
    midpoint = (int(starts.min()) + int(starts.max()) + span) // 2
    one = starts + span <= midpoint
    two = starts >= midpoint
    if int(one.sum()) < 8 or int(two.sum()) < 8:
        raise ValueError("middle split too short after full-window purge")
    if int((starts[one] + span).max()) > int(starts[two].min()):
        raise AssertionError("B1/B2 full-window overlap")
    return {key: value[one] if isinstance(value, np.ndarray) and len(value) == len(starts) else value
            for key, value in block.items()}, {
            key: value[two] if isinstance(value, np.ndarray) and len(value) == len(starts) else value
            for key, value in block.items()}


def flatten_history(prepared, mask):
    x = prepared["x"][mask].float()
    mean = prepared["hist_mean"][mask].float()
    scale = prepared["scale"][mask].float()
    return ((x - mean) / scale).permute(0, 2, 1).reshape(-1, x.shape[1]).numpy().astype(np.float32)


def soft_centers(y, q):
    return (q.T @ y) / np.maximum(q.sum(axis=0)[:, None], 1e-8)


def weighted_ridge(x, y, weight, alpha):
    if float(weight.sum()) < max(4, x.shape[1] / 4):
        return None
    return Ridge(alpha=alpha).fit(x, y, sample_weight=weight)


def predict_probe(kind, alpha, xfit, yfit, qfit, xquery, qquery):
    if kind == "Shared Mapping Probe":
        mass = qfit.sum(axis=0).clip(min=1)
        weight = (qfit / mass[None]).sum(axis=1)
        weight *= len(weight) / max(weight.sum(), 1e-9)
        return Ridge(alpha=alpha).fit(xfit, yfit, sample_weight=weight).predict(xquery)
    global_model = Ridge(alpha=alpha).fit(xfit, yfit)
    if kind == "Global Ridge":
        return global_model.predict(xquery)
    outputs = []
    fallback = global_model.predict(xquery)
    for k in range(8):
        local = weighted_ridge(xfit, yfit, qfit[:, k], alpha)
        outputs.append(local.predict(xquery) if local is not None else fallback)
    return np.einsum("nk,knt->nt", qquery, np.stack(outputs)).astype(np.float32)


def ridge_probes(method, blocks, q, common, span):
    rng = np.random.default_rng(SEED + 904)
    random_q = {}
    for name, block in blocks.items():
        random_q[name] = np.eye(8, dtype=np.float32)[rng.integers(8, size=len(block["y"]))]
    candidates, rows = [], []
    for kind in ("Global Ridge", "Prototype Ridge", "Shared Mapping Probe", "Random Group Ridge"):
        assignment = random_q if kind == "Random Group Ridge" else q
        scores = []
        for alpha in ALPHAS:
            prediction = predict_probe(kind, alpha, common["B1"], blocks["B1"]["y"],
                                       assignment["B1"], common["B2"], assignment["B2"])
            score = mean_sq(prediction, blocks["B2"]["y"])
            scores.append((score, alpha))
            candidates.append({"method": method, "probe": kind, "alpha": alpha, "B2_mse": score})
        select_mse, alpha = min(scores)
        pred_c = predict_probe(kind, alpha, common["B1"], blocks["B1"]["y"],
                               assignment["B1"], common["C"], assignment["C"])
        rows.append({"method": method, "probe": kind, "selected_alpha_B2": alpha,
                     "B2_mse": select_mse, "C_mse": mean_sq(pred_c, blocks["C"]["y"]),
                     "C_mae": mae(pred_c, blocks["C"]["y"]),
                     "C_independent_events": independent_count(blocks["C"]["ids"], span)})
    global_mse = next(row["C_mse"] for row in rows if row["probe"] == "Global Ridge")
    random_mse = next(row["C_mse"] for row in rows if row["probe"] == "Random Group Ridge")
    for row in rows:
        row["gain_vs_global_C"] = (global_mse - row["C_mse"]) / max(global_mse, 1e-12)
        row["gain_vs_random_C"] = (random_mse - row["C_mse"]) / max(random_mse, 1e-12)
    return rows, candidates


def fit_teachers(method, blocks, period, span, common, z, v, anchor_ops, ref):
    begin = time.perf_counter()
    teacher, routing, operator = {}, [], {"method": method}
    if method == "P0_Raw":
        km = MiniBatchKMeans(n_clusters=8, n_init=3, batch_size=256,
                             random_state=SEED, max_iter=150).fit(common["A"])
        temperature = float(np.median(np.min(squared_distance(common["A"], km.cluster_centers_), axis=1)) + .05)
        for name in blocks:
            teacher[name] = softmax_neg(squared_distance(common[name], km.cluster_centers_), temperature)
        q = teacher
        routing_time, route_parameters = 0.0, 0
        operator["teacher_source"] = "historical Raw KMeans; no future targets for routing"
        native_distance = {name: squared_distance(common[name], km.cluster_centers_).min(axis=1)
                           for name in blocks}
    elif method == "P1_Relation_Fingerprint":
        x_stats = time_statistics(blocks["A"]["history"], period)
        phase_floor = max(.03, float(np.quantile(x_stats[:, 2], .25)))
        fingerprint = {}
        phase_valid = {}
        for name, block in blocks.items():
            fingerprint[name], phase_valid[name] = relation_fingerprint(
                block["history"], block["y"], period, phase_floor)
        scaler = StandardScaler().fit(fingerprint["A"])
        fp = {name: scaler.transform(value).astype(np.float32) for name, value in fingerprint.items()}
        km = MiniBatchKMeans(n_clusters=8, n_init=3, batch_size=256,
                             random_state=SEED, max_iter=150).fit(fp["A"])
        temperature = float(np.median(np.min(squared_distance(fp["A"], km.cluster_centers_), axis=1)) + .05)
        for name in ("A", "B1", "B2"):
            teacher[name] = softmax_neg(squared_distance(fp[name], km.cluster_centers_), temperature)
        network, routing_time, route_parameters = train_router(
            np.concatenate((common["A"], common["B1"])),
            np.concatenate((teacher["A"], teacher["B1"])), seed=SEED)
        q = {name: route(network, common[name]) for name in blocks}
        operator["teacher_source"] = "TRAIN A relation fingerprint; B1 TRAIN labels only train student"
        operator["phase_valid_fraction_A"] = float(phase_valid["A"].mean())
        native_distance = {name: np.full(len(block["y"]), np.nan) for name, block in blocks.items()}
    else:
        reliable = [op for op in anchor_ops if op["reliable"]]
        response, support = operator_behaviors(reliable, ref)
        common_mask = support.mean(axis=0) >= .5 if len(support) else np.zeros(len(ref), bool)
        operator["reliable_train_operators"] = len(reliable)
        operator["common_reference_count"] = int(common_mask.sum())
        if len(reliable) < 8 or common_mask.sum() < 4:
            raise ValueError(f"P2 insufficient independent local operators ({len(reliable)}) or shared reference ({common_mask.sum()})")
        behavior = response[:, common_mask, :].reshape(len(reliable), -1)
        km = MiniBatchKMeans(n_clusters=8, n_init=3, batch_size=32,
                             random_state=SEED, max_iter=150).fit(behavior)
        temp = float(np.median(np.min(squared_distance(behavior, km.cluster_centers_), axis=1)) + .05)
        anchor_q = softmax_neg(squared_distance(behavior, km.cluster_centers_), temp)
        anchor_z = z["A"][[op["anchor"] for op in reliable]]
        # Nearest operator is selected by history state alone; future never routes a query.
        def near_q(name):
            return anchor_q[squared_distance(z[name], anchor_z).argmin(axis=1)]
        teacher["A"] = near_q("A")
        # B1/B2 offline teachers are TRAIN diagnostics, never used for C routing.
        for name in ("B1", "B2"):
            local = local_operators(z[name], v[name], blocks[name]["ids"], span,
                                    anchor_count=48, neighbor_pool=5000)
            local = [op for op in local if op["reliable"]]
            if local:
                local_response, local_support = operator_behaviors(local, ref[common_mask])
                eligible = local_support.sum(axis=1) >= 4
                if eligible.any():
                    behavior_new = local_response[eligible].reshape(int(eligible.sum()), -1)
                    local_z = z[name][[op["anchor"] for op, okay in zip(local, eligible) if okay]]
                    teacher[name] = softmax_neg(squared_distance(behavior_new,
                                                                   km.cluster_centers_), temp)[
                        squared_distance(z[name], local_z).argmin(axis=1)]
        fit_x, fit_q = [common["A"]], [teacher["A"]]
        if "B1" in teacher:
            fit_x.append(common["B1"]); fit_q.append(teacher["B1"])
        network, routing_time, route_parameters = train_router(
            np.concatenate(fit_x), np.concatenate(fit_q), seed=SEED)
        q = {name: route(network, common[name]) for name in blocks}
        native_distance = {name: np.full(len(block["y"]), np.nan) for name, block in blocks.items()}
        operator["teacher_source"] = "TRAIN A local 4x4 operators clustered by common-reference response"
        operator["reference_reliable_fraction"] = float(common_mask.mean())
    for name in ("A", "B1", "B2"):
        if name in teacher:
            routing.append({"method": method, "block": name, **teacher_quality(teacher[name], q[name]),
                            "teacher_uses_future_for_offline_label": method != "P0_Raw"})
        else:
            routing.append({"method": method, "block": name,
                            "top1_agreement": np.nan, "soft_kl": np.nan,
                            "reason": "independent local operator teacher not estimable"})
    routing.append({"method": method, "block": "C", "top1_agreement": np.nan,
                    "soft_kl": np.nan, "reason": "C future prohibited for routing; student only"})
    return q, teacher, routing, operator, native_distance, time.perf_counter() - begin, routing_time, route_parameters


def operator_audit(method, blocks, q, teacher, z, v, anchors, ref, span):
    reliable = [op for op in anchors if op["reliable"]]
    if len(reliable) < 2:
        return {"method": method, "operator_within": np.nan, "operator_between": np.nan,
                "operator_separation": np.nan, "operator_drift_C": np.nan,
                "reason": "fewer than 2 independent-support local operators",
                "reliable_operator_count_A": len(reliable),
                "operator_support_events_median_A": float(np.median([op["support"] for op in anchors])) if anchors else 0,
                "within_distance_distribution": [], "between_distance_distribution": []}
    response, support = operator_behaviors(reliable, ref)
    anchor_label = teacher["A"][[op["anchor"] for op in reliable]].argmax(axis=1)
    within, between, unsupported = [], [], 0
    for i in range(len(reliable)):
        for j in range(i + 1, len(reliable)):
            overlap = support[i] & support[j]
            if overlap.sum() < 4:
                unsupported += 1
                continue
            distance = mean_sq(response[i, overlap], response[j, overlap])
            (within if anchor_label[i] == anchor_label[j] else between).append(distance)
    result = {"method": method, "operator_within": float(np.mean(within)) if within else np.nan,
              "operator_between": float(np.mean(between)) if between else np.nan,
              "operator_separation": float(np.mean(between)/max(np.mean(within), 1e-12)) if within and between else np.nan,
              "operator_unsupported_pairs": unsupported, "operator_supported_pairs": len(within)+len(between),
              "reliable_operator_count_A": len(reliable),
              "operator_support_events_median_A": float(np.median([op["support"] for op in anchors])) if anchors else 0,
              "operator_condition_p95_A": float(np.quantile([op["condition"] for op in anchors], .95)) if anchors else np.nan}
    result["within_distance_distribution"] = within
    result["between_distance_distribution"] = between
    # TEST-block futures are used only for this offline drift audit, never for routing or training.
    late = local_operators(z["C"], v["C"], blocks["C"]["ids"], span,
                           anchor_count=64, neighbor_pool=5000)
    late = [op for op in late if op["reliable"]]
    result["reliable_operator_count_C"] = len(late)
    if not late:
        result["operator_drift_C"] = np.nan
        result["drift_reason"] = "insufficient independent C events"
        return result
    late_response, late_support = operator_behaviors(late, ref)
    distances = []
    for index, op in enumerate(late):
        label = q["C"][op["anchor"]].argmax()
        comparison = [j for j in range(len(reliable)) if anchor_label[j] == label]
        for j in comparison:
            overlap = late_support[index] & support[j]
            if overlap.sum() >= 4:
                distances.append(mean_sq(late_response[index, overlap], response[j, overlap]))
    result["operator_drift_C"] = float(np.mean(distances)) if distances else np.nan
    result["drift_reason"] = "ok" if distances else "no shared supported reference inputs across time"
    return result


def draw_figures(out, method, blocks, q, teacher, operator, pair_frame, per_proto):
    safe = method.lower()
    fig, ax = plt.subplots(figsize=(6, 4))
    if "A" in teacher:
        x = teacher["A"].mean(axis=0)
        y = q["A"].mean(axis=0)
        ax.bar(np.arange(8)-.2, x, .4, label="teacher TRAIN A")
        ax.bar(np.arange(8)+.2, y, .4, label="student TRAIN A")
    ax.set(xlabel="prototype", ylabel="mean assignment mass", title=method + " routing")
    ax.legend(); fig.tight_layout(); fig.savefig(out / (safe + "_teacher_student.png"), dpi=150); plt.close(fig)
    matrix = np.array([[next((r["mapping_gain"] for r in per_proto if r["method"] == method and
                              r["prototype"] == k and r["block"] == b), np.nan)
                        for b in ("A", "B1", "B2", "C")] for k in range(8)])
    fig, ax = plt.subplots(figsize=(6, 4)); im = ax.imshow(matrix, aspect="auto")
    ax.set(xticks=range(4), xticklabels=("A", "B1", "B2", "C"), yticks=range(8),
           xlabel="time block", ylabel="prototype", title=method + " mapping gain vs Global Ridge")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(out / (safe + "_prototype_time_heatmap.png"), dpi=150); plt.close(fig)
    if operator.get("within_distance_distribution") or operator.get("between_distance_distribution"):
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, key in (("within", "within_distance_distribution"),
                          ("between", "between_distance_distribution")):
            if operator.get(key):
                ax.hist(operator[key], bins=20, alpha=.5, label=name)
        ax.set(xlabel="common-supported operator response MSE", ylabel="pair count",
               title=method + " operator behavior distance")
        ax.legend(); fig.tight_layout()
        fig.savefig(out / (safe + "_operator_distance_distribution.png"), dpi=150); plt.close(fig)
    if pair_frame is not None:
        sim = pair_frame["history_similarity"]
        f = pair_frame["future_similarity"]
        edges = np.unique(np.quantile(sim, np.linspace(0, 1, 21)))
        if len(edges) > 2:
            bin_id = np.digitize(sim, edges[1:-1])
            x, y = [], []
            for k in range(len(edges)-1):
                valid = (bin_id == k) & pair_frame["same"]
                if valid.sum() >= 5:
                    x.append(float(sim[valid].mean())); y.append(float(f[valid].mean()))
            fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(x, y, marker="o")
            ax.set(xlabel="Raw history similarity exp(-MSE)",
                   ylabel="same-prototype future similarity exp(-MSE)",
                   title=method + " chronological C")
            fig.tight_layout(); fig.savefig(out / (safe + "_history_future_bins.png"), dpi=150); plt.close(fig)
    lab = q["C"].argmax(axis=1)
    selected = None
    for k in range(8):
        idx = np.flatnonzero((lab == k) & (blocks["C"]["channels"] == 0))
        if len(idx) < 2:
            continue
        chosen = idx[np.linspace(0, len(idx)-1, min(48, len(idx)), dtype=int)]
        d = squared_distance(blocks["C"]["raw"][chosen], blocks["C"]["raw"][chosen])
        i, j = np.unravel_index(np.argmax(d), d.shape)
        selected = (chosen[i], chosen[j], k)
        break
    if selected:
        i, j, k = selected
        fig, ax = plt.subplots(2, 1, figsize=(8, 5))
        for index, color in ((i, "tab:blue"), (j, "tab:orange")):
            ax[0].plot(blocks["C"]["history"][index], color=color, alpha=.8)
            ax[1].plot(blocks["C"]["y"][index], color=color, alpha=.8)
        ax[0].set_title(f"{method}, prototype {k}: dissimilar histories")
        ax[1].set_title("Associated true futures (offline visualization only)")
        fig.tight_layout(); fig.savefig(out / (safe + "_same_proto_different_history.png"), dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_run_config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--allow_legacy_reference_signature", action="store_true")
    cli = parser.parse_args()
    out = Path(cli.output)
    if (out / "complete").exists():
        print("ALREADY_COMPLETE", out, flush=True); return
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    source = json.loads(Path(cli.base_run_config).read_text(encoding="utf-8"))
    cli.seed = source["seed"]
    started = time.perf_counter()
    args, model, loaders, device, a0_sha = load_experiment(cli)
    if any(p.requires_grad for p in model.parameters()):
        raise AssertionError("frozen A0 required")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    raw = collect(model, loaders["train"].dataset.dataset, args, device, cli.stride)
    peak_gpu = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    floor_std = raw["x"].float().var(1, unbiased=False).add(1e-5).sqrt()
    floor = (floor_std.median(0).values * .25).clamp_min(.01)
    prepared = prepare(raw, floor)
    span = args.seq_len + args.pred_len
    masks, audit = partition(prepared, span)
    period, period_source = estimate_period(loaders["train"].dataset.dataset.data_x[:audit["cut_A"]])
    blocks = {name: rows(prepared, mask, period, args.seq_len) for name, mask in masks.items()}
    for name in blocks:
        blocks[name]["history"] = flatten_history(prepared, masks[name])
    blocks["B1"], blocks["B2"] = split_middle(blocks.pop("B"), span)
    raw_scale = StandardScaler().fit(blocks["A"]["raw"])
    common = {name: raw_scale.transform(block["raw"]).astype(np.float32)
              for name, block in blocks.items()}
    z_scale = StandardScaler().fit(time_statistics(blocks["A"]["history"], period))
    v_scale = StandardScaler().fit(time_statistics(blocks["A"]["y"], period))
    z = {name: z_scale.transform(time_statistics(block["history"], period)).astype(np.float32)
         for name, block in blocks.items()}
    v = {name: v_scale.transform(time_statistics(block["y"], period)).astype(np.float32)
         for name, block in blocks.items()}
    anchors = local_operators(z["A"], v["A"], blocks["A"]["ids"], span)
    center = np.median(z["A"], axis=0)
    candidate = np.argsort(np.linalg.norm(z["A"] - center, axis=1))[:min(128, len(z["A"]))]
    ref = z["A"][candidate[np.linspace(0, len(candidate)-1, min(32, len(candidate)), dtype=int)]]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run_config = {"seed": cli.seed, "prototype_seed": SEED, "a0_sha256": a0_sha,
                  "base_run_config": cli.base_run_config, "git_commit": commit,
                  "backbone": args.backbone, "dataset": args.dataset_name, "pred_len": args.pred_len,
                  "prototype_count": 8, "ridge_alphas": ALPHAS, "operator_dimension": 4,
                  "operator_ridge": 1.0, "student_epochs": 40, "phase_period": period,
                  "phase_period_source": period_source, "stride": cli.stride,
                  "temporal_audit": audit, "A0_training_scope": "full original TRAIN",
                  "TRAIN_only": True, "no_full_pipeline_OOF": True}
    (out / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    summary, per_proto, per_block, probes, candidates, routes, operators, failures = [], [], [], [], [], [], [], []
    for method in METHODS:
        print("METHOD_START", method, flush=True)
        method_start = time.perf_counter()
        try:
            q, teacher, routing, metadata, native, teacher_time, router_time, router_params = fit_teachers(
                method, blocks, period, span, common, z, v, anchors, ref)
        except ValueError as error:
            if method != "P2_Local_Operator" or not str(error).startswith("P2 insufficient independent"):
                raise
            failures.append({"method": method, "status": "not_estimable", "reason": str(error)})
            summary.append({"case": f"{args.backbone}/{args.dataset_name}-{args.pred_len}",
                            "seed": SEED, "method": method, "status": "not_estimable",
                            "reason": str(error), "a0_sha256": a0_sha,
                            "independent_events_A": independent_count(blocks["A"]["ids"], span),
                            "independent_events_C": independent_count(blocks["C"]["ids"], span)})
            print("METHOD_NOT_ESTIMABLE", method, error, flush=True)
            continue
        routes.extend(routing)
        operator = operator_audit(method, blocks, q, teacher, z, v, anchors, ref, span)
        operators.append({**metadata, **operator})
        clock = time.perf_counter()
        mapped, choices = ridge_probes(method, blocks, q, common, span)
        probe_time = time.perf_counter() - clock
        probes.extend(mapped); candidates.extend(choices)
        prototype_alpha = next(row["selected_alpha_B2"] for row in mapped if row["probe"] == "Prototype Ridge")
        global_alpha = next(row["selected_alpha_B2"] for row in mapped if row["probe"] == "Global Ridge")
        per_proto_gain = {}
        for block_name in ("B1", "B2", "C"):
            proto_pred = predict_probe("Prototype Ridge", prototype_alpha,
                common["B1"], blocks["B1"]["y"], q["B1"], common[block_name], q[block_name])
            global_pred = predict_probe("Global Ridge", global_alpha,
                common["B1"], blocks["B1"]["y"], q["B1"], common[block_name], q[block_name])
            for k in range(8):
                weight = q[block_name][:, k]
                denom = np.sum(weight[:, None] * (global_pred - blocks[block_name]["y"]) ** 2)
                numerator = np.sum(weight[:, None] * (proto_pred - blocks[block_name]["y"]) ** 2)
                per_proto_gain[(block_name, k)] = float(1 - numerator / max(denom, 1e-9))
        # Separate low-dimensional transfer audit; never compare its MSE numerically
        # with the full-horizon Ridge MSE above.
        transfer_models = [weighted_ridge(z["A"], v["A"], teacher["A"][:, k], 1.0)
                           for k in range(8)]
        transfer_global = Ridge(alpha=1.0).fit(z["A"], v["A"])
        fallback_v = transfer_global.predict(z["C"])
        transfer_output = np.einsum("nk,knd->nd", q["C"], np.stack([
            model_k.predict(z["C"]) if model_k is not None else fallback_v
            for model_k in transfer_models]))
        probes.append({"method": method, "probe": "Operator Transfer Probe",
                       "output_space": "four_train_standardized_future_statistics",
                       "selected_alpha_B2": 1.0, "C_mse": mean_sq(transfer_output, v["C"]),
                       "C_mae": mae(transfer_output, v["C"]),
                       "gain_vs_global_C": 1 - mean_sq(transfer_output, v["C"]) /
                           max(mean_sq(fallback_v, v["C"]), 1e-9)})
        future_center = soft_centers(blocks["A"]["y"], teacher["A"])
        global_center = blocks["A"]["y"].mean(axis=0)
        mass = teacher["A"].sum(axis=0)
        p = mass / max(mass.sum(), 1e-9)
        k_eff = float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0]))))
        future_between = float(np.sum(p[:, None] * (future_center - global_center) ** 2) / len(global_center))
        pair_c, frame_c = None, None
        for name in ("A", "B1", "B2", "C"):
            block = blocks[name]
            labels = q[name].argmax(axis=1)
            prediction = q[name] @ future_center
            baseline = np.mean((block["y"] - global_center) ** 2)
            mse = mean_sq(prediction, block["y"])
            pair, frame = pair_evidence(block, labels, span, seed=SEED + {"A":0,"B1":1,"B2":2,"C":3}[name])
            if name == "C": pair_c, frame_c = pair, frame
            own = native[name]
            common_centers = soft_centers(common["A"], teacher["A"])
            raw_distance = squared_distance(common[name], common_centers).min(axis=1)
            block_mass = q[name].mean(axis=0)
            per_block.append({"method": method, "block": name, "status": "ok",
                              "future_mse": mse, "fve": 1 - mse/max(baseline, 1e-9),
                              "future_lift": pair["lift"], "future_lift_reason": pair["reason"],
                              "future_lift_matched_pairs": pair.get("matched_pairs", 0),
                              "independent_events": independent_count(block["ids"], span),
                              "window_variable_rows": len(labels),
                              "own_distance_mean": float(np.nanmean(own)) if np.isfinite(own).any() else np.nan,
                              "raw_coverage_mean": float(raw_distance.mean()),
                              "raw_coverage_p90": float(np.quantile(raw_distance, .9)),
                              "raw_coverage_p95": float(np.quantile(raw_distance, .95)),
                              "raw_coverage_worst10_mean": float(raw_distance[raw_distance >= np.quantile(raw_distance, .9)].mean()),
                              "k_eff": float(np.exp(-np.sum(block_mass[block_mass > 0] * np.log(block_mass[block_mass > 0]))))})
            for k in range(8):
                hard = labels == k
                future_k = block["y"][hard].mean(axis=0) if hard.any() else np.full_like(global_center, np.nan)
                phase = block["phase"][hard]
                hist = np.histogram(phase, bins=np.linspace(0,1,9))[0]
                hist = (hist / max(hist.sum(), 1)).tolist()
                same_pairs = (frame["same"] & (frame["left_label"] == k)) if frame is not None else None
                similarity = frame["future_similarity"][same_pairs] if same_pairs is not None else np.array([])
                per_proto.append({"method": method, "block": name, "prototype": k,
                                  "soft_mass": float(q[name][:, k].sum()), "hard_windows": int(hard.sum()),
                                  "independent_event_support": independent_count(block["ids"][hard], span),
                                  "future_mse": mean_sq(block["y"][hard], future_center[k]) if hard.any() else np.nan,
                                  "future_center_drift_from_A": mean_sq(future_k, future_center[k]) if hard.any() else np.nan,
                                  "mapping_gain": per_proto_gain.get((name, k), np.nan),
                                  "future_similarity_pair_count": len(similarity),
                                  "future_similarity_p10": float(np.quantile(similarity, .1)) if len(similarity) >= 10 else np.nan,
                                  "future_similarity_median": float(np.median(similarity)) if len(similarity) >= 10 else np.nan,
                                  "future_similarity_p90": float(np.quantile(similarity, .9)) if len(similarity) >= 10 else np.nan,
                                  "phase_concentration": float(abs(np.exp(2j*np.pi*phase).mean())) if hard.any() else np.nan,
                                  "phase_histogram": hist})
        c = next(row for row in per_block if row["method"] == method and row["block"] == "C")
        proto = next(row for row in mapped if row["probe"] == "Prototype Ridge")
        random = next(row for row in mapped if row["probe"] == "Random Group Ridge")
        global_probe = next(row for row in mapped if row["probe"] == "Global Ridge")
        for block_row in per_block:
            if block_row["method"] != method:
                continue
            if block_row["block"] == "B2":
                block_row["mapping_gain_vs_global"] = 1 - proto["B2_mse"] / max(global_probe["B2_mse"], 1e-9)
            elif block_row["block"] == "C":
                block_row["mapping_gain_vs_global"] = proto["gain_vs_global_C"]
            else:
                block_row["mapping_gain_vs_global"] = np.nan
        cluster = cluster_metrics(common["A"], teacher["A"].argmax(axis=1))
        agreement = next((row["top1_agreement"] for row in routing if row["block"] == "B2"), np.nan)
        summary.append({"case": f"{args.backbone}/{args.dataset_name}-{args.pred_len}",
                        "seed": SEED, "method": method, "status": "ok",
                        "a0_sha256": a0_sha, "FVE": c["fve"], "FutureLift": c["future_lift"],
                        "FutureLift_reason": c["future_lift_reason"],
                        "Future_MSE": c["future_mse"],
                        "Future_Center_Distance": float(np.mean([np.mean((future_center[i]-future_center[j])**2)**.5
                            for i in range(8) for j in range(i+1,8)])),
                        "Between_Prototype_Variance": future_between,
                        "Operator_Within": operator["operator_within"],
                        "Operator_Between": operator["operator_between"],
                        "Operator_Separation": operator["operator_separation"],
                        "Operator_Drift": operator["operator_drift_C"],
                        "Route_Agreement": agreement, "K_eff": k_eff,
                        "Independent_Events": independent_count(blocks["C"]["ids"], span),
                        "Operator_Support_A": operator["operator_support_events_median_A"],
                        "Ridge_MSE": proto["C_mse"], "Ridge_MAE": proto["C_mae"],
                        "Gain_vs_Global": proto["gain_vs_global_C"],
                        "Gain_vs_Random": proto["gain_vs_random_C"],
                        "Random_Ridge_MSE": random["C_mse"],
                        "Raw_Coverage_Mean": c["raw_coverage_mean"],
                        "Raw_Coverage_P90": c["raw_coverage_p90"],
                        "Raw_Coverage_P95": c["raw_coverage_p95"],
                        "Raw_Coverage_Worst10": c["raw_coverage_worst10_mean"],
                        "Cluster_Silhouette_Raw": cluster["silhouette_raw"],
                        "Cluster_Davies_Bouldin_Raw": cluster["davies_bouldin_raw"],
                        "Cluster_Calinski_Harabasz_Raw": cluster["calinski_harabasz_raw"],
                        "Train_Time": teacher_time + probe_time,
                        "Prototype_Time": teacher_time - router_time,
                        "Router_Time": router_time, "Probe_Time": probe_time,
                        "Peak_VRAM": peak_gpu, "Router_Parameters": router_params,
                        "Peak_CPU_RSS_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        "Prototype_Memory_Floats": int(8 * (64 + args.pred_len)),
                        "Collapsed": bool((p < .02).any()),
                        "Phase_Concentration_Mean": float(np.nanmean([r["phase_concentration"] for r in per_proto
                            if r["method"] == method and r["block"] == "A"])),
                        "C_Future_Similarity_Within_Mean": pair_c["within_future_similarity_mean"],
                        "C_FutureLift_Matched_Pairs": pair_c.get("matched_pairs",0)})
        draw_figures(out, method, blocks, q, teacher, operator, frame_c, per_proto)
        save_csv(out / "summary.csv", summary)
        save_csv(out / "per_prototype.csv", per_proto)
        save_csv(out / "per_time_block.csv", per_block)
        save_csv(out / "mapping_probe.csv", probes)
        save_csv(out / "mapping_probe_candidates.csv", candidates)
        save_csv(out / "routing_diagnostics.csv", routes)
        save_csv(out / "operator_diagnostics.csv", operators)
        print("METHOD_COMPLETE", method, "Ridge_C", proto["C_mse"], "gain", proto["gain_vs_global_C"], flush=True)
    if failures:
        (out / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    save_csv(out / "summary.csv", summary)
    if not summary or all(row["status"] != "ok" for row in summary):
        raise RuntimeError("no estimable methods; see failures.json")
    run_config["elapsed_seconds"] = time.perf_counter() - started
    (out / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    (out / "complete").write_text("complete\n")
    print("COMPLETE", out, flush=True)


if __name__ == "__main__":
    main()
