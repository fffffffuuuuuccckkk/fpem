#!/usr/bin/env python3
"""Frozen A0 + history-calendar additive tokens + EIIL3/F03/R3/R3(trick).

Only the new projection is trained in the exogenous stage.  All downstream
heads are rebuilt from this selected A0-exog; old A0 feature/teacher caches
are never read.  TEST selection, if requested, is exploratory development.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.predictive_env.exogenous_embedding import ExogenousEmbedding
from models.predictive_env.eiil3_linear_correction import EnvironmentLinearCorrection, correction_design
from models.predictive_env.relation_prototypes import route
from tools.run_fpem_eiil3 import SPACES, dump, file_hash, valid_teacher
from tools.run_fpem_eiil3_cosa import fit as fit_variation, predict as predict_variation, router_from_checkpoint
from tools.run_fpem_eiil3_safe_full import parse_floats, run as run_safe_full
from tools.run_fpem_f03_r3_invariance import prepare
from tools.run_fpem_f03_safe import forecast, mse_mae
from tools.run_fpem_mechanism_fusion import flat_record
from tools.run_fpem_shared_factorization_g1r3 import load_experiment
from tools.run_predictive_env_iv_patchtst import IndexedDataset
from utils.timefeatures import time_features_from_frequency_str


def calendar_batch(dataset, ids, seq_len, device):
    stamp = np.asarray(dataset.data_stamp)
    starts = ids.detach().cpu().numpy().astype(np.int64)
    if starts.min() < 0 or starts.max() + seq_len > len(stamp):
        raise IndexError("calendar window is outside dataset history")
    mark = np.stack([stamp[i:i + seq_len] for i in starts])
    if not np.isfinite(mark).all():
        raise ValueError("calendar features contain NaN/Inf")
    return torch.from_numpy(mark).float().to(device)


def loader_for(dataset, args, stride, batch_size, *, shuffle=False, seed=2021):
    wrapped = IndexedDataset(dataset, cycle_len=args.cyclenet_cycle_len)
    selected = Subset(wrapped, range(0, len(dataset), stride))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(selected, batch_size=batch_size, shuffle=shuffle,
                      generator=generator, num_workers=0)


def score_adapter(adapter, dataset, args, device, stride, batch_size):
    adapter.base.eval()
    adapter.proj.eval()
    squared = absolute = 0.0
    count = 0
    with torch.no_grad():
        for x, y, ids, cycle in loader_for(dataset, args, stride, batch_size):
            x, y = x.float().to(device), y[:, -args.pred_len:].float().to(device)
            mark = calendar_batch(dataset, ids, args.seq_len, device)
            pred = adapter(x, mark, cycle.to(device))
            error = pred - y
            squared += float(error.square().sum())
            absolute += float(error.abs().sum())
            count += error.numel()
    return {"mse": squared/count, "mae": absolute/count, "evaluated_values": count}


def check_zero_identity(adapter, dataset, args, device):
    adapter.base.eval()
    x, _, ids, cycle = next(iter(loader_for(dataset, args, 1, min(args.batch_size, 4))))
    x, cycle = x.float().to(device), cycle.to(device)
    mark = calendar_batch(dataset, ids, args.seq_len, device)
    with torch.no_grad():
        hidden, mean, std = adapter.base.encode(x, cycle)
        original = adapter.base.frozen_h_head(hidden, cycle).transpose(1, 2)*std + mean
        augmented = adapter(x, mark, cycle)
    error = float((augmented-original).abs().max())
    if error > 1e-6:
        raise AssertionError(f"zero exogenous adapter changed A0: {error}")
    if any(module.training for module in adapter.base.modules()):
        raise AssertionError("frozen A0 unexpectedly entered training mode")
    return error


def prepare_adapter(cli, parent_asset, output):
    """Search only adapter LR/WD/epoch; save one best lightweight state."""
    source = output / "exog_selected.pt"
    report_path = output / "exog_search.json"
    parent_sha = parent_asset["A0_sha256"]
    search_signature = {"lrs": cli.exog_lrs, "weight_decays": cli.exog_weight_decays,
                        "epochs": cli.exog_epochs, "train_stride": cli.train_stride,
                        "val_stride": cli.val_stride, "test_stride": cli.test_stride,
                        "selection_split": cli.selection_split,
                        "max_train_batches": cli.max_train_batches}
    if cli.resume and source.exists() and report_path.exists():
        report = json.loads(report_path.read_text())
        if (report.get("parent_A0_sha256") == parent_sha
                and report.get("search_signature") == search_signature
                and report.get("checkpoint_sha256") == file_hash(source)):
            return report
    loader_cli = argparse.Namespace(base_run_config=parent_asset["base_run_config"],
                                    gpu=cli.gpu, seed=cli.seed,
                                    allow_legacy_reference_signature=True)
    args, base, loaders, device, actual_parent_sha = load_experiment(loader_cli)
    if actual_parent_sha != parent_sha:
        raise AssertionError("parent A0 identity changed")
    datasets = {name: loader.dataset.dataset for name, loader in loaders.items()}
    exog_dim = int(datasets["train"].data_stamp.shape[1])
    adapter = ExogenousEmbedding(base, args.backbone, args.seq_len,
                                 exog_dim, args.d_model).to(device)
    identity_error = check_zero_identity(adapter, datasets["val"], args, device)
    if cli.exog_source != "calendar_history":
        raise ValueError("only observed historical calendar is supported")
    selection = cli.selection_split
    selection_stride = cli.val_stride if selection == "val" else cli.test_stride
    train_batch = min(args.batch_size, cli.exog_batch_size)
    eval_batch = min(args.batch_size, cli.exog_eval_batch_size)
    baseline = score_adapter(adapter, datasets[selection], args, device,
                             selection_stride, eval_batch)
    best = (baseline["mse"], copy.deepcopy(adapter.proj.state_dict()),
            {"lr": None, "weight_decay": None, "epoch": 0,
             "selection": baseline})
    trials = []
    started = time.monotonic()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    for lr in parse_floats(cli.exog_lrs):
        for wd in [float(x) for x in cli.exog_weight_decays.split(",")]:
            if wd < 0:
                raise ValueError("weight decay must be nonnegative")
            torch.manual_seed(cli.seed)
            torch.nn.init.zeros_(adapter.proj.weight)
            optimizer = torch.optim.AdamW(adapter.proj.parameters(), lr=lr, weight_decay=wd)
            train = loader_for(datasets["train"], args, cli.train_stride,
                               train_batch, shuffle=True, seed=cli.seed)
            trial = {"lr": lr, "weight_decay": wd, "epochs": []}
            for epoch in range(1, cli.exog_epochs + 1):
                adapter.base.eval()
                adapter.proj.train()
                for step, (x, y, ids, cycle) in enumerate(train):
                    x = x.float().to(device)
                    y = y[:, -args.pred_len:].float().to(device)
                    mark = calendar_batch(datasets["train"], ids, args.seq_len, device)
                    prediction = adapter(x, mark, cycle.to(device))
                    loss = (prediction-y).square().mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite exogenous training loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    if cli.max_train_batches and step + 1 >= cli.max_train_batches:
                        break
                if epoch in (5, 10, 20) or epoch == cli.exog_epochs:
                    measured = score_adapter(adapter, datasets[selection], args, device,
                                             selection_stride, eval_batch)
                    trial["epochs"].append({"epoch": epoch, **measured})
                    if measured["mse"] < best[0] - 1e-9:
                        best = (measured["mse"], copy.deepcopy({
                            key: value.detach().cpu().clone()
                            for key, value in adapter.proj.state_dict().items()}),
                            {"lr": lr, "weight_decay": wd, "epoch": epoch,
                             "selection": measured})
            trials.append(trial)
            dump(output / "exog_search_progress.json", {"completed_trials": trials,
                 "selection_split": selection, "parent_A0_sha256": parent_sha})
    adapter.proj.load_state_dict(best[1])
    torch.save({"proj": best[1], "parent_A0_sha256": parent_sha,
                "backbone": args.backbone, "seq_len": args.seq_len,
                "exog_dim": exog_dim}, source)
    score_val = score_adapter(adapter, datasets["val"], args, device,
                              cli.val_stride, eval_batch)
    score_test = score_adapter(adapter, datasets["test"], args, device,
                               cli.test_stride, eval_batch)
    names = [type(f).__name__ for f in time_features_from_frequency_str(args.freq)]
    report = {"parent_A0_sha256": parent_sha, "checkpoint_sha256": file_hash(source),
              "search_signature": search_signature,
              "backbone": args.backbone, "dataset": args.dataset_name,
              "horizon": args.pred_len, "calendar_features": names,
              "calendar_origin": "dataset.data_stamp from observed history only",
              "alignment": {"patchtst": "same padded patches before encoder",
                            "moderntcn": "stem patch/downsample resolution before blocks",
                            "itransformer": "full-history calendar to variable tokens before encoder"}[args.backbone],
              "iTransformer_x_mark_tokens_in_A0": False,
              "zero_init_max_prediction_difference": identity_error,
              "selection_split": selection, "epoch0": baseline,
              "selected": best[2], "val": score_val, "test": score_test,
              "trials": trials, "added_parameters": adapter.proj.weight.numel(),
              "training_seconds": time.monotonic()-started,
              "peak_gpu_bytes": (torch.cuda.max_memory_allocated(device)
                                 if torch.cuda.is_available() else 0),
              "backbone_frozen_eval": True,
              "selection_notice": ("TEST is exploratory development data"
                                   if selection == "test" else "VAL-selected; TEST report-only")}
    dump(report_path, report)
    return report


@torch.no_grad()
def collect_adapter(adapter, dataset, args, device, stride, batch_size):
    chunks = {key: [] for key in ("x", "hidden", "mean", "std", "truth", "base", "ids")}
    adapter.base.eval()
    adapter.proj.eval()
    for x, y, ids, cycle in loader_for(dataset, args, stride, batch_size):
        x, cycle = x.float().to(device), cycle.to(device)
        mark = calendar_batch(dataset, ids, args.seq_len, device)
        hidden, mean, std = adapter.encode(x, mark, cycle)
        base = adapter.base.frozen_h_head(hidden, cycle).transpose(1, 2)*std + mean
        values = {"x": x, "hidden": hidden, "mean": mean, "std": std,
                  "truth": y[:, -args.pred_len:], "base": base, "ids": ids}
        for key, value in values.items():
            chunks[key].append(value.detach().cpu())
    return {key: torch.cat(parts) for key, parts in chunks.items()}


def get_exog_data(cli):
    asset = cli.asset
    parent_sha = asset["parent_A0_sha256"]
    loader_cli = argparse.Namespace(base_run_config=asset["base_run_config"],
                                    gpu=cli.gpu, seed=cli.seed,
                                    allow_legacy_reference_signature=True)
    args, base, loaders, device, actual_parent_sha = load_experiment(loader_cli)
    if actual_parent_sha != parent_sha:
        raise AssertionError("parent A0 SHA mismatch during feature extraction")
    datasets = {name: loader.dataset.dataset for name, loader in loaders.items()}
    saved = torch.load(cli.exog_checkpoint, map_location="cpu", weights_only=False)
    if saved["parent_A0_sha256"] != parent_sha:
        raise AssertionError("exogenous checkpoint parent A0 mismatch")
    adapter = ExogenousEmbedding(base, args.backbone, args.seq_len,
                                 saved["exog_dim"], args.d_model).to(device)
    adapter.proj.load_state_dict(saved["proj"])
    raw_train = collect_adapter(adapter, datasets["train"], args, device,
                                cli.train_stride, cli.exog_eval_batch_size)
    train_ids = raw_train["ids"].numpy().astype(np.int64)
    if cli.q_mode == "fixed_original" and not (cli.output/"teacher_q.npz").exists():
        source = cli.fixed_q_source_root/cli.case_key
        old_q = source/"teacher_q.npz"
        if old_q.is_file():
            with np.load(old_q) as stored:
                if not np.array_equal(stored["sample_id"], train_ids):
                    raise ValueError("fixed original q sample IDs differ from A0-exog TRAIN")
            shutil.copy2(old_q, cli.output/"teacher_q.npz")
            if (source/"teacher_q.json").is_file():
                shutil.copy2(source/"teacher_q.json", cli.output/"teacher_q.json")
            else:
                dump(cli.output/"teacher_q.json", {"source": "fixed_original_q",
                    "provenance": str(old_q)})
    teacher_window, teacher_meta = valid_teacher(cli, train_ids, raw_train)
    floor = raw_train["x"].float().var(1, unbiased=False).add(1e-5).sqrt().median(0).values.mul(.25).clamp_min(.01)
    train = flat_record(prepare(raw_train, floor), release_hidden=True)
    data = {space: train for space in SPACES[:4]}
    teacher_by_start = dict(zip(train_ids.tolist(), teacher_window))
    teacher_flat = np.stack([teacher_by_start[int(start)] for start in train["start"]])
    teacher = {space: teacher_flat for space in SPACES[:4]}
    del raw_train
    for space, stride in (("val", cli.val_stride), ("test", cli.test_stride)):
        raw = collect_adapter(adapter, datasets[space], args, device,
                              stride, cli.exog_eval_batch_size)
        data[space] = flat_record(prepare(raw, floor), release_hidden=True)
        del raw
    head = (base.h_forecast_head.weight.detach().cpu().numpy().astype(np.float32),
            base.h_forecast_head.bias.detach().cpu().numpy().astype(np.float32))
    check = (data["A"]["design"][:64] @ head[0].T + head[1])*data["A"]["ratio"][:64]
    init_error = float(np.max(np.abs(check-data["A"]["a0"][:64])))
    if init_error > 1e-3:
        raise AssertionError(f"A0-exog head/feature mismatch: {init_error}")
    meta = {"A0_sha256": asset["A0_sha256"], "parent_A0_sha256": parent_sha,
            "exog_checkpoint_sha256": file_hash(cli.exog_checkpoint),
            "q_sha256": file_hash(cli.output/"teacher_q.npz"),
            "teacher_source": teacher_meta["source"], "init_error_max": init_error,
            "dataset": args.dataset_name, "backbone": args.backbone,
            "horizon": args.pred_len, "features_persisted": False,
            "strides": {"train": cli.train_stride, "val": cli.val_stride,
                        "test": cli.test_stride}}
    dump(cli.output/"features.json", meta)
    return data, meta, head, teacher


def fit_r3_trick(cli, data):
    """Fit on Y-F03, but select/deploy A0 + R3; original R3 is untouched."""
    metrics_path = cli.output/"metrics.json"
    metrics = json.loads(metrics_path.read_text())
    if metrics.get("r3_trick", {}).get("status") == "complete":
        return
    device = torch.device(f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu")
    router = router_from_checkpoint(cli.output/"router.pt")
    q = {space: route(router, data[space]["key"]) for space in ("A", "val", "test")}
    state = torch.load(cli.output/"selected_safe_head.pt", map_location="cpu", weights_only=False)
    a0 = {space: data[space]["a0"] for space in q}
    f03 = {space: forecast(data[space], state) for space in q}
    features = {space: correction_design(a0[space], data[space]["key"]) for space in q}
    select = cli.selection_split
    best, trials = None, []
    for lr in parse_floats(cli.variation_lrs):
        model, info = fit_variation(features["A"], f03["A"], data["A"]["y"], q["A"],
                                    features[select], a0[select], data[select]["y"], q[select],
                                    lr, cli.epochs, cli.batch_size, cli.seed, device)
        trials.append(info)
        if best is None or info["B2_mse"] < best[0]:
            best = (info["B2_mse"], copy.deepcopy({key: value.cpu()
                   for key, value in model.state_dict().items()}), info)
    torch.save(best[1], cli.output/"r3_trick.pt")
    model = EnvironmentLinearCorrection(features["A"].shape[1],
                                         data["A"]["y"].shape[1]).to(device)
    model.load_state_dict(best[1])
    scores = {}
    for space in ("val", "test"):
        correction, _ = predict_variation(model, features[space], q[space], device)
        scores[space] = mse_mae(a0[space] + correction, data[space]["y"])
    metrics["r3_trick"] = {"status": "complete", "fit_target": "Y-F03-exog",
                           "inference_anchor": "A0-exog", "features": "A0-exog+history_key",
                           "selected": best[2], "trials": trials, "scores": scores,
                           "checkpoint_sha256": file_hash(cli.output/"r3_trick.pt")}
    metrics["versions"]["R3(trick)"] = scores
    dump(metrics_path, metrics)
    (cli.output/"exog_complete").write_text("complete\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--case-key", required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--train-stride", type=int, default=8)
    p.add_argument("--val-stride", type=int, default=8)
    p.add_argument("--test-stride", type=int, default=1)
    p.add_argument("--router-epochs", type=int, default=30)
    p.add_argument("--router-lr", type=float, default=.01)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--head-lrs", default="0.00002,0.0001,0.0003,0.001")
    p.add_argument("--lambda-values", default="0.0001,0.01")
    p.add_argument("--variation-lrs", default="0.0001,0.0003,0.001,0.003")
    p.add_argument("--selection-split", choices=("val", "test"), default="test")
    p.add_argument("--environment-supervision", default="predictive")
    p.add_argument("--use-exog-embedding", action="store_true")
    p.add_argument("--exog-source", choices=("calendar_history",), default="calendar_history")
    p.add_argument("--exog-lrs", default="0.0001,0.0003,0.001")
    p.add_argument("--exog-weight-decays", default="0,0.0001")
    p.add_argument("--exog-epochs", type=int, default=20)
    p.add_argument("--exog-batch-size", type=int, default=16)
    p.add_argument("--exog-eval-batch-size", type=int, default=16)
    p.add_argument("--max-train-batches", type=int, default=0,
                   help="test-only smoke limit; never set in final experiment")
    p.add_argument("--q-mode", choices=("rediscover", "fixed_original"), default="rediscover")
    p.add_argument("--fixed-q-source-root", type=Path, default=Path(
        "results/fpem_eiil3_fulltrain_testselect_20261113/cases"))
    p.add_argument("--resume", action="store_true")
    cli = p.parse_args()
    torch.set_num_threads(4)
    if not cli.use_exog_embedding:
        raise ValueError("this independent runner requires --use-exog-embedding")
    manifest = {row["case_key"]: row for row in
                (json.loads(line) for line in cli.manifest.read_text().splitlines())}
    parent = manifest[cli.case_key]
    if parent["status"] == "MISSING_A0":
        raise FileNotFoundError(f"parent A0 missing: {cli.case_key}")
    main_output = cli.output_root/"rediscover"/cli.case_key
    main_output.mkdir(parents=True, exist_ok=True)
    report = prepare_adapter(cli, parent, main_output)
    checkpoint = main_output/"exog_selected.pt"
    combined_sha = hashlib.sha256((parent["A0_sha256"] + report["checkpoint_sha256"]).encode()).hexdigest()
    asset = {**parent, "parent_A0_sha256": parent["A0_sha256"],
             "A0_sha256": combined_sha,
             "status": "READY" if cli.q_mode == "fixed_original" else "MISSING_ENV_Q"}
    cli.output_root = cli.output_root/cli.q_mode
    cli.output = cli.output_root/cli.case_key
    cli.exog_checkpoint = checkpoint
    cli.full_train = True
    cli.stop_on_gain = False
    cli.pilot_search = False
    cli.persist_features = False
    cli.generate_missing_q = cli.q_mode == "rediscover"
    cli.eiil_steps = 50
    cli.ridge_alpha = 100.0
    cli.lr_i3 = parse_floats(cli.head_lrs)[0]
    cache = {}
    def provider(args):
        if "result" not in cache:
            cache["result"] = get_exog_data(args)
        return cache["result"]
    run_safe_full(cli, data_loader=provider, asset_override=asset)
    fit_r3_trick(cli, provider(cli)[0])
    metrics = json.loads((cli.output/"metrics.json").read_text())
    dump(cli.output/"exog_run_config.json", {"parent_A0_sha256": parent["A0_sha256"],
         "A0_exog_sha256": combined_sha, "exog_checkpoint_sha256": report["checkpoint_sha256"],
         "q_mode": cli.q_mode, "selection_split": cli.selection_split,
         "calendar_features": report["calendar_features"],
         "exog_search": report["selected"]})
    print(json.dumps({"case": cli.case_key, "status": "complete",
                      "A0-exog": metrics["versions"]["A0"]["test"],
                      "F03+R3-exog": metrics["versions"]["F03+R3"]["test"],
                      "R3(trick)-exog": metrics["versions"]["R3(trick)"]["test"]}), flush=True)


if __name__ == "__main__":
    main()
