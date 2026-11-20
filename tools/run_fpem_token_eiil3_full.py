#!/usr/bin/env python3
"""Frozen global-periodic-token A0 -> EIIL3/F03/R3 and reverse-stage ablations.

One invocation handles one exact token checkpoint on its own host. The source
checkpoint is never modified. The existing full-TRAIN EIIL3 search supplies
the head, pure R3, and F03+R3; this runner adds R3(trick) and frozen-R3 then
F03. TEST selection is the historical exploratory protocol, not an independent
test estimate. Passing --selection-split val makes TEST report-only.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

_SCRIPT = Path(__file__).resolve()
ROOT = next((p for p in _SCRIPT.parents if (p / "models").is_dir()
             and (p / "tools" / "run_fpem_eiil3.py").is_file()), None)
if ROOT is None:
    raise RuntimeError("cannot locate FPEM repository root")
sys.path.insert(0, str(ROOT))

from models.PatchTST_PredictiveEnvIV import Model  # noqa: E402
from models.predictive_env.eiil3_linear_correction import (  # noqa: E402
    EnvironmentLinearCorrection, correction_design,
)
from models.predictive_env.periodic_exog import periodic_features_for_dataset  # noqa: E402
from models.predictive_env.relation_prototypes import route  # noqa: E402
from tools import run_predictive_env_iv_patchtst as legacy  # noqa: E402
from tools.run_fpem_eiil3 import SPACES, dump, file_hash, valid_teacher  # noqa: E402
from tools.run_fpem_eiil3_cosa import (  # noqa: E402
    fit as fit_variation, predict as predict_variation, router_from_checkpoint,
)
from tools.run_fpem_eiil3_r3_then_f03 import (  # noqa: E402
    corrected_record, fit_head_after_r3,
)
from tools.run_fpem_eiil3_safe_full import (  # noqa: E402
    parse_floats, run as run_safe_full, val_alpha_and_mse,
)
from tools.run_fpem_f03_r3_invariance import prepare  # noqa: E402
from tools.run_fpem_f03_safe import forecast, interpolation, mse_mae  # noqa: E402
from tools.run_fpem_mechanism_fusion import flat_record  # noqa: E402
from tools.run_fpem_periodic_token_a0 import make_datasets  # noqa: E402


class IndexedPeriodicDataset(Dataset):
    def __init__(self, periodic):
        self.periodic = periodic

    def __len__(self):
        return len(self.periodic)

    def __getitem__(self, index):
        x, y, marks = self.periodic[index]
        return x, y, marks, index


def _absolute(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _output_root(output, case_key):
    parts = Path(case_key).parts
    if any(part in (".", "..") for part in parts) or not parts:
        raise ValueError("case-key must be a relative case path")
    if tuple(output.parts[-len(parts):]) != parts:
        raise ValueError("output-dir must end in the complete case-key path")
    return output.parents[len(parts) - 1]


def _load_source(cli):
    run_dir = cli.a0_run_dir
    checkpoint = run_dir / "token" / "best_val.pt"
    protocol_path = run_dir / "protocol.json"
    comparison_path = run_dir / "comparison.json"
    if not all(path.is_file() for path in (checkpoint, protocol_path, comparison_path,
                                          run_dir / "complete")):
        raise FileNotFoundError(f"token A0 run is incomplete: {run_dir}")
    protocol = json.loads(protocol_path.read_text())
    comparison = json.loads(comparison_path.read_text())
    if protocol.get("max_train_batches") or protocol.get("max_eval_batches"):
        raise ValueError("truncated token A0 cannot enter a full downstream run")
    if protocol.get("periodic_mode") not in ("day", "week", "both", "timeF"):
        raise ValueError("unknown token calendar mode")
    sha = file_hash(checkpoint)
    if cli.expected_a0_sha256 and sha != cli.expected_a0_sha256:
        raise AssertionError("token A0 checkpoint SHA differs from requested identity")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("arm") != "token" or not isinstance(saved.get("model_config"), dict):
        raise ValueError("expected the token A0 {model_config,state_dict} checkpoint")
    if not isinstance(saved.get("state_dict"), dict):
        raise ValueError("token A0 state_dict missing")
    args = SimpleNamespace(**saved["model_config"])
    keys = saved["state_dict"].keys()
    global_marker = {"patchtst": "exog_token_fusion.global_embedding.weight",
                     "moderntcn": "modern_tcn_backbone.exog_fusion.global_embedding.weight"}
    if args.backbone in global_marker and global_marker[args.backbone] not in keys:
        raise ValueError("token checkpoint does not contain global-periodic-token fusion")
    for key, protocol_key in (("backbone", "backbone"), ("dataset_name", "dataset"),
                              ("seq_len", "seq_len"), ("pred_len", "pred_len"),
                              ("seed", "seed")):
        if getattr(args, key) != protocol[protocol_key]:
            raise AssertionError(f"token A0 model_config/protocol mismatch: {key}")
    if saved.get("selected_epoch") != comparison["token"]["selected_epoch"]:
        raise AssertionError("token A0 selected epoch differs from comparison")
    if abs(saved["validation"]["mse"] - comparison["token"]["val"]["mse"]) > 1e-9:
        raise AssertionError("token A0 validation metrics differ from comparison")
    csv = _absolute(Path(args.root_path) / args.data_path)
    if not csv.is_file() or file_hash(csv) != protocol["csv_sha256"]:
        raise AssertionError(f"token A0 dated input CSV changed: {csv}")
    args.root_path = str(csv.parent)
    if cli.seed is None:
        cli.seed = args.seed
    if cli.seed != args.seed:
        raise ValueError("downstream seed must equal the token A0 seed")
    cfg = legacy.model_config(args, "A0")
    cfg.exog_token_enabled = True
    cfg.exog_gate_init = float(protocol.get("fusion_gate_init", 0.0))
    model = Model(cfg)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    device = torch.device(f"cuda:{cli.gpu}" if cli.gpu >= 0 else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --gpu -1 for CPU")
    model.to(device)
    datasets = make_datasets(args, legacy, periodic_features_for_dataset,
                             protocol["periodic_mode"])
    if datasets["train"].calendar.shape[1] != protocol["calendar_features"]:
        raise AssertionError("calendar dimension differs from token A0 provenance")
    return model, args, device, datasets, protocol, sha


@torch.no_grad()
def _collect(model, dataset, args, device, stride, batch_size):
    subset = Subset(IndexedPeriodicDataset(dataset), range(0, len(dataset), stride))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    chunks = {name: [] for name in ("x", "hidden", "mean", "std", "truth", "base", "ids")}
    maximum_identity_error = 0.0
    model.eval()
    for x, y, marks, ids in loader:
        x = x.float().to(device)
        marks = marks.float().to(device)
        output = model.forward_components(x, exog=marks)
        hidden = output["hidden_tokens"]
        scale, shift = output["input_scale"], output["input_shift"]
        # The token warmup trains head_linear. h_forecast_head is an older,
        # separately frozen copy and is not the checkpoint's A0 prediction.
        design = model.head_flatten(hidden.permute(0, 1, 3, 2))
        reconstructed = model.head_linear(design).transpose(1, 2) * scale + shift
        base = output["prediction"]
        error = float((reconstructed - base).abs().max())
        maximum_identity_error = max(maximum_identity_error, error)
        if error > 2e-4:
            raise AssertionError(f"token A0 forward/head identity failed: {error}")
        values = {"x": x, "hidden": hidden, "mean": shift, "std": scale,
                  "truth": y[:, -args.pred_len:], "base": base, "ids": ids}
        for name, value in values.items():
            chunks[name].append(value.detach().cpu())
    if not chunks["x"]:
        raise ValueError("empty sampled split")
    raw = {name: torch.cat(parts) for name, parts in chunks.items()}
    return raw, maximum_identity_error


def _teacher(cli, train_ids, raw_train, a0_sha):
    identity_path = cli.output / "token_q_identity.json"
    cache_path = cli.output / "teacher_q.npz"
    if cache_path.exists() and not identity_path.exists():
        raise ValueError("teacher q cache exists without token A0 identity binding")
    if identity_path.exists():
        identity = json.loads(identity_path.read_text())
        q_mode = "fixed_original" if cli.fixed_q_case else "rediscover"
        if (identity.get("A0_sha256") != a0_sha
                or identity.get("train_stride") != cli.train_stride
                or identity.get("q_mode") != q_mode):
            raise ValueError("teacher q belongs to another token A0 or sampling protocol")
        if identity.get("q_sha256") != file_hash(cache_path):
            raise AssertionError("teacher q cache changed")
    if cli.fixed_q_case is not None and not cache_path.exists():
        source = cli.fixed_q_case / "teacher_q.npz"
        if not source.is_file():
            raise FileNotFoundError(source)
        with np.load(source) as stored:
            if not np.array_equal(stored["sample_id"], train_ids):
                raise ValueError("fixed q sample IDs differ from token A0 TRAIN windows")
            q = stored["q"].astype(np.float32)
        if q.shape != (len(train_ids), 3) or not np.allclose(q.sum(1), 1., atol=2e-4):
            raise ValueError("invalid fixed teacher q")
        np.savez_compressed(cache_path, sample_id=train_ids, q=q)
        dump(cli.output / "teacher_q.json", {
            "source": "fixed_original_q", "provenance": str(source),
            "source_sha256": file_hash(source), "sample_count": len(q),
        })
    teacher, meta = valid_teacher(cli, train_ids, raw_train)
    dump(identity_path, {"A0_sha256": a0_sha, "train_stride": cli.train_stride,
                         "q_mode": "fixed_original" if cli.fixed_q_case else "rediscover",
                         "q_sha256": file_hash(cache_path), "source": meta["source"]})
    return teacher, meta


def _provider(cli):
    model, args, device, datasets, protocol, a0_sha = _load_source(cli)
    raw_train, train_error = _collect(model, datasets["train"], args, device,
                                      cli.train_stride, cli.extract_batch_size)
    train_ids = raw_train["ids"].numpy().astype(np.int64)
    teacher_window, teacher_meta = _teacher(cli, train_ids, raw_train, a0_sha)
    floor = raw_train["x"].float().var(1, unbiased=False).add(1e-5).sqrt().median(0).values.mul(.25).clamp_min(.01)
    train = flat_record(prepare(raw_train, floor), release_hidden=True)
    channels = int(raw_train["x"].shape[-1])
    teacher_train = np.repeat(teacher_window, channels, axis=0)
    if not np.array_equal(train["start"].reshape(-1, channels)[:, 0], train_ids):
        raise AssertionError("TRAIN window/channel flatten order changed")
    del raw_train
    gc.collect()
    data = {space: train for space in SPACES[:4]}
    teacher = {space: teacher_train for space in SPACES[:4]}
    identity_errors = {"train": train_error}
    for space, stride in (("val", cli.val_stride), ("test", cli.test_stride)):
        raw, identity_errors[space] = _collect(model, datasets[space], args, device,
                                               stride, cli.extract_batch_size)
        data[space] = flat_record(prepare(raw, floor), release_hidden=True)
        del raw
        gc.collect()
    head = (model.head_linear.weight.detach().cpu().numpy().astype(np.float32),
            model.head_linear.bias.detach().cpu().numpy().astype(np.float32))
    for space in ("A", "val", "test"):
        record = data[space]
        end = min(64, len(record["design"]))
        rebuilt = (record["design"][:end] @ head[0].T + head[1]) * record["ratio"][:end]
        error = float(np.max(np.abs(rebuilt - record["a0"][:end])))
        if error > 2e-3:
            raise AssertionError(f"flat H/A0 identity failed on {space}: {error}")
        identity_errors[f"flat_{space}"] = error
    meta = {"A0_sha256": a0_sha, "q_sha256": file_hash(cli.output / "teacher_q.npz"),
            "dataset": args.dataset_name, "backbone": args.backbone,
            "horizon": args.pred_len, "seed": cli.seed,
            "strides": {"train": cli.train_stride, "val": cli.val_stride,
                        "test": cli.test_stride},
            "calendar_mode": protocol["periodic_mode"],
            "calendar_origin": "token A0 source protocol; input window history only",
            "A0_head": "trained head_linear from strict token checkpoint",
            "identity_error_max": identity_errors,
            "teacher_source": teacher_meta["source"], "features_persisted": False}
    dump(cli.output / "features.json", meta)
    del model, datasets
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return data, meta, head, teacher


def _selected_q(output, data):
    router = router_from_checkpoint(output / "router.pt")
    return {space: route(router, data[space]["key"]) for space in ("A", "val", "test")}


def _fit_trick(cli, data, q, original, f03):
    output = cli.output
    a0 = {space: data[space]["a0"] for space in q}
    base_features = {space: correction_design(a0[space], data[space]["key"])
                     for space in q}
    device = torch.device(f"cuda:{cli.gpu}" if cli.gpu >= 0 else "cpu")
    selection = cli.selection_split
    best, trials = None, []
    for lr in parse_floats(cli.variation_lrs):
        model, info = fit_variation(
            base_features["A"], f03["A"], data["A"]["y"], q["A"],
            base_features[selection], a0[selection], data[selection]["y"], q[selection],
            lr, cli.epochs, cli.batch_size, cli.seed, device)
        trials.append(info)
        if best is None or info["B2_mse"] < best[0]:
            best = (info["B2_mse"], copy.deepcopy(model.cpu().state_dict()), info)
    checkpoint = output / "r3_trick.pt"
    torch.save(best[1], checkpoint)
    model = EnvironmentLinearCorrection(base_features["A"].shape[1],
                                        data["A"]["y"].shape[1]).to(device)
    model.load_state_dict(best[1], strict=True)
    scores = {}
    for space in ("val", "test"):
        correction, _ = predict_variation(model, base_features[space], q[space], device)
        scores[space] = mse_mae(a0[space] + correction, data[space]["y"])
    return {"fit_target": "Y minus selected F03 full forecast",
            "input": "A0 forecast and observed history key", "inference_anchor": "A0",
            "selection_split": selection, "selected": best[2], "trials": trials,
            "scores": scores, "checkpoint_sha256": file_hash(checkpoint)}


def _fit_reverse(cli, data, q, original):
    output = cli.output
    r3_path = output / "A0_no_F03_r3.pt"
    r3_sha = file_hash(r3_path)
    device = torch.device(f"cuda:{cli.gpu}" if cli.gpu >= 0 else "cpu")
    a0 = {space: data[space]["a0"] for space in q}
    features = {space: correction_design(a0[space], data[space]["key"])
                for space in q}
    model = EnvironmentLinearCorrection(features["A"].shape[1],
                                        data["A"]["y"].shape[1]).to(device)
    model.load_state_dict(torch.load(r3_path, map_location="cpu", weights_only=False),
                          strict=True)
    model.eval().requires_grad_(False)
    correction = {space: predict_variation(model, features[space], q[space], device)[0]
                  for space in q}
    pure = {space: mse_mae(a0[space] + correction[space], data[space]["y"])
            for space in ("val", "test")}
    adjusted = {space: corrected_record(data[space], correction[space]) for space in q}
    best, trials = None, []
    grid = [("mse_only", 0.)] + [("mse_first_irm", lam)
                                 for lam in parse_floats(cli.lambda_values)]
    for mode, lam in grid:
        for lr in parse_floats(cli.head_lrs):
            state, info = fit_head_after_r3(cli, adjusted, q, original,
                                            mode, lam, lr, device)
            alpha, selection_mse = val_alpha_and_mse(
                adjusted[cli.selection_split], a0[cli.selection_split], state)
            trial = {**info, "alpha": alpha, "selection_mse": selection_mse}
            trials.append(trial)
            dump(output / "reverse_head_search.json", {"trials": trials,
                 "selection_split": cli.selection_split,
                 "frozen_r3_sha256": r3_sha})
            if best is None or selection_mse < best[0] - 1e-8:
                best = (selection_mse, alpha, state, trial)
    selected = interpolation(original, best[2], best[1])
    head_path = output / "selected_r3_aware_f03.pt"
    torch.save(selected, head_path)
    scores = {space: mse_mae(forecast(data[space], selected) + correction[space],
                             data[space]["y"]) for space in ("val", "test")}
    if scores[cli.selection_split]["mse"] > pure[cli.selection_split]["mse"] + 1e-5:
        raise AssertionError("reverse-stage alpha=0 protection failed")
    if file_hash(r3_path) != r3_sha:
        raise AssertionError("pure R3 checkpoint changed during reverse stage")
    return {"protocol": "freeze pure A0-conditioned R3; fit full H-to-Y-minus-R3 head",
            "selection_split": cli.selection_split, "selected": best[3],
            "trials": trials, "pure_R3": pure, "scores": scores,
            "frozen_r3_sha256": r3_sha, "head_sha256": file_hash(head_path)}


def run(cli):
    cli.a0_run_dir = _absolute(cli.a0_run_dir)
    cli.output = _absolute(cli.output_dir)
    cli.output_root = _output_root(cli.output, cli.case_key)
    cli.fixed_q_case = _absolute(cli.fixed_q_case) if cli.fixed_q_case else None
    if cli.gpu < -1 or min(cli.train_stride, cli.val_stride, cli.test_stride,
                           cli.extract_batch_size, cli.batch_size, cli.epochs,
                           cli.router_epochs) <= 0:
        raise ValueError("invalid GPU, stride, batch size, or epoch count")
    if cli.output == cli.a0_run_dir or cli.a0_run_dir in cli.output.parents:
        raise ValueError("downstream output must not overwrite token A0")
    checkpoint = cli.a0_run_dir / "token" / "best_val.pt"
    source_sha = file_hash(checkpoint)
    final = cli.output / "downstream_complete"
    metrics_path = cli.output / "metrics.json"
    if cli.resume and final.is_file() and metrics_path.is_file():
        previous = json.loads(metrics_path.read_text())
        if (previous.get("status") == "downstream_complete"
                and previous.get("A0_sha256") == source_sha
                and previous.get("selection_split") == cli.selection_split
                and previous.get("downstream_protocol", {}).get("q_mode") ==
                    ("fixed_original" if cli.fixed_q_case else "rediscover")):
            print(json.dumps({"case": cli.case_key, "status": "REUSED"}), flush=True)
            return
        raise ValueError("existing downstream completion belongs to another protocol")
    cli.output.mkdir(parents=True, exist_ok=True)
    if metrics_path.is_file():
        previous = json.loads(metrics_path.read_text())
        if previous.get("A0_sha256") != source_sha:
            raise ValueError("existing output belongs to another token A0")
    cli.full_train = True
    cli.environment_supervision = "predictive"
    cli.stop_on_gain = False
    cli.manifest = None
    cli.resume = True
    cli.asset = {"status": "MISSING_ENV_Q", "A0_sha256": source_sha}
    cache = {}

    def provider(request):
        if "data" not in cache:
            cache["data"] = _provider(request)
        return cache["data"]

    run_safe_full(cli, data_loader=provider, asset_override=cli.asset)
    metrics = json.loads(metrics_path.read_text())
    if metrics.get("A0_sha256") != source_sha or metrics.get("selection_split") != cli.selection_split:
        raise AssertionError("EIIL3 result identity/protocol mismatch")
    data, meta, raw_head, _ = provider(cli)
    q = _selected_q(cli.output, data)
    original = {"weight": torch.from_numpy(raw_head[0]),
                "bias": torch.from_numpy(raw_head[1])}
    selected = torch.load(cli.output / "selected_safe_head.pt", map_location="cpu",
                          weights_only=False)
    f03 = {space: forecast(data[space], selected) for space in q}
    metrics["r3_trick"] = _fit_trick(cli, data, q, original, f03)
    metrics["versions"]["R3(trick)"] = metrics["r3_trick"]["scores"]
    dump(metrics_path, metrics)
    metrics["reverse"] = _fit_reverse(cli, data, q, original)
    metrics["versions"]["F03+R3(forward-control)"] = metrics["versions"]["F03+R3"]
    metrics["versions"]["F03+R3"] = metrics["reverse"]["scores"]
    metrics["versions"]["F03+R3(reverse)"] = metrics["reverse"]["scores"]
    metrics["A0_source_checkpoint"] = str(checkpoint)
    metrics["A0_source_protocol_sha256"] = file_hash(cli.a0_run_dir / "protocol.json")
    metrics["downstream_protocol"] = {
        "q_mode": "fixed_original" if cli.fixed_q_case else "rediscover",
        "fixed_q_case": str(cli.fixed_q_case) if cli.fixed_q_case else None,
        "full_train": True, "selection_split": cli.selection_split,
        "selection_notice": ("TEST is exploratory development data" if cli.selection_split == "test"
                             else "VAL selected; TEST report-only"),
        "calendar_history_only": True, "features_persisted": False,
        "head_source": "token checkpoint head_linear", "A0_identity": meta["identity_error_max"],
    }
    metrics["status"] = "downstream_complete"
    dump(metrics_path, metrics)
    dump(cli.output / "token_downstream_run_config.json", {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(cli).items() if key != "asset"
    })
    final.write_text("complete\n")
    print(json.dumps({"case": cli.case_key, "status": "downstream_complete",
                      "A0_sha256": source_sha,
                      "versions_test": {name: value["test"] for name, value in metrics["versions"].items()}}),
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-key", required=True)
    parser.add_argument("--a0-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-a0-sha256")
    parser.add_argument("--fixed-q-case", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--val-stride", type=int, default=8)
    parser.add_argument("--test-stride", type=int, default=1)
    parser.add_argument("--extract-batch-size", type=int, default=8)
    parser.add_argument("--router-epochs", type=int, default=30)
    parser.add_argument("--router-lr", type=float, default=.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--head-lrs", default="0.00002,0.0001,0.0003,0.001")
    parser.add_argument("--lambda-values", default="0.0001,0.01")
    parser.add_argument("--variation-lrs", default="0.0001,0.0003,0.001,0.003")
    parser.add_argument("--selection-split", choices=("val", "test"), default="test")
    parser.add_argument("--resume", action="store_true")
    cli = parser.parse_args()
    torch.set_num_threads(4)
    run(cli)


if __name__ == "__main__":
    main()
