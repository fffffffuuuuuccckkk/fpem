#!/usr/bin/env python3
"""Read-only inventory of A0 and genuine K=3 TRAIN environment assignments.

This deliberately distinguishes a previous PredictiveEnvIV-stage teacher from
one computed from the frozen A0. It never silently substitutes 168 for 192.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Electricity", "Weather",
            "Traffic", "ExchangeRate")
BACKBONES = ("patchtst", "itransformer", "moderntcn")
HORIZONS = (96, 192, 336, 720)
Q_FIELDS = ("q_env_0", "q_env_1", "q_env_2")


def digest(path):
    sha = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def source_run_config(path):
    for parent in (path.parent.parent, path.parent.parent.parent, path.parent):
        config = parent / "run_config.json"
        if config.is_file():
            try:
                return json.loads(config.read_text()), config
            except (ValueError, OSError):
                pass
    return None, None


def key(config):
    try:
        return (str(config["backbone"]).lower(), str(config["dataset_name"]),
                int(config["pred_len"]), int(config.get("seq_len", 96)),
                int(config.get("seed", 2021)))
    except (KeyError, TypeError, ValueError):
        return None


def resolve(repo, name):
    if not name:
        return None
    path = Path(name)
    return path if path.is_absolute() else repo / path


def verify_q(path, train_rows, seq_len, horizon):
    try:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"sample_id", *Q_FIELDS}.issubset(reader.fieldnames or []):
                return None, "MISSING_Q_COLUMNS"
            records = [(int(row["sample_id"]), *(float(row[col]) for col in Q_FIELDS))
                       for row in reader]
        values = np.asarray(records, dtype=np.float64)
        n_valid = max(0, train_rows - seq_len - horizon + 1)
        if (values.ndim != 2 or values.shape[1] != 4 or len(values) == 0 or
                len(np.unique(values[:, 0])) != len(values) or
                values[:, 0].min() < 0 or values[:, 0].max() >= n_valid or
                not np.isfinite(values).all() or np.min(values[:, 1:]) < -1e-6 or
                not np.allclose(values[:, 1:].sum(axis=1), 1., atol=2e-4)):
            return None, "INVALID_Q_VALUES_OR_IDS"
        return {"q_shape": [len(values), 3],
                "q_ids": [int(values[:, 0].min()), int(values[:, 0].max())],
                "q_mass": values[:, 1:].mean(axis=0).tolist(),
                "q_unique_ids": len(values), "train_window_count": n_valid}, None
    except (OSError, ValueError, OverflowError):
        return None, "UNREADABLE_Q"


def train_row_count(path, dataset):
    with path.open("rb") as stream:
        total = sum(block.count(b"\n") for block in iter(lambda: stream.read(1 << 20), b"")) - 1
    if dataset.startswith("ETTh"):
        return min(total, 12 * 30 * 24)
    if dataset.startswith("ETTm"):
        return min(total, 12 * 30 * 24 * 4)
    return int(total * .7)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("results/fpem_eiil3_20261110"))
    parser.add_argument("--seed", type=int, default=2021)
    cli = parser.parse_args()
    repo = cli.repo.resolve()
    output = (repo / cli.output).resolve() if not cli.output.is_absolute() else cli.output
    output.mkdir(parents=True, exist_ok=True)
    configs = {}
    for path in (repo / "results").rglob("*run_config.json"):
        if output in path.parents:
            continue
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        name = key(value)
        if name is None:
            continue
        a0 = resolve(repo, value.get("reference_checkpoint"))
        if a0 is None or not a0.is_file():
            continue
        priority = (path.name == "base_run_config.json", "test_best_backbones_a0" in str(a0),
                    "g1_r3_full_point" in str(path), -len(str(path)))
        configs.setdefault(name, []).append((priority, path, value, a0))
    assignments = {}
    for path in (repo / "results").rglob("environment_gradient_final.csv"):
        if output in path.parents:
            continue
        cfg, config_path = source_run_config(path)
        name = key(cfg) if cfg else None
        if name is None or cfg.get("env_num") != 3:
            continue
        assignments.setdefault(name, []).append((path, cfg, config_path))
    records = []
    a0_hashes = {}
    data_rows = {}
    for backbone in BACKBONES:
        for dataset in DATASETS:
            for horizon in HORIZONS:
                name = (backbone, dataset, horizon, 96, cli.seed)
                options = sorted(configs.get(name, []), key=lambda item: item[0], reverse=True)
                record = {"backbone": backbone, "dataset": dataset, "pred_len": horizon,
                          "seq_len": 96, "seed": cli.seed, "K": 3,
                          "case_key": f"{backbone}/{dataset}/pred_{horizon}/seed_{cli.seed}",
                          "status": "MISSING_A0", "missing_reason": "no exact A0 run_config/checkpoint"}
                if not options:
                    records.append(record)
                    continue
                chosen = None
                chosen_q = None
                for option in options:
                    _, option_path, option_cfg, option_a0 = option
                    if option_a0 not in a0_hashes:
                        a0_hashes[option_a0] = digest(option_a0)
                    exact = []
                    for path, q_config, q_config_path in assignments.get(name, []):
                        q_a0 = resolve(repo, q_config.get("reference_checkpoint"))
                        if q_a0 is None or not q_a0.is_file():
                            continue
                        if q_a0 not in a0_hashes:
                            a0_hashes[q_a0] = digest(q_a0)
                        if a0_hashes[q_a0] == a0_hashes[option_a0]:
                            exact.append((path, q_config, q_config_path))
                    if exact:
                        chosen = option
                        chosen_q = exact
                        break
                if chosen is None:
                    chosen = options[0]
                    chosen_q = []
                _, config_path, cfg, a0 = chosen
                data_path = resolve(repo, str(Path(cfg.get("root_path", "")) / cfg.get("data_path", "")))
                record.update(base_run_config=str(config_path), A0_ckpt=str(a0),
                              data_path=str(data_path), data_class=cfg.get("data_class"),
                              a0_source="existing_reference_checkpoint")
                if not data_path.is_file():
                    record.update(status="MISSING_DATA", missing_reason="configured data_path missing")
                    records.append(record)
                    continue
                if a0 not in a0_hashes:
                    a0_hashes[a0] = digest(a0)
                record["A0_sha256"] = a0_hashes[a0]
                if data_path not in data_rows:
                    data_rows[data_path] = train_row_count(data_path, dataset)
                candidate_q = []
                for path, q_config, q_config_path in chosen_q:
                    validation, error = verify_q(path, data_rows[data_path], 96, horizon)
                    if error is None:
                        candidate_q.append((path, q_config, q_config_path, validation))
                if candidate_q:
                    candidate_q.sort(key=lambda x: ("compact_future_5e-4" in str(x[0]),
                                                   x[3]["q_unique_ids"]), reverse=True)
                    path, q_config, q_config_path, validation = candidate_q[0]
                    record.update(status="READY", missing_reason="", env_artifact=str(path),
                                  env_artifact_sha256=digest(path), q_source_stage="A2_final",
                                  q_provenance="legacy PredictiveEnvIV TRAIN teacher; refreshed during stages, not frozen-A0 teacher",
                                  q_config=str(q_config_path), **validation)
                else:
                    record.update(status="MISSING_ENV_Q", missing_reason="no exact A0/seed/horizon matched, validated K=3 TRAIN q")
                records.append(record)
    with (output / "manifest.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    counts = {status: sum(row["status"] == status for row in records)
              for status in sorted({row["status"] for row in records})}
    report = ["# EIIL3 asset audit", "", f"Exact 96-case scan: {counts}.",
              "Only exact backbone/dataset/96-step-history/horizon/seed and A0 SHA match is READY.",
              "Old A2 teacher q was updated during PredictiveEnvIV training; it is NOT frozen-A0 gradient q.",
              "No 168-step artifact was accepted for a 192-step case. Original files were not changed.", ""]
    for status in counts:
        report.append(f"## {status} ({counts[status]})")
        report.extend(f"- {row['case_key']}: {row.get('missing_reason', '')}" for row in records if row["status"] == status)
    (output / "audit_report.md").write_text("\n".join(report) + "\n")
    extras = [("Solar", repo/"dataset/solar/solar_AL.csv", "raw data exists, but no exact seq96/horizon96-720 A0+EIIL3 baseline is in the audited eight-dataset protocol"),
              ("ILI", repo/"dataset/all_datasets/illness/national_illness.csv", "data exists, but long-horizon 96/192/336/720 baseline and independent TRAIN event support need separate protocol"),
              ("PEMS-SF", repo/"dataset/all_datasets/PEMS-SF/PEMS-SF_TRAIN.ts", "classification-format .ts data; no compatible forecasting A0 configuration"),
              ("PEMS03/04/07/08", repo/"dataset/all_datasets/PEMS03", "no compatible local forecasting files or A0 configs found")]
    with (output/"excluded_datasets.csv").open("w",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(("dataset","data_path","data_exists","reason"))
        for dataset,path,reason in extras:
            writer.writerow((dataset,str(path),path.exists(),reason))
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
