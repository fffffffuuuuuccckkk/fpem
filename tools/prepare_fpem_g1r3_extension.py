"""Plan/reuse A0 references for G1+R3 Electricity/Weather extensions.

Only missing references are warmed up; all outputs live in a new result root.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/fpem_dual_pattern_memory/g1_r3_extended_test_search"
DATA = {
    "Electricity": ("electricity", "electricity.csv", "h", 168),
    "Weather": ("weather", "weather.csv", "10min", 144),
    "Solar": ("solar", "solar_AL.csv", "10min", 144),
    "ExchangeRate": ("exchange_rate", "exchange_rate.csv", "d", 7),
    "ILI": ("illness", "national_illness.csv", "w", 52),
}
HORIZONS = (96, 192, 336, 720)
BACKBONES = ("patchtst", "itransformer", "moderntcn")


def read_signature(path):
    return torch.load(path, map_location="cpu", weights_only=False)["signature"]


def candidates(backbone, dataset=None, horizon=None):
    for path in (ROOT / "results").rglob("shared_reference.pt"):
        parts = set(path.parts)
        if backbone not in parts or (dataset and dataset not in parts):
            continue
        if horizon is not None and f"pred_{horizon}" not in parts:
            continue
        yield path


def preferred(paths):
    return sorted(paths, key=lambda p: (
        "server2_imports" in p.parts,
        "fpem_final" not in str(p) and "fpem_no_future_anchor" not in str(p),
        len(str(p)), str(p)))[0]


def plan(backbone, datasets, horizons):
    archive = (ROOT / "dataset/all_datasets/archive_sha256.txt").read_text().split()[0]
    rows = []
    for dataset in datasets:
        subdir, file, freq, cycle = DATA[dataset]
        csv_path = ROOT / "dataset/all_datasets" / subdir / file
        if dataset == "Solar":
            csv_path = ROOT / "dataset/solar/solar_AL.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        enc_in = len(pd.read_csv(csv_path, nrows=1).columns) - 1
        same_dataset = list(candidates(backbone, dataset))
        templates = same_dataset or list(candidates(backbone))
        if not templates:
            raise FileNotFoundError(f"no A0 template for {backbone}")
        template = read_signature(preferred(templates))
        for horizon in horizons:
            case = OUT / backbone / dataset / f"pred_{horizon}"
            case.mkdir(parents=True, exist_ok=True)
            exact = list(candidates(backbone, dataset, horizon))
            source = preferred(exact) if exact else None
            ref = source or case / "shared_reference.pt"
            sig = dict(read_signature(source) if source else template)
            source_hash = archive
            if dataset == "Solar":
                source_hash = json.loads(csv_path.with_suffix(".provenance.json").read_text())["archive_sha256"]
            sig.update({"backbone": backbone, "dataset_name": dataset,
                        "dataset_archive_sha256": source_hash,
                        "data_class": "custom", "data_path": file, "target": "OT",
                        "freq": freq, "seq_len": 96, "pred_len": horizon,
                        "enc_in": enc_in, "cyclenet_cycle_len": cycle,
                        "seed": 2021, "warmup_epochs": 3,
                        "protocol_version": 4})
            config = dict(sig)
            config.pop("protocol_version", None)
            root_path = (f"./dataset/solar" if dataset == "Solar"
                         else f"./dataset/all_datasets/{subdir}")
            config.update({"root_path": root_path,
                           "reference_checkpoint": str(ref.resolve()),
                           "require_reference_checkpoint": True,
                           "batch_size": 16 if dataset == "Electricity" else 32,
                           "num_workers": 0, "experiments": "A0", "epochs": 0})
            (case / "base_run_config.json").write_text(
                json.dumps(config, indent=2) + "\n", encoding="utf-8")
            rows.append({"backbone": backbone, "dataset": dataset,
                         "pred_len": horizon, "case_dir": str(case),
                         "base_run_config": str(case / "base_run_config.json"),
                         "reference_checkpoint": str(ref.resolve()),
                         "reference_source": "reused" if source else "new_warmup",
                         "ready": bool(source)})
    label = "_".join(datasets) + "_" + "_".join(map(str, horizons))
    with (OUT / f"manifest_{backbone}_{label}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(backbone, "reused", sum(x["ready"] for x in rows),
          "missing", sum(not x["ready"] for x in rows), flush=True)


def ensure(case_dir, gpu):
    case = Path(case_dir)
    config = json.loads((case / "base_run_config.json").read_text())
    reference = Path(config["reference_checkpoint"])
    if reference.is_file():
        print("REUSE", reference, flush=True)
        return
    cmd = [sys.executable, str(ROOT / "tools/run_predictive_env_iv_patchtst.py")]
    keys = ("backbone", "root_path", "dataset_name", "dataset_archive_sha256",
            "data_class", "data_path", "target", "freq", "cyclenet_cycle_len",
            "seq_len", "pred_len", "d_model", "d_ff", "n_heads", "e_layers",
            "dropout", "seed", "warmup_epochs", "batch_size",
            "modern_tcn_patch_size", "modern_tcn_patch_stride",
            "modern_tcn_num_stages", "modern_tcn_ffn_ratio",
            "modern_tcn_large_size", "modern_tcn_small_size",
            "modern_tcn_downsample_ratio", "modern_tcn_head_dropout")
    for key in keys:
        if key in config:
            cmd.extend((f"--{key}", str(config[key])))
    cmd.extend(("--gpu", str(gpu), "--reference_checkpoint", str(reference),
                "--output", str(case / "reference_prepare"),
                "--prepare_reference_only", "--num_workers", "0"))
    subprocess.run(cmd, check=True, cwd=ROOT)
    if not reference.is_file():
        raise RuntimeError("A0 reference preparation returned without checkpoint")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=BACKBONES)
    p.add_argument("--ensure_case")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--datasets", default="Electricity,Weather")
    p.add_argument("--horizons", default="96,192,336,720")
    cli = p.parse_args()
    if cli.ensure_case:
        ensure(cli.ensure_case, cli.gpu)
    elif cli.backbone:
        plan(cli.backbone, cli.datasets.split(","),
             tuple(map(int, cli.horizons.split(","))))
    else:
        p.error("--backbone or --ensure_case is required")
