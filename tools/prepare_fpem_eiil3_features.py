#!/usr/bin/env python3
"""Prepare TRAIN-only EIIL q and reusable frozen-A0 features when storage-safe.

The feature cache is exact float32; oversized caches are not written. This
preserves the repository's low-storage policy without changing computation.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_fpem_eiil3 import SPACES, dump, get_data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--case-key", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max-cache-gb", type=float, default=8.)
    cli = p.parse_args()
    cli.asset = next(row for row in (json.loads(line) for line in cli.manifest.read_text().splitlines())
                     if row["case_key"] == cli.case_key)
    cli.seed = cli.asset["seed"]
    cli.train_stride = 8
    cli.val_stride = 8
    cli.test_stride = 1
    cli.eiil_steps = 50
    cli.generate_missing_q = True
    cli.persist_features = False
    cli.output.mkdir(parents=True, exist_ok=True)
    data, meta, _, teacher = get_data(cli)
    fields = {f"{space}_{name}": value for space in SPACES
              for name, value in data[space].items()}
    fields.update({f"q_teacher_{space}": teacher[space] for space in SPACES[:4]})
    bytes_needed = sum(value.nbytes for value in fields.values())
    free = shutil.disk_usage(cli.output).free
    limit = int(cli.max_cache_gb * (1024 ** 3))
    cached = bool(bytes_needed <= limit and free > 2 * bytes_needed + 10 * (1024 ** 3))
    if cached:
        np.savez_compressed(cli.output / "features.npz", **fields)
        meta["features_persisted"] = True
        dump(cli.output / "features.json", meta)
    status = {"case": cli.case_key, "status": "prepared", "A0_sha256": meta["A0_sha256"],
              "q_sha256": meta["q_sha256"], "feature_cache": cached,
              "uncompressed_feature_bytes": bytes_needed,
              "max_cache_gb": cli.max_cache_gb,
              "reason_if_not_cached": None if cached else "size_or_free_space_limit"}
    dump(cli.output / "assets_status.json", status)
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
