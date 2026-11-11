#!/usr/bin/env python3
"""Only remove this completed run's owned scratch files after winner validation."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import yaml

from .common import active_file,sha256


def prune(run,policy,dry_run=True):
    run=run.resolve(strict=True)
    rules=yaml.safe_load(Path(policy).read_text())
    if not (run/"complete").exists():raise ValueError("run not complete")
    if rules["trick_search"]["best_checkpoints_per_comparison"]!=1:
        raise ValueError("only top-1 supported by this pruning tool")
    manifest=json.loads((run/"best_manifest.json").read_text())
    best=run/manifest["checkpoint"]
    if not best.is_file() or sha256(best)!=manifest["sha256"]:
        raise ValueError("best checkpoint hash mismatch")
    if torch.load(best,map_location="cpu",weights_only=False) is None:
        raise ValueError("best checkpoint not loadable")
    if not (run/"trials.jsonl").exists():raise ValueError("trial ledger absent")
    if (run/"formal_baseline").exists() or (run/"formal_ablation").exists():
        raise ValueError("formal baseline/ablation cannot be pruned by trick tool")
    scratch=run/"scratch"
    if not scratch.exists():return []
    if scratch.is_symlink():raise ValueError("scratch symlink")
    result=[]
    for path in scratch.iterdir():
        if not path.is_file() or path.is_symlink() or path.suffix not in (".pt",".tmp"):
            raise ValueError(f"unknown scratch artifact {path}")
        if active_file(path) is not None:raise ValueError(f"scratch file active/unknown {path}")
        result.append({"path":str(path),"bytes":path.stat().st_size})
        if not dry_run:path.unlink()
    if not dry_run and not any(scratch.iterdir()):scratch.rmdir()
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True)
    p.add_argument("--policy",type=Path,default=Path("config/storage_policy.yaml"))
    p.add_argument("--apply",action="store_true")
    a=p.parse_args();print(json.dumps(prune(a.run,a.policy,not a.apply)))
