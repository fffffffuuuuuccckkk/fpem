"""One best weight per independent comparison; every trial keeps metrics."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch
import yaml

from .common import atomic_json,sha256


def require_free_space(path,min_free_gib=20,min_free_ratio=0.05,policy_path=None):
    """Fail before a new run writes artifacts when the target volume is too full."""
    if policy_path is not None:
        safety=yaml.safe_load(Path(policy_path).read_text())["safety"]
        min_free_gib=float(safety["min_free_gb_to_start"])
        min_free_ratio=float(safety["min_free_ratio_to_start"])
    path=Path(path)
    while not path.exists():path=path.parent
    usage=shutil.disk_usage(path)
    free=usage.free
    required=max(int(min_free_gib*2**30),int(usage.total*min_free_ratio))
    if free<required:
        raise OSError(f"low-storage preflight: {free/2**30:.2f} GiB free; "
                      f"need at least {required/2**30:.2f} GiB on {path}")
    return free


def record_trial(run,trial):
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    required=("trial_id","params","seed","epoch","status","objective","selected_by")
    missing=[name for name in required if name not in trial]
    if missing:raise ValueError("trial metadata missing: "+",".join(missing))
    path=run/"trials.jsonl"
    attempt=1
    if path.exists():
        for line in path.read_text().splitlines():
            previous=json.loads(line)
            if (previous.get("trial_id"),previous.get("comparison_key"))==(
                trial["trial_id"],trial.get("comparison_key")):
                attempt=max(attempt,int(previous.get("attempt",1))+1)
    trial={**trial,"attempt":attempt}
    with path.open("a",encoding="utf-8") as stream:
        stream.write(json.dumps(trial,sort_keys=True,default=str)+"\n")
        stream.flush();os.fsync(stream.fileno())


def promote_if_best(run,trial,source,comparison_key,metric_name,minimum_free_bytes=0):
    """Validate challenger before atomic best replacement. Never touch other runs."""
    run=Path(run);source=Path(source)
    if not source.is_file() or source.is_symlink():raise ValueError("source checkpoint invalid")
    if trial["status"]!="complete":return False
    if trial["objective"]!=metric_name:raise ValueError("objective/metric mismatch")
    if trial.get("comparison_key")!=comparison_key:raise ValueError("comparison unit mismatch")
    best_path=run/"best_model.pt";manifest_path=run/"best_manifest.json"
    existing=json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    value=float(trial[metric_name]);old=float(existing["metric_value"]) if existing else float("inf")
    if not value<old:return False
    free=shutil.disk_usage(run).free
    if free<source.stat().st_size+minimum_free_bytes:
        raise OSError("insufficient disk for atomic challenger validation")
    temporary=run/"best_model.pt.tmp"
    shutil.copyfile(source,temporary)
    old_weight=run/"best_model.pt.old"
    old_manifest=run/"best_manifest.json.old"
    if old_weight.exists() or old_manifest.exists():
        temporary.unlink()
        raise RuntimeError("previous interrupted promotion needs manual recovery")
    try:
        loaded=torch.load(temporary,map_location="cpu",weights_only=False)
        if loaded is None:raise ValueError("empty checkpoint")
        digest=sha256(temporary)
        manifest={"comparison_key":comparison_key,"objective":metric_name,
            "metric_value":value,"trial_id":trial["trial_id"],"checkpoint":"best_model.pt",
            "sha256":digest,"selected_by":trial["selected_by"],
            "test_used_for_selection":trial["selected_by"].upper()=="TEST"}
        if best_path.exists():os.link(best_path,old_weight)
        if manifest_path.exists():os.link(manifest_path,old_manifest)
        # Existing best remains untouched until the challenger is known loadable.
        os.replace(temporary,best_path)
        try:atomic_json(manifest_path,manifest)
        except Exception:
            if old_weight.exists():os.replace(old_weight,best_path)
            if old_manifest.exists():os.replace(old_manifest,manifest_path)
            raise
    finally:
        if temporary.exists():temporary.unlink()
        if old_weight.exists():old_weight.unlink()
        if old_manifest.exists():old_manifest.unlink()
    return True
