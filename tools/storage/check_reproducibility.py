#!/usr/bin/env python3
"""Rebuild the completed G1 frozen-A0 cache using its original producer."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .common import atomic_json,sha256,stat_fingerprint


def compare(a,b,label,differences):
    if isinstance(a,dict):
        if set(a)!=set(b):raise AssertionError(f"{label} keys differ")
        for key in a:compare(a[key],b[key],f"{label}.{key}",differences)
    elif isinstance(a,(np.ndarray,torch.Tensor)):
        x=np.asarray(a.detach().cpu() if isinstance(a,torch.Tensor) else a)
        y=np.asarray(b.detach().cpu() if isinstance(b,torch.Tensor) else b)
        if x.shape!=y.shape or x.dtype!=y.dtype:raise AssertionError(f"{label} shape/dtype differs")
        if np.issubdtype(x.dtype,np.number):
            index=np.r_[0:min(8,len(x)),max(0,len(x)-8):len(x)] if x.ndim else None
            error=float(np.max(np.abs(x[index]-y[index]))) if x.ndim and len(x) else float(abs(x-y))
            if not np.isfinite(error) or error>2e-4:raise AssertionError(f"{label} sample mismatch {error}")
            differences[label]=error
        elif not np.array_equal(x,y):raise AssertionError(f"{label} differs")
    elif a!=b:raise AssertionError(f"{label}: {a!r} != {b!r}")


def run(directory,report,gpu=0):
    directory=directory.resolve(strict=True)
    config=json.loads((directory/"run_config.json").read_text())
    result=json.loads((directory/"result.json").read_text())
    cache_path=directory/"frozen_a0_cache.pt"
    if not (directory/"complete").is_file() or not cache_path.is_file():
        raise ValueError("completed run and cache both required")
    if result["source_a0_sha256"]!=config["a0_sha256"]:raise AssertionError("A0 hash mismatch")
    from tools.run_fpem_g1_stability_search import _cache_source
    cli=SimpleNamespace(backbone=config["cli"]["backbone"],dataset=config["cli"]["dataset"],
                        pred_len=config["cli"]["pred_len"],seed=config["cli"]["seed"],gpu=gpu)
    original=torch.load(cache_path,map_location="cpu",weights_only=False)
    with tempfile.TemporaryDirectory(prefix="fpem_cache_rebuild_") as scratch:
        rebuilt,source,metrics=_cache_source(cli,Path(scratch))
        difference={}
        compare(original,rebuilt,"cache",difference)
        if rebuilt["a0_sha256"]!=config["a0_sha256"]:raise AssertionError("rebuilt A0 identity mismatch")
        if abs(rebuilt["a0"]["mse"]-result["a0"]["mse"])>2e-5:
            raise AssertionError("downstream A0 MSE changed")
        if abs(rebuilt["a0"]["mae"]-result["a0"]["mae"])>2e-5:
            raise AssertionError("downstream A0 MAE changed")
    verification={"status":"passed","cache_path":str(cache_path),
       "original_size":cache_path.stat().st_size,"original_sha256":sha256(cache_path),
       "fingerprint":stat_fingerprint(cache_path),"source_a0_sha256":config["a0_sha256"],
       "source_script":"tools/run_fpem_g1_stability_search.py:_cache_source",
       "run_complete":True,"metric_mse":result["a0"]["mse"],"metric_mae":result["a0"]["mae"],
       "numeric_tolerance":2e-4,"max_sample_error":max(difference.values(),default=0),
       "compared_fields":list(difference),"scratch_removed":True,
       "rebuild_command":f"python -m tools.run_fpem_g1_stability_search --backbone {cli.backbone} --dataset {cli.dataset} --pred_len {cli.pred_len} --seed {cli.seed} --gpu {gpu} --result_root {directory.parents[2]}"}
    atomic_json(report,verification)
    print(json.dumps({"status":"passed","cache":str(cache_path),"max_error":verification["max_sample_error"]}))


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True);p.add_argument("--gpu",type=int,default=0)
    a=p.parse_args()
    try:run(a.run,a.report,a.gpu)
    except (OSError,KeyError,ValueError,AssertionError,RuntimeError) as exc:
        result={"status":"BLOCKED","run":str(a.run),"reason":str(exc),
                "decision":"PROTECT until producer, consumers and downstream equality are proved"}
        atomic_json(a.report,result)
        print(json.dumps(result))


if __name__=="__main__":main()
