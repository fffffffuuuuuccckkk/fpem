"""Prepare existing A0 references and point-only G1+R3 cases; never train A0."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
DATASETS=("ETTh1","ETTh2","ETTm1","ETTm2","ExchangeRate")
HORIZONS=(96,192,336,720)
BASE=ROOT/"results/fpem_probabilistic"

def rows(path):
    with path.open(newline="",encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def baseline(backbone,dataset,horizon):
    if backbone=="patchtst":
        catalog=BASE/"test_best_patchtst_a0_20261013/selected_patchtst_a0_catalog.csv"
        ref=BASE/"test_best_patchtst_a0_20261013/references"/dataset/f"pred_{horizon}.pt"
    else:
        catalog=BASE/f"test_best_backbones_a0_20261013/selected_{backbone}_a0_catalog.csv"
        ref=(BASE/"test_best_backbones_a0_20261013/references/itransformer"/dataset/f"pred_{horizon}.pt"
             if backbone=="itransformer" else None)
    match=[row for row in rows(catalog) if row["dataset"]==dataset and
           int(row["pred_len"])==horizon]
    if len(match)!=1: raise ValueError(f"ambiguous A0 catalog: {backbone} {dataset} {horizon}")
    row=match[0]
    if backbone=="moderntcn": ref=Path(row["reference_path"])
    return row,ref

def convert_itransformer(row,ref,dataset,horizon):
    raw=Path(row["checkpoint_path"])
    if not raw.is_absolute(): raw=ROOT/raw
    if not raw.is_file(): return f"missing raw iTransformer checkpoint: {raw}"
    data_class="ett_hour" if dataset.startswith("ETTh") else "ett_minute" if dataset.startswith("ETTm") else "custom"
    freq="h" if data_class=="ett_hour" else "15min" if data_class=="ett_minute" else "d"
    archive=(ROOT/"dataset/all_datasets/archive_sha256.txt").read_text().split()[0]
    cmd=[sys.executable,str(ROOT/"tools/import_test_best_itransformer_a0.py"),
         "--checkpoint",str(raw),"--output",str(ref),"--dataset",dataset,
         "--horizon",str(horizon),"--data-class",data_class,
         "--data-path",f"{dataset}.csv" if dataset!="ExchangeRate" else "exchange_rate.csv",
         "--freq",freq,"--enc-in",str(8 if dataset=="ExchangeRate" else 7),
         "--cycle-len",str(24 if data_class=="ett_hour" else 96 if data_class=="ett_minute" else 7),
         "--archive-sha256",archive,"--test-mse",row["test_mse"],"--test-mae",row["test_mae"]]
    try: subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError as error: return f"iTransformer conversion failed: {error}"
    return None

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--backbones",default="patchtst,itransformer,moderntcn")
    p.add_argument("--output_root",default="results/fpem_dual_pattern_memory/g1_r3_full_point")
    p.add_argument("--convert_itransformer",action="store_true")
    args=p.parse_args()
    output=ROOT/args.output_root
    output.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for backbone in args.backbones.split(","):
        for dataset in DATASETS:
            for horizon in HORIZONS:
                row,ref=baseline(backbone,dataset,horizon)
                reason=""
                if not ref.is_file() and backbone=="itransformer" and args.convert_itransformer:
                    reason=convert_itransformer(row,ref,dataset,horizon) or ""
                if not ref.is_file() and not reason: reason=f"missing selected A0 reference: {ref}"
                case=output/backbone/dataset/f"pred_{horizon}"
                config=case/"base_run_config.json"
                if not reason:
                    payload=torch.load(ref,map_location="cpu",weights_only=False)
                    signature=payload.get("signature")
                    if not isinstance(signature,dict) or signature.get("backbone")!=backbone or signature.get("dataset_name")!=dataset or signature.get("pred_len")!=horizon:
                        reason=f"selected A0 signature mismatch: {ref}"
                    else:
                        config.parent.mkdir(parents=True,exist_ok=True)
                        source=dict(signature)
                        source.pop("protocol_version",None)
                        source.update({"root_path":"./dataset/all_datasets/ETT-small" if dataset!="ExchangeRate" else "./dataset/all_datasets/exchange_rate",
                                       "reference_checkpoint":str(ref.resolve()),
                                       "require_reference_checkpoint":True,"batch_size":32,
                                       "num_workers":0,"experiments":"A0","epochs":0})
                        config.write_text(json.dumps(source,indent=2)+"\n",encoding="utf-8")
                manifest.append({"backbone":backbone,"dataset":dataset,"pred_len":horizon,
                                 "case_dir":str(case),"base_run_config":str(config),
                                 "reference_checkpoint":str(ref),
                                 "status":"ready" if not reason else "missing",
                                 "reason":reason,"a0_catalog_test_mse":row["test_mse"],
                                 "a0_catalog_test_mae":row["test_mae"]})
    path=output/"manifest.csv"
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(manifest[0]))
        writer.writeheader();writer.writerows(manifest)
    print("ready",sum(row["status"]=="ready" for row in manifest),"missing",sum(row["status"]=="missing" for row in manifest),"manifest",path)
    for row in manifest:
        if row["status"]=="missing": print("MISSING",row["backbone"],row["dataset"],row["pred_len"],row["reason"])

if __name__=="__main__": main()
