"""Strict TRAIN-only time-extrapolated patch replacement benefit labels."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from tools.run_fpem_csif_probabilistic import subset_loader, core_forecast
from tools.run_fpem_pgir_probabilistic import base_data, prepare_core


@torch.no_grad()
def build_gain_cache(cli, full_dataset, full_bank, selector, args, device,
                     core_p4_dir, output, reference_hash):
    path=Path(output)
    if path.exists():
        saved=torch.load(path,map_location="cpu",weights_only=False)
        if saved["reference_sha256"]!=reference_hash:
            raise RuntimeError("OOF gain cache and A0 identity mismatch")
        return saved
    p4_cfg=json.loads((Path(core_p4_dir)/"run_config.json").read_text())
    core_cli=SimpleNamespace(base_run_config=cli.base_run_config,seed=cli.seed,
                             gpu=cli.gpu,core_epochs=p4_cfg["core_epochs"],
                             mask_epochs=p4_cfg["mask_epochs"],
                             lr_core=p4_cfg["lr_core"],lr_mask=p4_cfg["lr_mask"],
                             max_batches=0)
    core_args,core_datasets,core_device=base_data(core_cli)
    model,bank,_,metadata,cal=prepare_core(core_cli,core_args,core_datasets,
                                           core_device,Path(core_p4_dir))
    # Independent anchor blocks: stride seq_len, not near-identical windows.
    anchors=cal[::args.seq_len]
    labels=[];features=[];invariants=[];raw_gains=[]
    model.eval();bank.eval()
    for x,y,ids,cycle in subset_loader(core_datasets["train"],anchors,core_args,
                                       cli.seed,False):
        x=x.float().to(device);y=y[:,-args.pred_len:].float().to(device);cycle=cycle.to(device)
        query=bank.query(x)
        proto=query["x_proto"]
        baseline=core_forecast(model,x,cycle)
        baseline_error=(baseline-y).square().mean((1,2))
        b,length,channels=x.shape
        nseg=(length+selector.segment_len-1)//selector.segment_len
        all_delta=[]
        for j in range(nseg):
            start,end=j*selector.segment_len,min(length,(j+1)*selector.segment_len)
            variants=x[:,None].repeat(1,channels,1,1)
            for c in range(channels):
                variants[:,c,start:end,c]=proto[:,start:end,c]
            flat=variants.reshape(b*channels,length,channels)
            cyc=cycle.repeat_interleave(channels,dim=0)
            pieces=[]
            for lo in range(0,len(flat),32):
                pieces.append(core_forecast(model,flat[lo:lo+32],cyc[lo:lo+32]))
            pred=torch.cat(pieces).reshape(b,channels,*y.shape[1:])
            error=(pred-y[:,None]).square().mean((2,3))
            all_delta.append(error-baseline_error[:,None])
        delta=torch.stack(all_delta,-1)
        # Normalize only for the auxiliary regression scale; zero crossing
        # and benefit sign stay unchanged. Raw Δ is retained for diagnostics.
        target=(delta/(baseline_error[:,None,None]+1e-3)).clamp(-2.,2.)
        full_x=torch.from_numpy(np.stack([
            np.asarray(full_dataset.data_x[int(i):int(i)+args.seq_len],dtype=np.float32)
            for i in ids])).to(device)
        full_query=full_bank.query(full_x)
        feature,score,_=selector._features(full_x,full_query["x_proto"],
                                           full_query,full_bank)
        invariant=(score>=selector.invariant_threshold).float()
        labels.append(target.cpu());raw_gains.append(delta.cpu())
        features.append(feature.cpu());invariants.append(invariant.cpu())
    if not labels: raise ValueError("no OOF anchor labels")
    raw=torch.cat(raw_gains)
    data={"reference_sha256":reference_hash,"core_oof":metadata,
          "sample_count":len(anchors),"stride":args.seq_len,
          "target":torch.cat(labels),"features":torch.cat(features),
          "invariant":torch.cat(invariants),
          "raw_delta_mean":float(raw.mean()),"beneficial_fraction":float((raw<0).float().mean()),
          "clipped_fraction":float((torch.cat(labels).abs()>=2).float().mean())}
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(data,path)
    return data
