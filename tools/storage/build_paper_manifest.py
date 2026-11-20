#!/usr/bin/env python3
"""Conservative inventory of paper-critical result families, without modifying them."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime,timezone
from pathlib import Path

from .common import atomic_json,sha256

FAMILIES={
 "main_EIIL3":"fpem_eiil3_20261110",
 "COSA_comparison":"fpem_eiil3_cosa_20261110",
 "FOIL_ShifTS_A0":"foil_shifts_a0_protocol_20261011",
 "classic_baselines":"fpem_classic_baselines*",
 "A0_search":"fpem_final_patchtst_search",
 "paper_evidence":"fpem_paper_final_evidence_20261101",
 "paper_completion":"fpem_paper_completion_20261101",
 "probabilistic":"fpem_probabilistic",
 "environment_reference":"predictive_env_reference_all_datasets_k3",
 "introduction_figures":"fpem_intro_motivation",
 "history_future_similarity":"raw_history_future_similarity_20261106",
 "ABC_ablation":"fpem_abc_mechanism_search_20261031",
 "legacy_supervised_joint_private":"fpem_supervised_joint_private_env_20261019",
 "orthogonal_spectral":"fpem_orthogonal_spectral_full_testlr_20261020",
 "remote_mirror_pending_proof":"server2_imports",
}


def build(root,output):
    root=root.resolve(strict=True)
    families=[]
    for role,name in FAMILIES.items():
      for path in (sorted(root.glob(name)) or [root/name]):
        if not path.exists():
            families.append({"role":role,"path":str(path),"status":"missing; do not infer deletion"})
            continue
        count=bytes_used=0;models=[];completed=[];metrics=[]
        for current,dirs,names in os.walk(path,followlinks=False):
            dirs[:]=[d for d in dirs if not (Path(current)/d).is_symlink()]
            for filename in names:
                item=Path(current)/filename
                if item.is_symlink():continue
                try:stat=item.stat()
                except OSError:continue
                count+=1;bytes_used+=stat.st_blocks*512
                rel=str(item.relative_to(root))
                if filename in ("complete","point_complete","Flow_U1.complete"):
                    completed.append(rel)
                elif filename in ("point_metrics.json","metrics.json","result.json"):
                    metrics.append(rel)
                elif item.suffix in (".pt",".pth",".ckpt",".safetensors"):
                    model={"path":rel,"bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns}
                    if stat.st_size<=16<<20:model["sha256"]=sha256(item)
                    else:model["sha256_status"]="not computed on full disk; existing run manifest remains authoritative"
                    models.append(model)
        families.append({"role":role,"path":str(path),"status":"PROTECT",
                         "file_count":count,"allocated_bytes":bytes_used,
                         "completion_markers":completed,"metric_files":metrics,
                         "model_files":models})
    atomic_json(output,{"created_utc":datetime.now(timezone.utc).isoformat(),
                        "root":str(root),"policy":"No file in this manifest is auto-deletable",
                        "families":families})
    print(json.dumps({"families":len(families),"output":str(output)}))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path("results"))
    p.add_argument("--output",type=Path,required=True);a=p.parse_args();build(a.root,a.output)
