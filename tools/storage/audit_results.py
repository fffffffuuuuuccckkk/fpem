#!/usr/bin/env python3
"""Full read-only inventory. Only independently verified caches become candidates."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

from .common import CHECKPOINT_SUFFIXES,METADATA_SUFFIXES,active_file,atomic_json,stat_fingerprint,within
from .inventory_npz import inventory as npz_inventory


def run_state(path):
    for parent in (path.parent,*path.parents):
        names=("point_complete","complete","run_complete","Flow_U1.complete")
        if any((parent/name).exists() for name in names):return "complete"
        if parent.name=="results":break
    return "partial_or_unknown"


def audit(root,report,verified=None,top_n=50,recent_hours=48):
    root=root.resolve(strict=True);report=report.resolve(strict=False)
    if root.is_symlink() or not root.is_dir():raise ValueError("root must be a real results directory")
    report.mkdir(parents=True,exist_ok=True)
    verification=json.loads(Path(verified).read_text()) if verified else {}
    verified_target=verification.get("cache_path") if verification.get("status")=="passed" else None
    root_dev=root.stat().st_dev;now=time.time()
    inventory_file=report/"inventory.csv"
    headers=["path","logical_bytes","allocated_bytes","mtime_utc","inode","device","links",
             "suffix","run","role","run_state","referenced","unique_source","decision","reason"]
    protected=[];blocked=[];candidates=[];large=[];npz_members=[]
    folder_bytes=defaultdict(int);suffix_bytes=Counter();files=0
    with inventory_file.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=headers);writer.writeheader()
        for current,dirs,names in os.walk(root,topdown=True,followlinks=False):
            base=Path(current)
            dirs[:]=[d for d in dirs if not (base/d).is_symlink() and
                     not (report==base/d or report in (base/d).parents)]
            for name in names:
                path=base/name;relative=path.relative_to(root)
                if report==path or report in path.parents:continue
                st=stat_fingerprint(path);files+=1
                status=run_state(path);age=(now-st["mtime_ns"]/1e9)/3600
                suffix=path.suffix.lower();role="metadata" if suffix in METADATA_SUFFIXES else "artifact"
                decision="PROTECT";reason="unknown dependency; conservative default"
                if path.is_symlink():decision="BLOCKED";reason="symbolic link; never follow or delete"
                elif st["device"]!=root_dev:decision="BLOCKED";reason="different filesystem/mount"
                elif st["links"]!=1:decision="BLOCKED";reason="hard-linked asset"
                elif suffix in CHECKPOINT_SUFFIXES and name!="frozen_a0_cache.pt":
                    reason="checkpoint or model dependency"
                elif suffix in METADATA_SUFFIXES:
                    reason="paper metric/config/log/figure preserved"
                elif "fpem_eiil3_20261110" in relative.parts and name=="features.npz":
                    decision="BLOCKED"
                    reason=("recent <48h and unfinished EIIL3 run; Router/I3/R3 may consume cache"
                            if age<recent_hours or status!="complete" else
                            "EIIL3 cache dependency and regeneration not yet validated")
                elif name=="flow_input.npz":
                    decision="BLOCKED";reason="Flow or figure may still consume input; no exact rebuild proof"
                elif age<recent_hours and st["size"]>100<<20:
                    decision="BLOCKED";reason="large recent artifact <48h"
                if verified_target and path.resolve()==Path(verified_target).resolve() and decision!="BLOCKED":
                    if status=="complete" and age>=recent_hours and st["size"]==verification.get("original_size"):
                        decision="REGENERABLE_CACHE"
                        reason="full regeneration and downstream numerical equality verified; source A0 SHA fixed"
                        candidates.append({"path":str(path),"relative":str(relative),"kind":decision,
                            "reason":reason,"risk":"low","fingerprint":st,
                            "verification":str(verified),"expected_sha256":verification.get("original_sha256")})
                if decision=="PROTECT":
                    protected.append({"path":str(relative),"reason":reason})
                if decision=="BLOCKED" and (st["size"]>=100<<20 or path.is_symlink()):
                    blocked.append({"path":str(path),"reason":reason,"bytes":st["size"]})
                if st["size"]>=10<<20:large.append({"path":str(relative),"logical_bytes":st["size"],
                                                     "allocated_bytes":st["blocks"],"decision":decision})
                if name in ("features.npz","flow_input.npz") and not path.is_symlink():
                    try:
                        for member in npz_inventory(path):npz_members.append({"path":str(relative),**member})
                    except Exception as exc:blocked.append({"path":str(path),"reason":"NPZ central directory unreadable: "+str(exc),"bytes":st["size"]})
                writer.writerow({"path":str(relative),"logical_bytes":st["size"],
                    "allocated_bytes":st["blocks"],"mtime_utc":datetime.fromtimestamp(st["mtime_ns"]/1e9,timezone.utc).isoformat(),
                    "inode":st["inode"],"device":st["device"],"links":st["links"],"suffix":suffix,
                    "run":"/".join(relative.parts[:4]),"role":role,"run_state":status,
                    "referenced":"unknown; inspect dependency graph","unique_source":"unknown",
                    "decision":decision,"reason":reason})
                suffix_bytes[suffix or "<none>"]+=st["blocks"]
                for folder in (relative.parts[0],"/".join(relative.parts[:2]),"/".join(relative.parts[:3])):
                    folder_bytes[folder]+=st["blocks"]
    for name,rows in (("top_files.csv",sorted(large,key=lambda x:-x["allocated_bytes"])[:top_n]),
                      ("folder_sizes.csv",[{"folder":k,"allocated_bytes":v} for k,v in
                                            sorted(folder_bytes.items(),key=lambda item:-item[1])])):
        with (report/name).open("w",newline="",encoding="utf-8") as stream:
            if rows:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (report/"npz_members.jsonl").open("w") as stream:
        for row in npz_members:stream.write(json.dumps(row)+"\n")
    for name,rows in (("candidates.jsonl",candidates),("blocked.jsonl",blocked)):
        with (report/name).open("w") as stream:
            for row in rows:stream.write(json.dumps(row)+"\n")
    atomic_json(report/"protected_manifest.json",{"root":str(root),"files":protected})
    source=root.parent
    references=[]
    patterns=re.compile(r"features\.npz|flow_input\.npz|frozen_a0_cache\.pt|np\.load|torch\.load|base_run_config|cache_root|stage1_inputs|shared_reference|checkpoint",re.I)
    for source_dir in ("tools","models","scripts","config"):
        location=source/source_dir
        if not location.exists():continue
        for code in location.rglob("*"):
            if code.is_symlink() or not code.is_file() or code.suffix not in (".py",".sh",".yaml",".yml",".json"):
                continue
            try:
                if code.stat().st_size>2<<20:continue
                for line_no,line in enumerate(code.read_text(errors="replace").splitlines(),1):
                    if patterns.search(line):
                        references.append({"source":str(code.relative_to(source)),"line":line_no,
                                           "text":line.strip()[:240]})
            except OSError:continue
    dependencies={"EIIL3 features.npz":{
        "producer":"tools/run_fpem_eiil3.py:get_data (frozen A0 collect/prepare/flat_record)",
        "consumers":["same script I3/R3 fit", "tools/context_shift_viz/core.py", "reconstruction or paper figure re-evaluation"],
        "status":"cannot delete while point_complete absent or <48h"},
      "EIIL3 flow_input.npz":{"producer":"tools/run_fpem_eiil3.py",
        "consumers":["tools/run_fpem_eiil3_flow.py", "Flow ablation and CI re-evaluation"]},
      "G1 frozen_a0_cache.pt":{"producer":"tools/run_fpem_g1_stability_search.py:_cache_source",
        "consumers":["same search script when rerun"],"source_dependency":"frozen A0 checkpoint + base_run_config"},
      "other result artifacts":{"status":"unknown; no automatic deletion"},
      "source_references":references}
    atomic_json(report/"artifact_dependency_graph.json",dependencies)
    preview=["# Conservative cleanup preview","",f"Root: {root}",f"Files inventoried: {files}",
             f"Verified candidates: {len(candidates)}; blocked large/symlink items: {len(blocked)}.",
             "No entire experiment directory is a candidate.","",
             "## Candidates (dry run only)"]
    for row in candidates:preview.append(f"- {row['path']} — {row['fingerprint']['size']/2**30:.3f} GiB; {row['reason']}")
    preview += ["","## Highest-risk blocked artifacts"]
    for row in sorted(blocked,key=lambda x:-x["bytes"])[:top_n]:
        preview.append(f"- {row['path']} — {row['bytes']/2**30:.2f} GiB; {row['reason']}")
    preview += ["","## By file type allocated bytes"]
    for suffix,size in suffix_bytes.most_common():preview.append(f"- {suffix}: {size/2**30:.3f} GiB")
    (report/"cleanup_preview.md").write_text("\n".join(preview)+"\n")
    print(json.dumps({"files":files,"candidates":len(candidates),"blocked":len(blocked),
                      "report":str(report)}))


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path("results"))
    p.add_argument("--report-root",type=Path,required=True);p.add_argument("--verification",type=Path)
    p.add_argument("--top-n",type=int,default=50);p.add_argument("--recent-hours",type=int,default=48)
    a=p.parse_args();audit(a.root,a.report_root,a.verification,a.top_n,a.recent_hours)


if __name__=="__main__":main()
