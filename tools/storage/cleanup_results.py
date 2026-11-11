#!/usr/bin/env python3
"""Apply only audited, verified, unchanged per-file candidates."""
from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
from datetime import datetime,timezone
from pathlib import Path

from .common import active_file,sha256,stat_fingerprint,within


def cleanup(manifest,root,apply=False,recent_hours=48):
    manifest=manifest.resolve(strict=True);root=root.resolve(strict=True)
    journal=manifest.parent/"deletion_journal.jsonl"
    entries=[json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    report=[]
    for item in entries:
        target=Path(item["path"])
        reason=None
        if item.get("kind")!="REGENERABLE_CACHE":reason="kind not auto-approved"
        elif not within(target,root):reason="outside root"
        elif not target.is_file() or target.is_symlink():reason="missing or symlink"
        elif stat_fingerprint(target)!=item["fingerprint"]:reason="stat changed since audit"
        elif target.stat().st_nlink!=1:reason="hard link"
        elif datetime.now(timezone.utc).timestamp()-target.stat().st_mtime<recent_hours*3600:
            reason="recent file"
        elif active_file(target) is not None:reason="open/unknown process handle"
        elif not Path(item["verification"]).is_file():reason="verification absent"
        else:
            proof=json.loads(Path(item["verification"]).read_text())
            if proof.get("status")!="passed" or proof.get("cache_path")!=str(target):
                reason="verification not for exact target"
            elif proof.get("fingerprint")!=item["fingerprint"]:reason="verification stat changed"
            elif sha256(target)!=item.get("expected_sha256"):reason="SHA changed"
            elif not (target.parent/"complete").exists():reason="run no longer complete"
            elif not (target.parent/"run_config.json").exists():reason="run config absent"
            elif not (target.parent/"result.json").exists():reason="result metrics absent"
        before=shutil.disk_usage(root).free
        row={"path":str(target),"bytes":item["fingerprint"]["size"],"reason":item.get("reason"),
             "verification":item.get("verification"),"timestamp":datetime.now(timezone.utc).isoformat(),
             "operator":getpass.getuser(),"free_bytes_before":before,
             "status":"BLOCKED" if reason else "DRY_RUN"}
        if reason:row["block_reason"]=reason
        elif apply:
            target.unlink()
            row["status"]="DELETED";row["free_bytes_after"]=shutil.disk_usage(root).free
            with journal.open("a") as stream:
                stream.write(json.dumps(row)+"\n");stream.flush();os.fsync(stream.fileno())
        report.append(row)
        if reason and apply:break
    print(json.dumps({"apply":apply,"results":report}))
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--root",type=Path,default=Path("results"));p.add_argument("--dry-run",action="store_true")
    p.add_argument("--apply",action="store_true");p.add_argument("--recent-hours",type=int,default=48)
    a=p.parse_args();cleanup(a.manifest,a.root,a.apply,a.recent_hours)


if __name__=="__main__":main()
