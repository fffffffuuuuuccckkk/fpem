from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path


CHECKPOINT_SUFFIXES={".pt",".pth",".ckpt",".safetensors"}
METADATA_SUFFIXES={".csv",".json",".jsonl",".yaml",".yml",".md",".txt",".log",".pdf",".svg",".png"}


def sha256(path,chunk=8<<20):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda:stream.read(chunk),b""):
            digest.update(part)
    return digest.hexdigest()


def stat_fingerprint(path):
    s=Path(path).stat(follow_symlinks=False)
    return {"device":s.st_dev,"inode":s.st_ino,"size":s.st_size,
            "mtime_ns":s.st_mtime_ns,"links":s.st_nlink,"blocks":getattr(s,"st_blocks",0)*512}


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+".tmp")
    with temporary.open("w",encoding="utf-8") as stream:
        json.dump(value,stream,indent=2,default=str)
        stream.write("\n");stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def active_file(path):
    """Conservative /proc fd check; unreadable processes mean unknown."""
    target=Path(path).resolve(strict=True)
    proc=Path("/proc")
    if not proc.exists():return None
    unknown=False
    for directory in proc.iterdir():
        if not directory.name.isdigit():continue
        fd=directory/"fd"
        try:descriptors=list(fd.iterdir())
        except (PermissionError,FileNotFoundError):
            unknown=True;continue
        for item in descriptors:
            try:
                if item.resolve()==target:return int(directory.name)
            except (PermissionError,FileNotFoundError,OSError):continue
    if unknown:
        try:
            probe=subprocess.run(["fuser","-s",str(target)],capture_output=True,timeout=10)
            if probe.returncode==0:return "open_by_fuser"
            if probe.returncode==1:return None
        except (FileNotFoundError,subprocess.TimeoutExpired):pass
        return "unknown"
    return None


def within(path,root):
    path=Path(path).resolve(strict=False);root=Path(root).resolve(strict=True)
    return path!=root and root in path.parents
