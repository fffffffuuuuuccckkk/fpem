#!/usr/bin/env python
import os
import stat
from pathlib import Path

import paramiko


HOST = os.environ.get("SOURCE_HOST", "211.71.72.121")
USER = os.environ.get("SOURCE_USER", "OuXiaoyu")
PASSWORD = os.environ["SOURCE_PASSWORD"]
SRC = os.environ.get("SOURCE_DIR", "/data/OuXiaoyu/mystg/datasets")
DST = Path(os.environ.get("DEST_DIR", "/data/OuXiaoyu/datasets"))


def sync_dir(sftp, remote_dir, local_dir):
    local_dir.mkdir(parents=True, exist_ok=True)
    for item in sftp.listdir_attr(remote_dir):
        rpath = remote_dir.rstrip("/") + "/" + item.filename
        lpath = local_dir / item.filename
        if stat.S_ISDIR(item.st_mode):
            sync_dir(sftp, rpath, lpath)
        elif stat.S_ISREG(item.st_mode):
            if lpath.exists() and lpath.stat().st_size == item.st_size:
                continue
            tmp = lpath.with_suffix(lpath.suffix + ".part")
            print(f"GET {rpath} -> {lpath} ({item.st_size} bytes)", flush=True)
            sftp.get(rpath, str(tmp))
            tmp.replace(lpath)


client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASSWORD, look_for_keys=False, allow_agent=False, timeout=30)
try:
    with client.open_sftp() as sftp:
        sync_dir(sftp, SRC, DST)
finally:
    client.close()
print("dataset sync done", flush=True)
