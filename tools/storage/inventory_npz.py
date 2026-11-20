#!/usr/bin/env python3
"""Inspect ZIP central directory only; never decompress array payloads."""
import argparse
import json
import zipfile
from pathlib import Path


def inventory(path):
    with zipfile.ZipFile(path,"r",allowZip64=True) as archive:
        return [{"member":i.filename,"compressed_bytes":i.compress_size,
                 "uncompressed_bytes":i.file_size,"compression":i.compress_type}
                for i in archive.infolist()]


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("file",type=Path)
    a=p.parse_args();print(json.dumps(inventory(a.file),indent=2))
