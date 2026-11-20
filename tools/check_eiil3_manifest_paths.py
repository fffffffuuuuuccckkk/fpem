#!/usr/bin/env python3
"""Read-only current filesystem audit for EIIL3 A0/config/q asset paths."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args()
    for line in args.manifest.read_text().splitlines():
        row = json.loads(line)
        print(json.dumps({"case": row["case_key"], "A0": bool(row.get("A0_ckpt") and
            Path(row["A0_ckpt"]).is_file()), "config": bool(row.get("base_run_config") and
            Path(row["base_run_config"]).is_file()), "q": bool(row.get("env_artifact") and
            Path(row["env_artifact"]).is_file())}))


if __name__ == "__main__":
    main()
