#!/usr/bin/env python3
"""Rebind copied EIIL3 A0 configs after deleting split-run case outputs."""
import argparse
import json
import os
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    old = repo / "results" / "fpem_eiil3_safe_full_20261112"
    new = repo / "results" / "fpem_eiil3_fulltrain_20261113"
    manifest = new / "asset_audit" / "manifest.jsonl"
    if not manifest.is_file() or old.exists():
        raise ValueError("unexpected old/new result directory state")
    source = str(old / "asset_audit" / "asset_bindings")
    destination = str(new / "asset_audit" / "asset_bindings")
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    changed = 0
    for row in rows:
        config = row.get("base_run_config", "")
        if config.startswith(source + os.sep):
            replacement = destination + config[len(source):]
            if not Path(replacement).is_file():
                raise FileNotFoundError(replacement)
            row["base_run_config"] = replacement
            changed += 1
        if any(source in str(value) for value in row.values()):
            raise ValueError(f"unrebound old source in {row['case_key']}")
    backup = manifest.with_suffix(".pre_rebind.jsonl")
    if not backup.exists():
        shutil.copy2(manifest, backup)
    temporary = manifest.with_suffix(".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, manifest)
    print(json.dumps({"rows": len(rows), "rebound": changed,
                      "backup": str(backup), "manifest": str(manifest)}))


if __name__ == "__main__":
    main()
