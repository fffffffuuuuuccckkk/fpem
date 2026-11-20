#!/usr/bin/env python3
"""Bind an existing exact A0 checkpoint to an EIIL3 case without stale paths.

Only the reference path in a copied run configuration is changed. Any former
EIIL q is invalidated; it must be regenerated from this frozen A0 on TRAIN.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--case-key", required=True)
    p.add_argument("--a0-checkpoint", type=Path, required=True)
    p.add_argument("--template-config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    case = next(row for row in (json.loads(line) for line in args.source_manifest.read_text().splitlines())
                if row["case_key"] == args.case_key)
    if not args.a0_checkpoint.is_file():
        raise FileNotFoundError(args.a0_checkpoint)
    config = json.loads(args.template_config.read_text())
    for key, expected in (("backbone", case["backbone"]),
                          ("dataset_name", case["dataset"]),
                          ("pred_len", case["pred_len"]),
                          ("seq_len", case["seq_len"]),
                          ("seed", case["seed"])):
        if config.get(key) != expected:
            raise ValueError(f"template {key} mismatch: {config.get(key)} != {expected}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config["reference_checkpoint"] = str(args.a0_checkpoint.resolve())
    config_path = args.output_dir / "base_run_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    case.update(status="MISSING_ENV_Q", missing_reason="new A0 identity; rebuild TRAIN q",
                A0_ckpt=str(args.a0_checkpoint.resolve()),
                A0_sha256=hashlib.sha256(args.a0_checkpoint.read_bytes()).hexdigest(),
                base_run_config=str(config_path), a0_source="rebound_existing_exact_checkpoint")
    for key in ("env_artifact", "env_artifact_sha256", "q_config", "q_source_stage",
                "q_shape", "q_ids", "q_mass", "q_unique_ids", "q_provenance"):
        case.pop(key, None)
    manifest = args.output_dir / "case_manifest.jsonl"
    manifest.write_text(json.dumps(case) + "\n")
    print(json.dumps({"case": args.case_key, "manifest": str(manifest),
                      "A0_sha256": case["A0_sha256"]}))


if __name__ == "__main__":
    main()
