#!/usr/bin/env python3
"""Rebind existing, identity-checked A0 references after result relocation.

Never changes an existing checkpoint or upstream result. The generated config
only changes reference_checkpoint to the located copy of the same checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(cfg):
    try:
        return (str(cfg["backbone"]).lower(), str(cfg["dataset_name"]),
                int(cfg["pred_len"]), int(cfg.get("seq_len", 96)),
                int(cfg.get("seed", 2021)))
    except (KeyError, TypeError, ValueError):
        return None


def configs_for(checkpoint):
    parent = checkpoint.parent
    paths = [parent / "A0/run_config.json", parent / "reference_prepare/run_config.json",
             parent / "run_config.json", parent / "base_run_config.json"]
    paths += sorted(parent.glob("*/run_config.json"))
    for path in paths:
        if path.is_file():
            try:
                yield path, json.loads(path.read_text())
            except (OSError, ValueError):
                continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = [json.loads(line) for line in args.source_manifest.read_text().splitlines()]
    wanted = {(r["backbone"], r["dataset"], int(r["pred_len"]),
               int(r["seq_len"]), int(r["seed"])) for r in source}
    found = {key: [] for key in wanted}
    for checkpoint in (repo / "results").rglob("shared_reference.pt"):
        if output in checkpoint.parents or ".incomplete." in str(checkpoint):
            continue
        for config_path, cfg in configs_for(checkpoint):
            key = identity(cfg)
            if key not in wanted:
                continue
            found[key].append((checkpoint, config_path, cfg))
            break
    hashes = {}
    rows = []
    for source_row in source:
        row = dict(source_row)
        key = (row["backbone"], row["dataset"], int(row["pred_len"]),
               int(row["seq_len"]), int(row["seed"]))
        options = found[key]
        if not options:
            row.update(status="MISSING_A0", missing_reason="no local A0 checkpoint/config")
            rows.append(row)
            continue
        for checkpoint, _, _ in options:
            if checkpoint not in hashes:
                hashes[checkpoint] = sha(checkpoint)
        old = row.get("A0_sha256")
        options.sort(key=lambda item: (hashes[item[0]] != old,
                                        "server2_imports" in str(item[0]),
                                        len(str(item[0]))))
        checkpoint, config_path, cfg = options[0]
        chosen_sha = hashes[checkpoint]
        cfg = dict(cfg)
        cfg["reference_checkpoint"] = str(checkpoint.resolve())
        binding = output / "asset_bindings" / row["case_key"] / "base_run_config.json"
        binding.parent.mkdir(parents=True, exist_ok=True)
        binding.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
        row.update(A0_ckpt=str(checkpoint.resolve()), A0_sha256=chosen_sha,
                   base_run_config=str(binding), a0_source="rebound_existing_checkpoint",
                   a0_original_config=str(config_path.resolve()))
        q_path = Path(row.get("env_artifact", ""))
        q_valid = (row.get("status") == "READY" and old == chosen_sha
                   and q_path.is_file() and sha(q_path) == row.get("env_artifact_sha256"))
        if not q_valid:
            for field in ("env_artifact", "env_artifact_sha256", "q_config", "q_ids",
                          "q_mass", "q_shape", "q_unique_ids", "q_source_stage"):
                row.pop(field, None)
            row.update(status="MISSING_ENV_Q", missing_reason="generate EIIL3 TRAIN q for rebound A0")
        rows.append(row)
    manifest = output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    summary = {"total": len(rows), "available_A0": sum(r["status"] != "MISSING_A0" for r in rows),
               "ready_q": sum(r["status"] == "READY" for r in rows),
               "missing_A0": [r["case_key"] for r in rows if r["status"] == "MISSING_A0"]}
    (output / "asset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "missing_A0"}), flush=True)


if __name__ == "__main__":
    main()
