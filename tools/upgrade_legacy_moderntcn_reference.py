"""Wrap an older ModernTCN A0 state in the current exact reference signature.

No model weights are changed.  All overlapping signature fields must match;
newer metadata fields are filled from explicitly supplied dataset settings.
"""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--data-class", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--freq", required=True)
    parser.add_argument("--enc-in", type=int, required=True)
    parser.add_argument("--cycle-len", type=int, required=True)
    parser.add_argument("--patch-size", type=int, required=True)
    parser.add_argument("--patch-stride", type=int, required=True)
    parser.add_argument("--num-stages", type=int, required=True)
    parser.add_argument("--ffn-ratio", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--head-dropout", type=float, required=True)
    args = parser.parse_args()
    source_hash = sha256(args.legacy)
    if args.output.exists():
        old_output = torch.load(args.output, map_location="cpu", weights_only=False)
        if old_output.get("legacy_source_sha256") != source_hash:
            raise ValueError(f"Refusing to overwrite different upgraded reference: {args.output}")
        print(json.dumps({"status": "reuse", "output": str(args.output)}))
        return
    old = torch.load(args.legacy, map_location="cpu", weights_only=False)
    if not isinstance(old.get("state_dict"), dict) or not old["state_dict"]:
        raise ValueError("Missing ModernTCN state_dict")
    signature = dict(
        backbone="moderntcn", dataset_name=args.dataset,
        dataset_archive_sha256=args.archive_sha256,
        data_class=args.data_class, data_path=args.data_path, target="OT",
        freq=args.freq, seq_len=96, pred_len=args.horizon,
        d_model=64, d_ff=64 * args.ffn_ratio, n_heads=2, e_layers=1,
        dropout=args.dropout, seed=2021, warmup_epochs=3,
        enc_in=args.enc_in, cyclenet_cycle_len=args.cycle_len,
        protocol_version=4, modern_tcn_patch_size=args.patch_size,
        modern_tcn_patch_stride=args.patch_stride,
        modern_tcn_num_stages=args.num_stages,
        modern_tcn_ffn_ratio=args.ffn_ratio,
        modern_tcn_large_size=51, modern_tcn_small_size=5,
        modern_tcn_downsample_ratio=2,
        modern_tcn_head_dropout=args.head_dropout,
    )
    for key, value in old["signature"].items():
        if key in signature and signature[key] != value:
            raise ValueError(f"Legacy reference mismatch for {key}: {value} != {signature[key]}")
    payload = dict(signature=signature, state_dict=old["state_dict"],
                   legacy_source=str(args.legacy), legacy_source_sha256=source_hash,
                   caveat="Legacy metadata upgraded without modifying weights")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, suffix=".pt", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"status": "created", "output": str(args.output),
                      "legacy_sha256": source_hash}))


if __name__ == "__main__":
    main()
