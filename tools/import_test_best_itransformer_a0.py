"""Import a TEST-selected TSLib iTransformer A0 as an FPEM reference.

The TEST-selected protocol is intentionally labelled exploratory.  The
conversion verifies identical deterministic baseline predictions before save.
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_state(raw, horizon):
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Expected a nonempty iTransformer state_dict")
    replacements = {
        "enc_embedding.": "inverted_embedding.",
        "projection.": "head_linear.",
    }
    state = {}
    for key, value in raw.items():
        dest = key
        for old, new in replacements.items():
            if key.startswith(old):
                dest = new + key[len(old):]
                break
        if dest in state:
            raise ValueError(f"Duplicate converted key: {dest}")
        state[dest] = value
    if tuple(state["head_linear.weight"].shape)[0] != horizon:
        raise ValueError("Checkpoint prediction length does not match horizon")
    return state


def verify_forward(args, raw, state):
    from models.iTransformer import Model as TSLibiTransformer
    from models.PatchTST_PredictiveEnvIV import Model as FPEM
    from tools.run_predictive_env_iv_patchtst import model_config, parse_args

    old_argv = sys.argv
    try:
        sys.argv = ["verify_reference"]
        cfg = parse_args()
    finally:
        sys.argv = old_argv
    cfg.backbone = "itransformer"
    cfg.seq_len = 96
    cfg.pred_len = args.horizon
    cfg.d_model = args.d_model
    cfg.d_ff = args.d_ff
    cfg.n_heads = args.n_heads
    cfg.e_layers = args.e_layers
    cfg.dropout = args.dropout
    cfg.enc_in = args.enc_in
    cfg.task_name = "long_term_forecast"
    cfg.factor = 3
    cfg.activation = "gelu"
    cfg.embed = "timeF"
    cfg.freq = args.freq
    cfg.variant_fusion_mode = "off"
    baseline = TSLibiTransformer(cfg).eval()
    baseline.load_state_dict(raw, strict=True)
    fpem = FPEM(model_config(cfg, "A0")).eval()
    incompatible = fpem.load_state_dict(state, strict=False)
    expected_missing = set(fpem.state_dict()) - set(state)
    if incompatible.unexpected_keys or set(incompatible.missing_keys) != expected_missing:
        raise ValueError(f"Unexpected iTransformer conversion keys: {incompatible}")
    fpem.initialize_h_forecast_head()
    generator = torch.Generator().manual_seed(2021)
    x = torch.randn(2, 96, args.enc_in, generator=generator)
    with torch.no_grad():
        expected = baseline(x.clone(), None, None, None)
        actual = fpem.forward_components(x)["prediction"]
    error = (expected - actual).abs().max().item()
    if error > 1e-5:
        raise ValueError(f"iTransformer forward mismatch: max_abs_error={error}")
    return error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--data-class", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--freq", required=True)
    parser.add_argument("--enc-in", type=int, required=True)
    parser.add_argument("--cycle-len", type=int, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--test-mse", type=float, required=True)
    parser.add_argument("--test-mae", type=float, required=True)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--e-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2021)
    args = parser.parse_args()
    source_hash = sha256(args.checkpoint)
    if args.output.exists():
        old = torch.load(args.output, map_location="cpu", weights_only=False)
        if old.get("source_checkpoint_sha256") != source_hash:
            raise ValueError(f"Refusing to overwrite different reference: {args.output}")
        print(json.dumps({"reference": str(args.output), "status": "reuse", "source_sha256": source_hash}))
        return
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = convert_state(raw, args.horizon)
    if tuple(state["inverted_embedding.value_embedding.weight"].shape) != (args.d_model, 96):
        raise ValueError("Embedding shape does not match d_model and seq_len")
    if state["encoder.attn_layers.0.conv1.weight"].shape[0] != args.d_ff:
        raise ValueError("Encoder FFN size does not match d_ff")
    max_error = verify_forward(args, raw, state)
    signature = dict(
        backbone="itransformer", dataset_name=args.dataset,
        dataset_archive_sha256=args.archive_sha256,
        data_class=args.data_class, data_path=args.data_path, target="OT",
        freq=args.freq, seq_len=96, pred_len=args.horizon,
        d_model=args.d_model, d_ff=args.d_ff, n_heads=args.n_heads,
        e_layers=args.e_layers, dropout=args.dropout, seed=args.seed,
        warmup_epochs=0, enc_in=args.enc_in,
        cyclenet_cycle_len=args.cycle_len, protocol_version=4,
    )
    payload = dict(
        signature=signature, state_dict=state,
        source_checkpoint=str(args.checkpoint), source_checkpoint_sha256=source_hash,
        source_test_MSE=args.test_mse, source_test_MAE=args.test_mae,
        source_selection_split="test", forward_max_abs_error=max_error,
        caveat="TEST-selected A0: downstream TEST results are exploratory",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, suffix=".pt", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(
        {k: v for k, v in payload.items() if k != "state_dict"}, indent=2
    ))
    print(json.dumps({"reference": str(args.output), "status": "created",
                      "source_sha256": source_hash, "forward_max_abs_error": max_error}))


if __name__ == "__main__":
    main()
