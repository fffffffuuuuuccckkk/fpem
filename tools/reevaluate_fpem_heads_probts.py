#!/usr/bin/env python
"""Re-score saved shared-FPEM heads using ProbTS CRPS without retraining.

Results are written only to a new directory; the source checkpoints and
historical metrics are never modified. The full TEST split and 100 paths are
used by default, matching the aligned CSDI evaluation protocol.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.PatchTST_PredictiveEnvIV import Model
from models.predictive_env.probability_heads import HEAD_NAMES, ProbabilityHeadBank
from tools.run_fpem_head_comparison import evaluate_heads, write_summary
from tools.run_predictive_env_iv_patchtst import (
    IndexedDataset, build_data, model_config,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True,
                        help="Completed shared-head experiment directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--max_batches", type=int, default=0,
                        help="Diagnostic cap only; 0 evaluates the full split")
    parser.add_argument("--heads", nargs="+", choices=HEAD_NAMES,
                        default=list(HEAD_NAMES))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args_cli = parser.parse_args()
    if args_cli.output.exists():
        raise FileExistsError(args_cli.output)
    if not (args_cli.source / "shared_encoder.pt").is_file():
        raise FileNotFoundError("shared FPEM checkpoint is missing")
    checkpoint = torch.load(args_cli.source / "shared_encoder.pt",
                            map_location="cpu", weights_only=False)
    args = SimpleNamespace(**checkpoint["run_config"])
    args.prob_num_samples = args_cli.num_samples
    args.max_prob_eval_batches = args_cli.max_batches
    args.prob_shuffled_eval_batches = 0
    device = torch.device(f"cuda:{args_cli.gpu}" if torch.cuda.is_available()
                          else "cpu")
    datasets, _, _, test_loader = build_data(args)
    loader = test_loader if args_cli.split == "test" else DataLoader(
        IndexedDataset(datasets["val"], cycle_len=args.cyclenet_cycle_len),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    model = Model(model_config(args, "A2")).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.probability_detach_zvar_input = True
    bank = ProbabilityHeadBank(
        args.d_model, args.pred_len, args.future_patch_len,
        args.prob_condition_mode, args_cli.heads,
    ).to(device)
    bank.condition_encoder.load_state_dict(
        checkpoint["condition_encoder_state"], strict=True,
    )
    for name in args_cli.heads:
        head_payload = torch.load(
            args_cli.source / name / "head_checkpoint.pt",
            map_location="cpu", weights_only=False,
        )
        bank.heads[name].load_state_dict(head_payload["state_dict"], strict=True)
    args_cli.output.mkdir(parents=True)
    (args_cli.output / "rescore_config.json").write_text(json.dumps({
        "source": str(args_cli.source), "heads": args_cli.heads,
        "split": args_cli.split, "num_samples": args_cli.num_samples,
        "max_batches": args_cli.max_batches,
        "original_reference_sha256": checkpoint.get("reference_sha256"),
        "metric": "ProbTS Evaluator quantiles_num=20, per-sequence normalization",
    }, indent=2))
    results = evaluate_heads(model, bank, loader, args, device,
                             args_cli.output, args_cli.split)
    write_summary(args_cli.output, results)
    print("RESCORE_COMPLETE", args_cli.output,
          {name: result["ProbTS_CRPS"] for name, result in results.items()},
          flush=True)


if __name__ == "__main__":
    main()
