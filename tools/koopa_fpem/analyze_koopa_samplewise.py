import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


def add_common_args(p):
    p.add_argument("--task_name", default="long_term_forecast")
    p.add_argument("--is_training", type=int, default=0)
    p.add_argument("--model_id", default="ETTh1_96_96_koopa")
    p.add_argument("--model", default="Koopa")
    p.add_argument("--data", default="ETTh1")
    p.add_argument("--root_path", default="./dataset/ETT-small/")
    p.add_argument("--data_path", default="ETTh1.csv")
    p.add_argument("--features", default="M")
    p.add_argument("--target", default="OT")
    p.add_argument("--freq", default="h")
    p.add_argument("--checkpoints", default="./checkpoints/")
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--label_len", type=int, default=48)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--seasonal_patterns", default="Monthly")
    p.add_argument("--inverse", action="store_true", default=False)
    p.add_argument("--enc_in", type=int, default=7)
    p.add_argument("--dec_in", type=int, default=7)
    p.add_argument("--c_out", type=int, default=7)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--e_layers", type=int, default=2)
    p.add_argument("--d_layers", type=int, default=1)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--d_conv", type=int, default=4)
    p.add_argument("--factor", type=int, default=3)
    p.add_argument("--embed", default="timeF")
    p.add_argument("--distil", action="store_true", default=True)
    p.add_argument("--des", default="koopa_fpem")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--itr", type=int, default=1)
    p.add_argument("--train_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=0.001)
    p.add_argument("--loss", default="MSE")
    p.add_argument("--lradj", default="type1")
    p.add_argument("--use_amp", action="store_true", default=False)
    p.add_argument("--use_gpu", action="store_true", default=True)
    p.add_argument("--no_use_gpu", action="store_false", dest="use_gpu")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--gpu_type", default="cuda")
    p.add_argument("--use_multi_gpu", action="store_true", default=False)
    p.add_argument("--devices", default="0")
    p.add_argument("--augmentation_ratio", type=int, default=0)
    p.add_argument("--use_dtw", action="store_true", default=False)
    return p


def inv_only_forecast(model, x_enc):
    m = model.module if hasattr(model, "module") else model
    mean_enc = x_enc.mean(1, keepdim=True).detach()
    x = x_enc - mean_enc
    std_enc = (x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).sqrt().detach()
    x = x / std_enc
    residual, forecast = x, None
    for i in range(m.num_blocks):
        time_var_input, time_inv_input = m.disentanglement(residual)
        time_inv_output = m.time_inv_kps[i](time_inv_input)
        time_var_backcast, _ = m.time_var_kps[i](time_var_input)
        residual = residual - time_var_backcast
        forecast = time_inv_output if forecast is None else forecast + time_inv_output
    return forecast * std_enc + mean_enc


def build_exp(args, model_name, ckpt):
    args.model = model_name
    exp = Exp_Long_Term_Forecast(args)
    exp.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    exp.model.to(exp.device).eval()
    return exp


def main():
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--compare_model", default="")
    p.add_argument("--compare_ckpt", default="")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    exp = build_exp(args, args.model, args.ckpt)
    test_data, test_loader = exp._get_data("test")

    cmp_exp = None
    if args.compare_model and args.compare_ckpt:
        import copy
        cmp_args = copy.copy(args)
        cmp_exp = build_exp(cmp_args, args.compare_model, args.compare_ckpt)

    rows = []
    full_losses, inv_losses, cmp_losses = [], [], []
    with torch.no_grad():
        for bi, (bx, by, bxm, bym) in enumerate(test_loader):
            bx = bx.float().to(exp.device)
            by = by.float().to(exp.device)
            bxm = bxm.float().to(exp.device)
            bym = bym.float().to(exp.device)
            dec = torch.zeros_like(by[:, -args.pred_len:, :])
            dec = torch.cat([by[:, :args.label_len, :], dec], dim=1).float().to(exp.device)
            true = by[:, -args.pred_len:, :]
            full = exp.model(bx, bxm, dec, bym)[:, -args.pred_len:, :]
            inv = inv_only_forecast(exp.model, bx)[:, -args.pred_len:, :]
            lf = (full - true).abs().mean((1, 2)).cpu().numpy()
            li = (inv - true).abs().mean((1, 2)).cpu().numpy()
            full_losses.append(lf); inv_losses.append(li)
            lc = None
            if cmp_exp is not None:
                cmp = cmp_exp.model(bx.to(cmp_exp.device), bxm.to(cmp_exp.device), dec.to(cmp_exp.device), bym.to(cmp_exp.device))[:, -args.pred_len:, :]
                lc = (cmp - true.to(cmp_exp.device)).abs().mean((1, 2)).cpu().numpy()
                cmp_losses.append(lc)
            for j in range(len(lf)):
                row = [bi * args.batch_size + j, float(lf[j]), float(li[j]), "full" if lf[j] < li[j] else "inv_only"]
                if lc is not None:
                    row += [float(lc[j]), "compare" if lc[j] < lf[j] else "full"]
                rows.append(row)

    full_losses = np.concatenate(full_losses)
    inv_losses = np.concatenate(inv_losses)
    summary = {
        "n": int(len(full_losses)),
        "full_mae": float(full_losses.mean()),
        "inv_only_mae": float(inv_losses.mean()),
        "full_better_ratio": float((full_losses < inv_losses).mean()),
        "inv_only_better_ratio": float((inv_losses <= full_losses).mean()),
    }
    header = "sample,mae_full,mae_inv_only,better_full_or_inv"
    labels = ["full(use variant)", "inv-only(no variant)"]
    ratios = [summary["full_better_ratio"], summary["inv_only_better_ratio"]]
    if cmp_losses:
        cmp_losses = np.concatenate(cmp_losses)
        summary.update({
            "compare_model": args.compare_model,
            "compare_mae": float(cmp_losses.mean()),
            "compare_better_than_full_ratio": float((cmp_losses < full_losses).mean()),
            "full_better_than_compare_ratio": float((full_losses <= cmp_losses).mean()),
        })
        header += ",mae_compare,better_full_or_compare"
        labels += [args.compare_model]
        ratios += [summary["compare_better_than_full_ratio"]]

    np.savetxt(os.path.join(args.out_dir, "samplewise_compare.csv"), np.asarray(rows, dtype=object), fmt="%s", delimiter=",", header=header, comments="")
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    plt.figure(figsize=(6, 4))
    plt.bar(labels, ratios)
    plt.ylim(0, 1)
    plt.ylabel("sample ratio")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "samplewise_ratio.png"), dpi=160)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
