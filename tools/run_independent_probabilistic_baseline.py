#!/usr/bin/env python3
"""TRAIN-only project adapters of TimeGrad, TSFlow and TSDiff.

The independent baselines use exactly the FPEM CSV split, normalization,
history/horizon lengths and ProbTS-style CRPS evaluation.  They do not consume
FPEM features or future targets at inference.  See the run_config metadata
before interpreting them as upstream-paper benchmark numbers.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.predictive_env.independent_probabilistic_baselines import METHODS
from tools.evaluate_fpem_probabilistic import probabilistic_metrics
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom


DATA = {
    'ETTh1': (Dataset_ETT_hour, 'ETT-small', 'ETTh1.csv', 'h', 7),
    'ETTh2': (Dataset_ETT_hour, 'ETT-small', 'ETTh2.csv', 'h', 7),
    'ETTm1': (Dataset_ETT_minute, 'ETT-small', 'ETTm1.csv', '15min', 7),
    'ETTm2': (Dataset_ETT_minute, 'ETT-small', 'ETTm2.csv', '15min', 7),
    'ExchangeRate': (Dataset_Custom, 'exchange_rate', 'exchange_rate.csv', 'd', 8),
}


def build_loaders(args):
    klass, folder, file_name, frequency, channels = DATA[args.dataset]
    common = dict(args=SimpleNamespace(augmentation_ratio=0),
                  root_path=str(args.data_root / folder),
                  size=[args.seq_len, min(48, args.seq_len // 2), args.pred_len],
                  features='M', data_path=file_name, target='OT', scale=True,
                  timeenc=1, freq=frequency)
    datasets = {split: klass(flag=split, **common)
                for split in ('train', 'val', 'test')}
    observed = int(datasets['train'].data_x.shape[-1])
    if observed != channels:
        raise RuntimeError(f'unexpected channel count {observed} != {channels}')
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {split: DataLoader(dataset, batch_size=args.batch_size,
                                 shuffle=(split == 'train'),
                                 generator=generator if split == 'train' else None,
                                 num_workers=0, drop_last=False)
               for split, dataset in datasets.items()}
    return loaders, channels


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--dataset', choices=('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'ExchangeRate'), required=True)
    parser.add_argument('--pred_len', type=int, choices=(24, 48), required=True)
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--data_root', type=Path, default=ROOT / 'dataset/all_datasets')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--epochs', type=int, default=13)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--steps', type=int, default=0, help='0 selects method-specific default')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--sample_chunk', type=int, default=20)
    parser.add_argument('--max_train_batches', type=int, default=0)
    parser.add_argument('--max_test_windows', type=int, default=0)
    parser.add_argument('--resume_eval', action='store_true')
    return parser.parse_args()


def evaluate(model, loader, args, device):
    model.eval()
    totals, windows = {}, 0
    for batch_idx, (x, y, _, _) in enumerate(loader):
        if args.max_test_windows and windows >= args.max_test_windows:
            break
        x = x.float().to(device)
        y = y[:, -args.pred_len:].float().to(device)
        if args.max_test_windows:
            count = min(x.shape[0], args.max_test_windows - windows)
            x, y = x[:count], y[:count]
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + 25007 + batch_idx * 1009)
        samples = model.sample(x, args.num_samples, args.sample_chunk, generator)
        if samples.shape != (len(x), args.num_samples, args.pred_len, x.shape[-1]):
            raise RuntimeError('incorrect probabilistic sample shape')
        if not torch.isfinite(samples).all():
            raise FloatingPointError('nonfinite generated samples')
        naive = x[:, -1:, :].expand_as(y)
        metrics = probabilistic_metrics(samples, y, naive)
        aggregate = probabilistic_metrics(samples.sum(-1, keepdim=True),
                                           y.sum(-1, keepdim=True),
                                           naive.sum(-1, keepdim=True))
        scalar = {key: float(value) for key, value in metrics.items()
                  if torch.as_tensor(value).ndim == 0}
        scalar['ProbTS_CRPS_Sum'] = float(aggregate['ProbTS_CRPS'])
        scalar['CRPS_Sum'] = float(aggregate['CRPS'])
        for key, value in scalar.items():
            totals[key] = totals.get(key, 0.0) + len(x) * value
        if batch_idx == 0:
            np.savez_compressed(args.output / 'test_trajectory_examples.npz',
                                target=y[:2].cpu().numpy(),
                                y_samples=samples[:2, :min(20, args.num_samples)].cpu().numpy(),
                                y_mean=samples[:2].mean(1).cpu().numpy())
        windows += len(x)
        if batch_idx % 20 == 0:
            print('EVALUATE', args.method, args.dataset, args.pred_len, windows, flush=True)
    if not windows:
        raise RuntimeError('empty TEST evaluation')
    result = {key: value / windows for key, value in totals.items()}
    result.update({'method': args.method, 'dataset': args.dataset,
                   'pred_len': args.pred_len, 'num_samples': args.num_samples,
                   'evaluated_windows': windows, 'total_test_windows': len(loader.dataset),
                   'implementation': 'independent project PyTorch adapter, not bit-for-bit upstream',
                   'split': 'test', 'selection': 'fixed final epoch; no test selection'})
    (args.output / 'test_metrics.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    args = parse_args()
    if min(args.epochs, args.batch_size, args.num_samples, args.sample_chunk) <= 0:
        raise ValueError('epochs/batch/samples must be positive')
    if args.method not in METHODS or args.dataset not in DATA:
        raise ValueError('unknown method or dataset')
    if args.steps == 0:
        args.steps = {'timegrad': 20, 'tsflow': 16, 'tsdiff': 50}[args.method]
    if args.output.exists() and not args.resume_eval:
        raise FileExistsError(f'result directory exists: {args.output}')
    args.output.mkdir(parents=True, exist_ok=args.resume_eval)
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    loaders, channels = build_loaders(args)
    model = METHODS[args.method](channels, args.seq_len, args.pred_len,
                                  hidden=args.hidden, steps=args.steps).to(device)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update({'channels': channels, 'train_windows': len(loaders['train'].dataset),
                   'test_windows': len(loaders['test'].dataset),
                   'upstream_archive': {'timegrad': '/data/OuXiaoyu/TimeGrad-main.zip',
                                        'tsflow': '/data/OuXiaoyu/TSFlow-main.zip',
                                        'tsdiff': '/data/OuXiaoyu/unconditional-time-series-diffusion-main.zip'}[args.method],
                   'protocol': 'FPEM CSV TRAIN/VAL/TEST split; project adapter core objective; final-epoch evaluation'})
    if not args.resume_eval:
        (args.output / 'run_config.json').write_text(json.dumps(config, indent=2))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        history = []
        for epoch in range(args.epochs):
            model.train()
            total, count = 0.0, 0
            for step, (x, y, _, _) in enumerate(loaders['train']):
                if args.max_train_batches and step >= args.max_train_batches:
                    break
                x = x.float().to(device)
                y = y[:, -args.pred_len:].float().to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = model.training_loss(x, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite train loss')
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
                count += 1
            record = {'epoch': epoch + 1, 'train_loss': total / max(count, 1),
                      'train_batches': count}
            history.append(record)
            (args.output / 'history.json').write_text(json.dumps(history, indent=2))
            print('EPOCH', record, flush=True)
        torch.save({'state_dict': model.state_dict(), 'run_config': config},
                   args.output / 'final_epoch.pt')
    else:
        checkpoint = torch.load(args.output / 'final_epoch.pt', map_location=device,
                                weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
    result = evaluate(model, loaders['test'], args, device)
    (args.output / 'complete').write_text('complete\n')
    print('COMPLETE', result, flush=True)


if __name__ == '__main__':
    main()
