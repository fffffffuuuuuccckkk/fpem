#!/usr/bin/env python3
"""Raw-TRAIN-only history/actual-future motivation statistics; no model imports.

Each non-overlapping event contains a complete 96-step history and horizon.
Channels from the same event are correlated, so bootstrap units are *time
blocks of events*, not channel rows or pairs. Future values are used only as
offline outcomes; all historical groups and pair eligibility ignore future Y.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


DATASETS = {
    "ETTm2": ("dataset/ETT-small/ETTm2.csv", "15 min"),
    "Electricity": ("dataset/electricity/electricity.csv", "1 h"),
    "Weather": ("dataset/weather/weather.csv", "10 min"),
}
HORIZONS = (96, 192, 336, 720)
INDICATORS = ("trend_change", "recent_slope", "mean_shift", "vol_shift", "period_amp_shift")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def slope(a: np.ndarray) -> np.ndarray:
    t = np.arange(a.shape[-1], dtype=np.float64)
    centered = t - t.mean()
    return ((a - a.mean(axis=-1, keepdims=True)) @ centered / np.dot(centered, centered)).astype(np.float32)


def periodic_amplitude(a: np.ndarray, period: int) -> np.ndarray:
    if a.shape[-1] < period:
        return np.full(len(a), np.nan, np.float32)
    phase = np.exp(-2j * np.pi * np.arange(a.shape[-1]) / period)
    return (2 * np.abs(a @ phase) / a.shape[-1]).astype(np.float32)


def source_info(repo: Path, dataset: str):
    relative, cadence = DATASETS[dataset]
    path = repo / relative
    raw = pd.read_csv(path)
    dates = pd.to_datetime(raw.pop("date"), errors="raise")
    values = raw.to_numpy(dtype=np.float32)
    end = 12 * 30 * 24 * 4 if dataset == "ETTm2" else int(len(values) * .7)
    if end > len(values):
        raise ValueError(f"TRAIN boundary exceeds raw rows: {dataset}")
    return path, dates.iloc[:end].to_numpy(), values[:end], end, cadence, list(raw.columns)


def make_cache(repo, output, dataset, horizon, seq_len, channel_arg, seed, offset_fraction=0.):
    path, dates, values, end, cadence, names = source_info(repo, dataset)
    span = seq_len + horizon
    offset=int(span*offset_fraction)
    starts = np.arange(offset, end - span + 1, span, dtype=np.int32)
    if not len(starts):
        raise ValueError(f"no complete TRAIN windows for {dataset}-{horizon}")
    if channel_arg == "all":
        channels = list(range(values.shape[1]))
    else:
        channels = [int(c) for c in channel_arg.split(",")]
        if any(c < 0 or c >= values.shape[1] for c in channels):
            raise ValueError("channel out of range")
    key = {"dataset": dataset, "horizon": horizon, "seq_len": seq_len,
           "channels": channels, "seed": seed, "source_sha256": sha256(path),
           "train_end_exclusive": end, "window_stride": span,"window_offset":offset}
    cache_dir = output / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{dataset}_{horizon}_L{seq_len}_{hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()[:12]}.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as saved:
            return {k: saved[k] for k in saved.files}, key, cache_path, names, cadence
    x_indices = starts[:, None] + np.arange(seq_len)
    y_indices = starts[:, None] + seq_len + np.arange(horizon)
    x_parts, y_parts, c_parts, s_parts, scale_parts, metric_parts = [], [], [], [], [], []
    floor_std = np.nanstd(values.astype(np.float64), axis=0)
    dropped = 0
    period = {"ETTm2": 96, "Electricity": 24, "Weather": 144}[dataset]
    for channel in channels:
        x = values[x_indices, channel].astype(np.float32)
        y = values[y_indices, channel].astype(np.float32)
        valid = np.isfinite(x).all(1) & np.isfinite(y).all(1)
        dropped += int((~valid).sum())
        if not valid.any():
            continue
        x, y = x[valid], y[valid]
        event_starts = starts[valid]
        median = np.median(x, axis=1)
        mad = 1.4826 * np.median(np.abs(x - median[:, None]), axis=1)
        scale = np.maximum(mad, max(.05 * float(floor_std[channel]), 1e-5)).astype(np.float32)
        mean = x.mean(1).astype(np.float32)
        xn = (x - mean[:, None]) / scale[:, None]
        yn = (y - mean[:, None]) / scale[:, None]
        half = seq_len // 2
        early, recent = x[:, :half], x[:, half:]
        z = (slope(recent) - slope(early)) / scale
        v = (y.mean(1) - x[:, -1]) / scale
        future_slope = slope(y) / scale
        recent_slope = slope(x[:, -seq_len // 4:]) / scale
        mean_shift = (recent.mean(1) - early.mean(1)) / scale
        vol_shift = (recent.std(1) - early.std(1)) / scale
        periodic = (periodic_amplitude(recent, period) - periodic_amplitude(early, period)) / scale
        x_parts.append(xn.astype(np.float32))
        y_parts.append(yn.astype(np.float32))
        c_parts.append(np.full(len(x), channel, np.int16))
        s_parts.append(event_starts)
        scale_parts.append(scale)
        metric_parts.append(np.stack((z, v, future_slope, recent_slope,
                                      mean_shift, vol_shift, periodic), axis=1).astype(np.float32))
    if not x_parts:
        raise ValueError("all windows contain nonfinite values")
    x, y = np.concatenate(x_parts), np.concatenate(y_parts)
    channel, start = np.concatenate(c_parts), np.concatenate(s_parts)
    scale, metric = np.concatenate(scale_parts), np.concatenate(metric_parts)
    ordering = np.lexsort((channel, start))
    x, y, channel, start, scale, metric = (a[ordering] for a in (x,y,channel,start,scale,metric))
    d = pd.DatetimeIndex(dates[start])
    unique_starts = np.unique(start)
    rank = np.searchsorted(unique_starts, start)
    block = np.minimum(2, rank * 3 // len(unique_starts)).astype(np.int8)
    cache = dict(x=x, y=y, channel=channel, start=start, scale=scale,
                 z=metric[:, 0], v=metric[:, 1], future_slope=metric[:, 2],
                 recent_slope=metric[:, 3], mean_shift=metric[:, 4],
                 vol_shift=metric[:, 5], period_amp_shift=metric[:, 6],
                 block=block, event=rank.astype(np.int32), hour=d.hour.to_numpy(dtype=np.int8),
                 weekday=d.dayofweek.to_numpy(dtype=np.int8), month=d.month.to_numpy(dtype=np.int8),
                 timestamp=d.strftime("%Y-%m-%d %H:%M:%S").to_numpy(dtype="U19"),
                 future_timestamp=pd.DatetimeIndex(dates[start + seq_len]).strftime("%Y-%m-%d %H:%M:%S").to_numpy(dtype="U19"),
                 dropped=np.array(dropped, dtype=np.int32))
    np.savez_compressed(cache_path, **cache)
    return cache, key, cache_path, names, cadence


def rho(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[valid], y[valid]).statistic) if valid.sum() >= 8 else float("nan")


def event_block_bootstrap(values_x, values_y, event, rng, repetitions=200, event_block_len=3):
    """Circular moving-block bootstrap of *events*; channels move together."""
    event_values = np.unique(event)
    n = len(event_values)
    if n < 8:
        return np.array([], np.float32)
    result = []
    for _ in range(repetitions):
        origins = rng.integers(0, n, size=math.ceil(n / event_block_len))
        sampled_events = np.concatenate([(o + np.arange(event_block_len)) % n for o in origins])[:n]
        weights = np.bincount(sampled_events, minlength=n)
        selections = [np.flatnonzero((event == event_values[j])) for j in np.flatnonzero(weights)]
        rows = np.concatenate([np.tile(selections[k], weights[j]) for k,j in enumerate(np.flatnonzero(weights))])
        result.append(rho(values_x[rows], values_y[rows]))
    return np.asarray(result, np.float32)


def binned_median(x, y, event, lo, hi, rng, reps=120, bins=12):
    keep = np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x <= hi)
    if keep.sum() < bins * 8:
        return []
    edges = np.quantile(x[keep], np.linspace(0, 1, bins+1))
    rows = []
    for j in range(bins):
        mask = keep & (x >= edges[j]) & ((x <= edges[j+1]) if j == bins-1 else (x < edges[j+1]))
        ev = np.unique(event[mask])
        if mask.sum() < 8 or len(ev) < 3:
            continue
        yy = y[mask]
        # Bootstrap the conditional median by independent event blocks.
        med_boot = []
        by_event = [yy[event[mask] == value] for value in ev]
        for _ in range(reps):
            starts = rng.integers(0, len(ev), size=math.ceil(len(ev)/3))
            chosen = np.concatenate([(s + np.arange(3)) % len(ev) for s in starts])[:len(ev)]
            med_boot.append(float(np.median(np.concatenate([by_event[k] for k in chosen]))))
        rows.append({"x_median":float(np.median(x[mask])), "y_median":float(np.median(yy)),
                     "ci_low":float(np.quantile(med_boot,.025)), "ci_high":float(np.quantile(med_boot,.975)),
                     "n_samples":int(mask.sum()), "n_events":len(ev)})
    return rows


def visible_statistics(data, dataset, horizon, rng, bootstraps):
    z, v, event = data["z"], data["v"], data["event"]
    valid = np.isfinite(z) & np.isfinite(v)
    z,v,event = z[valid],v[valid],event[valid]
    channels=data["channel"][valid]
    rho_all = rho(z,v)
    channel_rhos=[]
    channel_centered_z=z.copy();channel_centered_v=v.copy()
    phase_centered_z=np.full(len(z),np.nan,np.float32)
    phase_centered_v=np.full(len(v),np.nan,np.float32)
    for channel in np.unique(channels):
        pick=channels==channel
        if pick.sum()>=8:channel_rhos.append(rho(z[pick],v[pick]))
        channel_centered_z[pick]-=z[pick].mean()
        channel_centered_v[pick]-=v[pick].mean()
    phase=data["hour"][valid].astype(int)*2+(data["weekday"][valid]>=5).astype(int)
    for channel in np.unique(channels):
        for slot in np.unique(phase[channels==channel]):
            pick=(channels==channel)&(phase==slot)
            if pick.sum()<8:continue
            phase_centered_z[pick]=z[pick]-z[pick].mean()
            phase_centered_v[pick]=v[pick]-v[pick].mean()
    boot = event_block_bootstrap(z,v,event,rng,bootstraps)
    by_block = [rho(z[data["block"][valid]==b],v[data["block"][valid]==b]) for b in range(3)]
    q33,q67 = np.quantile(z,[1/3,2/3])
    group = np.digitize(z,[q33,q67])
    low, high = v[group==0],v[group==2]
    real_delta = float(np.median(high)-np.median(low))
    # Preserve each time block and channel's temporal structure in the null:
    # permute entire event groups of historical z against fixed actual futures.
    null=[]
    full_z, full_v, full_e, full_block = data["z"],data["v"],data["event"],data["block"]
    for _ in range(bootstraps):
        shuffled=full_z.copy()
        for b in range(3):
            events=np.unique(full_e[full_block==b])
            replacement=rng.permutation(events)
            for src,dst in zip(events,replacement):
                src_rows=np.flatnonzero((full_e==src)&(full_block==b))
                dst_rows=np.flatnonzero((full_e==dst)&(full_block==b))
                # The windows have the same channel order unless missing raw values.
                if len(src_rows)==len(dst_rows) and np.array_equal(data["channel"][src_rows],data["channel"][dst_rows]):
                    shuffled[src_rows]=full_z[dst_rows]
                else:
                    per_channel={int(data["channel"][row]):full_z[row] for row in dst_rows}
                    shuffled[src_rows]=[per_channel.get(int(data["channel"][row]),np.nan) for row in src_rows]
        null.append(rho(shuffled,full_v))
    p_null=float((1+np.sum(np.abs(null)>=abs(rho_all)))/(1+len(null)))
    lo=max(np.quantile(data["z"][data["block"]==b],.05) for b in range(3))
    hi=min(np.quantile(data["z"][data["block"]==b],.95) for b in range(3))
    curves=[]
    if lo<hi:
        for b in range(3):
            pick=data["block"]==b
            for j,row in enumerate(binned_median(data["z"][pick],data["v"][pick],
                                                  data["event"][pick],lo,hi,rng,bootstraps)):
                curves.append({"dataset":dataset,"pred_len":horizon,"block":b,"bin":j,**row})
    outlier=np.abs(z)<=np.quantile(np.abs(z),.99)
    return {"dataset":dataset,"pred_len":horizon,"n_windows":len(z),
            "independent_events":len(np.unique(event)),"spearman":rho_all,
            "spearman_ci_low":float(np.nanquantile(boot,.025)) if len(boot) else np.nan,
            "spearman_ci_high":float(np.nanquantile(boot,.975)) if len(boot) else np.nan,
            "rho_block_0":by_block[0],"rho_block_1":by_block[1],"rho_block_2":by_block[2],
            "shuffle_p_two_sided":p_null,"median_v_high_minus_low":real_delta,
            "z_low_cut":float(q33),"z_high_cut":float(q67),
            "common_z_low":float(lo),"common_z_high":float(hi),
            "spearman_excluding_top1pct_abs_z_sensitivity":rho(z[outlier],v[outlier]),
            "future_trend_spearman":rho(data["z"],data["future_slope"]),
            "median_within_channel_spearman":float(np.nanmedian(channel_rhos)) if channel_rhos else np.nan,
            "fraction_channels_same_sign_as_pooled":float(np.mean(np.sign(channel_rhos)==np.sign(rho_all))) if channel_rhos else np.nan,
            "spearman_channel_demeaned":rho(channel_centered_z,channel_centered_v),
            "spearman_channel_hour_daytype_demeaned":rho(phase_centered_z,phase_centered_v),
            "phase_residualized_support_windows":int(np.isfinite(phase_centered_z).sum())},curves


def _similarity(a):
    norm=np.linalg.norm(a,axis=1,keepdims=True)
    unit=a/np.maximum(norm,1e-8)
    return np.clip(unit@unit.T,-1,1)


def pair_statistics(data,dataset,horizon,seq_len,rng,bootstraps):
    span=seq_len+horizon
    pairs=[]
    for channel in np.unique(data["channel"]):
        ids=np.flatnonzero(data["channel"]==channel)
        if len(ids)<3:continue
        x=data["x"][ids];y=data["y"][ids]
        sx=_similarity(x);sy=_similarity(y)
        ynorm=np.sum(y*y,axis=1)
        dist=np.sqrt(np.maximum(0,(ynorm[:,None]+ynorm[None,:]-2*(y@y.T))/horizon))
        i,j=np.triu_indices(len(ids),k=1)
        distant=(data["start"][ids[j]]-data["start"][ids[i]])>=2*span
        hour_diff=np.abs(data["hour"][ids[i]].astype(int)-data["hour"][ids[j]].astype(int))
        hour_diff=np.minimum(hour_diff,24-hour_diff)
        daytype=(data["weekday"][ids[i]]>=5)==(data["weekday"][ids[j]]>=5)
        phase=distant & (hour_diff<=2) & daytype
        if not phase.any():continue
        i,j=i[phase],j[phase]
        ii,jj=ids[i],ids[j]
        season_dist=np.abs(data["month"][ii].astype(int)-data["month"][jj].astype(int))
        same_season=np.minimum(season_dist,12-season_dist)<=1
        opposite=(data["future_slope"][ii]*data["future_slope"][jj]<0)
        pairs.append((ii,jj,sx[i,j],sy[i,j],dist[i,j],np.abs(data["v"][ii]-data["v"][jj]),
                      opposite,same_season))
    if not pairs:
        return {"dataset":dataset,"pred_len":horizon,"pair_status":"no_phase_matched_pairs"},[],[],None
    i,j,sx,sy,dist,delta,opposite,season=(np.concatenate([part[k] for part in pairs]) for k in range(8))
    sx,sy,dist,delta=sx.astype(np.float32),sy.astype(np.float32),dist.astype(np.float32),delta.astype(np.float32)
    threshold_x=float(np.quantile(sx,.9));threshold_y=float(np.quantile(sy,.1))
    high=sx>=threshold_x
    low=sx<=np.quantile(sx,.1)
    divergent=high & (sy<threshold_y)
    rate=float(np.mean(sy[high]<threshold_y))
    low_rate=float(np.mean(sy[low]<threshold_y))
    # Whole-event moving-block pairs bootstrap, preserving endpoint dependence.
    n_events=len(np.unique(data["event"]))
    ci=[]
    for _ in range(bootstraps):
        origins=rng.integers(0,n_events,size=math.ceil(n_events/3))
        sampled=np.concatenate([(o+np.arange(3))%n_events for o in origins])[:n_events]
        count=np.bincount(sampled,minlength=n_events)
        weights=count[data["event"][i]]*count[data["event"][j]]
        denom=np.sum(weights[high])
        if denom>0:ci.append(float(np.sum(weights[divergent])/denom))
    edges=np.quantile(sx,np.linspace(0,1,21))
    bins=[]
    for bin_id in range(20):
        mask=(sx>=edges[bin_id]) & ((sx<=edges[bin_id+1]) if bin_id==19 else (sx<edges[bin_id+1]))
        if not mask.any():continue
        bins.append({"dataset":dataset,"pred_len":horizon,"bin":bin_id,"n_pairs":int(mask.sum()),
                     "history_similarity_median":float(np.median(sx[mask])),
                     "future_similarity_median":float(np.median(sy[mask])),
                     "future_similarity_p10":float(np.quantile(sy[mask],.1)),
                     "future_similarity_p90":float(np.quantile(sy[mask],.9))})
    sensitivity=[]
    for qx in (.8,.9,.95):
        for qy in (.1,.25):
            tx=float(np.quantile(sx,qx));ty=float(np.quantile(sy,qy))
            mask=sx>=tx
            sensitivity.append({"dataset":dataset,"pred_len":horizon,"history_quantile":qx,
                                "future_quantile":qy,"tau_x":tx,"tau_y":ty,
                                "n_pairs":int(mask.sum()),"divergence_rate":float(np.mean(sy[mask]<ty))})
    selected=None
    if divergent.any():
        pool=np.flatnonzero(divergent)
        d25,d75=np.quantile(dist[pool],[.25,.75])
        central=pool[(dist[pool]>=d25)&(dist[pool]<=d75)]
        if len(central)>=5:pool=central
        med_x,med_y,med_d=np.median(sx[pool]),np.median(sy[pool]),np.median(dist[pool])
        spread_x=max(float(np.std(sx[pool])),1e-4)
        spread_y=max(float(np.std(sy[pool])),1e-4)
        spread_d=max(float(np.std(dist[pool])),1e-4)
        position=pool[np.argmin(((sx[pool]-med_x)/spread_x)**2+
                                ((sy[pool]-med_y)/spread_y)**2+
                                ((dist[pool]-med_d)/spread_d)**2)]
        selected={"dataset":dataset,"pred_len":horizon,"channel":int(data["channel"][i[position]]),
                  "first_window_start":int(data["start"][i[position]]),"second_window_start":int(data["start"][j[position]]),
                  "first_timestamp":str(data["timestamp"][i[position]]),"second_timestamp":str(data["timestamp"][j[position]]),
                  "history_similarity":float(sx[position]),"future_similarity":float(sy[position]),
                  "future_trajectory_rmse":float(dist[position]),"pair_count":len(sx),
                  "high_pair_count":int(high.sum()),"divergent_pair_count":int(divergent.sum()),
                  "selection_rule":"q90-history/q10-future subset; central 50% future RMSE; nearest 3D median of Sx,Sy,RMSE",
                  "cache_i":int(i[position]),"cache_j":int(j[position])}
    summary={"dataset":dataset,"pred_len":horizon,"pair_status":"ok","phase_matched_pairs":len(sx),
             "high_similarity_pairs":int(high.sum()),"divergent_high_pairs":int(divergent.sum()),
             "tau_x_q90":threshold_x,"tau_y_q10":threshold_y,"divergence_rate":rate,
             "divergence_ci_low":float(np.quantile(ci,.025)) if ci else np.nan,
             "divergence_ci_high":float(np.quantile(ci,.975)) if ci else np.nan,
             "low_similarity_divergence_rate":low_rate,
             "same_season_high_divergence_rate":float(np.mean(sy[high & season]<threshold_y)) if (high & season).sum() else np.nan,
             "same_season_high_pairs":int((high & season).sum()),
             "high_future_similarity_median":float(np.median(sy[high])),
             "high_future_similarity_p10":float(np.quantile(sy[high],.1)),
             "high_future_similarity_p90":float(np.quantile(sy[high],.9)),
             "high_trajectory_rmse_median":float(np.median(dist[high])),
             "low_trajectory_rmse_median":float(np.median(dist[low])),
             "high_mean_change_difference_median":float(np.median(delta[high])),
             "high_opposite_future_trend_rate":float(np.mean(opposite[high])),
             "independent_pair_event_count":n_events}
    return summary,bins,sensitivity,selected


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument("--output",type=Path,default=Path("results/fpem_intro_motivation"))
    ap.add_argument("--datasets",nargs="+",default=list(DATASETS),choices=DATASETS)
    ap.add_argument("--horizons",nargs="+",type=int,default=list(HORIZONS))
    ap.add_argument("--channels",default="all",help="all or comma-separated integer indices")
    ap.add_argument("--seq_len",type=int,default=96)
    ap.add_argument("--seed",type=int,default=2022)
    ap.add_argument("--bootstraps",type=int,default=120)
    ap.add_argument("--window_offset_fraction",type=float,default=0.,
                    help="Predeclared nonoverlapping-grid offset as fraction of full window; 0 or 0.5 for sensitivity")
    args=ap.parse_args()
    if not 0<=args.window_offset_fraction<1:
        raise ValueError("window_offset_fraction must lie in [0,1)")
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    config={"datasets":args.datasets,"horizons":args.horizons,"channels":args.channels,
            "seq_len":args.seq_len,"seed":args.seed,"bootstraps":args.bootstraps,
            "window_offset_fraction":args.window_offset_fraction,
            "window_stride":"seq_len + pred_len (no overlap)",
            "pair_eligibility":"same channel; gap >= 2 full windows; circular hour difference <= 2; same weekday/weekend",
            "pair_thresholds":"history cosine q90, future cosine q10 from all eligible TRAIN pairs",
            "history_scale":"per-window MAD*1.4826, lower bounded by .05*TRAIN-channel std or 1e-5",
            "bootstrap":"circular moving blocks of three independent full-window events"}
    (output/"config.json").write_text(json.dumps(config,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    rows=[];curves=[];pairs=[];bins=[];sensitivity=[]
    for dataset in args.datasets:
        for horizon in args.horizons:
            rng=np.random.default_rng(args.seed+int(horizon)+list(DATASETS).index(dataset)*1000)
            data,key,cache_path,names,cadence=make_cache(args.repo,output,dataset,horizon,
                                                           args.seq_len,args.channels,args.seed,
                                                           args.window_offset_fraction)
            visible,curve=visible_statistics(data,dataset,horizon,rng,args.bootstraps)
            pair,bin_rows,sens,selected=pair_statistics(data,dataset,horizon,args.seq_len,rng,args.bootstraps)
            visible.update(pair)
            visible.update({"channels_analyzed":len(np.unique(data["channel"])),
                            "channels_in_source":len(names),"sampling_interval":cadence,
                            "train_start":str(data["timestamp"][0]),
                            "train_last_window_start":str(data["timestamp"][-1]),
                            "train_end_exclusive_row":key["train_end_exclusive"],
                            "window_offset_steps":key["window_offset"],
                            "window_overlap":False,"dropped_nonfinite_windows":int(data["dropped"]),
                            "raw_csv_sha256":key["source_sha256"],"cache_file":str(cache_path)})
            rows.append(visible);curves+=curve;bins+=bin_rows;sensitivity+=sens
            if selected is not None:pairs.append(selected)
            print(f"DONE {dataset}-{horizon}: rho={visible['spearman']:.3f}, pairs={pair.get('phase_matched_pairs',0)}, divergence={pair.get('divergence_rate',float('nan')):.3f}",flush=True)
    pd.DataFrame(rows).to_csv(output/"statistics.csv",index=False)
    pd.DataFrame(curves).to_csv(output/"visible_curve_bins.csv",index=False)
    pd.DataFrame(bins).to_csv(output/"pair_similarity_bins.csv",index=False)
    pd.DataFrame(sensitivity).to_csv(output/"pair_divergence.csv",index=False)
    pd.DataFrame(pairs).to_csv(output/"selected_pairs.csv",index=False)


if __name__=="__main__":main()
