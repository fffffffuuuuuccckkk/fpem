#!/usr/bin/env python3
"""Publication and diagnostic figures from fpem_intro_raw_statistics.py caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASETS=("ETTm2","Electricity","Weather")
HORIZONS=(96,192,336,720)
TIME_COLORS=("#5277A3","#B77B3B","#54856A")
GROUP_COLORS=("#425F8C","#698E6B","#BE7845")
LABELS={"trend_change":"Historical trend change (per step / history scale)",
        "recent_slope":"Recent-quarter slope / history scale",
        "mean_shift":"Late minus early history mean / history scale",
        "vol_shift":"Late minus early history SD / history scale",
        "period_amp_shift":"Late minus early daily amplitude / history scale"}


def load_cache(row):
    with np.load(Path(row["cache_file"]),allow_pickle=False) as source:
        return {name:source[name] for name in source.files}


def save(fig,output,stem):
    fig.savefig(output/(stem+".pdf"),bbox_inches="tight")
    fig.savefig(output/(stem+".png"),dpi=300,bbox_inches="tight")
    plt.close(fig)


def visible_panel(ax,row,curves,data,rng,scatter=1100,legend=False):
    z,v=data["z"],data["v"]
    any_curve=False
    for block,color in enumerate(TIME_COLORS):
        indices=np.flatnonzero(data["block"]==block)
        if len(indices)>scatter:indices=rng.choice(indices,scatter,replace=False)
        ax.scatter(z[indices],v[indices],s=4,alpha=.09,color=color,rasterized=True)
        group=curves[(curves.dataset==row.dataset)&(curves.pred_len==row.pred_len)&(curves.block==block)]
        if len(group):
            any_curve=True
            group=group.sort_values("x_median")
            ax.plot(group.x_median,group.y_median,color=color,lw=1.7,label=f"TRAIN block {block+1}")
            ax.fill_between(group.x_median,group.ci_low,group.ci_high,color=color,alpha=.14)
    ax.axhline(0,color="#777777",lw=.6,alpha=.6)
    if not any_curve:
        ax.text(.02,.96,"Insufficient per-bin event support; scatter only",
                transform=ax.transAxes,va="top",fontsize=7,color="#555555")
    # Signed-log display retains every raw point while making the dense center
    # legible; this changes only axis geometry, never inclusion or statistics.
    ax.set_xscale("symlog",linthresh=.03)
    ax.set_yscale("symlog",linthresh=1.)
    ax.set_title(f"{row.dataset} · {row.pred_len}   ρ={row.spearman:.2f}, events={int(row.independent_events)}",fontsize=9)
    ax.set_xlabel("Visible trend change z (per step; symlog axis)")
    ax.set_ylabel("Actual future change v (symlog axis)")
    if legend:ax.legend(frameon=False,fontsize=8,ncol=3,loc="upper center")


def case_panel(ax_history,ax_future,row,pair,data,legend=False):
    a,b=int(pair.cache_i),int(pair.cache_j)
    length=data["x"].shape[1];horizon=data["y"].shape[1]
    ax_history.plot(np.arange(length),data["x"][a],color="#777777",lw=1.5,label="History A")
    ax_history.plot(np.arange(length),data["x"][b],color="#B5B5B5",lw=1.3,label="History B")
    ax_future.plot(np.arange(horizon),data["y"][a],color="#4D8A68",lw=1.4,label="Future A")
    ax_future.plot(np.arange(horizon),data["y"][b],color="#C48248",lw=1.4,label="Future B")
    ax_history.axvline(length-1,color="#666666",ls="--",lw=.8)
    ax_future.axvline(0,color="#666666",ls="--",lw=.8)
    ax_history.set_xlabel(f"Observed history ({length} steps)")
    ax_future.set_xlabel(f"Actual future ({horizon} steps)")
    ax_history.set_ylabel("History-scaled value")
    ax_future.set_yticklabels([])
    ax_history.set_title(f"History cosine = {pair.history_similarity:.2f}",fontsize=9)
    ax_future.set_title(f"Future cosine = {pair.future_similarity:.2f}",fontsize=9)
    both=np.r_[data["x"][a],data["x"][b],data["y"][a],data["y"][b]]
    bottom,top=np.quantile(both,[.005,.995]);margin=max(.1,(top-bottom)*.08)
    ax_history.set_ylim(bottom-margin,top+margin);ax_future.set_ylim(bottom-margin,top+margin)
    if legend:
        h1,l1=ax_history.get_legend_handles_labels();h2,l2=ax_future.get_legend_handles_labels()
        ax_history.figure.legend(h1+h2,l1+l2,loc="upper center",
                                 bbox_to_anchor=(.5,.85),ncol=4,frameon=False,fontsize=8)


def choose_visible(stats):
    # A priori support/significance rule, then fixed preference order. No visual ranking.
    order=[(d,h) for d in DATASETS for h in (336,192,96,720)]
    for d,h in order:
        row=stats[(stats.dataset==d)&(stats.pred_len==h)]
        if row.empty:continue
        r=row.iloc[0]
        correlations=[r.rho_block_0,r.rho_block_1,r.rho_block_2]
        if (r.independent_events>=30 and abs(r.spearman)>=.1 and r.shuffle_p_two_sided<=.05
                and sum(np.sign(c)==np.sign(r.spearman) for c in correlations)>=2
                and r.common_z_low<r.common_z_high):
            return r,"predeclared support/significance rule"
    choices=stats[stats.independent_events>=30]
    if choices.empty:choices=stats
    return choices.iloc[0],"fallback fixed order; no configuration met evidence rule"


def choose_pair(stats,selected):
    order=[(d,h) for d in DATASETS for h in (336,192,96,720)]
    for d,h in order:
        row=stats[(stats.dataset==d)&(stats.pred_len==h)]
        pair=selected[(selected.dataset==d)&(selected.pred_len==h)]
        if row.empty or pair.empty:continue
        r=row.iloc[0]
        if r.high_similarity_pairs>=50 and r.divergent_high_pairs>=5 and r.independent_events>=30:
            return r,pair.iloc[0],"predeclared support rule"
    for d,h in order:
        row=stats[(stats.dataset==d)&(stats.pred_len==h)]
        pair=selected[(selected.dataset==d)&(selected.pred_len==h)]
        if not row.empty and not pair.empty:
            return row.iloc[0],pair.iloc[0],"fallback fixed order; low support"
    return None,None,"no eligible true pair"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,default=Path("results/fpem_intro_motivation"))
    parser.add_argument("--seed",type=int,default=2022)
    args=parser.parse_args();out=args.input
    stats=pd.read_csv(out/"statistics.csv")
    curves=pd.read_csv(out/"visible_curve_bins.csv")
    bins=pd.read_csv(out/"pair_similarity_bins.csv")
    selected=pd.read_csv(out/"selected_pairs.csv")
    config=json.loads((out/"config.json").read_text(encoding="utf-8"))
    rng=np.random.default_rng(args.seed)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.spines.top":False,
                         "axes.spines.right":False,"pdf.fonttype":42,"ps.fonttype":42,
                         "figure.facecolor":"white","axes.labelsize":8})
    lead,lead_rule=choose_visible(stats)
    pair_row,pair,pair_rule=choose_pair(stats,selected)
    # Single-paper-width Figure 1(a): real sample scatter + equal-count median and CI.
    fig,ax=plt.subplots(figsize=(7.0,3.5))
    visible_panel(ax,lead,curves,load_cache(lead),rng,scatter=1600,legend=True)
    fig.tight_layout(rect=[0,0,1,.92]);save(fig,out,"fig1a_visible_variation")
    # Figure 1(b): horizontally separate regions, shared y-scale; no resampling of values.
    if pair is not None:
        fig,(left,right)=plt.subplots(1,2,figsize=(7.0,3.4),gridspec_kw={"wspace":.04})
        case_panel(left,right,pair_row,pair,load_cache(pair_row),legend=True)
        fig.suptitle(f"{pair.dataset}-{int(pair.pred_len)} · channel {int(pair.channel)} · "
                     f"{int(pair.high_pair_count)} high-similarity pairs (all channels)",y=.98,fontsize=10)
        fig.text(.5,.915,f"History starts: A {pair.first_timestamp}  |  B {pair.second_timestamp}",
                 ha="center",va="center",fontsize=7)
        fig.subplots_adjust(top=.69,bottom=.19);save(fig,out,"fig1b_unseen_variation")
    # Full diagnostic grids use the same fixed dataset/horizon order.
    fig,axes=plt.subplots(3,4,figsize=(15,9))
    for row_idx,d in enumerate(DATASETS):
        for col_idx,h in enumerate(HORIZONS):
            ax=axes[row_idx,col_idx];subset=stats[(stats.dataset==d)&(stats.pred_len==h)]
            if subset.empty:ax.axis("off");continue
            r=subset.iloc[0]
            visible_panel(ax,r,curves,load_cache(r),rng,scatter=600)
    fig.tight_layout();save(fig,out,"fig1a_all_configs")
    fig,axes=plt.subplots(3,4,figsize=(15,9))
    for ri,d in enumerate(DATASETS):
        for ci,h in enumerate(HORIZONS):
            ax=axes[ri,ci];subset=stats[(stats.dataset==d)&(stats.pred_len==h)]
            if subset.empty:ax.axis("off");continue
            r=subset.iloc[0];data=load_cache(r)
            z=data["z"];edges=np.quantile(z,[1/3,2/3]);groups=np.digitize(z,edges)
            for group,color in enumerate(GROUP_COLORS):
                future=data["y"][groups==group]
                if not len(future):continue
                x=np.arange(future.shape[1]);med=np.median(future,axis=0)
                ax.plot(x,med,lw=1.2,color=color,label=("Low","Mid","High")[group])
                ax.fill_between(x,np.quantile(future,.1,axis=0),np.quantile(future,.9,axis=0),
                                color=color,alpha=.1)
            ax.set_title(f"{d}-{h} · event n={int(r.independent_events)}",fontsize=9)
            ax.set_xlabel("Actual future step");ax.set_ylabel("Value / history scale")
    axes[0,0].legend(frameon=False,fontsize=7)
    fig.tight_layout();save(fig,out,"fig1a_future_trajectories")
    for indicator in ("recent_slope","mean_shift","vol_shift","period_amp_shift"):
        fig,axes=plt.subplots(3,4,figsize=(15,9))
        for ri,d in enumerate(DATASETS):
            for ci,h in enumerate(HORIZONS):
                ax=axes[ri,ci];subset=stats[(stats.dataset==d)&(stats.pred_len==h)]
                if subset.empty:ax.axis("off");continue
                r=subset.iloc[0];data=load_cache(r)
                x=data[indicator];v=data["v"];good=np.isfinite(x)&np.isfinite(v)
                if good.sum()<40:
                    ax.text(.5,.5,"Insufficient history cycles",ha="center",va="center",transform=ax.transAxes)
                    ax.set_axis_off();continue
                edges=np.quantile(x[good],np.linspace(0,1,13))
                mx=[];my=[]
                for j in range(12):
                    mask=good&(x>=edges[j])&((x<=edges[j+1]) if j==11 else (x<edges[j+1]))
                    if mask.sum()<8:continue
                    mx.append(np.median(x[mask]));my.append(np.median(v[mask]))
                ax.plot(mx,my,color="#526D90",marker="o",markersize=2,lw=1.2)
                ax.set_title(f"{d}-{h} · n={int(good.sum())}",fontsize=9)
                ax.set_xlabel(LABELS[indicator]);ax.set_ylabel("Actual future change v")
        fig.tight_layout();save(fig,out,f"fig1a_indicator_{indicator}")
    fig,axes=plt.subplots(3,4,figsize=(15,9))
    for ri,d in enumerate(DATASETS):
        for ci,h in enumerate(HORIZONS):
            ax=axes[ri,ci];pair_set=selected[(selected.dataset==d)&(selected.pred_len==h)]
            subset=stats[(stats.dataset==d)&(stats.pred_len==h)]
            if pair_set.empty or subset.empty:
                ax.text(.5,.5,"No pair meeting preset rule",ha="center",transform=ax.transAxes)
                ax.axis("off");continue
            r=subset.iloc[0];p=pair_set.iloc[0];data=load_cache(r)
            a,b=int(p.cache_i),int(p.cache_j)
            ax.plot(np.arange(len(data["x"][a])),data["x"][a],color="#777777",lw=.9)
            ax.plot(np.arange(len(data["x"][b])),data["x"][b],color="#AAAAAA",lw=.9)
            length=data["x"].shape[1]
            ax.plot(length+np.arange(len(data["y"][a])),data["y"][a],color="#4D8A68",lw=1)
            ax.plot(length+np.arange(len(data["y"][b])),data["y"][b],color="#C48248",lw=1)
            ax.axvline(length,color="#555555",ls="--",lw=.7)
            ax.set_title(f"{d}-{h} · Sx={p.history_similarity:.2f}, Sy={p.future_similarity:.2f}",fontsize=9)
            ax.set_xlabel("Actual steps (history | future)");ax.set_ylabel("History-scaled value")
    fig.tight_layout();save(fig,out,"fig1b_all_cases")
    fig,axes=plt.subplots(3,4,figsize=(15,9))
    for ri,d in enumerate(DATASETS):
        for ci,h in enumerate(HORIZONS):
            ax=axes[ri,ci]
            rows=bins[(bins.dataset==d)&(bins.pred_len==h)].sort_values("history_similarity_median")
            if rows.empty:ax.axis("off");continue
            ax.plot(rows.history_similarity_median,rows.future_similarity_median,color="#526D90",lw=1.5)
            ax.fill_between(rows.history_similarity_median,rows.future_similarity_p10,
                            rows.future_similarity_p90,color="#526D90",alpha=.18)
            ax.set_title(f"{d}-{h} · pairs={int(rows.n_pairs.sum())}",fontsize=9)
            ax.set_xlabel("History cosine similarity");ax.set_ylabel("Actual future cosine")
    fig.tight_layout();save(fig,out,"fig1b_conditional_quantiles")
    fig,axes=plt.subplots(3,4,figsize=(15,9))
    for ri,d in enumerate(DATASETS):
        for ci,h in enumerate(HORIZONS):
            ax=axes[ri,ci];subset=stats[(stats.dataset==d)&(stats.pred_len==h)]
            if subset.empty:ax.axis("off");continue
            r=subset.iloc[0]
            if pd.isna(r.divergence_rate):ax.axis("off");continue
            vals=[r.divergence_rate,r.low_similarity_divergence_rate,r.same_season_high_divergence_rate]
            ax.bar([0,1,2],vals,color=["#4D8A68","#888888","#C48248"])
            ax.errorbar([0],[r.divergence_rate],yerr=[[max(0,r.divergence_rate-r.divergence_ci_low)],
                                                       [max(0,r.divergence_ci_high-r.divergence_rate)]],
                        fmt="none",ecolor="black",capsize=3)
            ax.set_xticks([0,1,2],["High X","Low X","High X\nseason"],fontsize=7)
            ax.set_ylim(0,1);ax.set_title(f"{d}-{h} · high pairs={int(r.high_similarity_pairs)}",fontsize=9)
            ax.set_ylabel("Future divergence rate")
    fig.tight_layout();save(fig,out,"fig1b_divergence_rates")
    # Explicit interpretation limits, metric/population descriptions, and bilingual captions.
    readme=f"""# FPEM Introduction raw-data motivation figures

No A0, F03, R3, Flow, K-Means, checkpoint, prediction, or prototype library is used.
Sources: original TRAIN CSV intervals only. ETTm2 uses first 34,560 rows; Electricity/Weather use first floor(70% of rows), matching the repository's data loader. Input/output values are raw CSV units before per-window history normalization. All full history+future windows advance by `seq_len + pred_len`, so sampled windows have zero overlap. Different channels sharing one start are *one independent time event*, not independent replicates. Raw source SHA and TRAIN boundaries are in `statistics.csv`.

History scale: `max(1.4826*MAD(history), 0.05*SD(TRAIN channel), 1e-5)`. The future is always normalized with its own paired *history* mean and scale, never its future statistics. Main visible indicator `z=(slope(last half of X)-slope(first half of X))/history_scale`, where slope is linear least-squares per raw step. Outcome `v=(mean(Y)-last(X))/history_scale`. The 3 TRAIN time blocks and history-z terciles are formed without Y. Scatter displays a fixed-seed random subset for legibility; medians/CI use all eligible windows. Curves use 12 equal-count bins on the common 5th-95th percentile z support, with no smoothing. Both main-figure axes use signed-log display (linear center x=±0.03, y=±1) to retain all extreme points while making central structure legible; no value is trimmed, winsorized, or omitted from the statistics. The translucent curve region is a 95% moving-block bootstrap **median-estimation CI** using contiguous blocks of three independent full-window events. The separate trajectory P10-P90 region is the **distribution across actual futures**, not a CI. Spearman CI uses the same event-block bootstrap; the within-time-block permutation shuffles historical z event bundles against futures. Non-significant or contradictory relationships are retained.

History similarity is cosine of per-window history-scaled X; actual-future similarity is cosine of Y scaled by that paired X. Pair trajectory distance is RMSE in the same history-scale units. Pair eligibility is fixed *before* examining Y: same channel, at least one full-window gap, circular hour-of-day separation <=2 h, same weekend/weekday type. The same-month±1 control is reported separately. Across eligible TRAIN pairs, high history = cosine >= q90, divergent future = cosine < q10; both numeric thresholds and counts are saved. `pair_divergence.csv` contains predeclared q80/q90/q95 × future q10/q25 sensitivity. Future divergence-rate CI resamples moving blocks of independent events, preserving pair endpoint dependence. The conditional curve gives the median and P10-P90 **pair dispersion**, not a median CI. Displayed examples come from the central 50% of future-RMSE within the high-history/divergent-future subset and are closest to its joint (Sx,Sy,RMSE) median, never the most extreme; Y was used only for this disclosed offline example selection.

Main-figure selection rule is fixed in `plot_fpem_introduction_motivation.py`: dataset preference ETTm2, Electricity, Weather; horizon preference 336,192,96,720. Fig. 1(a) additionally requires >=30 independent events, |Spearman|>=0.1, shuffled-p<=0.05, at least two block correlations of the same sign, and overlapping z support; Fig. 1(b) requires >=50 high-history pairs, >=5 divergent pairs, >=30 events. A fallback is labeled if no case qualifies. `statistics.csv` also reports within-channel Spearman, channel-demeaned Spearman, and channel-plus-hour/daytype-demeaned Spearman as fixed sensitivity checks against cross-channel and calendar confounding; the grouping variables are historical only. Neither plot supports a causal claim, and the second does not identify an external environment as the cause.

Run: `python tools/fpem_intro_raw_statistics.py --repo . --output results/fpem_intro_motivation`, then `python tools/plot_fpem_introduction_motivation.py --input results/fpem_intro_motivation`. Select `--datasets`, `--horizons`, `--channels` and `--seed` in the statistics command as needed. `--window_offset_fraction 0.5` supplies a separately labeled nonoverlapping-grid sensitivity, not an alternative picked to optimize the figure. Cache files are deterministic by dataset/horizon/channels/source SHA/offset and never contain model predictions.

Figure 1(a) chosen: {lead.dataset}-{int(lead.pred_len)}, {lead_rule}. Figure 1(b) chosen: {pair_row.dataset if pair_row is not None else 'none'}-{int(pair_row.pred_len) if pair_row is not None else 'none'}, {pair_rule}. All results are descriptive and may change with another predeclared sampling design or dataset.
"""
    (out/"README.md").write_text(readme,encoding="utf-8")
    caption=f"""# Figure captions

## Figure 1(a) — English
History-identifiable variation in raw TRAIN data ({lead.dataset}, horizon {int(lead.pred_len)}). Each point is a non-overlapping history–actual-future window (display subsampled for visibility). The x-axis is the change in within-history slope divided by a history-only robust scale; the y-axis is the actual future mean relative to the last observation on the same scale. Both axes use a signed-log display to retain extreme points. Curves show equal-count-bin conditional medians for three chronological TRAIN blocks on common historical support; bands are 95% event-block bootstrap confidence intervals for the median. Spearman rho={lead.spearman:.3f} (95% CI {lead.spearman_ci_low:.3f} to {lead.spearman_ci_high:.3f}); independent event count={int(lead.independent_events)}. Association is predictive evidence, not causality.

## 图 1(a) — 中文
原始 TRAIN 数据中历史可辨识的变化（{lead.dataset}，预测长度 {int(lead.pred_len)}）。散点是互不重叠的历史—真实未来窗口（仅显示时固定种子抽样）；横轴为历史前后段斜率变化除以仅由历史计算的稳健尺度，纵轴为真实未来均值相对历史最后一个观测值的变化。双轴采用保留极值的有符号对数显示。三条曲线是三个连续 TRAIN 时间块在共同历史支持区间内的等样本量分箱条件中位数；阴影为中位数的事件块 Bootstrap 95% 置信区间。Spearman ρ={lead.spearman:.3f}，独立时间事件数={int(lead.independent_events)}。相关性不等于因果性。

## Figure 1(b) — English
History-unidentifiable future variation in raw TRAIN data ({pair_row.dataset if pair_row is not None else 'none'}, horizon {int(pair_row.pred_len) if pair_row is not None else 'none'}). Two same-channel, non-overlapping histories satisfy the predeclared phase and high-history-similarity rules, yet their actual futures fall in the low-future-similarity subset. Each trajectory is normalized by its own **history** mean and robust scale; left and right panels show all history and future steps at separate horizontal scales, not one common step scale. The example is chosen from the central 50% of future-RMSE among qualifying pairs, nearest their joint similarity/distance median, not an extreme. History cosine={pair.history_similarity if pair is not None else float('nan'):.3f}; future cosine={pair.future_similarity if pair is not None else float('nan'):.3f}; high-history pair count={int(pair.high_pair_count) if pair is not None else 0}. This demonstrates residual ambiguity, not its physical cause.

## 图 1(b) — 中文
原始 TRAIN 数据中历史难以辨识的未来差异（{pair_row.dataset if pair_row is not None else '无'}, 预测长度 {int(pair_row.pred_len) if pair_row is not None else '无'}）。同一通道两段互不重叠历史满足预设时相匹配和高历史相似规则，但真实未来位于低相似区。每条曲线均用自身历史均值和稳健尺度归一化；左右面板分别完整显示历史与未来，横向比例不同。案例来自达标样本对未来轨迹距离的中间 50%，再选最接近相似度和距离联合中位数的一对，而非极端异常对。历史余弦相似度={pair.history_similarity if pair is not None else float('nan'):.3f}，未来余弦相似度={pair.future_similarity if pair is not None else float('nan'):.3f}。该图只说明剩余不确定性，不能证明其外部环境成因。
"""
    (out/"figure_captions.md").write_text(caption,encoding="utf-8")
    print("MAIN",lead.dataset,int(lead.pred_len),"PAIR",pair_row.dataset if pair_row is not None else "none",
          int(pair_row.pred_len) if pair_row is not None else "none",flush=True)


if __name__=="__main__":main()
