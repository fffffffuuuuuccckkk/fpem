"""Model-free, causal raw-series diagnostics for history/future mapping relations.

This is an analog-forecasting diagnostic, not an estimator of a physical
Jacobian or the formal kernel conditional test in the cited papers.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SOURCE = {
    "ETTh1": "ETT-small/ETTh1.csv", "ETTm2": "ETT-small/ETTm2.csv",
    "Weather": "weather/weather.csv", "Electricity": "electricity/electricity.csv",
    "Traffic": "traffic/traffic.csv",
}


def bounds(name, n):
    if name == "ETTh1":
        a, b = 12*30*24, 4*30*24
        return ((0,a),(a,a+b),(a+b,min(n,a+2*b)))
    if name == "ETTm2":
        a, b = 12*30*24*4, 4*30*24*4
        return ((0,a),(a,a+b),(a+b,min(n,a+2*b)))
    a, c = int(n*.7), int(n*.2)
    return ((0,a),(a,n-c),(n-c,n))


def choose_channel(df, name, train_end, requested):
    cols = list(df.columns[1:])
    if requested != "auto":
        if requested not in cols:
            raise ValueError(f"channel {requested} not in {cols[:8]}...")
        return cols.index(requested)
    if "OT" in cols:
        return cols.index("OT")
    # TRAIN-only raw variance, deterministic tie-breaking.
    variances = df.iloc[:train_end,1:].var(numeric_only=True)
    return cols.index(str(variances.idxmax()))


def slope(x):
    t=np.linspace(-1.,1.,len(x)); return float(np.dot(x-x.mean(),t)/np.dot(t,t))


def build(raw, dates, channel, others, lo, hi, L, H, stride, floor, main_period):
    # Origin is the first point of X. Z, X and Y are wholly inside this split.
    starts=np.arange(lo+L, hi-L-H+1, stride, dtype=np.int64)
    out={k:[] for k in ("start","phase","week","z","x","y","x_window","dy_change",
                           "dy_increment","older","high","fft_high","diff","second_diff",
                           "slope","variance","level","shock","weekphase","periodphase",
                           "other")}
    for st in starts:
        z=raw[st-L:st,channel].astype(np.float32)
        x=raw[st:st+L,channel].astype(np.float32)
        y=raw[st+L:st+L+H,channel].astype(np.float32)
        scale=max(float(np.std(x)),float(floor),1e-4)
        smooth=np.convolve(x,np.ones(9)/9.,mode="same")
        high=x-smooth
        early,late=x[:L//2],x[L//2:]
        other=[]
        for c in others:
            q=raw[st:st+L,c]
            corr=float(np.corrcoef(x,q)[0,1]) if x.std()>1e-8 and q.std()>1e-8 else 0.
            lead=float(np.corrcoef(x[1:],q[:-1])[0,1]) if x[1:].std()>1e-8 and q[:-1].std()>1e-8 else 0.
            other.extend([float(q.mean()),float(slope(q)),float(q.std()),corr,lead])
        out["start"].append(int(st)); out["phase"].append(int(dates[st+L-1].hour))
        out["week"].append(int(dates[st+L-1].dayofweek))
        out["weekphase"].append(int(dates[st+L-1].dayofweek*24+dates[st+L-1].hour))
        out["periodphase"].append(int((st+L-1)%main_period))
        out["z"].append(z); out["x"].append(x); out["y"].append(y)
        out["x_window"].append((x-x.mean())/scale)
        out["dy_change"].append(y-x[-1]); out["dy_increment"].append(np.diff(y,prepend=x[-1]))
        out["older"].append(float(np.mean((z-x.mean())**2)))
        out["high"].append(high)
        spectrum=np.abs(np.fft.rfft(x-x.mean()))**2
        out["fft_high"].append(float(spectrum[len(spectrum)//2:].sum()/(spectrum.sum()+1e-8)))
        out["diff"].append(float(np.mean(np.abs(np.diff(x)))))
        out["second_diff"].append(float(np.mean(np.diff(x,n=2)**2)))
        out["slope"].append(float((slope(late)-slope(early))/scale))
        out["variance"].append(float(np.log((late.var()+floor**2)/(early.var()+floor**2))))
        out["level"].append(float((late.mean()-early.mean())/scale))
        out["shock"].append(float((x[-L//4:].mean()-x[:-L//4].mean())/scale))
        out["other"].append(other)
    return {k:np.asarray(v,dtype=np.float32 if k not in ("start","phase","week","weekphase","periodphase") else np.int64)
            for k,v in out.items()}


def dist(a,b):
    return np.mean((a-b)**2,axis=-1)


def nearest_indices(query, ref, k, origin_gap, same_hour=False):
    """Reference endpoint Y must precede query Z start (causal replay)."""
    found=[]
    for q in range(len(query["start"])):
        legal=ref["start"]+origin_gap <= query["start"][q]
        if same_hour:
            legal &= ref["phase"]==query["phase"][q]
        idx=np.flatnonzero(legal)
        if len(idx)<k:
            found.append((np.empty(0,dtype=int),np.empty(0))); continue
        d=dist(ref["x"][idx],query["x"][q])
        chosen=np.argpartition(d,min(k,len(d)-1))[:k]
        order=np.argsort(d[chosen])
        found.append((idx[chosen[order]],d[chosen[order]]))
    return found


def time_block_boot(values, origins, span, seed=2022, reps=300):
    values=np.asarray(values,dtype=float); origins=np.asarray(origins,dtype=int)
    finite=np.isfinite(values)
    values=values[finite]; origins=origins[finite]
    if len(values)==0: return (np.nan,np.nan,0)
    ids=np.unique(origins//span)
    blocks=[values[origins//span==e].mean() for e in ids]
    if len(blocks)<3: return (np.nan,np.nan,len(blocks))
    rng=np.random.default_rng(seed); a=np.asarray(blocks)
    sampled=np.asarray([a[rng.integers(len(a),size=len(a))].mean() for _ in range(reps)])
    return (*np.quantile(sampled,[.025,.975]).tolist(),len(blocks))


def spearman(a,b):
    if len(a)<5 or np.std(a)<1e-12 or np.std(b)<1e-12:return np.nan
    v=spearmanr(a,b).statistic
    return float(v) if np.isfinite(v) else np.nan


def mmd_energy(a,b,gamma):
    aa=cdist(a,a,"sqeuclidean"); bb=cdist(b,b,"sqeuclidean"); ab=cdist(a,b,"sqeuclidean")
    mmd=max(0.,float(np.exp(-gamma*aa).mean()+np.exp(-gamma*bb).mean()-2*np.exp(-gamma*ab).mean()))
    energy=max(0.,float(2*np.sqrt(ab).mean()-np.sqrt(aa).mean()-np.sqrt(bb).mean()))
    return mmd,energy


def write_rows(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def plot_save(fig,out,name):
    fig.tight_layout()
    for ext in ("pdf","png"):
        fig.savefig(out/f"{name}.{ext}",dpi=300,bbox_inches="tight")
    plt.close(fig)


def relation_stage(rows,ref,query,neigh,args,out,rng):
    """E01-E05 and E10: no fitted future mapping."""
    details=[]; triple=[]; null=[]
    features=("older","high","fft_high","diff","second_diff","slope","variance",
              "level","shock","phase","weekphase","periodphase","other")
    for qi,(idx,dx) in enumerate(neigh):
        if len(idx)<args.K:continue
        for rank,j in enumerate(idx):
            hist=ref["x"][j]; x=query["x"][qi]
            z=ref["z"][j]; zq=query["z"][qi]
            h=ref["high"][j]; hq=query["high"][qi]
            phase=min(abs(int(query["phase"][qi]-ref["phase"][j])),24-abs(int(query["phase"][qi]-ref["phase"][j])))/12.
            week=min(abs(int(query["weekphase"][qi]-ref["weekphase"][j])),
                     168-abs(int(query["weekphase"][qi]-ref["weekphase"][j])))/84.
            period=max(int(args.main_period),1)
            periodphase=min(abs(int(query["periodphase"][qi]-ref["periodphase"][j])),
                            period-abs(int(query["periodphase"][qi]-ref["periodphase"][j])))/max(period/2,1)
            dother=float(dist(query["other"][qi],ref["other"][j])) if query["other"].shape[1] else np.nan
            details.append(dict(q=qi,j=int(j),start=int(query["start"][qi]),dx=float(dx[rank]),
                dx_window=float(dist(query["x_window"][qi],ref["x_window"][j])),
                dy_raw=float(dist(query["y"][qi],ref["y"][j])),
                dy_change=float(dist(query["dy_change"][qi],ref["dy_change"][j])),
                dy_increment=float(dist(query["dy_increment"][qi],ref["dy_increment"][j])),
                older=float(dist(zq,z)),high=float(dist(hq,h)),
                fft_high=abs(float(query["fft_high"][qi]-ref["fft_high"][j])),
                diff=abs(float(query["diff"][qi]-ref["diff"][j])),
                second_diff=abs(float(query["second_diff"][qi]-ref["second_diff"][j])),
                slope=abs(float(query["slope"][qi]-ref["slope"][j])),
                variance=abs(float(query["variance"][qi]-ref["variance"][j])),
                level=abs(float(query["level"][qi]-ref["level"][j])),
                shock=abs(float(query["shock"][qi]-ref["shock"][j])),
                phase=phase,weekphase=week,periodphase=periodphase,other=dother))
        # Predefine a DX-equated triplet, using no future or feature Z.
        candidates=[(a,b) for a in range(len(idx)) for b in range(a+1,len(idx))
                    if abs(dx[a]-dx[b]) <= .1*max(dx[a],dx[b],1e-5)]
        if candidates:
            a,b=min(candidates,key=lambda ab:(abs(dx[ab[0]]-dx[ab[1]]),ab[0],ab[1]))
            j,k=idx[a],idx[b]
            base=dict(q=qi,start=int(query["start"][qi]),j=int(j),k=int(k),
                      dy_j=float(dist(query["y"][qi],ref["y"][j])),
                      dy_k=float(dist(query["y"][qi],ref["y"][k])),
                      dx_j=float(dx[a]),dx_k=float(dx[b]))
            for feature in features:
                if feature in ("older","high"):
                    arr=ref["z"] if feature=="older" else ref["high"]
                    qarr=query["z"] if feature=="older" else query["high"]
                    dj=float(dist(qarr[qi],arr[j])); dk=float(dist(qarr[qi],arr[k]))
                elif feature=="other":
                    if not query["other"].shape[1]:continue
                    dj=float(dist(query["other"][qi],ref["other"][j]));dk=float(dist(query["other"][qi],ref["other"][k]))
                elif feature in ("phase","weekphase","periodphase"):
                    cycle={"phase":24,"weekphase":168,"periodphase":max(int(args.main_period),1)}[feature]
                    dj=min(abs(int(query[feature][qi]-ref[feature][j])),cycle-abs(int(query[feature][qi]-ref[feature][j])))
                    dk=min(abs(int(query[feature][qi]-ref[feature][k])),cycle-abs(int(query[feature][qi]-ref[feature][k])))
                else:
                    dj=abs(float(query[feature][qi]-ref[feature][j]));dk=abs(float(query[feature][qi]-ref[feature][k]))
                if abs(dj-dk)>1e-8 and abs(base["dy_j"]-base["dy_k"])>1e-8:
                    triple.append(dict(**base,feature=feature,dz_j=dj,dz_k=dk,
                                       correct=int((dj<dk)==(base["dy_j"]<base["dy_k"])),
                                       x_correct=int((dx[a]<dx[b])==(base["dy_j"]<base["dy_k"]))))
    write_rows(out/"E01_E05_pairs.csv",details)
    write_rows(out/"E10_triplets.csv",triple)
    if not details:return
    p=pd.DataFrame(details)
    t=pd.DataFrame(triple)
    for f in features:
        if f not in p or p[f].notna().sum()<10:continue
        for outcome in ("dy_raw","dy_change","dy_increment"):
            r=spearman(p[f],p[outcome])
            if not np.isfinite(r):continue
            ctrl=np.column_stack((np.ones(len(p)),p.dx.to_numpy(),p.phase.to_numpy()))
            xres=p[f].to_numpy()-ctrl@np.linalg.lstsq(ctrl,p[f].to_numpy(),rcond=None)[0]
            yres=p[outcome].to_numpy()-ctrl@np.linalg.lstsq(ctrl,p[outcome].to_numpy(),rcond=None)[0]
            controlled=spearman(xres,yres)
            # Explanatory permutation: swap feature values only within coarse
            # time-event blocks. No calibrated p-value without exchangeability.
            order=np.arange(len(p)); groups=p["start"].to_numpy()//max(args.L+args.H,1)
            perm=[]
            for _ in range(args.permutations):
                shuffled=p[f].to_numpy().copy()
                for g in np.unique(groups):
                    ids=np.flatnonzero(groups==g)
                    if len(ids)>1:shuffled[ids]=shuffled[rng.permutation(ids)]
                perm.append(spearman(shuffled,p[outcome].to_numpy()))
            null.extend(dict(feature=f,outcome=outcome,rep=b,rho=v) for b,v in enumerate(perm))
            tf=t[t.feature==f] if len(t) else pd.DataFrame()
            acc=float(tf.correct.mean()) if len(tf) else np.nan
            ci=time_block_boot(tf.correct,tf.start,args.L+args.H,args.seed) if len(tf) else (np.nan,np.nan,0)
            rows.append(dict(experiment="E01_E05_E10",feature=f,mapping_metric=outcome,
                valid_samples=len(p),independent_blocks=int(p.start.floordiv(args.L+args.H).nunique()),
                rho=r,controlled_rho_dx_phase=controlled,p_value=np.nan,
                ci_low=ci[0],ci_high=ci[1],effect_size=r,
                pairwise_accuracy=acc,triples=len(tf),
                null_mean=float(np.nanmean(perm)) if np.isfinite(perm).any() else np.nan,
                null_p_empirical=float((1+np.sum(np.abs(perm)>=abs(r)))/(1+len(perm)))))
    if len(p):
        use=p.dx>=max(float(p.dx.quantile(.05)),1e-8)
        sensitivity=np.sqrt(p.loc[use,"dy_raw"])/np.sqrt(p.loc[use,"dx"])
        rows.append(dict(experiment="E07_finite_difference",feature="matched_XY_pairs",
                         mapping_metric="RMS_DY_over_RMS_DX",valid_samples=int(use.sum()),
                         effect_size=float(np.median(sensitivity)),
                         condition_number_median=np.nan,
                         conclusion="no-fitted finite difference; not physical Jacobian"))
    write_rows(out/"permutation_null.csv",null)
    if len(t):
        for f in features:
            tf=t[t.feature==f]
            if len(tf):
                rows.append(dict(experiment="E10_triplet",feature=f,mapping_metric="future_closer",
                                 valid_samples=len(tf),pairwise_accuracy=float(tf.correct.mean()),
                                 x_only_accuracy=float(tf.x_correct.mean())))
    # Figure A: fixed representative rule, not maximum divergence.
    choose=p[p.dx<=p.dx.quantile(.25)]
    if len(choose):
        exemplar=choose.iloc[(choose.dy_raw-choose.dy_raw.quantile(.75)).abs().argmin()]
        qi,j=int(exemplar.q),int(exemplar.j)
        fig,axs=plt.subplots(1,3,figsize=(10,2.7))
        for ax,key,title in zip(axs,("z","x","y"),("Earlier Z","Similar recent X","Observed future Y")):
            ax.plot(query[key][qi],color="#168a62",label="query")
            ax.plot(ref[key][j],color="#d99526",label="analog")
            ax.set_title(title)
        axs[0].legend(fontsize=7);plot_save(fig,out,"Figure_A_history_future")
        (out/"selected_example.json").write_text(json.dumps({"query_start":int(query["start"][qi]),
            "reference_start":int(ref["start"][j]),"rule":"bottom-quartile DX, nearest 75th-percentile DY",
            "dx":float(exemplar.dx),"dy":float(exemplar.dy_raw)},indent=2))
    fig,ax=plt.subplots(figsize=(5,3))
    if p.older.nunique()>5:
        p["dx_bin"]=pd.qcut(p.dx,3,duplicates="drop")
        for i,(_,sub) in enumerate(p.groupby("dx_bin",observed=True)):
            if sub.older.nunique()<5:continue
            sub=sub.copy();sub["dz_bin"]=pd.qcut(sub.older,5,duplicates="drop")
            grouped=sub.groupby("dz_bin",observed=True).agg(dz=("older","median"),dy=("dy_raw","mean"))
            ax.plot(grouped.dz,grouped.dy,marker="o",label=f"DX stratum {i+1}")
        ax.legend(fontsize=7)
    ax.set(xlabel="Earlier-history MSE DZ",ylabel="Mean future MSE DY")
    plot_save(fig,out,"Figure_B_DZ_DY")
    fig,ax=plt.subplots(figsize=(5,3))
    chosen=[r for r in rows if r.get("experiment")=="E01_E05_E10" and r.get("feature")=="older" and r.get("mapping_metric")=="dy_raw"]
    if chosen:
        vals=[n["rho"] for n in null if n["feature"]=="older" and n["outcome"]=="dy_raw"]
        ax.hist(vals,bins=30,color="#aab3bc");ax.axvline(chosen[0]["rho"],color="#d99526",label="observed");ax.legend()
    ax.set(xlabel="Spearman rho",ylabel="Block-constrained permutation count")
    plot_save(fig,out,"Figure_C_permutation")


def mapping_stage(rows,train,val,test,args,out,rng):
    """E06-E09/E11 on frozen early/late TRAIN analog banks; E12 causal replay."""
    span=2*args.L+args.H
    starts=train["start"]
    mid=int(np.median(starts))
    ai=np.flatnonzero(starts+span<mid); bi=np.flatnonzero(starts>mid+span)
    if min(len(ai),len(bi))<args.K:
        rows.append(dict(experiment="E06_E11",conclusion="insufficient purged TRAIN group support"));return
    A={k:v[ai] for k,v in train.items()};B={k:v[bi] for k,v in train.items()}
    # TRAIN-only common-support thresholds, cross-block analog distances.
    support_a=np.quantile(np.min(cdist(B["x"],A["x"])/np.sqrt(args.L),axis=1),.9)
    support_b=np.quantile(np.min(cdist(A["x"],B["x"])/np.sqrt(args.L),axis=1),.9)
    support=max(support_a,support_b)
    all_y=train["y"][:,::max(1,args.H//16)]
    bandwidth=max(float(np.median(pdist(all_y[::max(1,len(all_y)//300)]))),1e-5)
    gamma=1/(2*bandwidth**2)
    pca=PCA(n_components=min(6,args.L,args.K-2),random_state=args.seed).fit(train["x"])

    def evaluate(qset,split):
        saved=[];neighbor_a=[];neighbor_b=[];zmean_a=[];zmean_b=[]
        for qi in range(len(qset["start"])):
            x=qset["x"][qi]; y=qset["y"][qi]
            da=dist(A["x"],x); db=dist(B["x"],x)
            ja=np.argsort(da)[:args.K]; jb=np.argsort(db)[:args.K]
            if max(da[ja[-1]],db[jb[-1]])>support*2:continue
            if abs(float(np.mean(da[ja])-np.mean(db[jb])))>max(.5*support,1e-5):continue
            neighbor_a.append(ja);neighbor_b.append(jb)
            zmean_a.append(A["z"][ja].mean(0));zmean_b.append(B["z"][jb].mean(0))
            ya,yb=A["y"][ja],B["y"][jb]
            fa,fb=ya.mean(0),yb.mean(0)
            dz_a=dist(A["z"][ja],qset["z"][qi]).mean(); dz_b=dist(B["z"][jb],qset["z"][qi]).mean()
            select_a=bool(dz_a<=dz_b)
            x_select_a=bool(da[ja].mean()<=db[jb].mean())
            # Phase-only control: same definitions for both groups.
            pa=np.mean(np.abs(A["phase"][ja]-qset["phase"][qi]));pb=np.mean(np.abs(B["phase"][jb]-qset["phase"][qi]))
            phase_a=bool(pa<=pb)
            ea=float(np.mean((y-fa)**2));eb=float(np.mean((y-fb)**2))
            mmd,energy=mmd_energy(ya[:,::max(1,args.H//16)],yb[:,::max(1,args.H//16)],gamma)
            mmd_half,_=mmd_energy(ya[:,::max(1,args.H//16)],yb[:,::max(1,args.H//16)],gamma*.5)
            mmd_double,_=mmd_energy(ya[:,::max(1,args.H//16)],yb[:,::max(1,args.H//16)],gamma*2)
            # Local low-dimensional ridge; mark ill-conditioned estimates.
            xa=pca.transform(A["x"][ja]);xb=pca.transform(B["x"][jb])
            ca=np.linalg.cond(xa-xa.mean(0));cb=np.linalg.cond(xb-xb.mean(0))
            reg=.1
            def fitj(xx,yy):
                center=xx.mean(0);target=yy.mean(0); mat=xx-center
                return np.linalg.solve(mat.T@mat+reg*np.eye(mat.shape[1]),mat.T@(yy-target))
            J1,J2=fitj(xa,ya),fitj(xb,yb)
            sv1=np.linalg.svd(J1,full_matrices=False)[1][0]
            sv2=np.linalg.svd(J2,full_matrices=False)[1][0]
            saved.append(dict(q=qi,start=int(qset["start"][qi]),dx_a=float(da[ja].mean()),dx_b=float(db[jb].mean()),
                dz_a=float(dz_a),dz_b=float(dz_b),select_a=select_a,x_select_a=x_select_a,phase_a=phase_a,
                mse_a=ea,mse_b=eb,mse_match=ea if select_a else eb,mse_swap=eb if select_a else ea,
                mse_x=ea if x_select_a else eb,mse_phase=ea if phase_a else eb,
                mse_random=ea if rng.integers(2)==0 else eb,d_map=float(np.mean((fa-fb)**2)),
                d_cov=float(np.mean((ya.var(0)-yb.var(0))**2)),mmd=mmd,mmd_half=mmd_half,
                mmd_double=mmd_double,energy=energy,
                j_gap=float(np.linalg.norm(J1-J2)),j_sv_gap=float(abs(sv1-sv2)),
                cond_a=float(ca),cond_b=float(cb),
                var_a=float(ya.var(0).mean()),var_b=float(yb.var(0).mean()),
                quantile_gap=float(np.mean(np.abs(np.quantile(ya,[.1,.9],axis=0)-np.quantile(yb,[.1,.9],axis=0))))))
        write_rows(out/f"E06_E11_{split}.csv",saved)
        return pd.DataFrame(saved),neighbor_a,neighbor_b,np.asarray(zmean_a),np.asarray(zmean_b)

    V,_,_,_,_=evaluate(val,"val")
    T,ja_test,jb_test,za_test,zb_test=evaluate(test,"test")
    if V.empty or T.empty:
        rows.append(dict(experiment="E06_E11",conclusion="no common-support VAL or TEST queries"));return
    best_const_a=bool(V.mse_a.mean()<=V.mse_b.mean())
    # P1/P4: permute actual query Z (not the already-computed mapping choice).
    # P3: break reference X/Y pairs; clearly labeled nonstationary control.
    znull=[];blocknull=[];yshuffle=[]
    tq=T.q.to_numpy().astype(int)
    qz=test["z"][tq]
    phase=test["phase"][tq]
    event=test["start"][tq]//span
    for rep in range(args.permutations):
        permq=np.arange(len(T))
        for h in np.unique(phase):
            ids=np.flatnonzero(phase==h)
            if len(ids)>1:permq[ids]=rng.permutation(ids)
        select=np.mean((qz[permq]-za_test)**2,axis=1)<=np.mean((qz[permq]-zb_test)**2,axis=1)
        znull.append(float(np.where(select,T.mse_a,T.mse_b).mean()))
        # Circular event-block shift preserves within-block ordering but does
        # not establish a valid exchangeable null for nonstationary data.
        events=np.unique(event); shift=int(rng.integers(1,max(2,len(events))))
        shifted=qz.copy()
        if len(events)>1:
            for k,e in enumerate(events):
                dst=np.flatnonzero(event==e);src=np.flatnonzero(event==events[(k+shift)%len(events)])
                shifted[dst]=qz[np.resize(src,len(dst))]
        select2=np.mean((shifted-za_test)**2,axis=1)<=np.mean((shifted-zb_test)**2,axis=1)
        blocknull.append(float(np.where(select2,T.mse_a,T.mse_b).mean()))
        perm_a=rng.permutation(len(A["y"]));perm_b=rng.permutation(len(B["y"]))
        fakea=np.asarray([A["y"][perm_a[ids]].mean(0) for ids in ja_test])
        fakeb=np.asarray([B["y"][perm_b[ids]].mean(0) for ids in jb_test])
        fake=np.where(T.select_a.to_numpy()[:,None],fakea,fakeb)
        yshuffle.append(float(np.mean((test["y"][tq]-fake)**2)))
    write_rows(out/"E11_permutation.csv",[dict(rep=i,mse_permuted_Z=v,
        mse_block_shift_Z=blocknull[i],mse_shuffled_reference_Y=yshuffle[i]) for i,v in enumerate(znull)])
    mse_const=T.mse_a.mean() if best_const_a else T.mse_b.mean()
    match=T.mse_match.to_numpy();swap=T.mse_swap.to_numpy()
    gain=swap-match;ci=time_block_boot(gain,T.start,span,args.seed)
    rows.append(dict(experiment="E11_mapping_swap",feature="earlier_Z",mapping_metric="future_MSE",
        valid_samples=len(T),independent_blocks=ci[2],coverage=len(T)/max(1,len(test["start"])),
        mse_match=float(match.mean()),mse_swap=float(swap.mean()),mse_permuted=float(np.mean(znull)),
        mse_x_only=float(T.mse_x.mean()),mse_time_only=float(T.mse_phase.mean()),
        mse_best_constant=float(mse_const),mse_random=float(T.mse_random.mean()),
        selected_A_fraction=float(T.select_a.mean()),constant_A_selected=best_const_a,
        shuffled_reference_mse=float(np.mean(yshuffle)),block_shift_Z_mse=float(np.mean(blocknull)),
        gain_vs_baseline=float(mse_const-match.mean()),effect_size=float(gain.mean()),
        ci_low=ci[0],ci_high=ci[1],swap_win_rate=float(np.mean(gain>0)),
        p_value=np.nan,null_p_empirical=float((1+np.sum(np.asarray(znull)<=match.mean()))/(1+len(znull))),
        conclusion=("Z routing collapsed to one mapping; not adaptive evidence" if T.select_a.mean()<.05 or T.select_a.mean()>.95
                    else "candidate adaptive gain; requires independent replication" if match.mean()<min(mse_const,np.mean(znull))
                    else "no gain over constant/permuted control")))
    for exp,key in (("E06_local_mean","d_map"),("E07_local_jacobian","j_gap"),
                    ("E08_conditional_distribution","mmd"),("E08_conditional_energy","energy"),
                    ("E09_uncertainty","d_cov")):
        valarr=T[key].to_numpy();ci2=time_block_boot(valarr,T.start,span,args.seed)
        rows.append(dict(experiment=exp,feature="TRAIN_early_vs_late",mapping_metric=key,
            valid_samples=len(T),independent_blocks=ci2[2],coverage=len(T)/max(1,len(test["start"])),
            effect_size=float(valarr.mean()),ci_low=ci2[0],ci_high=ci2[1],
            condition_number_median=float(np.median(np.maximum(T.cond_a,T.cond_b))) if key=="j_gap" else np.nan,
            conclusion="local J unreliable" if key=="j_gap" and np.median(np.maximum(T.cond_a,T.cond_b))>1e5 else "descriptive only"))

    fig,ax=plt.subplots(figsize=(6,3))
    for bank,idx,color,label in ((A,ja_test[0],"#168a62","TRAIN early A"),
                                  (B,jb_test[0],"#d99526","TRAIN late B")):
        ys=bank["y"][idx]
        local_rng=np.random.default_rng(args.seed)
        boot=np.asarray([ys[local_rng.integers(len(ys),size=len(ys))].mean(0) for _ in range(200)])
        ax.plot(ys.mean(0),color=color,label=label)
        ax.fill_between(np.arange(args.H),*np.quantile(boot,[.025,.975],axis=0),color=color,alpha=.15)
    ax.set(xlabel="Future step",ylabel="TRAIN-standardized value");ax.legend();plot_save(fig,out,"Figure_D_local_mapping")
    fig,ax=plt.subplots(figsize=(6,3))
    ax.bar(["Z matched","swapped","permuted Z","best constant","X only"],
           [match.mean(),swap.mean(),np.mean(znull),mse_const,T.mse_x.mean()],color=["#168a62","#d99526","#aab3bc","#6e7b89","#82a7ba"])
    ax.tick_params(axis="x",rotation=20);ax.set(ylabel="TEST future MSE")
    plot_save(fig,out,"Figure_E_mapping_swap")
    fig,ax=plt.subplots(figsize=(5,3))
    ax.scatter(T.d_map,T.j_gap,s=8,alpha=.3,color="#168a62")
    ax.set(xlabel="Local future mean gap",ylabel="Projected local linear J gap")
    plot_save(fig,out,"Figure_F_local_linear")
    fig,ax=plt.subplots(figsize=(5,3))
    ax.scatter(T.mmd,T.energy,s=8,alpha=.3,color="#d99526")
    ax.set(xlabel="Neighborhood MMD² (fixed TRAIN bandwidth)",ylabel="Energy distance")
    plot_save(fig,out,"Figure_G_conditional_distribution")

    # E12: rolling reference includes only fully observed prior TEST windows.
    combined={k:np.concatenate((train[k],test[k]),axis=0) for k in train}
    risk=[]
    ntrain=len(train["start"])
    for qi in range(len(test["start"])):
        qstart=int(test["start"][qi]); legal=np.flatnonzero(combined["start"]+span<=qstart)
        if len(legal)<4*args.K:continue
        order=legal[np.argsort(combined["start"][legal])]
        old=order[:len(order)//2]; recent=order[len(order)//2:]
        if min(len(old),len(recent))<args.K:continue
        d1=dist(combined["x"][old],test["x"][qi]);d2=dist(combined["x"][recent],test["x"][qi])
        ja=old[np.argsort(d1)[:args.K]];jb=recent[np.argsort(d2)[:args.K]]
        fa=combined["y"][ja].mean(0);fb=combined["y"][jb].mean(0)
        s=float(np.mean((fa-fb)**2));r=float(np.mean((test["y"][qi]-fa)**2))
        risk.append(dict(start=qstart,signal=s,old_mapping_error=r,
                         dx_old=float(d1.min()),dx_recent=float(d2.min()),
                         history_variance=float(test["x"][qi].var())))
    write_rows(out/"E12_rolling_risk.csv",risk)
    if len(risk)>10:
        R=pd.DataFrame(risk);median=float(R.old_mapping_error.median())
        auc=roc_auc_score(R.old_mapping_error>median,R.signal) if R.signal.nunique()>1 else np.nan
        controls=np.column_stack((np.ones(len(R)),R.dx_old,R.dx_recent,R.history_variance,
                                  (R.start-R.start.mean())/max(R.start.std(),1.)))
        sres=R.signal.to_numpy()-controls@np.linalg.lstsq(controls,R.signal.to_numpy(),rcond=None)[0]
        rres=R.old_mapping_error.to_numpy()-controls@np.linalg.lstsq(controls,R.old_mapping_error.to_numpy(),rcond=None)[0]
        rows.append(dict(experiment="E12_causal_early_warning",feature="past_mapping_disagreement",
            mapping_metric="later_old_mapping_error",valid_samples=len(R),
            independent_blocks=int(R.start.floordiv(span).nunique()),
            rho=spearman(R.signal,R.old_mapping_error),controlled_rho=spearman(sres,rres),auc=auc,
            p_value=np.nan,conclusion="association only, not detected concept drift"))
        fig,ax=plt.subplots(figsize=(7,3))
        ax.plot(R.start,R.signal,label="past mapping disagreement",color="#168a62")
        ax.plot(R.start,R.old_mapping_error,label="later observed old-map error",color="#d99526",alpha=.7)
        ax.set(xlabel="Forecast origin",ylabel="Squared standardized units");ax.legend(fontsize=7)
        plot_save(fig,out,"Figure_H_mapping_risk")


def run(args):
    root=Path(args.project_root); out=root/"results/history_future_mapping_validation"/f"{args.dataset}_{args.channel}_L{args.L}_H{args.H}_K{args.K}_seed{args.seed}"
    out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(root/"dataset/all_datasets"/SOURCE[args.dataset])
    dates=pd.DatetimeIndex(pd.to_datetime(df.iloc[:,0]))
    intervals=bounds(args.dataset,len(df));ch=choose_channel(df,args.dataset,intervals[0][1],args.channel)
    raw=df.iloc[:,1:].to_numpy(dtype=np.float32)
    variance=raw[intervals[0][0]:intervals[0][1]].var(axis=0)
    others=[i for i in np.argsort(variance)[::-1] if i!=ch][:3]
    scaler=StandardScaler().fit(raw[intervals[0][0]:intervals[0][1]])
    raw=scaler.transform(raw).astype(np.float32)
    training_series=raw[intervals[0][0]:intervals[0][1],ch]
    power=np.abs(np.fft.rfft(training_series-training_series.mean()))**2
    freq=np.fft.rfftfreq(len(training_series))
    valid=(freq>=1/168.)&(freq<=1/6.)
    main_period=int(round(1/freq[np.flatnonzero(valid)[np.argmax(power[valid])]])) if valid.any() else 24
    args.main_period=main_period
    floor=max(float(np.std(raw[intervals[0][0]:intervals[0][1],ch])*.05),.01)
    stride=args.stride or (24 if args.dataset=="ETTh1" else 48 if args.dataset=="ETTm2" else 24)
    subsets={name:build(raw,dates,ch,others,*b,args.L,args.H,stride,floor,main_period)
             for name,b in zip(("train","val","test"),intervals)}
    train,val,test=(subsets[s] for s in ("train","val","test"))
    if min(len(train["x"]),len(val["x"]),len(test["x"]))<30:
        raise ValueError("insufficient wholly-contained windows")
    args.K=min(args.K,max(2,len(train["x"])//8))
    # Offline pair diagnosis: both ends within split and the reference future
    # must have ended before the query's preceding-Z interval begins.
    gap=2*args.L+args.H
    pairs=nearest_indices(test,test,args.K,gap)
    coverage=float(np.mean([len(i)==args.K for i,_ in pairs]))
    rows=[];rng=np.random.default_rng(args.seed)
    relation_stage(rows,test,test,pairs,args,out,rng)
    mapping_stage(rows,train,val,test,args,out,rng)
    for row in rows:
        row.update(dataset=args.dataset,channel=str(df.columns[ch+1]),L=args.L,H=args.H,K=args.K,
                   coverage=row.get("coverage",coverage),seed=args.seed,
                   p_value=row.get("p_value",np.nan),q_value=np.nan)
    write_rows(out/"summary.csv",rows)
    (out/"config.json").write_text(json.dumps(dict(dataset=args.dataset,channel=str(df.columns[ch+1]),
        L=args.L,H=args.H,K=args.K,seed=args.seed,stride=stride,train_val_test_bounds=intervals,
        counts={s:len(subsets[s]["x"]) for s in subsets},other_channels=[str(df.columns[i+1]) for i in others],
        train_dominant_period_steps=main_period,
        neighbor_rule="causal: reference Y end <= query Z start; same TEST split for pair statistics",
        mapping_banks="purged early/late TRAIN; TEST queries never in banks",
        scaler="TRAIN-only StandardScaler",permutations=args.permutations,
        permutation_interpretation="empirical control; no calibrated p without exchangeability"),indent=2))
    (out/"README.md").write_text("# History–future mapping validation\n\n"
        "All windows are real and wholly within TRAIN/VAL/TEST. Z is the 96 points before X; Y is future. "
        "TRAIN-only StandardScaler. Analog reference futures must finish before a query's Z starts. "
        "E01–E05/E10 use no fitted mapping; E06–E09/E11 use purged early/late TRAIN analog banks; "
        "E12 replays only fully observed past references. The projected local J is not a physical Jacobian. "
        "MMD is a matched-neighborhood diagnostic, not the formal NeurIPS conditional test. "
        "Permutation p-values are NA because exchangeability is not established; null_p_empirical is descriptive only. "
        "No result here proves concept drift. Output figures A–H are only produced where support exists.\n")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--project-root",default=".")
    p.add_argument("--dataset",choices=tuple(SOURCE),required=True)
    p.add_argument("--channel",default="auto")
    p.add_argument("--L",type=int,default=96);p.add_argument("--H",type=int,default=96)
    p.add_argument("--K",type=int,default=10);p.add_argument("--seed",type=int,default=2022)
    p.add_argument("--stride",type=int,default=0);p.add_argument("--permutations",type=int,default=999)
    run(p.parse_args())
