"""Lightweight relation-driven K=8 grouping, not causal-domain discovery.

G1..G6 are conditional mapping mixtures. G7/G8 cluster signed parameter
update tensors, never scalar MSE. All operations here are CPU NumPy/sklearn.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

K = 8


def weighted_linear(x, y, weights, alpha):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    w = np.maximum(np.asarray(weights, np.float64), 0)
    if w.sum() < 1e-8:
        return np.zeros((x.shape[1] + 1, y.shape[1]), np.float64)
    design = np.column_stack((x, np.ones(len(x))))
    gram = design.T @ (w[:, None] * design)
    penalty = np.eye(design.shape[1]) * alpha
    penalty[-1, -1] = alpha * .01
    rhs = design.T @ (w[:, None] * y)
    try:
        return np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram + penalty, rhs, rcond=1e-8)[0]


def predict_linear(x, coefficients):
    return np.column_stack((x, np.ones(len(x)))) @ coefficients


def posterior(x, y, models, priors, variance, temperature=1.0):
    predictions = np.stack([predict_linear(x, model) for model in models], axis=1)
    error = (y[:, None, :] - predictions) ** 2
    variance = np.clip(np.asarray(variance, np.float64), 1e-3, 1e3)
    if variance.ndim == 1:
        variance = variance[:, None]
    logp = -.5 * (error / variance[None]).sum(axis=2) - .5 * np.log(variance[None]).sum(axis=2)
    logp = logp / temperature + np.log(np.maximum(priors, 1e-5))[None]
    return np.exp(logp - logsumexp(logp, axis=1, keepdims=True)).astype(np.float32)


def _rff(x, center, scale, seed, width=32):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(x.shape[1], width)) / np.sqrt(x.shape[1])
    bias = rng.uniform(0, 2*np.pi, size=width)
    return np.sqrt(2/width)*np.cos(((x-center)/scale) @ matrix + bias), matrix, bias


def make_rff(train, values, seed=2022):
    center = np.mean(train, axis=0)
    scale = np.std(train, axis=0).clip(min=.1)
    _, matrix, bias = _rff(train, center, scale, seed)
    return {name: (np.sqrt(2/matrix.shape[1]) * np.cos(((value-center)/scale) @ matrix+bias)).astype(np.float32)
            for name,value in values.items()}


def _initialize(x, y, seed):
    global_model = weighted_linear(x, y, np.ones(len(x)), 10.)
    residual = y-predict_linear(x, global_model)
    # Directional relation fingerprint initializes the mixture without
    # seeding from historical feature clusters or scalar prediction errors.
    fingerprint = np.einsum("ni,nj->nij", residual, np.column_stack((x,np.ones(len(x))))).reshape(len(x),-1)
    components = min(8, len(x)-1, fingerprint.shape[1])
    pca = PCA(n_components=components, svd_solver="randomized",random_state=seed)
    projected = pca.fit_transform(fingerprint)
    norms = np.linalg.norm(projected,axis=1,keepdims=True).clip(min=1e-6)
    projected = projected/norms
    labels = MiniBatchKMeans(n_clusters=K,random_state=seed,n_init=3,batch_size=256).fit_predict(projected)
    q = np.eye(K,dtype=np.float32)[labels]*.85 + .15/K
    return q, global_model


def fit_mixture(method, x, y, *, seed=2022, iterations=10, alpha=10.0):
    """Fixed-budget EM variants; returns model and A posterior."""
    start=time.perf_counter()
    x=np.asarray(x,np.float64);y=np.asarray(y,np.float64)
    if len(x)<K*4: raise ValueError("too few training rows for K=8")
    q, shared = _initialize(x,y,seed)
    shared_var = np.mean((y-predict_linear(x,shared))**2,axis=0).clip(min=.01)
    models = [shared.copy() for _ in range(K)]
    variance=np.repeat(shared_var[None],K,axis=0)
    priors=np.ones(K)/K
    if method in ("G3_KerHRM_lite","G4_ANM_MM_lite"):
        raise ValueError("pass fixed RFF features to fit_mixture, not raw z")
    previous_q=q.copy()
    for iteration in range(iterations):
        for j in range(K):
            if method=="G2_HRM_lite":
                delta=weighted_linear(x,y-predict_linear(x,shared),q[:,j],alpha*10)
                models[j]=shared+delta
            elif method=="G5_SSCM_lite":
                delta=weighted_linear(x,y-predict_linear(x,shared),q[:,j],alpha*5)
                # Sparse group-specific deviation from one shared map.
                threshold=.02*np.maximum(np.std(delta[:-1]),1e-6)
                delta[:-1]=np.sign(delta[:-1])*np.maximum(np.abs(delta[:-1])-threshold,0)
                models[j]=shared+delta
            elif method=="G6_CCSL_proxy":
                full=weighted_linear(x,y,q[:,j],alpha)
                # Sparse *full* support pattern, not shared-map deviations.
                magnitudes=np.abs(full[:-1])
                cut=np.quantile(magnitudes,.7)
                full[:-1][magnitudes<cut]=0
                models[j]=full
            else:
                models[j]=weighted_linear(x,y,q[:,j],alpha)
        if method=="G4_ANM_MM_lite":
            for j in range(K):
                error=(y-predict_linear(x,models[j]))**2
                variance[j]=(np.sum(q[:,j,None]*error,axis=0)+10*shared_var)/(q[:,j].sum()+10)
            variance=np.clip(variance,shared_var[None]*.1,shared_var[None]*10)
        else:
            pooled=np.zeros_like(shared_var)
            for j in range(K):
                pooled+=np.sum(q[:,j,None]*(y-predict_linear(x,models[j]))**2,axis=0)
            pooled=(pooled/max(len(x),1)).clip(min=.01)
            variance[:]=pooled
        priors=(q.sum(axis=0)+2)/(len(q)+2*K)
        q=posterior(x,y,models,priors,variance,temperature=1.5)
        if not np.isfinite(q).all(): raise FloatingPointError("non-finite conditional posterior")
        previous_q=q
    matrix=np.stack(models)
    supports=np.abs(matrix[:,:-1])>.02
    support_difference=float(np.mean([np.mean(supports[i]!=supports[j])
                                    for i in range(K) for j in range(i+1,K)]))
    return {"models":matrix,"variance":variance,"priors":priors,"shared":shared,
            "q_train":previous_q,"fit_seconds":time.perf_counter()-start,
            "parameter_count":int(matrix.size+variance.size+priors.size),
            "support_difference":support_difference,"feature_definition":method}


def fit_rff_mixture(method, features, y, *, seed=2022, iterations=10):
    """G3 shared Gaussian noise, G4 group-specific shrunk diagonal noise."""
    base = "G4_ANM_MM_lite" if method=="G4_ANM_MM_lite" else "G3_KerHRM_lite"
    x=np.asarray(features,np.float64);y=np.asarray(y,np.float64)
    q,shared=_initialize(x,y,seed)
    global_var=np.mean((y-predict_linear(x,shared))**2,axis=0).clip(min=.01)
    variance=np.repeat(global_var[None],K,axis=0)
    models=[shared.copy() for _ in range(K)]
    prior=np.ones(K)/K
    start=time.perf_counter()
    for _ in range(iterations):
        for j in range(K):models[j]=weighted_linear(x,y,q[:,j],20.)
        if base=="G4_ANM_MM_lite":
            for j in range(K):
                e=(y-predict_linear(x,models[j]))**2
                variance[j]=(np.sum(q[:,j,None]*e,axis=0)+10*global_var)/(q[:,j].sum()+10)
            variance=np.clip(variance,global_var[None]*.1,global_var[None]*10)
        else:
            pooled=sum(np.sum(q[:,j,None]*(y-predict_linear(x,models[j]))**2,axis=0) for j in range(K))/len(x)
            variance[:]=np.maximum(pooled,.01)
        prior=(q.sum(axis=0)+2)/(len(q)+2*K)
        q=posterior(x,y,models,prior,variance,temperature=1.5)
    matrix=np.stack(models)
    return {"models":matrix,"variance":variance,"priors":prior,"shared":shared,
            "q_train":q,"fit_seconds":time.perf_counter()-start,
            "parameter_count":int(matrix.size+variance.size+prior.size),
            "support_difference":None,"feature_definition":method}


def fit_fingerprint(x, y, reference, *, seed=2022, signed_only=False):
    """Outer-product signed function update; reference is frozen Ridge or A0."""
    x=np.asarray(x,np.float64);y=np.asarray(y,np.float64);reference=np.asarray(reference,np.float64)
    design=np.column_stack((x,np.ones(len(x))))
    delta=np.einsum("ni,nj->nij",y-reference,design).reshape(len(x),-1)
    denom=(10+np.sum(design*design,axis=1))[:,None]
    delta=delta/denom
    if signed_only:delta=delta/np.linalg.norm(delta,axis=1,keepdims=True).clip(min=1e-6)
    return delta.astype(np.float32)


def fit_fingerprint_cluster(matrix, *, seed=2022):
    start=time.perf_counter()
    components=min(16,len(matrix)-1,matrix.shape[1])
    reducer=PCA(n_components=components,svd_solver="randomized",random_state=seed)
    projected=reducer.fit_transform(matrix)
    scale=np.std(projected,axis=0).clip(min=1e-6)
    projected=projected/scale
    km=MiniBatchKMeans(n_clusters=K,random_state=seed,n_init=3,batch_size=256).fit(projected)
    distance=((projected[:,None]-km.cluster_centers_[None])**2).mean(axis=2)
    temperature=np.median(distance.min(axis=1))+.05
    logits=-distance/temperature
    q=np.exp(logits-logsumexp(logits,axis=1,keepdims=True))
    return {"reducer":reducer,"scale":scale,"centers":km.cluster_centers_,
            "temperature":temperature,"q_train":q.astype(np.float32),
            "fit_seconds":time.perf_counter()-start,
            "parameter_count":int(reducer.components_.size+km.cluster_centers_.size),
            "feature_definition":"signed outer-product prediction update"}


def fingerprint_posterior(model, matrix):
    projected=model["reducer"].transform(matrix)/model["scale"]
    d=((projected[:,None]-model["centers"][None])**2).mean(axis=2)
    logits=-d/model["temperature"]
    return np.exp(logits-logsumexp(logits,axis=1,keepdims=True)).astype(np.float32)
