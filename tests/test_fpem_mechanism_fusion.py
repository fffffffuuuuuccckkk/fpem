import json

import numpy as np
import pytest

from models.predictive_env.mechanism_fusion import (FrozenGrouping, candidates,
                                                    fit_full_future_experts,
                                                    independent_support)
from tools.run_fpem_mechanism_fusion_flow import build_summary, mode_features, split_four_purged


def test_artifact_is_read_only_and_checks_identity(tmp_path):
    q=np.full((4,8),1/8,np.float32)
    np.savez_compressed(tmp_path/"assignments_train.npz",sample_id=np.array(
        ["E:TRAIN:0:0","E:TRAIN:8:0","E:TRAIN:16:0","E:TRAIN:24:0"]),
        q_teacher=q,time_block=np.array(["A","A","B1","B2"]),
        window_start=np.array([0,8,16,24]),channel=np.zeros(4,np.int16))
    (tmp_path/"prototype_meta.json").write_text(json.dumps({"A0_sha256":"abc",
        "dataset":"E","pred_len":96,"seed":2022,"K":8}))
    before=(tmp_path/"assignments_train.npz").read_bytes()
    grouping=FrozenGrouping(tmp_path,a0_sha="abc",dataset="E",horizon=96)
    joined=grouping.join("E",np.array([0,8,16,24]),np.zeros(4,int))
    assert [len(joined[b][0]) for b in ("A","B1","B2")]==[2,1,1]
    assert (tmp_path/"assignments_train.npz").read_bytes()==before
    with pytest.raises(ValueError):FrozenGrouping(tmp_path,a0_sha="wrong",dataset="E",horizon=96)


def test_full_future_experts_and_null_support():
    rng=np.random.default_rng(5);x=rng.normal(size=(64,4)).astype(np.float32)
    y=np.stack((x[:,0]+.2*x[:,1],x[:,2]-x[:,3]),1)
    q=np.eye(8,dtype=np.float32)[np.arange(64)%8]
    fit=fit_full_future_experts(x,y,q,alpha=1.,rank=2)
    output=candidates(fit,x)
    assert output.shape==(64,8,2) and np.isfinite(output).all()
    assert independent_support(np.arange(64)*2,q,span=8).shape==(8,)


def test_oof_summary_and_features_never_receive_current_target():
    starts=np.arange(20)*20
    blocks=split_four_purged(starts,span=20)
    for first,second in zip(blocks[:-1],blocks[1:]):
        assert max(starts[first]+20)<=min(starts[second])
    first=blocks[0]
    residual=np.ones((len(first),32),np.float32)
    q=np.full((len(first),8),1/8,np.float32)
    local,_,support=build_summary(residual,q,starts[first],20)
    query=np.full((3,8),1/8,np.float32)
    candidate=np.zeros((3,8,32),np.float32)
    point=np.zeros((3,32),np.float32)
    rho=np.ones((8,8),np.float32)
    a=mode_features(query,candidate,point,rho,local,support,"U2_oof_residual_summary")
    b=mode_features(query,candidate,point,rho,local,support,"U0_original_condition")
    assert a.shape==(3,32,14) and np.isfinite(a).all() and np.all(b==0)
