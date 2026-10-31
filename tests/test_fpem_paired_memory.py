import torch
from torch import nn
from torch.nn import functional as F

from models.predictive_env.paired_memory import PairedMemoryBank
from models.predictive_env.direct_concat import DirectConcatPredictor
from models.predictive_env.unseen_risk import (
    ConformalCalibrator, RiskScaleHead, interval_metrics, risk_features)


class DummyBackbone(nn.Module):
    def __init__(self,p,d,t):
        super().__init__()
        self.head_flatten=nn.Flatten(start_dim=-2)
        self.h_forecast_head=nn.Linear(p*d,t)

    def frozen_h_head(self,h,cycle=None):
        return self.h_forecast_head(self.head_flatten(h.permute(0,1,3,2)))

    def normalized_cycle_forecast(self,cycle,b,device):
        return None


def records(n=40,length=12,horizon=7,channels=3,p=4,d=8):
    torch.manual_seed(18)
    x=torch.randn(n,length,channels)
    y=torch.randn(n,horizon,channels)
    a0=torch.randn_like(y)
    return {"x":x,"target":y,"a0_pred":a0,
            "hidden":torch.randn(n,channels,p,d),
            "starts":torch.arange(n)*length}


def test_causal_pairing_and_no_self_future():
    r=records()
    bank=PairedMemoryBank(12,7,history_size=16)
    bank.fit(r)
    early=bank.retrieve(r["x"][:1],"all",3,2,torch.tensor([0]),"train")
    assert not early["valid"].any()
    assert torch.count_nonzero(early["hidden"])==0
    query=bank.retrieve(r["x"][5:6],"mixed",3,2,torch.tensor([60]),"train")
    assert torch.all(query["source_ends"][query["valid"]]<=60)
    assert not (query["indices"][query["valid"]]==5).any()
    original=query["future"].clone()
    r["target"][5]+=10000
    new_bank=PairedMemoryBank(12,7,16)
    new_bank.fit(r)
    again=new_bank.retrieve(r["x"][5:6],"mixed",3,2,torch.tensor([60]),"train")
    assert torch.allclose(original,again["future"])


def test_common_rare_order_and_deterministic_random():
    r=records();bank=PairedMemoryBank(12,7,16);bank.fit(r)
    x=r["x"][20:22];ids=torch.tensor([240,252])
    mixed=bank.retrieve(x,"mixed",2,2,ids,"train")
    assert torch.all(mixed["is_common"][:,:,:2] | ~mixed["valid"][:,:,:2])
    assert not mixed["is_common"][:,:,2:].any()
    left=bank.retrieve(x,"random",2,2,ids,"train")
    right=bank.retrieve(x,"random",2,2,ids,"train")
    assert torch.equal(left["indices"],right["indices"])
    assert torch.equal(left["valid"],right["valid"])


def test_val_cutoff_and_test_future_cannot_select_memory():
    r=records();bank=PairedMemoryBank(12,7,16);bank.fit(r)
    val=bank.retrieve(r["x"][20:21],"all",3,2,torch.tensor([20]),"val",200)
    assert torch.all(val["source_ends"][val["valid"]]<=200)
    test=bank.retrieve(r["x"][20:21],"all",3,2,torch.tensor([999]),"test")
    changed=r["target"].clone();changed[20]+=999
    again=bank.retrieve(r["x"][20:21],"all",3,2,torch.tensor([999]),"test")
    assert torch.equal(test["indices"],again["indices"])
    assert torch.equal(test["future"],again["future"])


def test_direct_concat_function_preserving_and_zero_residual():
    torch.manual_seed(10)
    b,c,p,d,t,k=2,3,4,8,7,5
    model=DummyBackbone(p,d,t)
    head=DirectConcatPredictor(model.h_forecast_head,p,d,t,k)
    h=torch.randn(b,c,p,d)
    memory={"hidden":torch.randn(b,c,k,p,d),"future":torch.randn(b,c,k,t),
            "residual":torch.randn(b,c,k,t),"valid":torch.ones(b,c,k,dtype=torch.bool)}
    mean=torch.zeros(b,1,c);std=torch.ones(b,1,c)
    full,point,correction=head(model,h,memory,mean,std,None,"future")
    a0=model.frozen_h_head(h).permute(0,2,1)
    assert torch.allclose(point,a0,atol=1e-7)
    assert torch.allclose(full,a0,atol=1e-7)
    assert torch.count_nonzero(correction)==0
    # New columns learn while copied A0 columns remain frozen.
    loss=F.mse_loss(full,torch.randn_like(full));loss.backward()
    g=head.hidden_head.head.weight.grad
    assert torch.count_nonzero(g[:,:p*d])==0
    assert torch.count_nonzero(g[:,p*d:])>0
    assert torch.count_nonzero(head.residual_head.head.weight.grad)>0


def test_zero_retrieval_slots_recover_a0_and_checkpoint_roundtrip():
    torch.manual_seed(11)
    b,c,p,d,t,k=1,2,1,10,5,3
    model=DummyBackbone(p,d,t)
    head=DirectConcatPredictor(model.h_forecast_head,p,d,t,k)
    h=torch.randn(b,c,p,d)
    memory={"hidden":torch.zeros(b,c,k,p,d),"future":torch.zeros(b,c,k,t),
            "residual":torch.zeros(b,c,k,t),"valid":torch.zeros(b,c,k,dtype=torch.bool)}
    mean=torch.zeros(b,1,c);std=torch.ones(b,1,c)
    prediction=head(model,h,memory,mean,std,None,"future")[0]
    assert torch.allclose(prediction,model.frozen_h_head(h).permute(0,2,1))
    restored=DirectConcatPredictor(model.h_forecast_head,p,d,t,k)
    restored.load_state_dict(head.state_dict())
    assert torch.equal(prediction,restored(model,h,memory,mean,std,None,"future")[0])


def test_k_zero_recovers_baseline():
    r=records();bank=PairedMemoryBank(12,7,16);bank.fit(r)
    retrieved=bank.retrieve(r["x"][:2],"mixed",0,0,torch.tensor([0,1]),"train")
    assert retrieved["hidden"].shape[2]==0
    model=DummyBackbone(4,8,7)
    head=DirectConcatPredictor(model.h_forecast_head,4,8,7,0)
    mean=torch.zeros(2,1,3);std=torch.ones(2,1,3)
    prediction=head(model,r["hidden"][:2],retrieved,mean,std,None,"none")[0]
    assert torch.allclose(prediction,model.frozen_h_head(r["hidden"][:2]).permute(0,2,1))


def test_risk_features_and_interval_calibration():
    r=records();bank=PairedMemoryBank(12,7,16);bank.fit(r)
    batch=bank.retrieve(r["x"][20:22],"mixed",2,2,torch.tensor([240,252]),"train")
    features=risk_features(batch)
    assert features.shape==(2,3,5)
    assert torch.isfinite(features).all()
    net=RiskScaleHead(6)
    scale=net(torch.cat((features,torch.zeros(2,3,1)),-1))
    pred=torch.randn(2,7,3);truth=torch.randn_like(pred)
    calibrator=ConformalCalibrator();calibrator.fit(pred,truth,scale)
    lo,hi=calibrator.interval(pred,scale)
    result=interval_metrics(pred,truth,scale,lo,hi)
    assert 0<=result["picp95_trajectory"]<=1
    assert result["mpiw95"]>0
