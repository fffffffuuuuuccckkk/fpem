import math
import tempfile

import pytest
import torch

from models.predictive_env.csif import CommonSpecificFactors, ConsensusLatentFusion
from models.predictive_env.conditional_innovation_prior import (
    ConditionalFutureInnovationPrior, InnovationConditionAdapter,
    strict_chronological_indices,
)


@pytest.mark.parametrize("shape", [(1,1,1,8), (2,3,5,12), (4,1,3,16)])
def test_orthogonal_reconstruction_and_initial_function(shape):
    torch.manual_seed(8)
    h = torch.randn(shape, requires_grad=True)
    model = CommonSpecificFactors(shape[-1], mode="memory")
    fusion = ConsensusLatentFusion(shape[-1], True)
    c, v, s, a, d = model(h)
    mixed, alpha = fusion(c, v, s)
    assert c.shape == v.shape == h.shape
    assert s.shape == (*shape[:-1],1)
    assert (c+v-h).abs().max() < 1e-5
    assert (mixed-h).abs().max() < 1e-3
    w = model.basis()
    assert (w.T@w-torch.eye(shape[-1])).abs().max() < 2e-5
    ((mixed*h).mean() + c.square().mean()).backward()
    assert model.reflection_vectors.grad is not None
    assert torch.isfinite(model.reflection_vectors.grad).all()
    assert fusion.gate[-1].weight.grad.abs().sum() > 0


def test_train_anchor_once_eval_frozen_and_checkpoint():
    model = CommonSpecificFactors(8, mode="memory", beta=.9)
    h = torch.randn(3,2,1,8)
    a = model(h)[3]
    model.train()
    model.observe_train_batch(a, torch.tensor([0,1,96]), 96)
    p_before = model.p_ema.clone()
    assert torch.equal(p_before, model.p_ema)
    info = model.finish_train_epoch()
    assert info["independent_anchor_groups"] == 4
    saved = model.state_dict()
    restored = CommonSpecificFactors(8, mode="memory")
    restored.load_state_dict(saved)
    assert torch.equal(restored.p_ema, model.p_ema)
    assert torch.equal(restored.scale_ema, model.scale_ema)
    model.eval()
    with pytest.raises(RuntimeError):
        model.observe_train_batch(a, torch.tensor([0,1,96]), 96)
    assert torch.equal(model.p_ema, restored.p_ema)


def test_dictionary_and_fixed_fusion_recover_hidden():
    h = torch.randn(2,1,3,8)
    f = CommonSpecificFactors(8, mode="dictionary")
    c,v,s,_,_ = f(h)
    fused,alpha = ConsensusLatentFusion(8,False)(c,v,s)
    assert (fused-h).abs().max() < 1e-6
    assert torch.equal(alpha,torch.full_like(alpha,.5))


def test_prior_and_adapter_preserve_flow_condition():
    b,c,p,d,horizon,patch = 2,3,1,8,25,16
    common,specific = torch.randn(b,c,p,d),torch.randn(b,c,p,d)
    support = torch.rand(b,c,p,1)
    prior = ConditionalFutureInnovationPrior(d,horizon)
    distribution = prior(common,specific,support,torch.rand(d))
    assert distribution["pi_novel"].shape == (b,c,horizon)
    assert bool((distribution["sigma_novel"] > distribution["sigma_normal"]).all())
    residual = torch.randn(b,c,horizon)
    loss = prior.nll(distribution,residual)
    assert torch.isfinite(loss)
    loss.backward()
    assert prior.net[-1].weight.grad.abs().sum() > 0
    condition = torch.randn(b,c,math.ceil(horizon/patch),d)
    adapter = InnovationConditionAdapter(d,patch)
    features = prior.features(distribution)
    assert torch.equal(adapter(condition,features),condition)
    assert torch.equal(adapter(condition,features,enabled=False),condition)
    adapter(condition,features).square().mean().backward()
    assert adapter.projection[-1].weight.grad.abs().sum() > 0


def test_strict_train_time_boundary():
    core,cal,cut = strict_chronological_indices(4000,96,720)
    assert max(i+96+720-1 for i in core) < cut
    assert min(i+96 for i in cal) >= cut
    assert set(core).isdisjoint(cal)
