import torch

from models.predictive_env.prototype_guided_invariant import (
    PrototypeGuidedSharedHead, prototype_h_centers, routing_control)
from models.predictive_env.prototype_invariance_losses import (
    prototype_risks, prototype_risk_penalty, shared_optimum_penalty,
    conditional_agreement)
from models.predictive_env.prototype_mapping_consensus import (
    fit_prototype_mappings, predict_mapping)
from models.predictive_env.prototype_shared_subspace import (
    PrototypeAdversarialHead, PrototypeSharedPrivateHead,
    pairwise_prototype_error_loss)


def test_variable_level_router_and_null():
    torch.manual_seed(1)
    h = torch.randn(8, 3, 2, 12)
    q = torch.rand(8, 3, 4).softmax(-1) * .7
    global_center, centers, info = prototype_h_centers(h, q)
    assert centers.shape == (4,12) and info["effective"].min() > 0
    for kind in ("real","shuffled","random","uniform","time"):
        other = routing_control(q,kind,history_ids=torch.arange(8))
        assert other.shape == q.shape
        torch.testing.assert_close(other.sum(-1),q.sum(-1),atol=1e-6,rtol=1e-6)
    head = PrototypeGuidedSharedHead(2,12,7,rank=4,reference_mode="prototype",prototypes=4)
    head.set_references(global_center, centers)
    assert head(h,q).shape == (8,3,7)


def test_full_future_prototype_losses_backward():
    h=torch.randn(12,2,3,8)
    q=torch.rand(12,2,4).softmax(-1)
    truth=torch.randn(12,2,6)
    head=PrototypeGuidedSharedHead(3,8,6,rank=4)
    pred=head(h,q)
    risk,rho,_=prototype_risks(pred,truth,q)
    assert risk.shape==rho.shape==(4,)
    for kind in ("variance","dro","incremental"):
        penalty,_=prototype_risk_penalty(pred,truth,q,kind,base=torch.zeros_like(pred))
        assert torch.isfinite(penalty)
    for kind in ("norm","variance","irm"):
        penalty,_=shared_optimum_penalty(pred,truth,q,head.head_weight,kind)
        assert torch.isfinite(penalty)
        penalty.backward(retain_graph=True)
    assert torch.isfinite(head.head_weight.grad).all()
    assert torch.isfinite(conditional_agreement(pred,truth,q,h.mean(2)))


def test_shared_mapping_uses_same_h_space():
    torch.manual_seed(3)
    h=torch.randn(14,2,1,5)
    q=torch.rand(14,2,3).softmax(-1)
    truth=torch.randn(14,2,4)
    fitted=fit_prototype_mappings(h,truth,q,ridge=1.0,max_rows=64)
    for kind in ("global","consensus","spectral"):
        output=predict_mapping(h,fitted,kind,0.5)
        assert output.shape==truth.shape and torch.isfinite(output).all()


def test_optional_pairwise_private_and_adversarial_controls():
    h=torch.randn(8,2,3,12)
    q=torch.rand(8,2,4).softmax(-1)
    truth=torch.randn(8,2,5)
    private=PrototypeSharedPrivateHead(3,12,5,rank=4,prototypes=4)
    pred=private(h,q)
    penalty=private.split_penalty(h)+pairwise_prototype_error_loss(
        pred,truth,q,torch.arange(8)*100,10)
    penalty.backward()
    assert torch.isfinite(penalty)
    assert private.private_map.weight.grad is not None
    adversarial=PrototypeAdversarialHead(3,12,5,rank=4,prototypes=4)
    logits=adversarial.adversarial_logits(h)
    assert logits.shape==(8,2,4)
    assert torch.isfinite(logits).all()
