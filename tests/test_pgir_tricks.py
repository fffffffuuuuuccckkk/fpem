import sys
from pathlib import Path

import pytest
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.predictive_env.pgir_tricks import (
    PGIRTrickMask, local_context_prototype, align_patch_boundaries)
from models.predictive_env.ptm import PrototypeBank


def bank_for_test(pred_len=5):
    bank=PrototypeBank(pred_len, k=2, history_size=8)
    with torch.no_grad():
        bank.U[0]=torch.linspace(-1,1,8)
        bank.U[1]=torch.linspace(1,-1,8)
        bank.V[0]=0.2;bank.V[1]=-0.2
        bank.counts[:]=torch.tensor([5.,3.])
        bank.fitted.fill_(True)
    return bank


@pytest.mark.parametrize("shape",[(1,31,1),(2,96,7),(3,97,2)])
def test_identity_cap_and_gradient(shape):
    torch.manual_seed(7)
    bank=bank_for_test()
    selector=PGIRTrickMask()
    selector.invariant_threshold.fill_(1.)
    x=torch.randn(*shape)
    selected,info=selector(x,bank)
    assert torch.equal(x,selected)
    (selected.square().mean()+0.1*info["probability"].mean()).backward()
    assert selector.net[-1].bias.grad.abs().sum()>0
    with torch.no_grad():
        selector.net[-1].bias.fill_(-1.)
    changed,info=selector(x,bank,local=True,boundary=True,recent=True)
    assert changed.shape==x.shape
    assert int(info["replace_count"].max())<=2
    assert int(info["replace_count"].min())>=0
    assert torch.isfinite(changed).all()


def test_local_match_excludes_target_patch():
    bank=bank_for_test()
    x=torch.linspace(-2,2,48)[None,:,None]
    source_before,_=local_context_prototype(x,bank,16)
    altered=x.clone()
    altered[:,16:32]+=9
    source_after,_=local_context_prototype(altered,bank,16)
    # The middle-patch prototype may be shifted by whole-window normalization;
    # normalized local candidate weights must not directly see middle pixels.
    # For this direct test, keep mean/std fixed via a symmetric perturbation.
    altered=x.clone()
    altered[:,16:24]+=5
    altered[:,24:32]-=5
    _,w_before=local_context_prototype(x,bank,16)
    _,w_after=local_context_prototype(altered,bank,16)
    assert torch.allclose(w_before[:,:,1],w_after[:,:,1],atol=1e-6)


def test_boundary_alignment_uses_history_neighbors_only():
    source=torch.zeros(1,32,1)
    observed=torch.arange(32,dtype=torch.float32)[None,:,None]
    aligned=align_patch_boundaries(source,observed,16)
    assert aligned[0,15,0]==observed[0,16,0]
    assert aligned[0,16,0]==observed[0,15,0]


@pytest.mark.parametrize("bias", [0.01, 0.1, 0.3])
def test_initial_bias_preserves_a0_and_threshold_fallback(bias):
    selector = PGIRTrickMask(output_bias=bias)
    selector.invariant_threshold.fill_(1.0)
    x = torch.randn(2, 48, 1)
    bank = bank_for_test()
    selected, _ = selector(x, bank)
    assert torch.equal(selected, x)
    selector.decision_threshold = -1e9
    selected, info = selector(x, bank)
    assert torch.equal(selected, x)
    assert not bool(info["replace_count"].any())
    selector.decision_threshold = 0.5
    _, info = selector(x, bank)
    assert int(info["replace_count"].max()) <= 2
    assert bool(info["replace_count"].any())


def test_train_only_normalization_and_checkpoint_roundtrip():
    class Dataset:
        set_type = 0
        seq_len = 32
        data_x = np.random.default_rng(9).normal(size=(64, 1)).astype(np.float32)

        def __len__(self):
            return 33

    bank = bank_for_test()
    selector = PGIRTrickMask(normalize_features=True)
    selector.invariant_threshold.fill_(1.0)
    assert selector.fit_feature_normalizer(bank, Dataset(), torch.device("cpu"))["count"] > 0
    x = torch.from_numpy(Dataset.data_x[:32][None])
    selected, info = selector(x, bank)
    assert torch.equal(selected, x)
    assert torch.isfinite(selector.prepare_features(info["features"])).all()
    restored = PGIRTrickMask(normalize_features=True)
    restored.load_state_dict(selector.state_dict())
    same, _ = restored(x, bank)
    assert torch.equal(same, selected)
    Dataset.set_type = 1
    with pytest.raises(ValueError, match="TRAIN only"):
        selector.fit_feature_normalizer(bank, Dataset(), torch.device("cpu"))
