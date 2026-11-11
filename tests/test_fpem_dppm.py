import torch

from models.predictive_env.dual_pattern_memory.prototype_bank import PatternBank, normalized_pairs
from models.predictive_env.dual_pattern_memory.forecast_fusion import (
    stable_forecast, variation_forecast, stable_direction_basis)
from models.predictive_env.dual_pattern_memory.uncertainty_memory import fit_uncertainty_memory
from tools.diagnose_fpem_dppm import purged_source_mask


def test_bank_queries_only_compact_prototypes_and_round_trip():
    torch.manual_seed(3)
    x = torch.randn(32, 2, 8)
    y = torch.randn(32, 2, 11)
    blocks = torch.arange(32) * 4 // 32
    bank = PatternBank(4, 11, 8, ridge=.1)
    bank.fit(x, y, blocks)
    assert bank.centers.shape == (4, 8)
    assert bank.mapping.shape == (4, 9, 11)
    restored = PatternBank.from_state(bank.state())
    one = bank.query(x[:1])
    two = restored.query(x[:1])
    assert one["prediction"].shape == (1, 2, 11)
    assert torch.allclose(one["prediction"], two["prediction"])
    assert not hasattr(bank, "train_features")
    assert not hasattr(bank, "train_target")


def test_history_only_normalization():
    x = torch.ones(2, 5, 3)
    y = torch.full((2, 7, 3), 9.)
    z, mean, scale = normalized_pairs(x, y, torch.full((3,), .1))
    assert z.shape == (2, 3, 7)
    assert torch.allclose(z, torch.full_like(z, 80.))
    assert torch.allclose(mean, torch.ones_like(mean))
    assert torch.isfinite(z).all()


def test_a0_and_null_fallback():
    base = torch.randn(3, 2, 9)
    memory = torch.randn_like(base)
    basis = torch.linalg.qr(torch.randn(9, 3)).Q
    assert torch.equal(stable_forecast(base, memory, basis, 0.), base)
    candidates = torch.randn(3, 2, 4, 9)
    weights = torch.zeros(3, 2, 4)
    rel = torch.ones(4, 3)
    output, beta = variation_forecast(base, candidates, weights, rel, basis,
                                      segment_count=3)
    assert torch.allclose(output, base)
    assert beta.max() == 0


def test_direction_and_uncertainty_are_finite():
    torch.manual_seed(6)
    base = torch.randn(8, 2, 12)
    truth = base + .1
    memory = truth.clone()
    blocks = torch.arange(8) // 4
    basis, audit = stable_direction_basis(memory, base, truth, blocks, 3)
    assert basis.shape[0] == 12
    assert torch.isfinite(basis).all()
    assert audit
    error = torch.randn(8, 2, 12)
    weight = torch.softmax(torch.randn(8, 2, 4), -1)
    stats = fit_uncertainty_memory(error, weight)
    assert stats["center"].shape == (4, 12)
    assert torch.isfinite(stats["mad"]).all()


def test_identical_history_future_mapping_and_linear_trend():
    length, horizon = 80, 13
    t = torch.linspace(-1, 1, length)
    x = t[None,None,:].expand(12,1,-1).clone()
    future = torch.linspace(1, 2, horizon)[None,None,:].expand(12,1,-1).clone()
    bank = PatternBank(2, horizon, length, ridge=.001)
    bank.fit(x, future, torch.arange(12)*4//12)
    pred = bank.query(x[:2])["prediction"]
    assert pred.shape == (2,1,horizon)
    assert torch.isfinite(pred).all()
    assert (pred-future[:2]).square().mean() < .1


def test_conflicting_futures_are_not_falsely_persistent():
    torch.manual_seed(1)
    x = torch.randn(20,1,8)*.001
    future = torch.ones(20,1,9)
    future[10:] = -1
    bank = PatternBank(2,9,8,ridge=.1)
    bank.fit(x,future,torch.arange(20)*4//20)
    pred = bank.query(x[:1])["prediction"]
    assert torch.isfinite(pred).all()
    assert pred.abs().mean() < .5


def test_single_reliable_pattern_and_null():
    base = torch.zeros(2,1,8)
    candidates = torch.stack((torch.ones_like(base),-torch.ones_like(base)),2)
    weight = torch.tensor([[[1.,0.]],[[0.,0.]]])
    reliability = torch.tensor([[1.,1.],[0.,0.]])
    full, beta = variation_forecast(base,candidates,weight,reliability,
                                    torch.zeros(8,0),segment_count=2)
    assert torch.allclose(full[0], torch.ones_like(full[0]))
    assert torch.allclose(full[1], base[1])
    assert float(beta.min()) >= 0. and float(beta.max()) <= 1.


def test_chronological_purge_excludes_future_overlap():
    group = torch.tensor([0,0,1,1,2,2])
    starts = torch.tensor([0,10,20,30,40,50])
    ends = starts + 15
    source = purged_source_mask(group,starts,ends,2)
    assert source.tolist() == [True,True,True,False,False,False]


def test_shared_future_basis_keeps_complete_future_and_checkpoint():
    torch.manual_seed(11)
    x=torch.randn(20,2,6)
    y=torch.randn(20,2,9)
    basis=torch.linalg.qr(torch.randn(9,3)).Q
    bank=PatternBank(3,9,6,ridge=.1)
    bank.fit(x,y,torch.arange(20)*4//20,output_basis=basis)
    assert bank.mapping.shape == (3,7,3)
    prediction=bank.query(x[:2])["prediction"]
    assert prediction.shape == (2,2,9)
    restored=PatternBank.from_state(bank.state())
    assert torch.allclose(prediction,restored.query(x[:2])["prediction"])


def test_next_pilot_feature_modes_and_block_shuffle():
    from tools.run_fpem_dppm_next import history_description,mapping_feature,block_shuffle
    torch.manual_seed(17)
    key=torch.randn(12,2,64)
    truth=torch.randn(12,2,9)
    block=torch.arange(12)*3//12
    stable=PatternBank(3,9,64,ridge=.1)
    stable.fit(key,truth,block)
    for mode in ("V0","V1","V2"):
        desc=history_description(key,stable,mode)
        assert desc.shape==key.shape
        assert torch.isfinite(desc).all()
        for input_mode in ("description","concat","history"):
            assert mapping_feature(key,desc,input_mode).shape==(12,2,128)
    shuffled=block_shuffle(key,block,91)
    assert not torch.equal(shuffled,key)
    for b in block.unique():
        original=key[block==b].flatten(1).sort(0).values
        changed=shuffled[block==b].flatten(1).sort(0).values
        assert torch.allclose(original,changed)
