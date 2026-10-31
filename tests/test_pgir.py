import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.predictive_env.pgir import PGIRMask
from models.predictive_env.ptm import PrototypeBank
from models.predictive_env.conditional_innovation_prior import strict_chronological_indices
from tools.run_fpem_pgir_probabilistic import CoreView, condition_features


def tiny_bank(pred_len=5, k=2):
    bank = PrototypeBank(pred_len, k=k, history_size=8)
    with torch.no_grad():
        bank.U[0] = torch.linspace(-1, 1, 8)
        bank.U[1] = torch.linspace(1, -1, 8)
        bank.V[0] = 0.2
        bank.V[1] = -0.2
        bank.counts[:] = torch.tensor([5., 3.])
        bank.fitted.fill_(True)
    return bank


@pytest.mark.parametrize("shape", [(1, 31, 1), (2, 32, 4), (3, 96, 7)])
def test_identity_binary_and_grad(shape):
    torch.manual_seed(13)
    bank = tiny_bank()
    selector = PGIRMask(16)
    selector.invariant_threshold.fill_(1.)
    x = torch.randn(*shape)
    selected, info = selector(x, bank)
    assert selected.shape == x.shape
    assert info["mask"].shape == (shape[0], shape[2], (shape[1]+15)//16)
    assert torch.equal(selected, x)  # exact A0 at initialization
    assert set(info["mask"].detach().unique().tolist()) <= {0., 1.}
    (selected.square().mean()+0.1*info["probability"].mean()).backward()
    assert selector.net[-1].bias.grad.abs().sum() > 0
    with torch.no_grad():
        selector.net[-1].bias.fill_(-1.)
    changed, info = selector(x, bank)
    assert info["replacement_fraction"] > 0
    assert not torch.equal(changed, x)
    kept, _ = selector(x, bank, mode="P1")
    assert torch.equal(kept, x)


def test_train_only_threshold_and_state_roundtrip():
    bank = tiny_bank()
    class Train:
        set_type = 0
        seq_len = 32
        pred_len = 5
        data_x = np.random.default_rng(4).normal(size=(96, 2)).astype("float32")
        data_y = data_x
        def __len__(self):
            return 96 - self.seq_len - self.pred_len + 1
    train = Train()
    selector = PGIRMask()
    selector.fit_threshold(bank, train, torch.device("cpu"))
    assert torch.isfinite(selector.invariant_threshold)
    clone = PGIRMask()
    clone.load_state_dict(selector.state_dict())
    x = torch.from_numpy(train.data_x[:32][None])
    assert torch.equal(selector(x, bank)[0], clone(x, bank)[0])
    train.set_type = 2
    with pytest.raises(ValueError, match="TRAIN"):
        selector.fit_threshold(bank, train, torch.device("cpu"))


def test_random_same_replacement_count():
    bank = tiny_bank()
    selector = PGIRMask()
    selector.invariant_threshold.fill_(1.)
    with torch.no_grad():
        selector.net[-1].bias.fill_(-1.)
    x = torch.randn(3, 64, 2)
    _, factual = selector(x, bank)
    _, random = selector(x, bank, mode="random",
                         generator=torch.Generator().manual_seed(8))
    assert torch.equal((factual["mask"] == 0).sum(-1),
                       (random["mask"] == 0).sum(-1))


def test_oof_core_and_history_only_condition():
    class Train:
        set_type = 0
        seq_len = 24
        pred_len = 8
        data_x = np.random.default_rng(5).normal(size=(240, 2)).astype("float32")
        data_y = data_x
        def __len__(self):
            return len(self.data_x)-self.seq_len-self.pred_len+1
    train = Train()
    core, calibration, cut = strict_chronological_indices(len(train),24,8)
    assert max(i+24+8-1 for i in core) < min(i+24 for i in calibration)
    view = CoreView(train,cut)
    assert len(view.data_x) == cut
    bank = tiny_bank(pred_len=8)
    selector = PGIRMask(16)
    selector.fit_threshold(bank,view,torch.device("cpu"))
    x = torch.from_numpy(train.data_x[None,:24])
    selected,info = selector(x,bank)
    original = condition_features(x,selected,info)
    train.data_y[-8:] = 9999.0
    changed = condition_features(x,selected,info)
    assert all(torch.equal(a,b) for a,b in zip(original,changed))
