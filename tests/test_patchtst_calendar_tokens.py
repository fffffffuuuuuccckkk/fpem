"""PatchTST periodic token integration and paired epoch-zero identity."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(model_file, repo_root, fusion_file):
    sys.path.insert(0, str(repo_root))
    load_module("models.predictive_env.exogenous_token_fusion", fusion_file)
    model_class = load_module("staged_patchtst", model_file).Model
    cfg = SimpleNamespace(
        pred_len=4, d_model=8, seq_len=16, enc_in=3,
        predictive_env_backbone="patchtst", predictive_env_ablation="A0",
        exog_token_enabled=False, dropout=0.0, factor=1,
        n_heads=2, d_ff=16, e_layers=1, activation="gelu",
        patch_len=4, predictive_env_stride=2,
    )
    torch.manual_seed(17)
    old = model_class(cfg).eval()
    cfg.exog_token_enabled = True
    torch.manual_seed(17)
    new = model_class(cfg).eval()
    for name, weight in old.state_dict().items():
        assert torch.equal(weight, new.state_dict()[name]), name
    values = torch.randn(2, 16, 3)
    calendar = torch.randn(2, 16, 4)
    with torch.no_grad():
        old_y = old.forward_components(values)["prediction"]
        new_y = new.forward_components(values, exog=calendar)["prediction"]
        assert torch.equal(old_y, new_y)
        new.exog_token_fusion.gate.fill_(0.25)
        changed = new.forward_components(values, exog=calendar)["prediction"]
    assert (changed - old_y).abs().max().item() > 0
    try:
        old.encode(values, exog=calendar)
    except ValueError:
        pass
    else:
        raise AssertionError("disabled PatchTST accepted exog")
    print("PatchTST calendar tokens: PASS; epoch-zero exact A0 identity")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: test_patchtst_calendar_tokens.py MODEL REPO FUSION")
    main(*(Path(arg) for arg in sys.argv[1:]))
