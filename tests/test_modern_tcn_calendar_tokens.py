"""Small-tensor integration test for ModernTCN calendar-token injection."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(model_file: Path, repo_root: Path, fusion_file: Path,
         modern_file: Path) -> None:
    sys.path.insert(0, str(repo_root))
    load_module("models.predictive_env.exogenous_token_fusion", fusion_file)
    load_module("models.predictive_env.modern_tcn_backbone", modern_file)
    model_module = load_module("staged_predictive_env", model_file)

    cfg = SimpleNamespace(
        pred_len=4, d_model=8, seq_len=16, enc_in=3,
        predictive_env_backbone="moderntcn", predictive_env_ablation="A0",
        exog_token_enabled=True, dropout=0.0, factor=1,
        n_heads=2, d_ff=16, e_layers=1, activation="gelu",
        modern_tcn_patch_size=4, modern_tcn_patch_stride=2,
        modern_tcn_large_size=5, modern_tcn_small_size=3,
    )
    torch.manual_seed(29)
    model = model_module.Model(cfg).eval()
    assert model.modern_tcn_backbone.exog_token_enabled
    assert not hasattr(model.modern_tcn_backbone, "exog_stem")
    assert model.modern_tcn_backbone.exog_fusion.global_embedding.in_features == 16
    values = torch.randn(2, 16, 3)
    calendar = torch.randn(2, 16, 4)

    with torch.no_grad():
        old, old_mean, old_std = model.encode(values)
        empty, empty_mean, empty_std = model.encode(values, exog=None)
        initial, _, _ = model.encode(values, exog=calendar)
    assert torch.equal(old, empty)
    assert torch.equal(old_mean, empty_mean)
    assert torch.equal(old_std, empty_std)
    assert old.shape == (2, 3, model.modern_tcn_backbone.output_tokens(16), 8)
    assert torch.equal(initial, old)  # zero fusion gate preserves the A0 output

    with torch.no_grad():
        model.modern_tcn_backbone.exog_fusion.gate.fill_(0.5)
        changed, _, _ = model.encode(values, exog=calendar)
    delta = (changed - old).abs().max().item()
    assert delta > 0

    for bad in (torch.randn(2, 15, 4), torch.full((2, 16, 4), float("nan"))):
        try:
            model.encode(values, exog=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid ModernTCN exog was accepted")

    cfg.exog_token_enabled = False
    torch.manual_seed(29)
    legacy = model_module.Model(cfg).eval()
    assert not any(k.startswith("exog_") for k in legacy.modern_tcn_backbone.state_dict())
    enabled_state = model.state_dict()
    for name, tensor in legacy.state_dict().items():
        assert torch.equal(tensor, enabled_state[name]), name
    with torch.no_grad():
        # The enabled model's gate was changed above; restore it for this check.
        model.modern_tcn_backbone.exog_fusion.gate.zero_()
        assert torch.equal(
            legacy.forward_components(values)["prediction"],
            model.forward_components(values, exog=calendar)["prediction"],
        )
    try:
        legacy.encode(values, exog=calendar)
    except ValueError:
        pass
    else:
        raise AssertionError("disabled ModernTCN accepted exog")

    print(f"ModernTCN calendar tokens: PASS; legacy no-exog path unchanged; "
          f"enabled_delta={delta:.8f}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: test_modern_tcn_calendar_tokens.py MODEL REPO FUSION MODERN"
        )
    main(*(Path(arg) for arg in sys.argv[1:]))
