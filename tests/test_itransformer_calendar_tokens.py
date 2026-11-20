"""Small-tensor smoke test for the optional iTransformer calendar-token path."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn


def main(model_file: Path, repo_root: Path, fusion_file: Path) -> None:
    sys.path.insert(0, str(repo_root))
    fusion_name = "models.predictive_env.exogenous_token_fusion"
    fusion_spec = importlib.util.spec_from_file_location(fusion_name, fusion_file)
    assert fusion_spec is not None and fusion_spec.loader is not None
    fusion_module = importlib.util.module_from_spec(fusion_spec)
    sys.modules[fusion_name] = fusion_module
    fusion_spec.loader.exec_module(fusion_module)
    spec = importlib.util.spec_from_file_location("staged_predictive_env", model_file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from layers.Embed import DataEmbedding_inverted
    from layers.SelfAttention_Family import AttentionLayer, FullAttention
    from layers.Transformer_EncDec import Encoder, EncoderLayer

    torch.manual_seed(17)
    model = module.Model.__new__(module.Model)
    nn.Module.__init__(model)
    model.backbone = "itransformer"
    model.inverted_embedding = DataEmbedding_inverted(8, 6, dropout=0)
    model.encoder = Encoder(
        [EncoderLayer(
            AttentionLayer(FullAttention(False, attention_dropout=0), 6, 2),
            6, 12, dropout=0, activation="gelu",
        )],
        norm_layer=nn.LayerNorm(6),
    )
    model.eval()
    values = torch.randn(2, 8, 3)
    calendar = torch.randn(2, 8, 4)

    old_tokens, old_mean, old_std = model.encode(values)
    same_tokens, same_mean, same_std = model.encode(values, exog=None)
    assert torch.equal(old_tokens, same_tokens)
    assert torch.equal(old_mean, same_mean)
    assert torch.equal(old_std, same_std)
    manual_old_embedding = model.inverted_embedding(
        (values - old_mean) / old_std, None
    )
    manual_old_encoded, _ = model.encoder(manual_old_embedding, attn_mask=None)
    assert torch.equal(old_tokens, manual_old_encoded[:, :3, :].unsqueeze(2))

    with_tokens, mean, std = model.encode(values, exog=calendar)
    normalized = (values - mean) / std
    manual_embedding = model.inverted_embedding.value_embedding(
        torch.cat((normalized.transpose(1, 2), calendar.transpose(1, 2)), dim=1)
    )
    expected, _ = model.encoder(manual_embedding, attn_mask=None)
    assert with_tokens.shape == (2, 3, 1, 6)
    assert torch.allclose(with_tokens, expected[:, :3, :].unsqueeze(2))
    epoch0_delta = (with_tokens - old_tokens).abs().max().item()
    assert epoch0_delta > 0

    for bad in (torch.randn(2, 7, 4), torch.full((2, 8, 4), float("nan"))):
        try:
            model.encode(values, exog=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid calendar was accepted")

    print(f"iTransformer calendar tokens: PASS; old path unchanged; "
          f"epoch0_max_delta={epoch0_delta:.8f}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: test_itransformer_calendar_tokens.py MODEL_FILE REPO_ROOT FUSION_FILE"
        )
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
