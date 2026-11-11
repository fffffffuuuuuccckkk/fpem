import torch

from models.predictive_env.predictive_memory import PredictiveMemory
from models.predictive_env.retrieval_guided_adapter import RetrievalGuidedAdapter


def records(seed=11):
    g = torch.Generator().manual_seed(seed)
    n, length, channels, patches, width, horizon, envs = 20, 16, 3, 4, 12, 5, 3
    x = torch.randn(n, length, channels, generator=g)
    h = torch.randn(n, channels, patches, width, generator=g)
    r = torch.randn(n, channels, horizon, generator=g)
    q = torch.softmax(torch.randn(n, envs, generator=g), -1)
    starts = torch.arange(n)*length
    return x, h, r, q, starts


def test_oof_response_and_hidden_purge():
    x, h, r, q, starts = records()
    bank = PredictiveMemory(4, 8, 5, 12, 3)
    bank.fit_keys(x)
    info = bank.fit_values(x, h, r, q, starts, forbidden_interval=(64, 128))
    changed_r, changed_h = r.clone(), h.clone()
    # Starts 3..7 have full intervals intersecting the forbidden interval.
    forbidden = (starts < 128) & (starts + 21 > 64)
    changed_r[forbidden] += 10000
    changed_h[forbidden] += 10000
    other = PredictiveMemory(4, 8, 5, 12, 3)
    other.keys.copy_(bank.keys)
    other.fit_values(x, changed_h, changed_r, q, starts,
                     forbidden_interval=(64, 128))
    assert info["source_anchors"] == int((~forbidden).sum())
    for name in ("response", "env_delta", "hidden_value", "within", "between"):
        torch.testing.assert_close(getattr(bank, name), getattr(other, name))


def test_history_only_retrieval_and_identity():
    x, h, r, q, starts = records()
    bank = PredictiveMemory(4, 8, 5, 12, 3)
    bank.fit_keys(x)
    bank.fit_values(x, h, r, q, starts)
    adapter = RetrievalGuidedAdapter(12, 3)
    adapter.eval()
    query1 = bank.query(x[:2])
    r[0] += 999  # No fit: TEST/VAL future cannot alter an already-frozen bank.
    query2 = bank.query(x[:2])
    torch.testing.assert_close(query1["env_response"], query2["env_response"])
    bypass = adapter(h[:2], None)
    torch.testing.assert_close(bypass["h_fused"], h[:2], rtol=0, atol=0)
    out = adapter(h[:2], query1, "P3")
    assert out["h_fused"].shape == h[:2].shape
    assert torch.isfinite(out["h_fused"]).all()
    out["h_fused"].square().mean().backward()
    assert any(p.grad is not None for p in adapter.parameters())


def test_fallback_zero_support_and_non_divisible_horizon():
    x, h, r, q, starts = records()
    bank = PredictiveMemory(4, 8, 5, 12, 3)
    bank.fit_keys(x); bank.fit_values(x, h, r, q, starts)
    bank.support.zero_(); bank.env_support.zero_()
    query = bank.query(x[:1])
    assert torch.equal(query["c_inv"], torch.zeros_like(query["c_inv"]))
    assert torch.equal(query["c_var"], torch.zeros_like(query["c_var"]))


if __name__ == "__main__":
    test_oof_response_and_hidden_purge()
    test_history_only_retrieval_and_identity()
    test_fallback_zero_support_and_non_divisible_horizon()
    print("predictive-memory unit tests passed")
