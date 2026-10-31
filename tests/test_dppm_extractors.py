import torch

from models.predictive_env.dual_pattern_memory.extractors import PatternExtractor
from models.predictive_env.dual_pattern_memory.prototype_bank import PatternBank


def test_extractor_shapes_and_history_only_query():
    generator = torch.Generator().manual_seed(32)
    n, c, horizon = 45, 2, 12
    key = torch.randn(n, c, 64, generator=generator)
    future = torch.randn(n, c, horizon, generator=generator)
    blocks = (torch.arange(n)*5//n).clamp_max(4)
    starts = torch.arange(n)*96
    for mode in ("P0", "P1", "P2", "P3", "P4", "P6", "P7", "P8"):
        ex = PatternExtractor(mode, ms=4, mr=3, seed=7, span=108)
        ex.fit(key, future, future, blocks, starts)
        stable_key = ex.stable_key(key)
        assert stable_key.shape == key.shape
        stable = PatternBank(4, horizon, ridge=.05, seed=7)
        stable.fit(stable_key, future, blocks, preset_centers=ex.stable_centers)
        stable_query = stable.query(ex.stable_key(key[:2]))
        rare_key = ex.rare_key(key[:2], stable_query)
        assert rare_key.shape == (2, c, 64)
        rare = PatternBank(3, horizon, ridge=.05, seed=18)
        train_rare = ex.rare_key(key, stable.query(stable_key))
        rare.fit(train_rare, future, blocks, torch.cat((key,train_rare),-1),
                 preset_centers=ex.rare_centers)
        output = rare.query(rare_key,torch.cat((key[:2],rare_key),-1))
        assert output["prediction"].shape == (2,c,horizon)
        assert torch.isfinite(output["prediction"]).all()


if __name__ == "__main__":
    test_extractor_shapes_and_history_only_query()
    print("extractor synthetic shape/history-only smoke passed")
