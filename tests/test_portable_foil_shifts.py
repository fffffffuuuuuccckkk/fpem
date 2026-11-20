import unittest
from types import SimpleNamespace

import torch
from torch import nn

from models.PatchTST_PredictiveEnvIV import Model
from models.predictive_env.portable_foil_shifts import (
    FOILAdapter, ShifTSAdapter, train_only_aligned_windows,
)


class TinyBackbone(nn.Module):
    def __init__(self, channels=3, seq_len=8, pred_len=5, width=4):
        super().__init__()
        self.embed = nn.Linear(seq_len, width)
        self.head = nn.Linear(width, pred_len)

    def encode(self, x):
        mean = x.mean(1, keepdim=True).detach()
        std = x.var(1, keepdim=True, unbiased=False).add(1e-5).sqrt()
        token = self.embed(((x - mean) / std).transpose(1, 2)).unsqueeze(2)
        return token, mean, std

    def forecast_head(self, token):
        return self.head(token.squeeze(2))


class PortableBaselinesTest(unittest.TestCase):
    def test_itransformer_backbone_contract(self):
        cfg = SimpleNamespace(
            seq_len=8, pred_len=5, d_model=16, d_ff=32, n_heads=2,
            e_layers=1, dropout=0.0, factor=3, activation="gelu", enc_in=3,
            predictive_env_backbone="itransformer", predictive_env_ablation="A0",
            variant_fusion_mode="legacy", predictive_env_num=3,
        )
        x = torch.randn(2, 8, 3)
        y = torch.randn(2, 5, 3)
        foil = FOILAdapter(Model(cfg), d_model=16, pred_len=5, env_num=3)
        self.assertEqual(tuple(foil.predict(x).shape), (2, 5, 3))
        self.assertEqual(tuple(foil.environment_predictions(x).shape), (2, 3, 5, 3))
        self.assertTrue(bool(torch.isfinite(foil.environment_head_loss(x, y))))
        shifts = ShifTSAdapter(Model(cfg), seq_len=8, channels=3)
        aligned = train_only_aligned_windows(torch.randn(20, 3), torch.arange(2), 8, 5)
        prediction, aux = shifts.predict_with_aux(x, aligned)
        self.assertEqual(tuple(prediction.shape), (2, 5, 3))
        self.assertTrue(bool(torch.isfinite(aux)))

    def test_foil_shapes_train_only_assignment_and_gradients(self):
        torch.manual_seed(7)
        model = FOILAdapter(TinyBackbone(), d_model=4, pred_len=5, env_num=3)
        x = torch.randn(6, 8, 3)
        y = torch.randn(6, 5, 3)
        self.assertEqual(tuple(model.predict(x).shape), (6, 5, 3))
        self.assertEqual(tuple(model.environment_predictions(x).shape), (6, 3, 5, 3))
        labels = model.infer_environment(x, y)
        self.assertEqual(tuple(labels.shape), (6,))
        env_loss = model.environment_head_loss(x, y)
        env_loss.backward()
        self.assertIsNotNone(model.env_heads.grad)
        model.zero_grad(set_to_none=True)
        final, _ = model.final_loss(x, y, labels)
        self.assertTrue(bool(torch.isfinite(final)))
        final.backward()
        self.assertIsNotNone(model.backbone.embed.weight.grad)
        self.assertIsNone(model.env_heads.grad)

    def test_shifts_training_only_windows_and_test_independence(self):
        torch.manual_seed(11)
        values = torch.randn(30, 3)
        ids = torch.tensor([0, 2, 17])
        aligned = train_only_aligned_windows(values, ids, seq_len=8, pred_len=5)
        self.assertEqual(tuple(aligned.shape), (3, 9, 5, 3))
        self.assertTrue(torch.equal(aligned[0, 8], values[8:13]))
        with self.assertRaises(ValueError):
            train_only_aligned_windows(values, torch.tensor([18]), 8, 5)
        model = ShifTSAdapter(TinyBackbone(), seq_len=8, channels=3)
        x = torch.randn(3, 8, 3)
        pred, aux = model.predict_with_aux(x, aligned)
        self.assertEqual(tuple(pred.shape), (3, 5, 3))
        self.assertTrue(bool(torch.isfinite(aux)))
        (pred.square().mean() + aux).backward()
        self.assertIsNotNone(model.mask.grad)
        self.assertIsNotNone(model.backbone.embed.weight.grad)
        # Inference never accepts a teacher; only historical x is passed.
        self.assertTrue(torch.allclose(model.predict(x), model.predict_with_aux(x)[0]))


if __name__ == "__main__":
    unittest.main()
