from copy import deepcopy
import sys

import torch

from models.PatchTST_PredictiveEnvIV import Model
from tools.run_predictive_env_iv_patchtst import model_config, parse_args


def _config(experiment="A2", fusion="inv_only", backbone="patchtst"):
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]]
        args = parse_args()
    finally:
        sys.argv = original_argv
    args.backbone = backbone
    args.decomposition_type = "supervised_joint_private"
    args.representation_constraint = "classification"
    args.variant_fusion_mode = fusion
    args.seq_len = 96
    args.pred_len = 24
    args.d_model = 64 if backbone == "moderntcn" else 32
    args.d_ff = 64
    args.n_heads = 2
    args.env_num = 3
    args.dropout = 0.0
    args.enc_in = 3
    return model_config(args, experiment)


def test_environment_supervised_shared_private_gradient_and_checkpoint():
    torch.manual_seed(2021)
    model = Model(_config()).train()
    x = torch.randn(4, 96, 3)
    target = torch.randn(4, 24, 3)
    q = torch.softmax(torch.randn(4, 3), dim=-1)
    output = model.forward_components(x)
    assert output["z_inv_tokens"].shape == output["z_var_tokens"].shape
    assert output["decomposition_gate"].shape == output["z_inv_tokens"].shape
    assert set(output["decomposition_aux_losses"]) == {
        "reconstruction", "decorrelation"
    }
    env_loss, _ = model.environment_classification_loss(
        output["z_inv"], output["z_var"], q
    )
    loss = ((output["invariant_prediction"] - target).square().mean()
            + 0.1 * env_loss
            + sum(output["decomposition_aux_losses"].values()))
    loss.backward()
    for name in ("shared", "private"):
        grad = getattr(model.decomposer, name).weight.grad
        assert grad is not None and torch.isfinite(grad).all()
        assert grad.abs().sum() > 0
    assert not q.requires_grad
    replica = Model(_config())
    replica.load_state_dict(deepcopy(model.state_dict()), strict=True)


def test_a0_output_ignores_new_decomposition():
    model = Model(_config(experiment="A0")).eval()
    with torch.no_grad():
        output = model.forward_components(torch.randn(2, 96, 3))
    assert torch.equal(output["hidden_tokens"], output["z_inv_tokens"])
    assert not output["decomposition_aux_losses"]


def test_backbone_interfaces_and_probability_condition():
    for backbone in ("patchtst", "moderntcn", "itransformer"):
        model = Model(_config(fusion="prob_affine_flow", backbone=backbone)).eval()
        with torch.no_grad():
            output = model.forward_components(torch.randn(2, 96, 3))
        assert output["invariant_prediction"].shape == (2, 24, 3)
        assert output["probabilistic_condition"] is not None
        assert output["z_inv_tokens"].shape == output["z_var_tokens"].shape
