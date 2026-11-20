import torch

from models.predictive_env.eiil3_mse_priority import priority_step


def test_priority_modes_and_clip():
    torch.manual_seed(3)
    x = torch.randn(32, 5)
    y = torch.randn(32, 4)
    ratio = torch.ones(32, 1)
    q = torch.softmax(torch.randn(32, 3), 1)
    initial = torch.nn.Linear(5, 4)
    for mode in ("V0", "V1", "V2", "V3", "V4", "V5"):
        head = torch.nn.Linear(5, 4)
        head.load_state_dict(initial.state_dict())
        opt = torch.optim.SGD(head.parameters(), lr=1e-4)
        info = priority_step(head, opt, x, y, ratio, q, mode, rho=.05)
        assert all(torch.isfinite(value).all() for value in head.parameters())
        assert torch.isfinite(torch.tensor(info["gradient_cosine"]))
        if mode == "V4":
            assert info["effective_aux_ratio"] <= .050001
        if mode == "V5":
            assert torch.allclose(head.weight.grad,
                (2 / y.numel()) * (initial(x) - y).T @ x, atol=1e-5)


def test_projected_auxiliary_not_opposed_to_mse():
    torch.manual_seed(4)
    x = torch.randn(16, 3)
    y = torch.randn(16, 2)
    q = torch.softmax(torch.randn(16, 3), 1)
    ratio = torch.ones(16, 1)
    head = torch.nn.Linear(3, 2)
    before = (head(x) - y).square().mean().item()
    info = priority_step(head, torch.optim.SGD(head.parameters(), lr=1e-6),
                         x, y, ratio, q, "V4", rho=.1)
    after = (head(x) - y).square().mean().item()
    assert after <= before + 1e-7
    assert info["effective_aux_ratio"] <= .100001
