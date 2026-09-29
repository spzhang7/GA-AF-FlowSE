import importlib.util

import pytest


torch_missing = importlib.util.find_spec("torch") is None
pytestmark = pytest.mark.skipif(torch_missing, reason="PyTorch is only installed on server")


def test_flowse_clean_prediction_time_direction():
    import torch

    from rl.common.flow_matching import (
        clean_prediction,
        flow_interpolate,
    )

    noise = torch.tensor([[[1.0, -1.0]]])
    clean = torch.tensor([[[5.0, 3.0]]])
    time = torch.tensor([0.25])
    state = flow_interpolate(noise, clean, time)
    velocity = clean - noise
    torch.testing.assert_close(clean_prediction(state, velocity, time), clean)


def test_completed_square_gradient_matches_expanded_objective():
    import torch

    from rl.common.flow_matching import (
        build_completed_square_target,
        completed_square_loss,
    )

    current = torch.randn(3, 4, 2, requires_grad=True)
    rollout = torch.randn_like(current)
    old = torch.randn_like(current)
    reference = torch.randn_like(current)
    advantage = torch.tensor([-0.8, 0.0, 0.9])
    lam = 0.25
    target, curvature = build_completed_square_target(
        rollout, old, reference, advantage, lam
    )
    completed = completed_square_loss(current, target, curvature)
    completed_gradient = torch.autograd.grad(completed, current, retain_graph=True)[0]

    a = advantage[:, None, None]
    gamma = 1.0 - a
    expanded_per_element = (
        a * (current - rollout).square()
        + gamma * (current - old).square()
        + lam * (current - reference).square()
    )
    expanded = expanded_per_element.mean(dim=(1, 2)).mean()
    expanded_gradient = torch.autograd.grad(expanded, current)[0]
    torch.testing.assert_close(completed_gradient, expanded_gradient)
    torch.testing.assert_close(curvature, torch.full_like(curvature, 1.0 + lam))


def test_curvature_margin_is_enforced():
    import torch

    from rl.common.flow_matching import (
        build_completed_square_target,
    )

    value = torch.zeros(1, 2, 1)
    with pytest.raises(ValueError, match="curvature below margin"):
        build_completed_square_target(
            value, value, value, advantage=torch.tensor([-1.0]),
            lambda_reference=0.0, gamma=torch.tensor([0.0])
        )


def test_explicit_euler_matches_torchdiffeq_euler():
    import torch
    from torchdiffeq import odeint

    from rl.common.flow_matching import euler_terminal

    initial = torch.tensor([1.0, -0.5])

    def vector_field(time, state):
        return 0.25 * state + time

    nfe = 10
    expected = odeint(
        vector_field,
        initial,
        torch.linspace(0.0, 1.0, nfe + 1),
        method="euler",
    )[-1]
    actual = euler_terminal(vector_field, initial, nfe)
    torch.testing.assert_close(actual, expected)

