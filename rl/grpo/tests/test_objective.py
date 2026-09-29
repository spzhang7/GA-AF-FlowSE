import torch

from rl.grpo.objective import grpo_objective, plain_grpo_ratio


def test_preupdate_plain_ratio_is_one():
    log_prob = torch.tensor([[-2.0, -3.0], [-1.0, -4.0]])
    assert torch.equal(plain_grpo_ratio(log_prob, log_prob), torch.ones_like(log_prob))


def test_ppo_clipping_matches_hand_calculation_without_dt_scaling():
    old = torch.zeros(2, 2)
    ratio = torch.tensor([[1.5, 1.0], [0.5, 1.0]])
    current = ratio.log().requires_grad_()
    advantages = torch.tensor([1.0, -1.0])
    output = grpo_objective(
        current,
        old,
        advantages,
        torch.zeros_like(old),
        clip_epsilon=0.2,
        beta=0.0,
    )
    # Positive A clips high ratio to 1.2; negative A clips low ratio to 0.8.
    expected = torch.tensor((-1.2 - 1.0 + 0.8 + 1.0) / 4.0)
    assert torch.allclose(output.policy_loss, expected)
    assert output.diagnostics["clip_fraction"] == 0.5
    output.loss.backward()
    assert torch.isfinite(current.grad).all()


def test_reference_kl_is_added_with_beta():
    current = torch.zeros(1, 2, requires_grad=True)
    kl = torch.tensor([[2.0, 4.0]])
    output = grpo_objective(
        current,
        torch.zeros_like(current),
        torch.tensor([0.0]),
        kl,
        beta=0.25,
    )
    assert torch.allclose(output.reference_kl, torch.tensor(3.0))
    assert torch.allclose(output.loss, torch.tensor(0.75))


def test_gradient_increases_positive_and_decreases_negative_likelihood():
    current = torch.zeros(2, 1, requires_grad=True)
    output = grpo_objective(
        current,
        torch.zeros_like(current),
        torch.tensor([1.0, -1.0]),
        torch.zeros_like(current),
        beta=0.0,
    )
    output.loss.backward()
    # Gradient descent raises current[0] and lowers current[1].
    assert current.grad[0].item() < 0.0
    assert current.grad[1].item() > 0.0
