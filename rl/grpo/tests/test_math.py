import math

import torch

from rl.grpo.math import (
    compute_group_advantages,
    gaussian_transition_log_prob,
    reference_gaussian_kl,
    sde_transition_stats,
)


def test_eq6_matches_hand_calculation_and_is_float32():
    state = torch.tensor([[[1.0, -2.0]]], dtype=torch.float64)
    velocity = torch.tensor([[[0.5, 3.0]]], dtype=torch.float64)
    time = 0.25
    dt = 0.1
    result = sde_transition_stats(state, velocity, time, dt, diffusion=0.4)
    sigma = 0.4 * math.sqrt((1.0 - time) / time)
    expected = state.float() + (
        velocity.float()
        + sigma**2 / (2.0 * (1.0 - time))
        * (-state.float() + time * velocity.float())
    ) * dt
    assert result.mean.dtype == torch.float32
    assert torch.allclose(result.mean, expected)
    assert torch.allclose(result.std, torch.tensor(sigma * math.sqrt(dt)))


def test_gaussian_sum_and_mean_differ_only_by_valid_dimensions():
    mean = torch.zeros(2, 3, 2)
    action = torch.ones_like(mean)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    summed = gaussian_transition_log_prob(
        action, mean, 0.5, mask, reduction="sum_valid"
    )
    averaged = gaussian_transition_log_prob(
        action, mean, 0.5, mask, reduction="mean_valid"
    )
    assert summed.valid_dimensions.tolist() == [4, 2]
    assert torch.allclose(
        averaged.value,
        summed.value / summed.valid_dimensions.to(summed.value.dtype),
    )


def test_padding_is_excluded_from_logprob_and_kl():
    mean = torch.zeros(1, 2, 2, requires_grad=True)
    action = torch.tensor([[[1.0, -1.0], [1000.0, -1000.0]]])
    reference = torch.tensor([[[0.5, 0.5], [999.0, 999.0]]])
    mask = torch.tensor([[True, False]])
    log_prob = gaussian_transition_log_prob(
        action, mean, 1.0, mask, reduction="mean_valid"
    ).value
    kl = reference_gaussian_kl(
        mean, reference, 1.0, mask, reduction="mean_valid"
    )
    assert torch.allclose(kl, torch.tensor([0.125]))
    (log_prob + kl).sum().backward()
    assert torch.equal(mean.grad[:, 1], torch.zeros_like(mean.grad[:, 1]))


def test_group_advantage_population_std_and_zero_std_discard():
    rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0], [7.0, 7.0, 7.0, 7.0]])
    result = compute_group_advantages(rewards, correction=0)
    assert result.valid_groups.tolist() == [True, False]
    assert torch.allclose(result.advantages[0].mean(), torch.tensor(0.0))
    assert torch.allclose(
        result.advantages[0].std(correction=0), torch.tensor(1.0)
    )
    assert torch.equal(result.advantages[1], torch.zeros(4))
    assert not result.eligible_candidates[1].any()
