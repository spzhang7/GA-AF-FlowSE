import torch
import pytest

from rl.grpo.math import gaussian_transition_log_prob
from rl.grpo.rollout import (
    FlowSEWindowedSDESampler,
    WindowSpec,
    sample_window_spec,
)


def constant_velocity(state, time, mask):
    del time, mask
    return torch.full_like(state, 0.25)


def test_window_sampling_is_deterministic_and_legal():
    first = sample_window_spec(1234)
    second = sample_window_spec(1234)
    assert first == second
    assert 7 <= first.nfe <= 10
    assert 1 <= first.start_step <= 3
    first.validate()


def test_a_zero_strictly_degenerates_to_euler():
    initial = torch.randn(3, 4, 2, generator=torch.Generator().manual_seed(2))
    spec = WindowSpec(nfe=7, start_step=1, window_size=2)
    sampler = FlowSEWindowedSDESampler(diffusion=0.0)
    result = sampler.rollout_group(
        initial,
        frame_mask=torch.ones(3, 4, dtype=torch.bool),
        spec=spec,
        velocity_fn=constant_velocity,
        initial_latent_seeds=[1, 2, 3],
        brownian_seeds=[4, 5, 6],
        retain_full_trajectory=True,
    )
    assert torch.allclose(result.terminal, initial + 0.25)
    assert all(not item.transitions for item in result.trajectories)


def test_fixed_latent_and_brownian_seeds_replay_exactly():
    initial = torch.randn(2, 5, 3, generator=torch.Generator().manual_seed(11))
    kwargs = dict(
        frame_mask=torch.tensor(
            [[True, True, True, False, False], [True, True, True, True, False]]
        ),
        spec=WindowSpec(nfe=7, start_step=1, window_size=2),
        velocity_fn=constant_velocity,
        initial_latent_seeds=[101, 102],
        brownian_seeds=[201, 202],
        retain_full_trajectory=True,
    )
    sampler = FlowSEWindowedSDESampler(diffusion=0.4)
    first = sampler.rollout_group(initial, **kwargs)
    second = sampler.rollout_group(initial, **kwargs)
    assert torch.equal(first.terminal, second.terminal)
    assert len(first.trajectories[0].transitions) == 2
    for left, right in zip(
        first.trajectories[0].transitions,
        second.trajectories[0].transitions,
        strict=True,
    ):
        assert torch.allclose(left.next_state, left.old_mean + left.std * left.epsilon)
        assert torch.equal(left.next_state, right.next_state)
        assert torch.equal(left.old_log_prob, right.old_log_prob)
        replayed = gaussian_transition_log_prob(
            left.next_state,
            left.old_mean,
            left.std,
            kwargs["frame_mask"][0:1],
            reduction="mean_valid",
        )
        assert torch.equal(left.old_log_prob, replayed.value)


def test_padding_does_not_change_mean_valid_old_logprob():
    initial = torch.zeros(1, 4, 1)
    mask = torch.tensor([[True, True, False, False]])
    result = FlowSEWindowedSDESampler(diffusion=0.4).rollout_group(
        initial,
        frame_mask=mask,
        spec=WindowSpec(nfe=7, start_step=1, window_size=2),
        velocity_fn=constant_velocity,
        initial_latent_seeds=[1],
        brownian_seeds=[2],
    )
    assert all(record.valid_dimensions == 2 for record in result.trajectories[0].transitions)


def test_repeated_initial_latent_seed_requires_explicit_diagnostic_opt_in():
    initial = torch.zeros(2, 4, 1)
    kwargs = dict(
        frame_mask=torch.ones(2, 4, dtype=torch.bool),
        spec=WindowSpec(nfe=7, start_step=1, window_size=2),
        velocity_fn=constant_velocity,
        initial_latent_seeds=[11, 11],
        brownian_seeds=[21, 22],
    )
    sampler = FlowSEWindowedSDESampler(diffusion=0.4)
    with pytest.raises(ValueError, match="unique for GRPO groups"):
        sampler.rollout_group(initial, **kwargs)

    result = sampler.rollout_group(
        initial,
        **kwargs,
        allow_repeated_initial_latent_seeds=True,
    )
    assert result.trajectories[0].initial_latent_seed == 11
    assert result.trajectories[1].initial_latent_seed == 11
    assert not torch.equal(result.terminal[0], result.terminal[1])
