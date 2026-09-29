"""FlowSE ``0 -> 1`` Euler/windowed-SDE sampler with exact replay records."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import torch

from .math import gaussian_transition_log_prob, sde_transition_stats


VelocityFunction = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class WindowSpec:
    nfe: int
    start_step: int
    window_size: int = 2

    def validate(self) -> None:
        if self.nfe < 2:
            raise ValueError("nfe must be at least two")
        if self.window_size < 1:
            raise ValueError("window_size must be positive")
        if self.start_step < 1:
            raise ValueError("the SDE window cannot start at singular step zero")
        if self.start_step + self.window_size > self.nfe:
            raise ValueError("the SDE window extends beyond the integration grid")


@dataclass(frozen=True)
class TransitionRecord:
    trajectory_index: int
    step_index: int
    time: float
    dt: float
    state: torch.Tensor
    next_state: torch.Tensor
    old_mean: torch.Tensor
    std: torch.Tensor
    epsilon: torch.Tensor
    old_log_prob: torch.Tensor
    old_log_prob_sum: torch.Tensor
    valid_dimensions: int


@dataclass(frozen=True)
class TrajectoryRollout:
    trajectory_index: int
    initial_latent_seed: int
    brownian_seed: int
    transitions: tuple[TransitionRecord, ...]
    terminal: torch.Tensor


@dataclass(frozen=True)
class GroupRollout:
    spec: WindowSpec
    trajectories: tuple[TrajectoryRollout, ...]
    terminal: torch.Tensor
    full_states: tuple[torch.Tensor, ...] | None


def sample_window_spec(
    seed: int,
    *,
    nfe_minimum: int = 7,
    nfe_maximum: int = 10,
    start_minimum: int = 1,
    start_maximum: int = 3,
    window_size: int = 2,
) -> WindowSpec:
    """Deterministically sample one NFE/window pair for a complete group."""

    if nfe_minimum > nfe_maximum:
        raise ValueError("nfe_minimum cannot exceed nfe_maximum")
    generator = random.Random(int(seed))
    nfe = generator.randint(int(nfe_minimum), int(nfe_maximum))
    legal_maximum = min(int(start_maximum), nfe - int(window_size))
    legal_minimum = max(1, int(start_minimum))
    if legal_minimum > legal_maximum:
        raise ValueError("configuration contains no legal SDE window")
    spec = WindowSpec(
        nfe=nfe,
        start_step=generator.randint(legal_minimum, legal_maximum),
        window_size=int(window_size),
    )
    spec.validate()
    return spec


class FlowSEWindowedSDESampler:
    """Roll out one logical GRPO group under a frozen old policy."""

    def __init__(
        self,
        *,
        diffusion: float = 0.4,
        logprob_reduction: str = "mean_valid",
        offload_records_to_cpu: bool = True,
    ) -> None:
        if diffusion < 0.0:
            raise ValueError("diffusion must be non-negative")
        if logprob_reduction not in {"sum_valid", "mean_valid"}:
            raise ValueError("unsupported log-prob reduction")
        self.diffusion = float(diffusion)
        self.logprob_reduction = str(logprob_reduction)
        self.offload_records_to_cpu = bool(offload_records_to_cpu)

    @staticmethod
    def _stored(tensor: torch.Tensor, *, cpu: bool) -> torch.Tensor:
        value = tensor.detach().clone()
        return value.cpu() if cpu else value

    @staticmethod
    def _brownian_noise(
        state: torch.Tensor, generators: list[torch.Generator]
    ) -> torch.Tensor:
        return torch.stack(
            [
                torch.randn(
                    state[index].shape,
                    generator=generators[index],
                    device=state.device,
                    dtype=torch.float32,
                )
                for index in range(state.shape[0])
            ],
            dim=0,
        )

    def rollout_group(
        self,
        initial_states: torch.Tensor,
        *,
        frame_mask: torch.Tensor,
        spec: WindowSpec,
        velocity_fn: VelocityFunction,
        initial_latent_seeds: list[int],
        brownian_seeds: list[int],
        retain_full_trajectory: bool = False,
        allow_repeated_initial_latent_seeds: bool = False,
    ) -> GroupRollout:
        """Generate candidates sharing NFE/window with audited randomness.

        Production GRPO groups require unique initial latent seeds.  The explicit
        opt-in is reserved for matched-latent diagnostics in which several
        Brownian continuations deliberately start from the exact same latent.
        """

        spec.validate()
        if initial_states.ndim != 3:
            raise ValueError("initial_states must have shape [group, frames, channels]")
        group = initial_states.shape[0]
        if len(initial_latent_seeds) != group:
            raise ValueError("initial latent seeds must match group size")
        if (
            not allow_repeated_initial_latent_seeds
            and len(set(initial_latent_seeds)) != group
        ):
            raise ValueError("initial latent seeds must be unique for GRPO groups")
        if len(brownian_seeds) != group or len(set(brownian_seeds)) != group:
            raise ValueError("Brownian seeds must be unique and match group size")
        mask = torch.as_tensor(frame_mask, device=initial_states.device, dtype=torch.bool)
        if mask.shape != initial_states.shape[:2]:
            raise ValueError("frame_mask must have shape [group, frames]")

        state = initial_states.detach().float().clone()
        dt = 1.0 / spec.nfe
        generators = []
        for seed in brownian_seeds:
            generator = torch.Generator(device=state.device)
            generator.manual_seed(int(seed))
            generators.append(generator)
        per_trajectory: list[list[TransitionRecord]] = [[] for _ in range(group)]
        full_states = [self._stored(state, cpu=self.offload_records_to_cpu)] if retain_full_trajectory else None

        with torch.no_grad():
            for step in range(spec.nfe):
                time_value = step / spec.nfe
                time = torch.full(
                    (group,), time_value, device=state.device, dtype=torch.float32
                )
                velocity = velocity_fn(state, time, mask)
                if velocity.shape != state.shape:
                    raise ValueError("velocity_fn returned the wrong shape")
                stochastic = (
                    self.diffusion > 0.0
                    and spec.start_step <= step < spec.start_step + spec.window_size
                )
                if stochastic:
                    stats = sde_transition_stats(
                        state, velocity, time, dt, diffusion=self.diffusion
                    )
                    epsilon = self._brownian_noise(state, generators)
                    next_state = stats.mean + stats.std * epsilon
                    log_prob = gaussian_transition_log_prob(
                        next_state,
                        stats.mean,
                        stats.std,
                        mask,
                        reduction=self.logprob_reduction,
                    )
                    for index in range(group):
                        per_trajectory[index].append(
                            TransitionRecord(
                                trajectory_index=index,
                                step_index=step,
                                time=float(time_value),
                                dt=float(dt),
                                state=self._stored(
                                    state[index : index + 1],
                                    cpu=self.offload_records_to_cpu,
                                ),
                                next_state=self._stored(
                                    next_state[index : index + 1],
                                    cpu=self.offload_records_to_cpu,
                                ),
                                old_mean=self._stored(
                                    stats.mean[index : index + 1],
                                    cpu=self.offload_records_to_cpu,
                                ),
                                std=self._stored(
                                    stats.std[index : index + 1]
                                    if stats.std.ndim > 0
                                    else stats.std,
                                    cpu=self.offload_records_to_cpu,
                                ),
                                epsilon=self._stored(
                                    epsilon[index : index + 1],
                                    cpu=self.offload_records_to_cpu,
                                ),
                                old_log_prob=self._stored(
                                    log_prob.value[index : index + 1],
                                    cpu=self.offload_records_to_cpu,
                                ),
                                old_log_prob_sum=self._stored(
                                    log_prob.sum_valid[index : index + 1],
                                    cpu=self.offload_records_to_cpu,
                                ),
                                valid_dimensions=int(
                                    log_prob.valid_dimensions[index].item()
                                ),
                            )
                        )
                else:
                    next_state = state + velocity.float() * dt
                state = next_state.detach()
                if full_states is not None:
                    full_states.append(
                        self._stored(state, cpu=self.offload_records_to_cpu)
                    )

        trajectories = tuple(
            TrajectoryRollout(
                trajectory_index=index,
                initial_latent_seed=int(initial_latent_seeds[index]),
                brownian_seed=int(brownian_seeds[index]),
                transitions=tuple(per_trajectory[index]),
                terminal=self._stored(
                    state[index], cpu=self.offload_records_to_cpu
                ),
            )
            for index in range(group)
        )
        if self.diffusion > 0.0 and any(
            len(item.transitions) != spec.window_size for item in trajectories
        ):
            raise AssertionError("each trajectory must retain the complete SDE window")
        terminal = self._stored(state, cpu=self.offload_records_to_cpu)
        return GroupRollout(
            spec=spec,
            trajectories=trajectories,
            terminal=terminal,
            full_states=(tuple(full_states) if full_states is not None else None),
        )
