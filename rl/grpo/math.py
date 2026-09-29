"""Auditable tensor math for FlowSE-GRPO Eq. (6) and Eq. (8).

The public controlled baseline uses FlowSE time ``0 -> 1`` and performs all
transition statistics in float32.  Gaussian reductions always exclude padded
mel frames and include the channel dimension in the valid-element count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch


GaussianReduction = Literal["sum_valid", "mean_valid"]


@dataclass(frozen=True)
class SDETransitionStats:
    """Mean and standard deviation of one public Eq. (6) transition."""

    mean: torch.Tensor
    std: torch.Tensor
    sigma: torch.Tensor


@dataclass(frozen=True)
class GaussianLogProb:
    """Reduced and unreduced log density for a batch of transitions."""

    value: torch.Tensor
    sum_valid: torch.Tensor
    valid_dimensions: torch.Tensor


@dataclass(frozen=True)
class GroupAdvantageResult:
    """Eq. (8) advantages plus explicit zero-variance group eligibility."""

    advantages: torch.Tensor
    group_mean: torch.Tensor
    group_std: torch.Tensor
    valid_groups: torch.Tensor
    eligible_candidates: torch.Tensor


def _batch_time_factor(
    value: float | torch.Tensor,
    reference: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    """Return a float32 scalar or ``[B,1,...]`` factor for ``reference``."""

    factor = torch.as_tensor(value, device=reference.device, dtype=torch.float32)
    if factor.ndim == 0:
        return factor
    if factor.ndim != 1 or factor.shape[0] != reference.shape[0]:
        raise ValueError(f"{name} must be scalar or have shape [batch]")
    return factor.reshape(factor.shape[0], *([1] * (reference.ndim - 1)))


def sde_transition_stats(
    state: torch.Tensor,
    velocity: torch.Tensor,
    time: float | torch.Tensor,
    dt: float | torch.Tensor,
    *,
    diffusion: float = 0.4,
) -> SDETransitionStats:
    """Compute FlowSE-GRPO Eq. (6) in the public ``0 -> 1`` direction.

    ``time`` must be strictly inside ``(0, 1)``.  Step zero is therefore never
    a legal SDE-window step.  Casts to float32 intentionally preserve the
    gradient path from ``mean`` back to ``velocity``.
    """

    if state.shape != velocity.shape:
        raise ValueError("state and velocity must have identical shapes")
    if state.ndim < 2:
        raise ValueError("state must include batch and feature dimensions")
    if not math.isfinite(float(diffusion)) or diffusion < 0.0:
        raise ValueError("diffusion must be finite and non-negative")

    state32 = state.float()
    velocity32 = velocity.float()
    time32 = _batch_time_factor(time, state32, name="time")
    dt32 = _batch_time_factor(dt, state32, name="dt")
    if bool(torch.any((time32 <= 0.0) | (time32 >= 1.0)).item()):
        raise ValueError("SDE time must lie strictly inside (0, 1)")
    if bool(torch.any(dt32 <= 0.0).item()):
        raise ValueError("dt must be positive")

    sigma = float(diffusion) * torch.sqrt((1.0 - time32) / time32)
    correction = sigma.square() / (2.0 * (1.0 - time32))
    mean = state32 + (
        velocity32 + correction * (-state32 + time32 * velocity32)
    ) * dt32
    std = sigma * torch.sqrt(dt32)
    return SDETransitionStats(mean=mean, std=std, sigma=sigma)


def _expanded_mask(values: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
    if values.ndim < 2:
        raise ValueError("values must have shape [batch, frames, ...]")
    mask = torch.as_tensor(frame_mask, device=values.device, dtype=torch.bool)
    if mask.ndim == 1 and values.shape[0] == 1 and mask.shape[0] == values.shape[1]:
        mask = mask.unsqueeze(0)
    if mask.ndim != 2 or mask.shape != values.shape[:2]:
        raise ValueError("frame_mask must have shape [batch, frames]")
    for _ in range(values.ndim - 2):
        mask = mask.unsqueeze(-1)
    return mask.expand_as(values)


def masked_reduce(
    values: torch.Tensor,
    frame_mask: torch.Tensor,
    reduction: GaussianReduction,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce per-element values while excluding all padded dimensions."""

    if reduction not in {"sum_valid", "mean_valid"}:
        raise ValueError("reduction must be 'sum_valid' or 'mean_valid'")
    mask = _expanded_mask(values, frame_mask)
    finite_values = torch.where(mask, values, torch.zeros_like(values))
    per_batch_sum = finite_values.reshape(values.shape[0], -1).sum(dim=-1)
    valid_dimensions = mask.reshape(values.shape[0], -1).sum(dim=-1)
    if bool(torch.any(valid_dimensions <= 0).item()):
        raise ValueError("every transition must contain at least one valid mel value")
    if reduction == "sum_valid":
        reduced = per_batch_sum
    else:
        reduced = per_batch_sum / valid_dimensions.to(per_batch_sum.dtype)
    return reduced, per_batch_sum, valid_dimensions


def gaussian_transition_log_prob(
    next_state: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor | float,
    frame_mask: torch.Tensor,
    *,
    reduction: GaussianReduction = "mean_valid",
) -> GaussianLogProb:
    """Log-density of the actually executed Gaussian transition.

    ``next_state`` is detached by construction so gradients can only flow
    through the recomputed current-policy mean (and, if supplied, variance).
    """

    action = next_state.detach().float()
    mean32 = mean.float()
    if action.shape != mean32.shape:
        raise ValueError("next_state and mean must have identical shapes")
    std32 = torch.as_tensor(std, device=mean32.device, dtype=torch.float32)
    try:
        action, mean32, std32 = torch.broadcast_tensors(action, mean32, std32)
    except RuntimeError as exc:
        raise ValueError("std is not broadcastable to transition shape") from exc
    if bool(torch.any(std32 <= 0.0).item()) or not bool(torch.isfinite(std32).all().item()):
        raise ValueError("Gaussian std must be finite and strictly positive")
    log_prob = -0.5 * (
        ((action - mean32) / std32).square()
        + 2.0 * torch.log(std32)
        + math.log(2.0 * math.pi)
    )
    value, total, valid = masked_reduce(log_prob, frame_mask, reduction)
    return GaussianLogProb(value=value, sum_valid=total, valid_dimensions=valid)


def reference_gaussian_kl(
    current_mean: torch.Tensor,
    reference_mean: torch.Tensor,
    std: torch.Tensor | float,
    frame_mask: torch.Tensor,
    *,
    reduction: GaussianReduction = "mean_valid",
) -> torch.Tensor:
    """Same-variance ``KL(current || released-base reference)``."""

    current32 = current_mean.float()
    reference32 = reference_mean.detach().float()
    if current32.shape != reference32.shape:
        raise ValueError("current and reference means must have identical shapes")
    std32 = torch.as_tensor(std, device=current32.device, dtype=torch.float32)
    try:
        current32, reference32, std32 = torch.broadcast_tensors(
            current32, reference32, std32
        )
    except RuntimeError as exc:
        raise ValueError("std is not broadcastable to transition shape") from exc
    if bool(torch.any(std32 <= 0.0).item()) or not bool(torch.isfinite(std32).all().item()):
        raise ValueError("Gaussian std must be finite and strictly positive")
    per_element = (current32 - reference32).square() / (2.0 * std32.square())
    value, _, _ = masked_reduce(per_element, frame_mask, reduction)
    return value


def compute_group_advantages(
    rewards: torch.Tensor,
    *,
    correction: int = 0,
    epsilon: float = 1.0e-8,
) -> GroupAdvantageResult:
    """Compute Eq. (8) independently for every row/group.

    Zero-variance groups are marked ineligible; their placeholder advantages
    are zero and must not enter an optimizer batch.
    """

    values = torch.as_tensor(rewards)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("rewards must have shape [groups, group_size>=2]")
    if not values.is_floating_point():
        values = values.float()
    values = values.float()
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("rewards must be finite")
    if correction < 0 or correction >= values.shape[1]:
        raise ValueError("correction must be in [0, group_size)")
    if not math.isfinite(float(epsilon)) or epsilon < 0.0:
        raise ValueError("epsilon must be finite and non-negative")

    group_mean = values.mean(dim=1)
    group_std = values.std(dim=1, correction=int(correction))
    valid_groups = group_std > float(epsilon)
    safe_std = torch.where(valid_groups, group_std, torch.ones_like(group_std))
    advantages = (values - group_mean[:, None]) / safe_std[:, None]
    advantages = torch.where(valid_groups[:, None], advantages, torch.zeros_like(advantages))
    eligible = valid_groups[:, None].expand_as(advantages).clone()
    return GroupAdvantageResult(
        advantages=advantages,
        group_mean=group_mean,
        group_std=group_std,
        valid_groups=valid_groups,
        eligible_candidates=eligible,
    )
