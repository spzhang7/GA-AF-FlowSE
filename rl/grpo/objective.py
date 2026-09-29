"""Plain-ratio PPO/GRPO Eq. (9), without RatioNorm or loss/dt scaling."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GRPOObjectiveOutput:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    reference_kl: torch.Tensor
    ratio: torch.Tensor
    diagnostics: dict[str, float]


def plain_grpo_ratio(
    current_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    *,
    log_ratio_clamp: float | None = 20.0,
) -> torch.Tensor:
    """Return ``exp(logp_current - logp_old)`` with overflow-only protection."""

    if current_log_prob.shape != old_log_prob.shape:
        raise ValueError("current and old log-prob shapes differ")
    log_ratio = current_log_prob - old_log_prob.detach()
    if log_ratio_clamp is not None:
        if not math.isfinite(float(log_ratio_clamp)) or log_ratio_clamp <= 0.0:
            raise ValueError("log_ratio_clamp must be finite and positive")
        log_ratio = log_ratio.clamp(
            -float(log_ratio_clamp), float(log_ratio_clamp)
        )
    return torch.exp(log_ratio)


def grpo_objective(
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    reference_kl_values: torch.Tensor,
    *,
    clip_epsilon: float = 0.2,
    beta: float = 0.0,
    log_ratio_clamp: float | None = 20.0,
) -> GRPOObjectiveOutput:
    """Compute transition -> trajectory -> minibatch Eq. (9) reduction."""

    if current_log_probs.ndim != 2:
        raise ValueError("log probabilities must have shape [trajectories, transitions]")
    if current_log_probs.shape != old_log_probs.shape:
        raise ValueError("current and old log-prob shapes differ")
    if reference_kl_values.shape != current_log_probs.shape:
        raise ValueError("reference KL must match the log-prob shape")
    if advantages.shape != (current_log_probs.shape[0],):
        raise ValueError("advantages must have shape [trajectories]")
    if not 0.0 < float(clip_epsilon) < 1.0:
        raise ValueError("clip_epsilon must lie in (0, 1)")
    if not math.isfinite(float(beta)) or beta < 0.0:
        raise ValueError("beta must be finite and non-negative")
    tensors = (current_log_probs, old_log_probs, advantages, reference_kl_values)
    if not all(bool(torch.isfinite(value).all().item()) for value in tensors):
        raise ValueError("objective inputs must be finite")
    if bool(torch.any(reference_kl_values < -1.0e-7).item()):
        raise ValueError("reference KL cannot be negative")

    raw_log_ratio = current_log_probs - old_log_probs.detach()
    ratio = plain_grpo_ratio(
        current_log_probs,
        old_log_probs,
        log_ratio_clamp=log_ratio_clamp,
    )
    lower = 1.0 - float(clip_epsilon)
    upper = 1.0 + float(clip_epsilon)
    clipped_ratio = ratio.clamp(lower, upper)
    advantage = advantages[:, None]
    unclipped_loss = -advantage * ratio
    clipped_loss = -advantage * clipped_ratio
    transition_policy_loss = torch.maximum(unclipped_loss, clipped_loss)
    policy_loss = transition_policy_loss.mean(dim=1).mean()
    reference_kl = reference_kl_values.mean(dim=1).mean()
    loss = policy_loss + float(beta) * reference_kl

    clipped = (ratio < lower) | (ratio > upper)
    positive = advantage > 0
    negative = advantage < 0
    overflow = (
        torch.zeros_like(raw_log_ratio, dtype=torch.bool)
        if log_ratio_clamp is None
        else raw_log_ratio.abs() > float(log_ratio_clamp)
    )

    def fraction(mask: torch.Tensor) -> float:
        return float(mask.float().mean().detach().item()) if mask.numel() else 0.0

    diagnostics = {
        "loss": float(loss.detach().item()),
        "policy_loss": float(policy_loss.detach().item()),
        "reference_kl": float(reference_kl.detach().item()),
        "weighted_reference_kl": float((float(beta) * reference_kl).detach().item()),
        "ratio_mean": float(ratio.detach().mean().item()),
        "ratio_std": float(ratio.detach().std(correction=0).item()),
        "log_ratio_mean": float(raw_log_ratio.detach().mean().item()),
        "log_ratio_std": float(raw_log_ratio.detach().std(correction=0).item()),
        "log_ratio_abs_max": float(raw_log_ratio.detach().abs().max().item()),
        "approx_kl": float((0.5 * raw_log_ratio.detach().square()).mean().item()),
        "clip_fraction": fraction(clipped),
        "positive_clip_fraction": fraction(clipped & positive.expand_as(clipped)),
        "negative_clip_fraction": fraction(clipped & negative.expand_as(clipped)),
        "overflow_clamp_fraction": fraction(overflow),
    }
    return GRPOObjectiveOutput(
        loss=loss,
        policy_loss=policy_loss,
        reference_kl=reference_kl,
        ratio=ratio,
        diagnostics=diagnostics,
    )
